"""No-add masking, objective compatibility, checkpoint metadata and reporting."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from analysis import aggregate_experiments as aggregate
from scripts import train_refiner as runner


class NoAddPenaltyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_lambda_preserves_loss_gradients_rng_and_skips_sigmoid(self):
        torch.manual_seed(21)
        old_outputs = [torch.randn(2, 1, 3, 3, requires_grad=True) for _ in range(2)]
        outputs = [value.detach().clone().requires_grad_() for value in old_outputs]
        targets = torch.randint(0, 2, (2, 3, 3)).float()
        valid = torch.ones(2, 3, 3, dtype=torch.bool)
        valid[0, 0, 0] = False
        expected, old_steps = runner.deep_supervision(old_outputs, targets, valid)
        expected_gradients = torch.autograd.grad(expected, old_outputs)
        rng = torch.get_rng_state().clone()
        with patch.object(runner, 'noadd_penalties', side_effect=AssertionError('Penalty must be skipped')), \
                patch.object(torch.Tensor, 'sigmoid', side_effect=AssertionError('No extra sigmoid')):
            total, steps, bce, penalty, penalties = runner.refiner_objective(
                outputs, targets, valid, object(), noadd_lambda=0.)
        self.assertIs(total, bce)
        self.assertTrue(torch.equal(total, expected))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(steps, old_steps)))
        self.assertEqual((penalty, penalties), (0., [0., 0.]))
        gradients = torch.autograd.grad(total, outputs)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(gradients, expected_gradients)))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_mask_excludes_road_ignore_and_initial_probability_at_threshold(self):
        raw = torch.tensor([[[1, 3], [0, 4]]])
        targets, valid = (raw == 3).double(), raw != 0
        initial = torch.tensor([[[[.25, .25], [.25, .5]]]], dtype=torch.float64)
        draft = torch.logit(initial).requires_grad_()
        answer = torch.logit(torch.tensor([[[[.5, .6], [.7, .8]]]], dtype=torch.float64)).requires_grad_()
        penalty, _ = runner.noadd_penalties([answer], draft, targets, valid)
        self.assertAlmostEqual(penalty.item(), (.5-.25)**2 / 3, places=14)
        penalty.backward()
        self.assertAlmostEqual(answer.grad[0, 0, 0, 0].item(), 1/24, places=14)
        self.assertTrue(torch.equal(answer.grad.flatten()[1:], torch.zeros(3, dtype=torch.float64)))
        self.assertIsNone(draft.grad)

    def test_equal_or_lower_probability_has_zero_penalty_and_gradient(self):
        draft = torch.logit(torch.full((1, 1, 2, 2), .25, dtype=torch.float64)).requires_grad_()
        values = torch.tensor([[[[.25, .1], [.2, .05]]]], dtype=torch.float64)
        answer = torch.logit(values).requires_grad_()
        penalty, _ = runner.noadd_penalties([answer], draft, torch.zeros(1, 2, 2),
                                           torch.ones(1, 2, 2, dtype=torch.bool))
        self.assertEqual(penalty.item(), 0.)
        penalty.backward()
        self.assertTrue(torch.equal(answer.grad, torch.zeros_like(answer)))
        self.assertIsNone(draft.grad)

    def test_hand_computed_two_step_penalty_and_weighted_objective(self):
        raw = torch.tensor([[[1, 3], [0, 4]]])
        targets, valid = (raw == 3).double(), raw != 0
        draft = torch.logit(torch.tensor([[[[.25, .25], [.25, .5]]]], dtype=torch.float64))
        first = torch.zeros_like(draft, requires_grad=True)
        second = torch.full_like(draft, torch.logit(torch.tensor(.75, dtype=torch.float64)).item(), requires_grad=True)
        expected_steps = ((.5-.25)**2 / 3, (.75-.25)**2 / 3)
        total, steps, bce, penalty, penalties = runner.refiner_objective(
            [first, second], targets, valid, draft, noadd_lambda=10.)
        for observed, expected in zip(penalties, expected_steps):
            self.assertAlmostEqual(observed.item(), expected, places=14)
        self.assertAlmostEqual(penalty.item(), sum(expected_steps)/2, places=14)
        self.assertTrue(torch.equal(bce, torch.stack(steps).mean()))
        self.assertTrue(torch.equal(total, bce+10*penalty))

    def test_zero_output_and_all_ignored_masks_have_exactly_zero_penalty(self):
        draft = torch.randn(2, 1, 4, 4)
        answer = draft.clone().requires_grad_()
        for valid in (torch.ones(2, 4, 4, dtype=torch.bool), torch.zeros(2, 4, 4, dtype=torch.bool)):
            penalty, steps = runner.noadd_penalties([answer, answer], draft, torch.zeros(2, 4, 4), valid)
            self.assertEqual(penalty.item(), 0.)
            self.assertTrue(all(value.item() == 0 for value in steps))

    def test_lambda_validation_and_checkpoint_restoration(self):
        self.assertEqual(runner.build_parser().parse_args([]).noadd_lambda, 0.)
        for value in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                runner.validate_noadd_lambda(value)
        original = json.loads((aggregate.ROOT/'results/exp/e0_v2_s0/metrics.json').read_text())

        class StopBeforeData(RuntimeError):
            pass

        for mode in ('--eval-only', '--resume'):
            for saved_value in (None, 10.):
                config = copy.deepcopy(original['config'])
                if saved_value is not None:
                    config['noadd_lambda'] = saved_value
                checkpoint = dict(state_dict={}, config=config, epoch=1)
                argv = ['train_refiner.py', mode]
                if mode == '--eval-only':
                    argv += ['--ckpt', 'runs/noadd_trial/mock.pt']
                captured = []

                def validate(value):
                    captured.append(value)
                    raise StopBeforeData

                with patch.object(sys, 'argv', argv), patch.object(Path, 'is_file', return_value=True), \
                        patch.object(torch, 'load', return_value=checkpoint), \
                        patch.object(runner, 'validate_noadd_lambda', side_effect=validate):
                    with self.assertRaises(StopBeforeData):
                        runner.main()
                self.assertEqual(captured, [saved_value or 0.])


class NoAddAggregateTests(unittest.TestCase):
    def test_previous_summary_bytes_and_appendix_are_preserved_in_memory(self):
        captured = {}

        def capture(path, text, **kwargs):
            captured[path.name] = text

        with patch.object(sys, 'argv', ['aggregate_experiments.py']), patch('builtins.print'), \
                patch.object(Path, 'mkdir'), patch.object(Path, 'write_text', autospec=True, side_effect=capture):
            aggregate.main()
        old_md = (aggregate.ROOT/'results/exp/summary.md').read_text()
        # Once e5 is committed, compare the complete report, including e5 itself.
        already_reported = '| e5_noadd |' in old_md
        new_md = captured['summary.md'] if already_reported else ''.join(
            line for line in captured['summary.md'].splitlines(keepends=True)
            if not line.startswith('| e5_noadd'))
        self.assertEqual(new_md, old_md)
        self.assertIn('## Untied vs shared', new_md)
        new = json.loads(captured['summary.json'])
        if not already_reported:
            for key in ('rows', 'table1'):
                new[key] = [row for row in new[key] if not row['experiment'].startswith('e5_noadd')]
        self.assertEqual(json.dumps(new, indent=2, allow_nan=False)+'\n',
                         (aggregate.ROOT/'results/exp/summary.json').read_text())

    def test_noadd_supports_k6_k8_and_rerun_is_excluded(self):
        data = json.loads((aggregate.ROOT/'results/exp/e0_v2_s0/metrics.json').read_text())
        data['config']['noadd_lambda'] = 10.
        with patch.object(Path, 'read_text', return_value=json.dumps(data)):
            row = aggregate.read_run('e5_noadd_s0', Path('metrics.json'))
        self.assertEqual(row['status'], 'complete')
        for k in (6, 8):
            self.assertEqual(row[f'A_K{k}'], data['evaluation']['groups']['Overall'][str(k)]['scores']['A'])
        with patch.object(Path, 'glob', return_value=[Path('e0_v2_s0_rerun/metrics.json')]), \
                patch.object(Path, 'read_text', side_effect=FileNotFoundError):
            result = aggregate.aggregate_directory(Path('absent'))
        self.assertEqual(len(result['rows']), 4*len(aggregate.EXPERIMENTS))
        self.assertFalse(any('rerun' in row['experiment'] for row in result['rows']))


if __name__ == '__main__':
    unittest.main()
