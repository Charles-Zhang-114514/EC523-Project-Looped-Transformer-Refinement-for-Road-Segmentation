"""Finite-stack routing, exact shared compatibility and additive reporting."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from analysis import aggregate_experiments as aggregate
from looproad.models.refiner import LoopRefiner
from scripts import train_refiner as runner


class UntiedRefinerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def small(**kwargs):
        return LoopRefiner(8, 16, heads=4, image_size=32, **kwargs)

    def test_shared_initialization_rng_and_nonzero_trajectory_match_main(self):
        # Recorded on main 6186cb2 before editing; no tensor fixture is committed.
        references = {
            'patch_linear': (
                '56e181f91d4be9ff23bd65a1b27b639a405e294b5e3ee0ece0cb1ad9fda2bb7f',
                '5b41f7dd8750c9408a00bb3558b0ae2b56260df79819d7402722f3be8695b9e0',
                '50cda0363a15b28867b1a26ee40982786eb93080dab2af0cba5cd7897b871b07'),
            'conv_upsample': (
                '68f260aee3ae528ad45fdfe39e91a9c37d1e6d40b0e59eb163f82f343512b21b',
                '1cf445d3595824df46ab28f02acfdfe86fe415359899ba674d7c8474545897b9',
                'cc2fd367f719f5482d06d403beba60a941a20cf9dc4ee92b039411e66e524a12')}
        for head, expected in references.items():
            for explicit in ({}, {'untied_steps': 0}):
                with self.subTest(head=head, explicit=explicit):
                    torch.manual_seed(1701)
                    model = self.small(head_type=head, **explicit)
                    state_digest = hashlib.sha256()
                    for key, value in model.state_dict().items():
                        state_digest.update(key.encode())
                        state_digest.update(value.numpy().tobytes())
                    self.assertEqual(state_digest.hexdigest(), expected[0])
                    self.assertEqual(hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(), expected[1])
                    features, draft = torch.randn(2, 8, 2, 2), torch.randn(2, 1, 32, 32)
                    last = model.output_head[-1] if head == 'patch_linear' else model.output_head.final
                    torch.nn.init.normal_(last.weight, std=.02)
                    state, tokens = model.init(features, draft)
                    trajectory = hashlib.sha256()
                    for _ in range(4):
                        state, draft = model.step(state, draft, tokens)
                        trajectory.update(state.detach().numpy().tobytes())
                        trajectory.update(draft.detach().numpy().tobytes())
                    self.assertEqual(trajectory.hexdigest(), expected[2])

    def test_independent_cores_route_in_order_reset_and_fail_on_step_five(self):
        torch.manual_seed(7)
        model = self.small(untied_steps=4)
        identities = [{id(p) for p in core.parameters()} for core in model.core]
        for i in range(4):
            for j in range(i):
                self.assertFalse(identities[i] & identities[j])
                self.assertFalse(torch.equal(model.core[i][0].qkv.weight, model.core[j][0].qkv.weight))
        seen = []
        hooks = [core.register_forward_hook(lambda *_, index=i: seen.append(index))
                 for i, core in enumerate(model.core)]
        features, draft = torch.randn(2, 8, 2, 2), torch.randn(2, 1, 32, 32)
        with self.assertRaisesRegex(RuntimeError, 'Call init'):
            model.step(torch.zeros(2, 4, 16), draft, torch.zeros(2, 4, 16))
        state, tokens = model.init(features, draft)
        for _ in range(4):
            state, answer = model.step(state, draft, tokens)
            self.assertTrue(torch.equal(answer, draft))
        self.assertEqual(seen, [0, 1, 2, 3])
        with self.assertRaisesRegex(RuntimeError, 'step 5 exceeds untied_steps=4'):
            model.step(state, draft, tokens)
        self.assertEqual(seen, [0, 1, 2, 3])
        state, tokens = model.init(features, draft)
        model.step(state, draft, tokens)
        self.assertEqual(seen, [0, 1, 2, 3, 0])
        for hook in hooks:
            hook.remove()

    def test_zeroing_one_core_only_changes_its_isolated_step(self):
        model = self.small(untied_steps=4)
        altered = copy.deepcopy(model)
        for parameter in altered.core[2].parameters():
            parameter.data.zero_()
        features, draft = torch.randn(2, 8, 2, 2), torch.randn(2, 1, 32, 32)
        state, tokens = model.init(features, draft)
        altered.init(features, draft)
        # Use identical input state at each depth to isolate routing from propagation.
        for depth in range(4):
            expected, _ = model.step(state, draft, tokens)
            observed, _ = altered.step(state, draft, tokens)
            self.assertEqual(torch.equal(observed, expected), depth != 2)

    def test_parameter_budget_and_four_step_gradients(self):
        model = LoopRefiner(head_type='conv_upsample', untied_steps=4)
        self.assertEqual(model.parameter_counts(), {
            'core': 3_159_040, 'feature_projection': 262_400,
            'answer_projection': 65_792, 'initializer': 65_792,
            'output_head': 9_281, 'trainable_total': 3_562_305})
        small = self.small(head_type='conv_upsample', untied_steps=4)
        torch.nn.init.normal_(small.output_head.final.weight, std=.02)
        outputs = runner.refine_steps(small, torch.randn(2, 8, 2, 2), torch.randn(2, 1, 32, 32), 4)
        loss, losses = runner.deep_supervision(outputs, torch.ones(2, 32, 32),
                                               torch.ones(2, 32, 32, dtype=torch.bool))
        loss.backward()
        self.assertEqual(len(losses), 4)
        for core in small.core:
            self.assertTrue(torch.isfinite(core[0].qkv.weight.grad).all())
            self.assertGreater(core[0].qkv.weight.grad.abs().sum().item(), 0)

    def test_cli_depth_validation_and_inference_guards(self):
        args = runner.build_parser().parse_args(['--untied-steps', '4', '--eval-k', '1,2,4'])
        runner.validate_untied_settings(args)
        for field, value in (('eval_k', [1, 2, 4, 8]), ('k_train', 5),
                             ('selection_ks', {'1': .5, '6': .5}), ('untied_steps', -1)):
            invalid = copy.copy(args)
            setattr(invalid, field, value)
            with self.assertRaises(ValueError):
                runner.validate_untied_settings(invalid)
        shared = runner.build_parser().parse_args([])
        self.assertEqual(shared.untied_steps, 0)
        runner.validate_untied_settings(shared)
        model = self.small(untied_steps=4)
        with self.assertRaisesRegex(ValueError, 'evaluation K=8'):
            runner.evaluate(None, None, model, {}, [1, 8], torch.device('cpu'))
        for mode in runner.INFER_ABLATIONS:
            with self.assertRaisesRegex(ValueError, 'inference K=8'):
                list(runner.inference_steps(model, None, None, [8], torch.device('cpu'), mode))

    def test_eval_only_and_resume_restore_untied_depth_and_legacy_default(self):
        data = json.loads((aggregate.ROOT/'results/exp/e0_v2_s0/metrics.json').read_text())

        class StopBeforeData(RuntimeError):
            pass

        for mode in ('--eval-only', '--resume'):
            for saved_depth in (None, 4):
                config = copy.deepcopy(data['config'])
                if saved_depth is not None:
                    config['untied_steps'] = saved_depth
                checkpoint = dict(state_dict={}, config=config, epoch=1)
                argv = ['train_refiner.py', mode, '--eval-k', '1,2,4']
                if mode == '--eval-only':
                    argv += ['--ckpt', 'runs/untied_trial/mock.pt']
                captured = []

                def validate(args):
                    runner.check_untied_limit(args.untied_steps, args.eval_k, 'test K')
                    captured.append(args.untied_steps)
                    raise StopBeforeData

                with patch.object(sys, 'argv', argv), patch.object(Path, 'is_file', return_value=True), \
                        patch.object(torch, 'load', return_value=checkpoint), \
                        patch.object(runner, 'validate_untied_settings', side_effect=validate):
                    with self.assertRaises(StopBeforeData):
                        runner.main()
                self.assertEqual(captured, [saved_depth or 0])


class UntiedAggregateTests(unittest.TestCase):
    def test_untied_complete_run_has_na_at_k6_k8_and_complete_mean(self):
        data = json.loads((aggregate.ROOT/'results/exp/e0_v2_s0/metrics.json').read_text())
        data['config']['untied_steps'] = 4
        for variants in data['evaluation']['groups'].values():
            for key in ('6', '8'):
                variants.pop(key)
        rows = []
        for seed in range(3):
            data['config']['seed'] = seed
            with patch.object(Path, 'read_text', return_value=json.dumps(data)):
                row = aggregate.read_run(f'e4_untied_s{seed}', Path('metrics.json'))
            self.assertEqual(row['status'], 'complete')
            self.assertIsNone(row['A_K6'])
            self.assertIsNone(row['A_K8'])
            rows.append(row)
        mean = aggregate.mean_row(rows, 'e4_untied')
        self.assertEqual(mean['status'], 'complete')
        self.assertIsNone(mean['A_K6'])
        self.assertIsNone(mean['A_K8'])
        self.assertEqual(aggregate.display(mean['A_K8']), 'NA')

    def test_original_four_experiments_render_identically_without_writes(self):
        captured = {}

        def capture(path, text, **kwargs):
            captured[path.name] = text

        with patch.object(sys, 'argv', ['aggregate_experiments.py']), patch('builtins.print'), \
                patch.object(Path, 'mkdir'), patch.object(Path, 'write_text', autospec=True, side_effect=capture):
            aggregate.main()
        # The candidate is rendered only in memory; existing result files stay untouched.
        old_md = (aggregate.ROOT/'results/exp/summary.md').read_text()
        strip_e4 = lambda text: '\n'.join(line for line in text.splitlines() if not line.startswith('| e4_untied'))
        self.assertEqual(strip_e4(captured['summary.md']), strip_e4(old_md))
        old = json.loads((aggregate.ROOT/'results/exp/summary.json').read_text())
        new = json.loads(captured['summary.json'])
        for report in (old, new):
            for key in ('rows', 'table1'):
                report[key] = [row for row in report[key] if not row['experiment'].startswith('e4_untied')]
        self.assertEqual(json.dumps(old, indent=2), json.dumps(new, indent=2))


if __name__ == '__main__':
    unittest.main()
