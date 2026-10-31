"""Staff-runner intervention semantics, shared drafts and inference reporting."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scripts import train_refiner as runner
from looproad.models.refiner import LoopRefiner
from analysis import aggregate_experiments as aggregate


class InlinePool:
    """Run real official scoring synchronously for small synthetic-mask tests."""
    def __init__(self, workers, initializer, initargs):
        initializer(*initargs)

    def apply_async(self, function, args):
        result = function(*args)
        return SimpleNamespace(get=lambda: result)

    def close(self):
        pass

    join = close
    terminate = close


class InferAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(29)
        self.device = torch.device('cpu')
        self.features = torch.randn(1, 8, 4, 4)
        self.draft = torch.randn(1, 1, 16, 16)
        self.model = LoopRefiner(8, 16, heads=4, patch=4, image_size=16).eval()
        nn.init.normal_(self.model.output_head[-1].weight, std=.1)

    def trajectory(self, mode):
        return [y.clone() for _, y, _, _ in runner.inference_steps(
            self.model, self.features, self.draft, [1, 2, 3, 4], self.device, mode)]

    def test_none_logits_are_bitwise_identical_to_existing_loop(self):
        with torch.no_grad():
            legacy = runner.refine_steps(self.model, self.features, self.draft, 4)
        for observed, expected in zip(self.trajectory('none'), legacy):
            self.assertTrue(torch.equal(observed, expected))

    def test_reset_reruns_initializer_and_supplies_initial_state_before_every_core(self):
        states, core_inputs, step_args = [], [], []
        initialize, step = self.model.init, self.model.step

        def record_init(features, draft):
            state, tokens = initialize(features, draft)
            states.append(state.clone())
            return state, tokens

        def record_step(state, answer, tokens):
            step_args.append((state.clone(), answer.clone(), tokens.clone()))
            return step(state, answer, tokens)

        hook = self.model.core.register_forward_pre_hook(lambda _, args: core_inputs.append(args[0].clone()))
        try:
            with patch.object(self.model, 'init', side_effect=record_init), patch.object(self.model, 'step', side_effect=record_step):
                self.trajectory('reset-state')
        finally:
            hook.remove()
        self.assertEqual(len(states), 5)
        self.assertEqual(len(step_args), 4)
        for state in states:
            self.assertTrue(torch.equal(state, states[0]))
        for args, core_input in zip(step_args, core_inputs):
            state, answer, tokens = args
            self.assertTrue(torch.equal(state, states[0]))
            expected = states[0] + tokens + self.model.answer_tokens(answer) + self.model.position_encoding
            self.assertTrue(torch.equal(core_input, expected))
        self.assertFalse(torch.equal(step_args[1][1], self.draft))

    def test_no_feedback_passes_initial_logits_as_answer_and_residual_base(self):
        normal = self.trajectory('none')
        answers, results = [], []
        original = self.model.step

        def record(state, answer, tokens):
            answers.append(answer.clone())
            result = original(state, answer, tokens)
            results.append(result[1].clone())
            return result

        with patch.object(self.model, 'step', side_effect=record):
            observed = self.trajectory('no-mask-feedback')
        self.assertEqual(len(answers), 4)
        self.assertTrue(all(torch.equal(answer, self.draft) for answer in answers))
        self.assertTrue(torch.equal(observed[0], normal[0]))
        self.assertFalse(torch.equal(observed[1], normal[1]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(results, observed)))

    def test_all_modes_match_the_actual_staff_predict_tensor(self):
        path = Path(os.environ['EC523_STARTER_DIR'])/'predict.py'
        spec = importlib.util.spec_from_file_location('staff_infer_ablation_test', path)
        staff = importlib.util.module_from_spec(spec)
        with patch.object(sys, 'path', [str(path.parent), *sys.path]):
            spec.loader.exec_module(staff)

        class Adapter(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            @property
            def core(self):
                return self.inner.core

            def initialize(self, features, initial_logits):
                return self.inner.init(features, initial_logits)

            def step(self, features, state, previous_logits):
                z, tokens = state
                next_z, answer = self.inner.step(z, previous_logits, tokens)
                return (next_z, tokens), answer

        base = lambda _: (self.features.clone(), self.draft.clone())
        for mode in runner.INFER_ABLATIONS:
            trace, calls = staff.predict_tensor(base, Adapter(self.model), torch.zeros(1, 3, 16, 16), 4, mode)
            self.assertEqual(calls, 4)
            for y, probability in zip(self.trajectory(mode), trace[1:]):
                np.testing.assert_array_equal(y.sigmoid().numpy(), probability)

    def test_shared_encoder_and_draft_once_per_image_with_identical_none_scores(self):
        class Encoder(nn.Module):
            def forward(self, images):
                return images[:, :1, ::16, ::16].repeat(1, 8, 1, 1)

        class Head(nn.Module):
            def forward(self, features):
                return torch.nn.functional.interpolate(features[:, :1], size=(32, 32)) - .5

        encoder, head = Encoder(), Head()
        model = LoopRefiner(8, 16, heads=4, image_size=32, head_type='conv_upsample').eval()
        nn.init.normal_(model.output_head.final.weight, std=.1)
        raw = np.ones((1, 32, 32), dtype=np.uint8)
        raw[:, 14:18, 2:30] = 3
        batch = (np.full((1, 3, 32, 32), .5, dtype=np.float32), None, None, ['synthetic'], raw)
        datasets = {domain: object() for domain in runner.DOMAINS}
        context = SimpleNamespace(Pool=InlinePool)
        with patch.object(runner.mp, 'get_context', return_value=context), patch.object(runner, 'make_loader', return_value=[batch]):
            legacy = runner.evaluate(encoder, head, model, datasets, [1, 2, 4], self.device, 0, 1, eval_extra=True)
            with patch.object(encoder, 'forward', wraps=encoder.forward) as enc, patch.object(head, 'forward', wraps=head.forward) as hd:
                modes = runner.evaluate(encoder, head, model, datasets, [1, 2, 4], self.device, 0, 1,
                                        eval_extra=True, infer_ablations=list(runner.INFER_ABLATIONS))
            self.assertEqual(enc.call_count, 2)
            self.assertEqual(hd.call_count, 2)
        self.assertEqual(legacy['images'], modes['images'])
        for domain in (*runner.DOMAINS, 'Overall'):
            for key, item in legacy['groups'][domain].items():
                observed = modes['ablations']['none']['groups'][domain][key]
                for field in ('scores', 'structure', 'extra', 'added_road_pixels', 'deleted_road_pixels',
                              'added_road_fraction', 'deleted_road_fraction'):
                    self.assertEqual(item[field], observed[field])
            a = modes['ablations']['none']['groups'][domain]['1']
            b = modes['ablations']['no-mask-feedback']['groups'][domain]['1']
            self.assertEqual(a['scores'], b['scores'])
            self.assertEqual(a['extra'], b['extra'])

    def test_mode_parser_default_and_invalid_choices(self):
        self.assertIsNone(runner.build_parser().parse_args([]).infer_ablations)
        self.assertEqual(runner.parse_infer_ablations('none,reset-state,no-mask-feedback'), list(runner.INFER_ABLATIONS))
        for value in ('', 'none,none', 'no_mask_feedback', 'zero-state'):
            with self.assertRaises(argparse.ArgumentTypeError):
                runner.parse_infer_ablations(value)

    def test_inference_aggregate_deltas_sample_std_and_missing_seed(self):
        def report(seed):
            def item(value):
                return {'scores': {'A': value, 'road_iou': value, 'cldice': value, 'boundary_iou': value},
                        'added_road_fraction': .01,
                        'extra': {'changes': {'added_correct_fraction': .5},
                                  'tracking': {'gap_repair_fraction': .2, 'new_break_fraction': .03}}}
            ablations = {}
            for offset, mode in enumerate(aggregate.INFER_MODES):
                groups = {domain: {'y0_draft': item(.1), **{str(k): item(.2 + seed*.1 + offset*.01)
                          for k in (1, 2, 4, 6, 8)}} for domain in ('Urban', 'Rural', 'Overall')}
                ablations[mode] = {'groups': groups}
            return {'config': {'seed': seed}, 'protocol': 'val_sel_rep', 'evaluated_epoch': seed+1,
                    'evaluation': {'ablations': ablations}}

        def read_text(path, *args, **kwargs):
            return json.dumps(report(int(path.parent.name[-1])))

        with patch.object(Path, 'read_text', autospec=True, side_effect=read_text):
            result = aggregate.aggregate_inference(Path('unused'))
        self.assertEqual(len(result['rows']), 12)
        row = result['rows'][1]
        self.assertAlmostEqual(row['delta_none_A_K4'], .01)
        self.assertAlmostEqual(row['delta_IoU_K4'], .11)
        for row in result['rows'][9:]:
            self.assertEqual(row['status'], 'complete')
            self.assertAlmostEqual(row['A_K1']['std'], .1)
        self.assertAlmostEqual(result['rows'][9]['A_K1']['mean'], .3)

        def missing(path, *args, **kwargs):
            if path.parent.name.endswith('s2'):
                raise FileNotFoundError
            return read_text(path)

        with patch.object(Path, 'read_text', autospec=True, side_effect=missing):
            result = aggregate.aggregate_inference(Path('unused'))
        self.assertTrue(all(row['status'] == 'incomplete' for row in result['rows'][9:]))


if __name__ == '__main__':
    unittest.main()
