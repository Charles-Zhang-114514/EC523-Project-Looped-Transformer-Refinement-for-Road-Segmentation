"""Submission serialization, exact adapter math, CLI IDs and staff-loop parity."""
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from submission import predict
from submission.model import DEFAULT_CONFIG, build_refiner, refiner_from_config, load_refiner
from submission.baseline import FrozenBaseline
from looproad.models.refiner import LoopRefiner
from looproad.models.linear_probe import LinearProbe
from scripts.export_submission import export_weights

SMALL = dict(dimension=16, heads=4, image_size=16, patch=4, head_type='patch_linear')


class SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        (ROOT/'runs/submission_tests').mkdir(parents=True, exist_ok=True)

    def setUp(self):
        torch.manual_seed(9)
        self.features = torch.randn(1, 8, 4, 4)
        self.draft = torch.randn(1, 1, 16, 16)
        self.original = LoopRefiner(8, **SMALL).eval()
        nn.init.normal_(self.original.output_head[-1].weight, std=.02)
        self.model = refiner_from_config(8, SMALL).eval()
        self.model.load_state_dict(self.original.state_dict())

    def test_adapter_and_original_are_exact_at_every_update(self):
        self.assertEqual(list(inspect.signature(build_refiner).parameters), ['feature_channels'])
        with torch.no_grad():
            z, tokens = self.original.init(self.features, self.draft)
            state = self.model.initialize(self.features, self.draft)
            y, wrapped = self.draft, self.draft
            self.assertTrue(torch.equal(state[0], z))
            self.assertTrue(torch.equal(state[1], tokens))
            for _ in range(4):
                z, y = self.original.step(z, y, tokens)
                state, wrapped = self.model.step(self.features, state, wrapped)
                self.assertTrue(torch.equal(state[0], z))
                self.assertTrue(torch.equal(wrapped, y))
        for method in (self.model.initialize, self.model.step):
            self.assertFalse({'steps', 'K', 'k_train'} & inspect.signature(method).parameters.keys())
        self.assertEqual(list(self.model.state_dict()), list(self.original.state_dict()))

    def test_export_contains_only_model_tensors_and_reloads_exactly(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs/submission_tests') as directory:
            root = Path(directory)
            model = build_refiner(1024)
            config = dict(DEFAULT_CONFIG, seed=0)
            torch.save(dict(state_dict=model.state_dict(), config=config, epoch=7,
                            optimizer={'unused': 1}, scheduler={'unused': 2}, rng={'unused': 3}), root/'trusted.pt')
            probe = LinearProbe(1024, 'bn_linear')
            torch.save(dict(state_dict=probe.state_dict(), encoder='dinov3_vitl16_web',
                            head='bn_linear', pos_weight=4), root/'probe.pt')
            reports = export_weights(root/'trusted.pt', root/'probe.pt', root/'export')
            self.assertEqual(reports[0]['parameters'], 1_193_025)
            # Own the tensor storage so NFS can remove the temporary export
            # immediately; live safetensors memory maps can retain .nfs files.
            tensors = {name: value.clone() for name, value in load_file(str(root/'export/model.safetensors')).items()}
            self.assertEqual(set(tensors), set(model.state_dict()))
            for name, tensor in tensors.items():
                self.assertTrue(torch.equal(tensor, model.state_dict()[name]))
            loaded = load_refiner(root/'export/model.safetensors', 1024, torch.device('cpu'))
            self.assertEqual(list(loaded.state_dict()), list(model.state_dict()))
            with self.assertRaises(FileExistsError):
                export_weights(root/'trusted.pt', root/'probe.pt', root/'export')

    def test_actual_export_matches_best_pt_outputs(self):
        checkpoint = ROOT/'runs/exp/e0_v2_s0/best.pt'
        exported = ROOT/'submission/model.safetensors'
        if not checkpoint.exists() or not exported.exists():
            self.skipTest('Author-only check requires the trusted training artifact and its export')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        original = LoopRefiner(1024, head_type='conv_upsample').eval()
        original.load_state_dict(saved['state_dict'])
        loaded = load_refiner(exported, 1024, torch.device('cpu'))
        features, y = torch.randn(1, 1024, 64, 64), torch.randn(1, 1, 1024, 1024)
        with torch.no_grad():
            z, tokens = original.init(features, y)
            state = loaded.initialize(features, y)
            wrapped = y.clone()
            for _ in range(2):
                z, y = original.step(z, y, tokens)
                state, wrapped = loaded.step(features, state, wrapped)
                self.assertTrue(torch.equal(y, wrapped))

    def test_interventions_match_actual_staff_runner_and_k0_skips_initialize(self):
        file = Path(os.environ['EC523_STARTER_DIR'])/'predict.py'
        spec = importlib.util.spec_from_file_location('staff_submission_test', file)
        staff = importlib.util.module_from_spec(spec)
        with patch.object(sys, 'path', [str(file.parent), *sys.path]):
            spec.loader.exec_module(staff)
        base = lambda _: (self.features.clone(), self.draft.clone())
        image = torch.zeros(1, 3, 16, 16)
        for mode in predict.MODES:
            a, calls = predict.predict_tensor(base, self.model, image, 4, mode)
            b, official_calls = staff.predict_tensor(base, self.model, image, 4, mode)
            self.assertEqual((calls, official_calls), (4, 4))
            for ours, official in zip(a, b):
                np.testing.assert_array_equal(ours, official)
        with patch.object(self.model, 'initialize', side_effect=AssertionError('K0 initialized')):
            trace, calls = predict.predict_tensor(base, self.model, image, 0)
        self.assertEqual(calls, 0)
        np.testing.assert_array_equal(trace[0], self.draft.sigmoid().numpy())

    def test_single_image_and_directory_cli_outputs_are_identical(self):
        class TinyBaseline(nn.Module):
            feature_channels = 8

            def forward(self, image):
                return image[:, :1, ::16, ::16].repeat(1, 8, 1, 1), image[:, :1] - .5

        with tempfile.TemporaryDirectory(dir=ROOT/'runs/submission_tests') as directory:
            root = Path(directory)
            (root/'images').mkdir()
            pixels = np.random.default_rng(7).integers(0, 256, (1024, 1024, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root/'images/tile.png')
            (root/'encoder').mkdir()
            for name in ('config.json', 'preprocessor_config.json', 'model.safetensors'):
                (root/'encoder'/name).write_bytes(b'unit test encoder placeholder')
            (root/'probe.safetensors').write_bytes(b'unit test probe placeholder')
            config = dict(dimension=16, heads=4, image_size=1024, patch=16, head_type='conv_upsample')
            model = refiner_from_config(8, config)
            save_file(model.state_dict(), str(root/'model.safetensors'), metadata={'config': json.dumps(config)})
            common = ['--checkpoint', str(root/'model.safetensors'), '--baseline', str(root/'probe.safetensors'),
                      '--encoder-dir', str(root/'encoder'), '--steps', '4', '--save-trace', '--device', 'cpu', '--single-pass']
            with patch.object(predict, 'load_baseline', return_value=TinyBaseline()):
                for name, source in [('single', root/'images/tile.png'), ('folder', root/'images')]:
                    report = predict.main(common+['--input', str(source), '--output', str(root/name)])
                    self.assertEqual(report['images'][0]['id'], 'tile')
                    self.assertEqual(report['images'][0]['core_calls'], 4)
            for p in (root/'single').rglob('*'):
                if p.is_file() and p.name != 'run.json':
                    self.assertEqual(p.read_bytes(), (root/'folder'/p.relative_to(root/'single')).read_bytes())

    def test_invalid_ids_nonfinite_outputs_and_occupied_directory_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs/submission_tests') as directory:
            root = Path(directory)
            for name in ('tile.png', 'tile.jpg'):
                Image.new('RGB', (16, 16)).save(root/name)
            with self.assertRaises(ValueError):
                predict.input_files(root)
            args = ['--input', str(root/'tile.png'), '--output', str(root), '--checkpoint', 'unused',
                    '--baseline', 'unused', '--encoder-dir', 'unused', '--steps', '1']
            with self.assertRaises(ValueError):
                predict.main(args)
        bad = lambda _: (self.features, self.draft * float('nan'))
        with self.assertRaises(ValueError):
            predict.predict_tensor(bad, self.model, torch.zeros(1, 3, 16, 16), 1)

    def test_baseline_frozen_eval_and_original_frozen_draft_math(self):
        from scripts.train_refiner import frozen_draft

        class TinyEncoder(nn.Module):
            def forward(self, images):
                return images[:, :1, ::16, ::16].repeat(1, 1024, 1, 1)

        encoder, probe = TinyEncoder(), LinearProbe(1024, 'bn_linear')
        baseline = FrozenBaseline(encoder, probe)
        baseline.train()
        self.assertFalse(baseline.training or probe.training)
        self.assertFalse(any(p.requires_grad for p in baseline.parameters()))
        image = torch.randn(1, 3, 1024, 1024)
        expected = frozen_draft(encoder, probe, image)
        observed = baseline(image)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(observed, expected)))


if __name__ == '__main__':
    unittest.main()
