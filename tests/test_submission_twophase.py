"""Float32 feature round trips, cache lifetime and the TA 0/1 mask contract."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from PIL import Image
import torch
from torch import nn
from safetensors.torch import save_file

from submission import predict
from submission.model import refiner_from_config
from scripts.package_submission import package, ROOT


class TinyEncoder(nn.Module):
    def forward(self, image):
        return image[:, :1, ::16, ::16].repeat(1, 8, 1, 1)


class TinyProbe(nn.Module):
    def forward(self, features):
        return torch.nn.functional.interpolate(features[:, :1], size=(1024, 1024), mode='bilinear', align_corners=False) - .5


class TinyBaseline(nn.Module):
    feature_channels = 8

    def __init__(self):
        super().__init__()
        self.encoder, self.probe = TinyEncoder(), TinyProbe()

    def forward(self, image):
        features = self.encoder(image)
        return features, self.probe(features)


class TwoPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        # Scratch is outside the repository, matching the public cache rule.
        self.directory = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root/'images').mkdir()
        Image.fromarray(np.random.default_rng(4).integers(0, 256, (1024, 1024, 3), dtype=np.uint8)).save(self.root/'images/tile.png')
        (self.root/'encoder').mkdir()
        for name in ('config.json', 'preprocessor_config.json', 'model.safetensors'):
            (self.root/'encoder'/name).write_bytes(b'test fixture')
        fixture_hashes = {name: predict.sha256(self.root/'encoder'/name) for name in ('config.json', 'preprocessor_config.json', 'model.safetensors')}
        pinned = patch.dict('submission.fetch_encoder.EXPECTED_SHA256', fixture_hashes, clear=True)
        pinned.start()
        self.addCleanup(pinned.stop)
        (self.root/'baseline.safetensors').write_bytes(b'test fixture')
        config = dict(dimension=16, heads=4, image_size=1024, patch=16, head_type='conv_upsample')
        torch.manual_seed(19)
        model = refiner_from_config(8, config)
        nn.init.normal_(model.output_head.final.weight, std=.03)
        save_file(model.state_dict(), str(self.root/'model.safetensors'), metadata={'config': json.dumps(config)})
        self.common = ['--checkpoint', str(self.root/'model.safetensors'), '--baseline', str(self.root/'baseline.safetensors'),
                       '--encoder-dir', str(self.root/'encoder'), '--input', str(self.root/'images'), '--steps', '2',
                       '--cache-dir', str(self.root/'cache'), '--device', 'cpu']

    def run_cli(self, name, extra=()):
        with patch.object(predict, 'load_baseline', side_effect=lambda *args: TinyBaseline()):
            return predict.main(self.common+['--output', str(self.root/name), *extra])

    def test_two_phase_single_pass_and_road_masks_exact(self):
        references = []

        def make(*args):
            base = TinyBaseline()
            references.append(weakref.ref(base.encoder))
            return base

        with patch.object(predict, 'load_baseline', side_effect=make):
            two = predict.main(self.common+['--output', str(self.root/'two')])
        self.assertIsNone(references[0]())
        one = self.run_cli('one', ['--single-pass'])
        for folder in ('probabilities', 'masks', 'road_masks'):
            suffix = 'npy' if folder == 'probabilities' else 'png'
            a, b = self.root/'two'/folder/f'tile.{suffix}', self.root/'one'/folder/f'tile.{suffix}'
            self.assertEqual(a.read_bytes(), b.read_bytes())
        with Image.open(self.root/'two/road_masks/tile.png') as image:
            road = np.asarray(image)
        with Image.open(self.root/'two/masks/tile.png') as image:
            mask = np.asarray(image)
        self.assertEqual(road.dtype, np.uint8)
        self.assertLessEqual(set(np.unique(road)), {0, 1})
        np.testing.assert_array_equal(road, mask//255)
        self.assertEqual(road.shape, (1024, 1024))
        self.assertEqual((two['mode'], one['mode']), ('two-phase', 'single-pass'))
        self.assertGreater(two['cache_bytes'], 0)
        self.assertTrue(two['cache_removed'])
        self.assertEqual(list((self.root/'cache').iterdir()), [])

    def test_cache_removed_after_encoding_and_refinement_errors(self):
        for failure in ('encode', 'refine'):
            def baseline(*args):
                base = TinyBaseline()
                if failure == 'encode':
                    base.encoder.forward = lambda *_: (_ for _ in ()).throw(RuntimeError('injected encoder failure'))
                return base
            with patch.object(predict, 'load_baseline', side_effect=baseline):
                with patch.object(predict, 'refine_tensor', side_effect=RuntimeError('injected refiner failure')):
                    with self.assertRaisesRegex(RuntimeError, 'injected'):
                        predict.main(self.common+['--output', str(self.root/failure)])
            self.assertEqual(list((self.root/'cache').iterdir()), [])

    def test_keep_cache_preserves_original_float32_features(self):
        result = self.run_cli('keep', ['--keep-cache'])
        directories = list((self.root/'cache').iterdir())
        self.assertEqual(len(directories), 1)
        feature = np.load(directories[0]/'00000000.npy', allow_pickle=False)
        expected = TinyEncoder()(predict.image_tensor(self.root/'images/tile.png', torch.device('cpu'))).numpy()
        self.assertEqual(feature.dtype, np.float32)
        np.testing.assert_array_equal(feature, expected)
        self.assertTrue(result['cache_kept'])
        self.assertFalse(result['cache_removed'])

    def test_non_1024_input_cache_location_and_space_checks(self):
        Image.new('RGB', (17, 19)).save(self.root/'images/bad.png')
        with self.assertRaisesRegex(ValueError, '1024x1024'):
            self.run_cli('wrong-size')
        with self.assertRaisesRegex(ValueError, 'outside'):
            predict.create_cache(ROOT/'runs', self.root/'output', 1)
        with patch.object(predict.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(OSError, 'Insufficient'):
                predict.create_cache(self.root/'cache2', self.root/'output', 1)

    def test_package_rejects_repository_destination(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            package(ROOT/'unignored_package', ROOT/'submission/encoder')


if __name__ == '__main__':
    unittest.main()
