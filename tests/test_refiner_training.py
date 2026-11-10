"""Exercise real optimization/checkpoints with tiny synthetic frozen features."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scripts import train_refiner as runner
from scripts.six_run_batch import VARIANTS
from looproad.models.refiner import LoopRefiner


class TinyRoadDataset:
    def __init__(self, split, domain, **kwargs):
        self.split, self.domain = split, domain
        self.pairs = [(Path(f'{split}_{domain}_{i}.png'), Path('mask')) for i in range(2)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        raw = np.ones((32, 32), dtype=np.uint8)
        raw[8:24] = 3
        target = (raw == 3).astype(np.float32)
        return np.stack([target] * 3), target, np.ones_like(target, dtype=bool), str(index), raw


class TinyEncoder(nn.Module):
    out_channels = 8

    def forward(self, images):
        return torch.nn.functional.adaptive_avg_pool2d(images[:, :1], 2).repeat(1, 8, 1, 1)


class TinyProbe(nn.Module):
    def forward(self, features):
        return features.new_full((len(features), 1, 32, 32), -1.)


class TrainerIntegrationTests(unittest.TestCase):
    def test_all_six_recipes_train_save_and_evaluate_without_training_on_val(self):
        torch.set_num_threads(1)
        for variant in VARIANTS:
            with self.subTest(variant=variant['name']), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                (root/'split.json').write_text('{}')
                train_calls, evaluation_calls = [], []

                def dataset(split, domain, **kwargs):
                    train_calls.append(split)
                    self.assertEqual(split, 'Train')
                    return TinyRoadDataset(split, domain)

                def evaluate(encoder, head, model, datasets, ks, device, *args, **kwargs):
                    for item in datasets.values():
                        self.assertIsInstance(item, TinyRoadDataset)
                        self.assertIn(item.split, ('Val-sel', 'Val-rep'))
                    evaluation_calls.append(next(iter(datasets.values())).split)
                    model.eval()
                    with torch.no_grad():
                        images = torch.zeros(2, 3, 32, 32)
                        features, draft = runner.frozen_draft(encoder, head, images)
                        outputs = runner.refine_steps(model, features, draft, max(ks))
                        self.assertTrue(all(torch.isfinite(x).all() for x in outputs))
                    # Selection increases monotonically, so best.pt is the latest epoch.
                    score = .5 + .001 * len(evaluation_calls)
                    return {'wall_seconds': 0., 'groups': {'Overall': {
                        key: {'scores': {'A': score}} for key in ['y0_draft', *map(str, ks)]}}}

                common = ['train_refiner.py', '--device', 'cpu', '--batch-size', '2',
                          '--dimension', '16', '--heads', '4', '--k-train', '2',
                          '--head-type', 'conv_upsample', '--loader-workers', '0',
                          '--eval-loader-workers', '0', '--metric-workers', '1',
                          '--probe-ckpt', str(root/'probe.pt'), '--val-split', str(root/'split.json'),
                          '--ckpt-dir', str(root/'runs/checkpoints')]
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(runner, 'ROOT', root))
                    stack.enter_context(patch.dict(os.environ, {'NSLOTS': '0'}))
                    stack.enter_context(patch.object(runner.probe, 'RoadDataset', side_effect=dataset))
                    stack.enter_context(patch.object(runner.y0, 'load_frozen_probe',
                                                    side_effect=lambda *a: (TinyEncoder(), TinyProbe(), {'head': 'bn_linear'})))
                    stack.enter_context(patch.object(runner, 'LoopRefiner',
                                                    side_effect=lambda *a, **kw: LoopRefiner(*a, image_size=32, **kw)))
                    stack.enter_context(patch.object(runner, 'protocol_datasets', side_effect=lambda split, part, limit=None:
                                                    {d: TinyRoadDataset(part, d) for d in runner.DOMAINS}))
                    stack.enter_context(patch.object(runner, 'evaluate', side_effect=evaluate))
                    stack.enter_context(patch.object(runner, 'summary_markdown', return_value='Synthetic test\n'))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    arguments = common + ['--epochs', '2', '--out-dir', str(root/'reports'),
                                          '--loss', variant['loss'], '--pos-weight', str(variant['pos_weight']),
                                          '--augmentation', variant['augmentation'], '--skeleton-iterations', '3',
                                          '--false-add-warmup-steps', '1', '--false-add-calibration-batches', '1']
                    with patch('sys.argv', arguments):
                        runner.main()
                    saved = torch.load(root/'runs/checkpoints/last.pt', weights_only=True)
                    self.assertEqual(saved['epoch'], 2)
                    self.assertGreater(float(saved['state_dict']['output_head.final.weight'].abs().sum()), 0.)
                    self.assertEqual(train_calls, ['Train', 'Train'])
                    self.assertEqual(evaluation_calls, ['Val-sel', 'Val-sel', 'Val-rep'])
                    for key in ('loss', 'pos_weight', 'augmentation'):
                        self.assertEqual(saved['config'][key], variant[key])
                    report = json.loads((root/'reports/metrics.json').read_text())
                    self.assertEqual(report['protocol'], 'val_sel_rep')
                    self.assertEqual(report['evaluated_epoch'], 2)
                    self.assertEqual(len(report['epochs']), 2)
                    if variant['loss'] == 'false_add':
                        state = saved['config']['objective_state']
                        self.assertGreater(state['effective_aux_weight'], 0.)
                        with patch('sys.argv', common + ['--resume', '--epochs', '3', '--out-dir', str(root/'resumed')]):
                            runner.main()
                        resumed = torch.load(root/'runs/checkpoints/last.pt', weights_only=True)
                        self.assertEqual(resumed['config']['objective_state'], state)
                        self.assertEqual(resumed['epoch'], 3)
                    before = len(train_calls)
                    with patch('sys.argv', common + ['--eval-only', '--ckpt', str(root/'runs/checkpoints/best.pt'),
                                                    '--out-dir', str(root/'evaluated')]):
                        runner.main()
                    self.assertEqual(len(train_calls), before)


if __name__ == '__main__':
    unittest.main()
