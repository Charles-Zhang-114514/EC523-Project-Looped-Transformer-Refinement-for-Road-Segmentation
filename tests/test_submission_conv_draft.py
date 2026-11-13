"""Frozen-head tensor serialization and the stable Midway package layout."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from looproad.models.draft_head import ConvUpDraftHead
from looproad.models.linear_probe import LinearProbe
from submission.baseline import load_probe, probe_from_metadata
from submission.model import DEFAULT_CONFIG, build_refiner
from scripts.export_submission import export_weights
from scripts import package_submission as packaging

ROOT = packaging.ROOT
MIDWAY_FILES = (
    'requirements.txt', 'src/looproad/__init__.py', 'src/looproad/data/__init__.py',
    'src/looproad/data/road_dataset.py', 'src/looproad/encoders/__init__.py',
    'src/looproad/encoders/huggingface.py', 'src/looproad/encoders/resnet.py',
    'src/looproad/eval/__init__.py', 'src/looproad/eval/structured_errors.py',
    'src/looproad/models/__init__.py', 'src/looproad/models/linear_probe.py',
    'src/looproad/models/refiner.py', 'src/looproad/train/__init__.py',
    'submission/README.md', 'submission/__init__.py', 'submission/baseline.py',
    'submission/baseline.safetensors', 'submission/encoder/LICENSE.md',
    'submission/encoder_sha256.json', 'submission/fetch_encoder.py',
    'submission/model.py', 'submission/model.safetensors', 'submission/predict.py')


class SubmissionConvDraftTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / 'runs')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_legacy_bn_safetensors_load_is_bitwise_unchanged(self):
        original = LinearProbe(1024, head_type='bn_linear').eval()
        source = ROOT / 'submission/baseline.safetensors'
        if source.exists():
            original.load_state_dict(load_file(str(source)), strict=True)
        else:
            source = self.root / 'legacy.safetensors'
            save_file(original.state_dict(), str(source), metadata={'head': 'bn_linear'})
        restored = load_probe(source)
        feature = torch.randn(1, 1024, 4, 4)
        with torch.no_grad():
            self.assertTrue(torch.equal(original(feature), restored(feature)))
        # Older files without metadata retain the exact legacy interpretation.
        bare = self.root / 'bare.safetensors'
        save_file(original.state_dict(), str(bare))
        with torch.no_grad():
            self.assertTrue(torch.equal(original(feature), load_probe(bare)(feature)))

    def test_conv_export_reloads_exact_checkpoint_outputs(self):
        model = build_refiner(1024)
        torch.save(dict(state_dict=model.state_dict(), config=dict(DEFAULT_CONFIG, seed=0), epoch=1),
                   self.root / 'refiner.pt')
        original = ConvUpDraftHead(1024, 1).eval()
        checkpoint = ROOT / 'runs/draft_head/A1-i/best.pt'
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            original.load_state_dict(saved['state_dict'], strict=True)
        else:
            saved = dict(state_dict=original.state_dict(), encoder='dinov3_vitl16_web',
                         head='conv_up', pos_weight=4, draft_layers='last', epoch=6)
        torch.save(saved, self.root / 'draft.pt')
        with contextlib.redirect_stdout(io.StringIO()):
            export_weights(self.root / 'refiner.pt', self.root / 'draft.pt', self.root / 'export')
        restored = load_probe(self.root / 'export/baseline.safetensors')
        with safe_open(str(self.root / 'export/baseline.safetensors'), framework='pt') as reader:
            self.assertEqual(reader.metadata()['head'], 'conv_up')
            self.assertEqual(json.loads(reader.metadata()['head_config'])['draft_layers'], 'last')
        feature = torch.randn(1, 1024, 4, 4)
        with torch.no_grad():
            self.assertTrue(torch.equal(original({'draft': (feature,), 'refiner': feature}), restored(feature)))
        self.assertFalse(restored.training)
        self.assertFalse(any(p.requires_grad for p in restored.parameters()))

    def test_unknown_or_multilayer_head_is_rejected(self):
        with self.assertRaises(ValueError):
            probe_from_metadata({'head': 'unknown'})
        with self.assertRaises(ValueError):
            probe_from_metadata({'head': 'conv_up', 'head_config': json.dumps(
                dict(in_channels=1024, input_layers=4, draft_layers='6,12,18,24'))})

    def test_default_package_file_list_matches_midway(self):
        repo = self.root / 'repo'
        for name in MIDWAY_FILES:
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# fixture\n')
        (repo / 'submission/encoder_sha256.json').write_bytes((ROOT / 'submission/encoder_sha256.json').read_bytes())
        for name in ('draft_head.py', 'refiner_losses.py'):
            (repo / 'src/looproad/models' / name).write_text('# excluded by default\n')
        (repo / 'src/looproad/data/augmentation.py').write_text('# training only\n')
        with patch.object(packaging, 'ROOT', repo), contextlib.redirect_stdout(io.StringIO()):
            archive = packaging.package(self.root / 'midway', repo / 'submission/encoder', no_encoder=True)
        with zipfile.ZipFile(archive) as zipped:
            self.assertEqual(sorted(zipped.namelist()), sorted(MIDWAY_FILES))
        old = ROOT / 'runs/submission_packages/gradescope_final_v2.zip'
        if old.exists():
            with zipfile.ZipFile(old) as zipped:
                self.assertEqual(sorted(zipped.namelist()), sorted(MIDWAY_FILES))


if __name__ == '__main__':
    unittest.main()
