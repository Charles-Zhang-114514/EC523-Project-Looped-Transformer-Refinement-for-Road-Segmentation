"""Draft-head shapes, block indexing and checkpoint compatibility, offline."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import AutoModel, DINOv3ViTConfig

from analysis import y0_error_analysis as y0
from looproad.encoders.huggingface import DinoV3WebViTL16Encoder, patch_tokens_to_map
from looproad.models.draft_head import (ConvUpDraftHead, DraftFeatureEncoder,
                                       build_draft, parse_draft_layers)
from looproad.models.linear_probe import LinearProbe
from scripts.train_refiner import frozen_draft


ROOT = Path(__file__).resolve().parents[1]


class DraftHeadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def encoder(self):
        config = DINOv3ViTConfig(hidden_size=16, num_hidden_layers=4, num_attention_heads=4,
                                intermediate_size=32, num_register_tokens=4, patch_size=16)
        model = AutoModel.from_config(config, attn_implementation="sdpa")
        return DinoV3WebViTL16Encoder(model, (.5,) * 3, (.25,) * 3, autocast_bf16=False)

    def test_full_resolution_and_parameter_counts(self):
        for layers, expected in ((1, 656865), (4, 1449441)):
            head = ConvUpDraftHead(1024, layers).eval()
            self.assertEqual(sum(p.numel() for p in head.parameters()), expected)
            with torch.no_grad():
                result = head(tuple(torch.randn(1, 1024, 64, 64) for _ in range(layers)))
            self.assertEqual(result.shape, (1, 1, 1024, 1024))
            self.assertTrue(torch.isfinite(result).all())

    def test_layer_selection(self):
        self.assertEqual(parse_draft_layers("last"), "last")
        self.assertEqual(parse_draft_layers("6,12,18,24"), (6, 12, 18, 24))
        for invalid in ("", "0", "1,1", "2,1", "x", "1,"):
            with self.subTest(value=invalid), self.assertRaises(argparse.ArgumentTypeError):
                parse_draft_layers(invalid)
        with self.assertRaisesRegex(ValueError, "only 4 blocks"):
            DraftFeatureEncoder(self.encoder(), "6,12,18,24")

    def test_hidden_state_indices_normalization_and_frozen_encoder(self):
        base = self.encoder()
        wrapper = DraftFeatureEncoder(base, (1, 2, 3, 4)).train(True)
        images = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            output = base.model(pixel_values=base.normalize(images), output_hidden_states=True)
            result = wrapper(images)
            self.assertEqual(len(output.hidden_states), 5)
            for k, actual in enumerate(result["draft"], 1):
                normalized = base.model.norm(output.hidden_states[k])
                expected = patch_tokens_to_map(normalized[:, 5:, :], 2, 2)
                self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(result["refiner"], base(images)))
        self.assertFalse(any(module.training for module in wrapper.modules()))
        self.assertFalse(any(p.requires_grad for p in wrapper.parameters()))
        self.assertFalse(result["refiner"].requires_grad)

    def test_old_checkpoint_frozen_draft_matches_original_path(self):
        base, original = self.encoder(), LinearProbe(16, "bn_linear").eval()
        # The legacy head always interpolates to 1024, even for this tiny input.
        images = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            features = base(images)
            expected = original(features)
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            path = Path(directory) / "probe.pt"
            torch.save(dict(encoder=y0.ENCODER, head="bn_linear", pos_weight=4.,
                            state_dict=original.state_dict()), path)
            with patch.object(y0.probe, "build_encoder", return_value=base):
                encoder, head, _ = y0.load_frozen_probe(path, torch.device("cpu"))
            actual_features, actual = frozen_draft(encoder, head, images)
        self.assertTrue(torch.equal(features, actual_features))
        self.assertTrue(torch.equal(expected, actual))

    def test_new_checkpoint_roundtrip_through_frozen_draft(self):
        for layers in ("last", "1,2,3,4"):
            with self.subTest(layers=layers):
                base = self.encoder()
                with patch("looproad.models.draft_head.build_encoder", return_value=base):
                    encoder, head = build_draft(layers)
                head.eval()
                images = torch.rand(1, 3, 32, 32)
                expected_features, expected = frozen_draft(encoder, head, images)
                with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
                    path = Path(directory) / "best.pt"
                    torch.save(dict(encoder=y0.ENCODER, head="conv_up", pos_weight=4.,
                                    draft_layers=layers, state_dict=head.state_dict(),
                                    config={"draft_layers": layers, "head": "conv_up"}), path)
                    with patch("looproad.models.draft_head.build_encoder", return_value=copy.deepcopy(base)):
                        loaded_encoder, loaded_head, _ = y0.load_frozen_probe(path, torch.device("cpu"))
                    features, actual = frozen_draft(loaded_encoder, loaded_head, images)
                self.assertEqual(features.shape, (1, 16, 2, 2))
                self.assertEqual(actual.shape, (1, 1, 32, 32))
                self.assertTrue(torch.equal(expected_features, features))
                self.assertTrue(torch.equal(expected, actual))
                self.assertFalse(loaded_head.training)
                self.assertFalse(any(p.requires_grad for p in loaded_head.parameters()))


if __name__ == "__main__":
    unittest.main()
