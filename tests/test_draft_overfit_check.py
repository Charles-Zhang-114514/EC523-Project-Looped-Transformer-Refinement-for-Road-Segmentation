"""Train-only deterministic sampling, gap arithmetic and shared encoding."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from analysis import draft_overfit_check as audit
from looproad.models.draft_head import ConvUpDraftHead


def group(value):
    return {"Overall": {"y0_draft": {"scores": {"A": value}}}}


class DraftOverfitTests(unittest.TestCase):
    def test_sampling_is_reproducible_and_train_only(self):
        with tempfile.TemporaryDirectory(dir=audit.ROOT / "runs") as directory:
            root = Path(directory)
            for split in ("Train", "Val"):
                for domain in audit.DOMAINS:
                    for kind in ("images_png", "masks_png"):
                        destination = root / split / domain / kind
                        destination.mkdir(parents=True)
                        for i in range(8):
                            (destination / f"{split}_{i}.png").touch()
            with patch.dict(os.environ, {"EC523_DATA_DIR": str(root)}):
                first = audit.train_subset(3, 0)
                second = audit.train_subset(3, 0)
            expected = np.sort(np.random.default_rng(0).choice(8, size=3, replace=False))
            for domain in audit.DOMAINS:
                self.assertEqual(first[domain].pairs, second[domain].pairs)
                self.assertEqual(first[domain].split, "Train")
                self.assertEqual([p.name for p, _ in first[domain].pairs],
                                 [f"Train_{i}.png" for i in expected])
                self.assertTrue(all(root / "Train" in p.parents for p, _ in first[domain].pairs))

    def test_gap_difference_uses_each_heads_own_val_score(self):
        train = {"linear": group(.65), "A1-i": group(.72)}
        val = {"linear": group(.55), "A1-i": group(.58)}
        result = audit.gap_comparison(train, val)
        self.assertAlmostEqual(result["gap"]["linear"], .10)
        self.assertAlmostEqual(result["gap"]["A1-i"], .14)
        self.assertAlmostEqual(result["gap_difference"], .04)

    def test_empty_gap_and_invalid_sample_size(self):
        values = {"linear": group(.5), "A1-i": group(.6)}
        self.assertEqual(audit.gap_comparison(values, values)["gap_difference"], 0.)
        with self.assertRaises(ValueError):
            audit.train_subset(0)

    def test_shared_encoder_called_once_and_conv_wrapper_is_equivalent(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, image):
                self.calls += 1
                return image

        encoder = Encoder()
        linear = nn.Conv2d(4, 1, 1).eval().requires_grad_(False)
        conv = ConvUpDraftHead(4, 1).eval().requires_grad_(False)
        features = torch.randn(1, 4, 2, 2)
        actual_features, drafts = audit.frozen_pair(encoder, {"linear": linear, "A1-i": conv}, features)
        self.assertEqual(encoder.calls, 1)
        self.assertTrue(torch.equal(actual_features, features))
        with torch.no_grad():
            expected = conv({"draft": (features,), "refiner": features})
        self.assertTrue(torch.equal(drafts["A1-i"], expected))
        self.assertFalse(drafts["A1-i"].requires_grad)


if __name__ == "__main__":
    unittest.main()
