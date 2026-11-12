"""Train-only deterministic sampling, gap arithmetic and shared encoding."""
import os
import contextlib
import io
import json
import subprocess
import sys
import types
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
    def test_threshold_choice_uses_only_selection_and_ties_choose_lower(self):
        def scores(best):
            return {d: {str(t): {"scores": {"A": 1. if t == best else .1}}
                        for t in audit.THRESHOLDS} for d in ("Overall", *audit.DOMAINS)}
        evaluations = {split: {"groups": {h: scores(.4 if split == "Val-sel" else .7) for h in audit.HEADS}}
                       for split in ("Train", "Val-sel", "Val-rep")}
        recorded = {h: {"Overall": {"y0_draft": {"scores": {"A": .5}}, "4": {"scores": {"A": .6}}}}
                    for h in audit.HEADS}
        result = audit.threshold_comparison(evaluations, recorded, True)
        self.assertEqual(result["selected_tau"], {h: .4 for h in audit.HEADS})
        self.assertTrue(all(r["tau"] == .7 for r in result["posthoc"]))
        self.assertEqual(result["hypotheses"]["P3"]["met"], False)
        equal = scores(.1)
        self.assertEqual(audit.select_threshold(equal), .3)

    def test_gate_failure_is_recorded_without_raising(self):
        groups = {h: {d: {str(k): {"scores": {"A": .6}} for k in (1, 2, 4)}
                      for d in ("Overall", *audit.DOMAINS)} for h in audit.HEADS}
        recorded = json.loads(json.dumps(groups))
        groups["linear"]["Overall"]["2"]["scores"]["A"] += .001
        with contextlib.redirect_stdout(io.StringIO()):
            result = audit.verify_report({"groups": groups}, recorded, False)
            smoke = audit.verify_report({"groups": groups}, recorded, True)
        self.assertFalse(result["G"])
        self.assertEqual(sum(r["status"] == "FAIL" for r in result["checks"]), 1)
        self.assertIsNone(smoke["G"])

    def test_disabled_options_match_original_output_bytes(self):
        original = subprocess.check_output(["git", "show", "39087f1:analysis/draft_overfit_check.py"],
                                           cwd=audit.ROOT, text=True)
        legacy = types.ModuleType("legacy_draft_check")
        legacy.__file__ = str(audit.ROOT / "analysis/draft_overfit_check.py")
        exec(compile(original, legacy.__file__, "exec"), legacy.__dict__)
        scores = {name: .5 for name in audit.SCORES}
        evaluation = {"groups": {h: {d: {k: {"scores": dict(scores), "counts": {}}
                                             for k in ("y0_draft", "1", "2", "4")}
                                      for d in ("Overall", *audit.DOMAINS)} for h in audit.HEADS},
                      "files": {d: ["fixture.png"] for d in audit.DOMAINS}, "wall_seconds": 1., "forward_seconds": .5}
        with tempfile.TemporaryDirectory(dir=audit.ROOT / "runs") as directory:
            outputs = []
            for i, module in enumerate((legacy, audit)):
                out = Path(directory) / str(i)
                with patch.object(sys, "argv", ["audit", "--device", "cpu", "--out-dir", str(out)]), \
                        patch.object(module, "load_pipeline", return_value=(None, {}, {}, {})), \
                        patch.object(module, "train_subset", return_value={d: [None] for d in audit.DOMAINS}), \
                        patch.object(module, "evaluate", return_value=evaluation), contextlib.redirect_stdout(io.StringIO()):
                    module.main()
                outputs.append([(out / name).read_bytes() for name in ("metrics.json", "summary.md")])
                self.assertEqual(sorted(p.name for p in out.iterdir()), ["metrics.json", "summary.md"])
            self.assertEqual(outputs[0], outputs[1])

    def test_threshold_half_matches_default_official_counts(self):
        class Sample:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                image = np.arange(48, dtype=np.float32).reshape(3, 4, 4) / 48
                raw = np.ones((4, 4), np.uint8)
                raw[:, 1:3] = 3
                return image, (raw == 3).astype(np.float32), raw != 0, "fixture", raw

        torch.manual_seed(0)
        encoder = nn.Identity()
        heads = {h: nn.Conv2d(3, 1, 1).requires_grad_(False).eval() for h in audit.HEADS}
        sets = {d: Sample() for d in audit.DOMAINS}
        baseline = audit.evaluate(encoder, heads, {}, sets, [], torch.device("cpu"), 0, 1)
        sweep = audit.threshold_evaluate(encoder, heads, sets, torch.device("cpu"), 0, 1)
        for h in audit.HEADS:
            for d in ("Overall", *audit.DOMAINS):
                self.assertEqual(baseline["groups"][h][d]["y0_draft"], sweep["groups"][h][d]["0.5"])

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
