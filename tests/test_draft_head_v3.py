"""Draft v3 numerical parity, D4 inference, and decision rule coverage."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from analysis import evaluate_draft_tta as evaluator
from analysis import summarize_draft_head_v3 as summary
from looproad.data.augmentation import d4_transform
from looproad.models.draft_head import ConvUpDraftHead
from scripts import train_draft_head as trainer


def tile(index, domain):
    image = np.random.default_rng(index).random((3, 32, 32), dtype=np.float32)
    labels = np.where(image[0] >= .5, 3, 1).astype(np.uint8)
    labels[:2, :2] = 0
    return image, (labels == 3).astype(np.float32), labels != 0, f"{domain}_{index}", labels


class DraftV3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_all_d4_transforms_match_training_and_invert_on_a_rectangle(self):
        image = torch.arange(30).view(1, 2, 3, 5)
        for code in range(8):
            with self.subTest(code=code):
                transformed = evaluator.d4(image, code)
                np.testing.assert_array_equal(transformed.numpy(), d4_transform(image.numpy(), code))
                self.assertTrue(torch.equal(evaluator.d4(transformed, code, inverse=True), image))

    def test_d4_averages_inverse_logits_and_reuses_identity(self):
        logits = torch.arange(-30, 30, dtype=torch.float32).view(1, 1, 6, 10)
        views = []

        def identity(image):
            views.append(image)
            return image

        self.assertTrue(torch.equal(evaluator.average_d4_logits(identity, logits, logits), logits))
        self.assertEqual(len(views), 7)
        for code, view in enumerate(views, 1):
            self.assertTrue(torch.equal(view, evaluator.d4(logits, code)))

        values = iter([2, 3, 4, -5, -6, 7, 8])
        averaged = evaluator.average_d4_logits(
            lambda image: torch.full_like(image, next(values)), logits, torch.ones_like(logits))
        self.assertTrue(torch.equal(averaged, torch.full_like(logits, 1.75)))

    def test_cosine_lr_uses_300_update_warmup_and_ends_at_zero(self):
        steps, lr = 3160, .001
        self.assertAlmostEqual(trainer.cosine_lr(0, steps, lr), lr / 300, places=15)
        self.assertEqual(trainer.cosine_lr(299, steps, lr), lr)
        self.assertLess(trainer.cosine_lr(300, steps, lr), lr)
        self.assertAlmostEqual(trainer.cosine_lr(1729, steps, lr), lr / 2, places=15)
        self.assertEqual(trainer.cosine_lr(steps - 1, steps, lr), 0.)
        self.assertEqual([trainer.cosine_lr(i, 2, lr) for i in range(2)], [lr, 0.])

    def test_default_one_batch_matches_pre_v3_loss_weights_bn_and_rng(self):
        old = types.ModuleType("pre_v3_draft")
        old.__file__ = trainer.__file__
        source = subprocess.check_output(
            ["git", "show", "ecb4632:scripts/train_draft_head.py"], cwd=trainer.ROOT, text=True)
        exec(compile(source, old.__file__, "exec"), old.__dict__)

        def build(_layers):
            encoder = nn.Sequential(nn.AvgPool2d(16), nn.Conv2d(3, 8, 1)).requires_grad_(False)
            return encoder, ConvUpDraftHead(in_channels=8)

        datasets = {"Urban": [tile(0, "Urban")], "Rural": [tile(1, "Rural")]}
        partition = {domain: types.SimpleNamespace(pairs=[]) for domain in trainer.DOMAINS}
        scores = dict(road_iou=.5, cldice=.5, boundary_iou=.5, quality=.5)
        results = []
        with tempfile.TemporaryDirectory(dir=trainer.ROOT / "runs") as tmp:
            for version, module in enumerate((old, trainer)):
                parent = Path(tmp) / str(version)
                argv = ["train_draft_head.py", "--device", "cpu", "--epochs", "1", "--batch-size", "2",
                        "--loader-workers", "0", "--eval-loader-workers", "0", "--metric-workers", "1",
                        "--out-dir", str(parent / "report"), "--ckpt-dir", str(parent / "checkpoints")]
                with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), \
                     patch.object(module, "build_draft", build), \
                     patch.object(module, "protocol_datasets", return_value=partition), \
                     patch.object(module.probe, "RoadDataset", side_effect=lambda split, d, **_: datasets[d]), \
                     patch.object(module, "evaluate", return_value={"groups": dict.fromkeys(evaluator.DOMAINS, scores)}):
                    module.main()
                output = json.loads((parent / "report/metrics.json").read_text())
                saved = torch.load(parent / "checkpoints/best.pt", map_location="cpu", weights_only=True)
                self.assertFalse((parent / "checkpoints/last.pt").exists())
                batch = output["epochs"][0]["batches"][0]
                config = {k: v for k, v in saved["config"].items() if k not in ("ckpt_dir", "out_dir")}
                results.append((batch["loss"], saved["state_dict"], config, torch.get_rng_state(),
                                random.getstate(), np.random.get_state()))
        a, b = results
        self.assertEqual(a[0], b[0])
        self.assertTrue(all(torch.equal(a[1][name], b[1][name]) for name in a[1]))
        self.assertEqual(a[2], b[2])
        self.assertTrue(torch.equal(a[3], b[3]))
        self.assertEqual(a[4], b[4])
        self.assertEqual(a[5][0], b[5][0])
        np.testing.assert_array_equal(a[5][1], b[5][1])
        self.assertEqual(a[5][2:], b[5][2:])

    def test_tta_stream_scores_official_counts_with_original_features_and_exact_k(self):
        from scripts.train_refiner import LoopRefiner, inference_steps

        class Features(nn.Module):
            def forward(self, image):
                values = nn.functional.avg_pool2d(image, 4).repeat(1, 3, 1, 1)[:, :8]
                return {"draft": image[:, :1] - .5, "refiner": values}

        class Head(nn.Module):
            def forward(self, inputs):
                return inputs["draft"]

            def refiner_features(self, inputs):
                return inputs["refiner"]

        model = LoopRefiner(8, dimension=16, heads=4, patch=4, image_size=32).eval()
        encoder, head = Features(), Head()
        urban, rural = tile(0, "Urban"), tile(1, "Rural")
        datasets = {"Urban": [urban], "Rural": [rural]}
        calls = []
        with patch.object(evaluator.runner, "inference_steps", wraps=inference_steps) as infer:
            handle = model.core.register_forward_hook(lambda *_: calls.append(1))
            try:
                report = evaluator.evaluate(encoder, head, model, datasets, (1, 2, 4, 8),
                                            torch.device("cpu"), 0, 1, "d4")
            finally:
                handle.remove()
        self.assertEqual(len(calls), 2 * 2 * 8)
        for index, item in enumerate((urban, rural)):
            expected = encoder(torch.tensor(item[0]).unsqueeze(0))["refiner"]
            for invocation in infer.call_args_list[2 * index:2 * index + 2]:
                self.assertTrue(torch.equal(invocation.args[1], expected))
        for mode in ("none", "d4"):
            for key in ("y0_draft", "1", "2", "4", "8"):
                observed = report["groups"][mode]["Overall"][key]["scores"]
                self.assertEqual(observed["quality"], 1.)
        self.assertEqual(len(report["images"]), 2)
        self.assertTrue(all(row["draft_seconds"]["d4"] > row["draft_seconds"]["none"] for row in report["images"]))

    def test_preregistered_rules_use_all_expected_checkpoint_and_k_comparisons(self):
        training = {}
        averages = {}
        for arm, baseline in zip(summary.ARMS, (.530, .532, .534)):
            averages[arm] = {"Overall": {"quality": baseline}}
            for seed in (0, 1):
                training[arm, seed] = dict(val_rep={"Overall": {"quality": baseline + (seed - .5) * .001}},
                                           val_sel_curve=[.52, .53, .529], best_selection_A=.53,
                                           best_epoch=6 + seed)
        tta_rows = [dict(arm=arm, seed=seed, split="Val-rep", domain="Overall", delta={"quality": .005})
                    for arm, seed in training]
        refiner = [dict(K=k, domain="Overall", delta_vs_recorded={"quality": .003 if k != "8" else -.01})
                   for k in summary.KS]
        observed = summary.decide(training, averages, tta_rows, refiner)
        self.assertTrue(all(observed[f"P{i}"]["met"] for i in range(1, 6)))
        self.assertEqual(observed["R2"]["selected_arm"], "v3b")
        self.assertTrue(observed["R3"]["train_to_last"])
        self.assertTrue(observed["R4"]["include_tta"])
        training["v3a", 0]["val_sel_curve"][-1] = .52
        tta_rows[-1]["delta"]["quality"] = .002
        refiner[2]["delta_vs_recorded"]["quality"] = .0015
        observed = summary.decide(training, averages, tta_rows, refiner)
        self.assertFalse(observed["P3"]["met"])
        self.assertFalse(observed["P4"]["met"])
        self.assertTrue(observed["P5"]["met"])
        self.assertEqual(observed["R3"]["median_best_epoch"], 6.5)
        self.assertEqual(observed["R3"]["fixed_epochs"], 7)
        self.assertFalse(observed["R4"]["include_tta"])
