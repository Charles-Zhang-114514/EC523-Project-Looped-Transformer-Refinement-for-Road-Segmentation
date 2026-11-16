"""CPU memorization checks for the complete, unmodified conv_up head.

The approved fast-test resolution is 16x16 features -> 256x256 logits;
all channels, layers, optimizer settings and 300 updates remain unchanged.
This checks representational capacity, not validation quality. Diagnostic
logits use eval mode, matching selection/reporting after a training epoch.
"""
import json
import os
import time
import unittest

import numpy as np
import torch

from looproad.models.draft_head import ConvUpDraftHead
from looproad.models.linear_probe import masked_bce_with_logits
from scripts.train_linear_probe import official_metrics


def logit_statistics(logits):
    return {"mean": float(logits.mean()), "maximum": float(logits.max()),
            "minimum": float(logits.min()),
            "predicted_road_fraction": float((logits.sigmoid() >= .5).float().mean())}


class DraftHeadOverfitTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
        torch.set_num_threads(min(8, available))

    def run_overfit(self, input_layers):
        started = time.perf_counter()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            # Independent Gaussian inputs for both samples; no encoder or
            # label-derived feature channels. Four layers get their own BN.
            features = torch.randn(2, 1024 * input_layers, 16, 16)
            maps = features.chunk(input_layers, dim=1)
            head = ConvUpDraftHead(1024, input_layers)
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0)
            target = torch.zeros(2, 256, 256)
            target[0, 64:76, :] = 1
            target[0, :, 160:172] = 1
            target[1, 144:156, :] = 1
            target[1, :, 80:92] = 1
            valid = torch.ones_like(target, dtype=torch.bool)
            head.eval()
            with torch.no_grad():
                before = logit_statistics(head(maps))
            losses, first_train_statistics = [], None
            for step in range(1, 301):
                head.train()
                optimizer.zero_grad(set_to_none=True)
                logits = head(maps)
                loss = masked_bce_with_logits(logits, target, valid, pos_weight=4.)
                self.assertTrue(torch.isfinite(loss).item(), f"Nonfinite loss at step {step}")
                losses.append(float(loss.detach()))
                if step == 1:
                    first_train_statistics = logit_statistics(logits.detach())
                loss.backward()
                optimizer.step()
                if step == 1:
                    head.eval()
                    with torch.no_grad():
                        after_one = logit_statistics(head(maps))
            head.eval()
            with torch.no_grad():
                final = head(maps)
                final_loss = float(masked_bce_with_logits(final, target, valid, pos_weight=4.))
            self.assertEqual(tuple(final.shape), (2, 1, 256, 256))
            self.assertTrue(torch.isfinite(final).all().item())
            prediction = (final.sigmoid() >= .5).squeeze(1).numpy().astype(np.float32)
            labels = np.where(target.numpy() == 1, 3, 1).astype(np.uint8)
            metrics = official_metrics()
            counts = metrics.Counts()
            for predicted, raw in zip(prediction, labels):
                counts = counts + metrics.image_counts(predicted, raw)
            iou = metrics.summarize(counts)["road_iou"]
            result = {"input_layers": input_layers, "feature_channels": 1024 * input_layers,
                      "steps": 300, "road_fraction": float(target.mean()),
                      "initial_train_loss": losses[0], "last_train_loss": losses[-1],
                      "final_eval_loss": final_loss, "final_iou": iou,
                      "final_maximum_logit": float(final.max()),
                      "before_eval": before, "after_step1_eval": after_one,
                      "step1_train_mode": first_train_statistics,
                      "loss_curve": {str(k): losses[k - 1] for k in (1, 2, 10, 50, 100, 200, 300)},
                      "seconds": time.perf_counter() - started}
            print("OVERFIT " + json.dumps(result, allow_nan=False), flush=True)
            self.assertLess(losses[-1], .3 * losses[0])
            self.assertLess(final_loss, .3 * losses[0])
            self.assertGreater(iou, .8)
            self.assertGreater(float(final.max()), 0.)

    def test_last_layer_overfits_fixed_random_features(self):
        self.run_overfit(1)

    def test_four_layers_overfit_fixed_random_features(self):
        self.run_overfit(4)


if __name__ == "__main__":
    unittest.main()
