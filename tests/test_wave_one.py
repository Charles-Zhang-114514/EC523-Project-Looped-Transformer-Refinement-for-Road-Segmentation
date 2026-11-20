"""Wave-1 adapter gradients, output identity and shared-encoder TTA regression."""
import copy
import subprocess
import types
import unittest

import torch
from torch import nn

from analysis import evaluate_draft_tta as tta
from looproad.models.lora import LoRALinear, LoRAEncoder
from looproad.encoders.huggingface import DinoV3WebViTL16Encoder


class WaveOneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)

    def test_zero_B_matches_linear_and_only_adapters_get_gradients(self):
        linear = nn.Linear(12, 8)
        wrapper = LoRALinear(copy.deepcopy(linear))
        inputs = torch.randn(2, 5, 12)
        self.assertTrue(torch.equal(linear(inputs), wrapper(inputs)))
        wrapper(inputs).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in wrapper.base.parameters()))
        self.assertIsNotNone(wrapper.lora_A.grad)
        self.assertIsNotNone(wrapper.lora_B.grad)
        self.assertGreater(wrapper.lora_B.grad.abs().sum().item(), 0.)

    def test_tiny_hf_encoder_checkpointing_has_adapter_gradients(self):
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        config = DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                               num_attention_heads=4, patch_size=16, num_register_tokens=0)
        base = DinoV3WebViTL16Encoder(DINOv3ViTModel(config), (.5,)*3, (.5,)*3, autocast_bf16=False)
        image = torch.randn(2, 3, 32, 32)
        reference = base(image)
        lora = LoRAEncoder(base)
        lora.eval()
        self.assertTrue(torch.equal(reference, lora(image)["refiner"]))
        lora.train()
        lora(image)["refiner"].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in lora.parameters() if not p.requires_grad))
        self.assertTrue(all(p.grad is not None for p in lora.parameters() if p.requires_grad))
        self.assertEqual(len(lora.names), 8)
        self.assertTrue(all("lora_" in key for key in lora.adapter_state()))

    def test_two_cpu_images_single_head_and_zero_bias_match_historical_tta(self):
        old = types.ModuleType("historical_tta")
        old.__file__ = tta.__file__
        source = subprocess.check_output(["git", "show", "cf1db5c:analysis/evaluate_draft_tta.py"],
                                         cwd=tta.ROOT, text=True)
        exec(compile(source, old.__file__, "exec"), old.__dict__)

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, image):
                self.calls += 1
                return image

        encoder = Encoder()
        head = nn.Conv2d(3, 1, 3, padding=1).eval()
        for image in torch.randn(2, 3, 32, 32).split(1):
            with torch.no_grad():
                original = head(encoder(image))
                reference = old.average_d4_logits(lambda view: head(encoder(view)), image, original)
                features, actual = tta.ensemble_drafts(encoder, [head], image)
            self.assertTrue(torch.equal(reference, actual[0]))
            self.assertTrue(torch.equal(features, image))
            self.assertIs(tta.output_bias(actual[0], 0), actual[0])
            self.assertTrue((old.binary_mask(reference) == tta.binary_mask(actual[0])).all())
        encoder.calls = 0
        tta.ensemble_drafts(encoder, [head, head], torch.randn(1, 3, 32, 32))
        self.assertEqual(encoder.calls, 8)

    def test_small_ensemble_bias_pipeline_uses_official_metrics_and_cleans_cache(self):
        import argparse
        import json
        import os
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        import numpy as np
        from scripts.train_refiner import LoopRefiner

        class Encoder(nn.Module):
            def forward(self, image):
                return nn.functional.avg_pool2d(image, 4).repeat(1, 3, 1, 1)[:, :8]

        encoder = Encoder().eval()
        head = nn.Sequential(nn.Conv2d(8, 1, 1), nn.Upsample(scale_factor=4, mode="bilinear",
                                                            align_corners=False)).eval()
        model = LoopRefiner(8, dimension=16, heads=4, patch=4, image_size=32).eval()
        datasets = {}
        for index, domain in enumerate(tta.runner.DOMAINS):
            image = np.random.default_rng(index).random((3, 32, 32), dtype=np.float32)
            raw = np.where(image[0] >= .5, 3, 1).astype(np.uint8)
            datasets[domain] = [(image, (raw == 3).astype(np.float32), raw != 0, domain, raw)]
        with tempfile.TemporaryDirectory(dir=tta.ROOT/"runs") as temporary:
            root = Path(temporary)
            scratch = root/"scratch"
            scratch.mkdir()
            checkpoint = root/"head.pt"
            torch.save(dict(encoder=tta.runner.y0.ENCODER, head="conv_up", draft_layers="last",
                            state_dict=head.state_dict()), checkpoint)
            args = argparse.Namespace(device="cpu", heads=[checkpoint, checkpoint], loader_workers=0,
                                      metric_workers=1, out_dir=root/"report")
            with patch.object(tta.runner.y0, "load_frozen_probe", return_value=(encoder, head, {})), \
                 patch.object(tta.previous.shared, "load_model", return_value=(model, 1)), \
                 patch.object(tta.runner, "protocol_datasets", return_value=datasets), \
                 patch.dict(os.environ, {"TMPDIR": str(scratch)}):
                tta.ensemble_bias_run(args)
            report = json.loads((args.out_dir/"metrics.json").read_text())
            self.assertEqual(list(scratch.glob("ec523-wave1-*")), [])
            self.assertEqual(len(report["bias_selection_L"]), 31)
            selected = report["selected_ensemble"]
            self.assertEqual(report["bias_selection_groups"]["0.0"],
                             report["splits"]["Val-sel"]["groups"][selected])
            self.assertTrue(all(row["masks_equal"] for row in report["regression"]))
