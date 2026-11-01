"""Frozen local DINOv3-web + BN probe, aligned with baseline.py:1-6.

Reuse the project's encoder and probe forward implementations, as used by
scripts/train_refiner.py:frozen_draft. No image processor resizes inputs, no
repository-id lookup occurs and inference never opens an annotation file.
"""
import json
from pathlib import Path
import sys

import torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from looproad.encoders.huggingface import DinoV3WebViTL16Encoder
from looproad.models.linear_probe import LinearProbe

ENCODER_FILES = ('config.json', 'preprocessor_config.json', 'model.safetensors')


class FrozenBaseline(nn.Module):
    feature_channels = 1024

    def __init__(self, encoder, probe):
        super().__init__()
        self.encoder, self.probe = encoder, probe
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Freeze BN running statistics even when a caller invokes train().
        super().train(False)
        return self

    def forward(self, images):
        if images.shape[1:] != (3, 1024, 1024):
            raise ValueError('The submitted model requires full 1024x1024 RGB images; no resizing is applied')
        with torch.no_grad():
            features = self.encoder(images)
            initial_logits = self.probe(features).float()
        if features.shape[1:] != (1024, 64, 64) or initial_logits.shape != (images.shape[0], 1, 1024, 1024):
            raise ValueError('Frozen baseline output shape violates the feature/logit contract')
        return features, initial_logits


def load_baseline(checkpoint, encoder_dir, device):
    """Load only explicitly supplied files, including normalization statistics."""
    directory = Path(encoder_dir).resolve(strict=True)
    if not directory.is_dir() or any(not (directory/name).is_file() for name in ENCODER_FILES):
        raise FileNotFoundError('Encoder directory must contain config.json, preprocessor_config.json and model.safetensors')
    processor = json.loads((directory/'preprocessor_config.json').read_text())
    mean, std = processor['image_mean'], processor['image_std']
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError('Invalid three-channel normalization configuration')
    backbone = AutoModel.from_pretrained(str(directory), local_files_only=True, attn_implementation='sdpa')
    encoder = DinoV3WebViTL16Encoder(backbone, mean, std, autocast_bf16=True)
    if encoder.out_channels != 1024 or encoder.patch_size != 16:
        raise ValueError('Expected DINOv3 ViT-L/16 features with 1024 channels')
    probe = LinearProbe(1024, head_type='bn_linear')
    probe.load_state_dict(load_file(str(checkpoint), device='cpu'), strict=True)
    return FrozenBaseline(encoder, probe).to(device).eval()
