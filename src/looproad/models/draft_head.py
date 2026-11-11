"""Convolutional draft heads and frozen multi-layer DINOv3 features.

The existing LinearProbe and encoder forward methods are unchanged. Block
numbers are one-based: hidden_states[0] is the input to block 1, and entry k
is block k's output (transformers 4.57.6 utils/generic.py:1023-1033). Selected
blocks use the pretrained frozen final LayerNorm, just as last_hidden_state
does for the final block (modeling_dinov3_vit.py:529-534).
"""
import argparse

import torch
from torch import nn

from looproad.encoders.huggingface import patch_tokens_to_map
from looproad.encoders.resnet import build_encoder


ENCODER = "dinov3_vitl16_web"


def parse_draft_layers(value):
    """Return 'last' or unique, increasing one-based transformer block numbers."""
    if value == "last":
        return "last"
    try:
        layers = tuple(int(part) for part in value.split(","))
        if not layers or min(layers) < 1 or tuple(sorted(set(layers))) != layers:
            raise ValueError
        return layers
    except (ValueError, AttributeError) as error:
        raise argparse.ArgumentTypeError("Use last or increasing unique block numbers, e.g. 6,12,18,24") from error


def layer_text(layers):
    return "last" if layers == "last" else ",".join(map(str, layers))


class DraftFeatureEncoder(nn.Module):
    """Keep the original last-layer feature for refiner compatibility.

The draft maps are ephemeral batch-local tensors; no feature cache or forward
hook is used. CPU uses float32; CUDA uses the original encoder's bf16 policy.
"""
    stride = 16

    def __init__(self, encoder, layers="last"):
        super().__init__()
        self.encoder = encoder
        self.layers = parse_draft_layers(layers) if isinstance(layers, str) else layers
        self.out_channels = encoder.out_channels
        self.draft_channels = encoder.out_channels
        count = encoder.model.config.num_hidden_layers
        if self.layers != "last" and max(self.layers) > count:
            raise ValueError(f"Requested block {max(self.layers)} but the encoder has only {count} blocks")
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, images):
        with torch.no_grad():
            if self.layers == "last":
                features = self.encoder(images)
                return {"draft": (features,), "refiner": features}
            base = self.encoder
            normalized = base.normalize(images)
            with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16,
                                enabled=base.autocast_bf16 and images.device.type == "cuda"):
                output = base.model(pixel_values=normalized, output_hidden_states=True)
                height, width = (size // base.patch_size for size in images.shape[-2:])

                def spatial(tokens):
                    patches = tokens[:, 1 + base.num_register_tokens:, :]
                    return patch_tokens_to_map(patches, height, width).float()

                last = spatial(output.last_hidden_state)
                # hidden_states[-1] is pre-final-LN in this transformers model.
                maps = tuple(last if k == base.model.config.num_hidden_layers else
                             spatial(base.model.norm(output.hidden_states[k])) for k in self.layers)
            return {"draft": maps, "refiner": last}


class ConvUpDraftHead(nn.Module):
    """Per-layer BN -> concat -> 1x1 -> four convolutional upsampling stages."""
    head_type = "conv_up"

    def __init__(self, in_channels=1024, input_layers=1):
        super().__init__()
        if in_channels < 1 or input_layers < 1:
            raise ValueError("Feature channels and layer count must be positive")
        self.normalizations = nn.ModuleList(nn.BatchNorm2d(in_channels) for _ in range(input_layers))
        self.projection = nn.Conv2d(in_channels * input_layers, 256, 1)
        channels = (256, 128, 64, 32, 16)
        self.stages = nn.Sequential(*[nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(source, target, 3, padding=1), nn.BatchNorm2d(target), nn.GELU())
            for source, target in zip(channels, channels[1:])])
        self.output = nn.Conv2d(16, 1, 1)

    @staticmethod
    def refiner_features(features):
        """Expose only the original 1024-channel final-layer map to LoopRefiner."""
        return features["refiner"]

    def forward(self, features):
        maps = features["draft"] if isinstance(features, dict) else features
        if torch.is_tensor(maps):
            maps = (maps,)
        if len(maps) != len(self.normalizations):
            raise ValueError("Feature layer count does not match the draft head")
        values = [normalization(value) for normalization, value in zip(self.normalizations, maps)]
        return self.output(self.stages(self.projection(torch.cat(values, dim=1))))


def build_draft(layers="last"):
    layers = parse_draft_layers(layers) if isinstance(layers, str) else layers
    encoder = DraftFeatureEncoder(build_encoder(ENCODER), layers)
    head = ConvUpDraftHead(encoder.draft_channels, 1 if layers == "last" else len(layers))
    return encoder, head


def load_frozen_conv_draft(checkpoint, device):
    """Load the new checkpoint format; legacy loading is left in its old path."""
    if checkpoint.get("encoder") != ENCODER or checkpoint.get("head") != "conv_up":
        raise ValueError("Expected a DINOv3-web conv_up draft checkpoint")
    if float(checkpoint.get("pos_weight", -1)) != 4:
        raise ValueError("Expected a draft checkpoint trained with pos_weight=4")
    encoder, head = build_draft(checkpoint["draft_layers"])
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    return encoder.to(device).requires_grad_(False).eval(), head.to(device).requires_grad_(False).eval()
