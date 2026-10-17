"""Frozen ResNet feature encoders."""
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Layer3(nn.Module):
    """ImageNet-normalized ResNet-50 stem through layer3 (output stride 16)."""

    out_channels = 1024
    stride = 16

    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V2, autocast_bf16=True):
        super().__init__()
        if weights is not None:
            checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Expected cached weights at {checkpoint}; downloads are disabled")
        model = resnet50(weights=weights)
        self.features = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                                      model.layer1, model.layer2, model.layer3)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.autocast_bf16 = bool(autocast_bf16)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        super().train(False)

    def train(self, mode=True):
        """Keep all batch normalization layers in inference mode."""
        super().train(False)
        return self

    def forward(self, images):
        images = (images - self.mean) / self.std
        use_bf16 = self.autocast_bf16 and images.device.type == "cuda"
        with torch.no_grad(), torch.autocast(device_type=images.device.type, dtype=torch.bfloat16,
                                             enabled=use_bf16):
            features = self.features(images)
        return features.float()


def build_encoder(name="resnet50_layer3", **kwargs):
    if name == "resnet50_layer3":
        return ResNet50Layer3(**kwargs)
    from .huggingface import build_huggingface_encoder

    return build_huggingface_encoder(name, **kwargs)
