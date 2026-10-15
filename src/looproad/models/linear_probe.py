"""One-layer spatial linear probe."""
import torch.nn.functional as F
from torch import nn


class LinearProbe(nn.Module):
    def __init__(self, in_channels=1024):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, features):
        logits = self.projection(features)
        return F.interpolate(logits, size=(1024, 1024), mode="bilinear", align_corners=False)


def masked_bce_with_logits(logits, target, valid, pos_weight=1.0):
    weight = logits.new_tensor(float(pos_weight))
    loss = F.binary_cross_entropy_with_logits(logits, target.unsqueeze(1),
                                               pos_weight=weight, reduction="none")
    valid = valid.unsqueeze(1).to(dtype=loss.dtype)
    return (loss * valid).sum() / valid.sum().clamp_min(1)
