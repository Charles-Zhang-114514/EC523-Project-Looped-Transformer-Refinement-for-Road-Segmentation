"""Batch-local feature/RGB carrier; no persistence or encoder ownership."""
from dataclasses import dataclass
import torch


@dataclass
class DetailFeatures:
    features: torch.Tensor
    rgb: torch.Tensor

    @property
    def shape(self):
        return self.features.shape

    def float(self):
        return DetailFeatures(self.features.float(), self.rgb.float())

    def clone(self):
        return DetailFeatures(self.features.clone(), self.rgb.clone())
