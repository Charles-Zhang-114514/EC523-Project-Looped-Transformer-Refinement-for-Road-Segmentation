"""Trainable low-rank adapters for the existing offline DINOv3 spatial encoder."""
import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=16, alpha=16):
        super().__init__()
        self.base = linear.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs):
        update = nn.functional.linear(nn.functional.linear(inputs, self.lora_A), self.lora_B)
        return self.base(inputs) + update * self.scale


class LoRAEncoder(nn.Module):
    def __init__(self, base, checkpointing=True):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.out_channels = base.out_channels
        self.names = []
        for index, block in enumerate(base.model.layer):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                original = getattr(block.attention, name)
                setattr(block.attention, name, LoRALinear(original))
                self.names.append(f"layer.{index}.attention.{name}")
        if checkpointing:
            base.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def train(self, mode=True):
        super().train(mode)
        self.base.model.train(mode)
        return self

    def forward(self, images):
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16,
                            enabled=images.device.type == "cuda"):
            features = self.base.extract(self.base.normalize(images))
        return {"draft": (features.float(),), "refiner": features.float()}

    def adapter_state(self):
        return {name: parameter.detach().cpu().clone() for name, parameter in self.named_parameters()
                if parameter.requires_grad}

    def load_adapters(self, values):
        parameters = dict(self.named_parameters())
        expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
        if set(values) != expected:
            raise ValueError("Adapter checkpoint does not match the installed projections")
        with torch.no_grad():
            for name, value in values.items():
                parameters[name].copy_(value)
