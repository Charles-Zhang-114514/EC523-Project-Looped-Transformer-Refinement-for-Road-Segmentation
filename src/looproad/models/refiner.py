"""A shared global-attention correction core with externally controlled looping."""
import math

import torch
from torch import nn
import torch.nn.functional as F


def patchify(answer, patch=16):
    """[B,1,H,W] -> [B,(H/p)*(W/p),p*p], in row-major patch order."""
    if answer.ndim != 4 or answer.shape[1] != 1:
        raise ValueError("Expected a single-channel [B,1,H,W] answer")
    if answer.shape[-2] % patch or answer.shape[-1] % patch:
        raise ValueError("Answer dimensions must be divisible by patch size")
    return F.unfold(answer, kernel_size=patch, stride=patch).transpose(1, 2)


def unpatchify(tokens, height=1024, width=1024, patch=16):
    """Invert patchify without interpolation, overlap or a learned decoder."""
    if height % patch or width % patch or tokens.ndim != 3:
        raise ValueError("Invalid patch grid")
    if tokens.shape[1:] != ((height // patch) * (width // patch), patch * patch):
        raise ValueError("Token shape does not match the requested image grid")
    return F.fold(tokens.transpose(1, 2), output_size=(height, width),
                  kernel_size=patch, stride=patch)


def sine_position_encoding(grid, dimension):
    if dimension % 4:
        raise ValueError("2D sinusoidal encoding requires dimension divisible by four")
    axis = torch.arange(grid, dtype=torch.float32)
    row, col = torch.meshgrid(axis, axis, indexing="ij")
    frequency = torch.exp(-math.log(10000.) * torch.arange(dimension // 4, dtype=torch.float32)
                          / (dimension // 4))
    x, y = col.flatten()[:, None] * frequency, row.flatten()[:, None] * frequency
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1).unsqueeze(0)


class TransformerBlock(nn.Module):
    """Pre-LN global attention followed by a pre-LN GELU MLP, without dropout."""
    def __init__(self, dimension, heads, mlp_ratio):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dimension)
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.projection = nn.Linear(dimension, dimension)
        self.norm2 = nn.LayerNorm(dimension)
        hidden = int(dimension * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, dimension))

    def forward(self, tokens):
        batch, count, dimension = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(batch, count, 3, self.heads, dimension // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0., is_causal=False)
        tokens = tokens + self.projection(attended.transpose(1, 2).reshape(batch, count, dimension))
        return tokens + self.mlp(self.norm2(tokens))


class LoopRefiner(nn.Module):
    def __init__(self, in_channels=1024, dimension=256, layers=1, heads=8,
                 mlp_ratio=4., patch=16, image_size=1024,
                 no_mask_feedback=False, reset_state=False):
        super().__init__()
        if min(in_channels, dimension, layers, heads, patch, image_size) < 1 or mlp_ratio <= 0:
            raise ValueError("Model dimensions must be positive")
        if dimension % heads or dimension % 4 or image_size % patch:
            raise ValueError("Incompatible attention dimension or patch grid")
        self.image_size, self.patch = image_size, patch
        self.grid = image_size // patch
        self.no_mask_feedback, self.reset_state = no_mask_feedback, reset_state
        self.feature_projection = nn.Linear(in_channels, dimension)
        self.answer_projection = nn.Linear(patch * patch, dimension)
        self.initializer = nn.Linear(dimension, dimension)
        self.register_buffer("position_encoding", sine_position_encoding(self.grid, dimension))
        self.core = nn.Sequential(*(TransformerBlock(dimension, heads, mlp_ratio) for _ in range(layers)))
        self.output_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, patch * patch))
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)
        self._initial_state = None
        self.parameter_counts()

    def answer_tokens(self, logits):
        if logits.shape[1:] != (1, self.image_size, self.image_size):
            raise ValueError("Answer shape does not match the configured image size")
        return self.answer_projection(patchify(logits.sigmoid(), self.patch))

    def init(self, features, draft):
        if features.ndim != 4 or features.shape[-2:] != (self.grid, self.grid):
            raise ValueError("Feature grid must match image_size / patch (64x64 by default)")
        feature_tokens = self.feature_projection(features.flatten(2).transpose(1, 2))
        initial_state = self.initializer(feature_tokens + self.answer_tokens(draft))
        # Retain the graph for the reset-state ablation; never detach across steps.
        self._initial_state = initial_state
        return initial_state, feature_tokens

    def step(self, state, answer, feature_tokens):
        if self.reset_state:
            if self._initial_state is None:
                raise RuntimeError("Call init before using reset_state")
            state = self._initial_state
        feedback = (torch.zeros_like(feature_tokens) if self.no_mask_feedback
                    else self.answer_tokens(answer))
        next_state = self.core(state + feature_tokens + feedback + self.position_encoding)
        correction = unpatchify(self.output_head(next_state), self.image_size, self.image_size, self.patch)
        return next_state, answer + correction

    def parameter_counts(self):
        names = ("core", "feature_projection", "answer_projection", "initializer", "output_head")
        counts = {name: sum(p.numel() for p in getattr(self, name).parameters()) for name in names}
        counts["trainable_total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if counts["core"] > 5_000_000 or counts["trainable_total"] > 10_000_000:
            raise ValueError(f"Refiner parameter budget exceeded: {counts}")
        return counts
