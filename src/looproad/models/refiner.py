"""Shared recurrence, with an optional finite untied stack for comparison."""
import math

import torch
from torch import nn
import torch.nn.functional as F
from .detail_features import DetailFeatures


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


class ConvUpsampleHead(nn.Module):
    """Mix neighboring token features while decoding a smooth full-size delta."""
    def __init__(self, dimension, detail_head=False):
        super().__init__()
        self.detail_head = detail_head
        self.norm = nn.LayerNorm(dimension)
        self.project = nn.Linear(dimension, 16)
        self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.final = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)
        if detail_head:
            self.detail_encoder = nn.Sequential(
                nn.Conv2d(3,16,3,stride=2,padding=1), nn.GELU(),
                nn.Conv2d(16,32,3,stride=2,padding=1), nn.GELU(),
                nn.Conv2d(32,32,3,padding=1), nn.GELU())
            self.detail_fusion = nn.Sequential(nn.Conv2d(49,16,3,padding=1), nn.GELU())
            self.detail_gate = nn.Conv2d(16,16,1)
            nn.init.zeros_(self.detail_gate.weight)
            nn.init.zeros_(self.detail_gate.bias)

    def forward(self, tokens, grid, rgb=None, answer=None):
        values = self.project(self.norm(tokens)).transpose(1, 2).reshape(tokens.shape[0], 16, grid, grid)
        values = F.interpolate(values, scale_factor=4, mode="bilinear", align_corners=False)
        if self.detail_head:
            if rgb is None or answer is None:
                raise ValueError('Detail output requires RGB and current answer')
            details = self.detail_encoder(rgb)
            feedback = F.avg_pool2d(answer.sigmoid(),4)
            fused = self.detail_fusion(torch.cat((values,details,feedback),dim=1))
            values = values + self.detail_gate(fused)
        values = F.gelu(self.conv1(values))
        values = F.interpolate(values, scale_factor=4, mode="bilinear", align_corners=False)
        return self.final(F.gelu(self.conv2(values)))


class LoopRefiner(nn.Module):
    def __init__(self, in_channels=1024, dimension=256, layers=1, heads=8,
                 mlp_ratio=4., patch=16, image_size=1024,
                 no_mask_feedback=False, reset_state=False, head_type="patch_linear",
                 untied_steps=0, detail_head=False):
        super().__init__()
        if min(in_channels, dimension, layers, heads, patch, image_size) < 1 or mlp_ratio <= 0:
            raise ValueError("Model dimensions must be positive")
        if dimension % heads or dimension % 4 or image_size % patch:
            raise ValueError("Incompatible attention dimension or patch grid")
        if not isinstance(untied_steps, int) or untied_steps < 0:
            raise ValueError("untied_steps must be a nonnegative integer")
        if detail_head and (head_type != "conv_upsample" or untied_steps):
            raise ValueError("Detail recovery requires the shared conv_upsample head")
        self.detail_head = detail_head
        self.untied_steps = untied_steps
        self._step_index = 0
        self.image_size, self.patch = image_size, patch
        self.grid = image_size // patch
        if head_type not in ("patch_linear", "conv_upsample") or (head_type == "conv_upsample" and patch != 16):
            raise ValueError("Unknown head type or conv_upsample patch size other than 16")
        self.head_type = head_type
        self.no_mask_feedback, self.reset_state = no_mask_feedback, reset_state
        self.feature_projection = nn.Linear(in_channels, dimension)
        self.answer_projection = nn.Linear(patch * patch, dimension)
        self.initializer = nn.Linear(dimension, dimension)
        self.register_buffer("position_encoding", sine_position_encoding(self.grid, dimension))
        if untied_steps:
            self.core = nn.ModuleList([
                nn.Sequential(*(TransformerBlock(dimension, heads, mlp_ratio) for _ in range(layers)))
                for _ in range(untied_steps)
            ])
        else:
            # Preserve legacy registration keys, initialization order and RNG use.
            self.core = nn.Sequential(*(TransformerBlock(dimension, heads, mlp_ratio) for _ in range(layers)))
        if head_type == "patch_linear":
            self.output_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, patch * patch))
            nn.init.zeros_(self.output_head[-1].weight)
            nn.init.zeros_(self.output_head[-1].bias)
        else:
            self.output_head = ConvUpsampleHead(dimension, detail_head=detail_head)
        self._initial_state = None
        self.parameter_counts()

    def answer_tokens(self, logits):
        if logits.shape[1:] != (1, self.image_size, self.image_size):
            raise ValueError("Answer shape does not match the configured image size")
        return self.answer_projection(patchify(logits.sigmoid(), self.patch))

    def init(self, features, draft):
        rgb = None
        if self.detail_head:
            if not isinstance(features, DetailFeatures):
                raise ValueError('Detail recovery requires features plus full-resolution RGB')
            features, rgb = features.features, features.rgb
            if rgb.shape != (draft.shape[0],3,self.image_size,self.image_size):
                raise ValueError('RGB must match full-resolution answer')
        if features.ndim != 4 or features.shape[-2:] != (self.grid, self.grid):
            raise ValueError("Feature grid must match image_size / patch (64x64 by default)")
        feature_tokens = self.feature_projection(features.flatten(2).transpose(1, 2))
        initial_state = self.initializer(feature_tokens + self.answer_tokens(draft))
        # Retain the graph for the reset-state ablation; never detach across steps.
        self._initial_state = initial_state
        self._step_index = 0
        return initial_state, (DetailFeatures(feature_tokens,rgb) if self.detail_head else feature_tokens)

    def step(self, state, answer, feature_tokens):
        rgb = None
        if self.detail_head:
            feature_tokens, rgb = feature_tokens.features, feature_tokens.rgb
        if self.untied_steps:
            if self._initial_state is None:
                raise RuntimeError("Call init before using the untied stack")
            if self._step_index >= self.untied_steps:
                raise RuntimeError(f"Untied stack exhausted: step {self._step_index + 1} "
                                   f"exceeds untied_steps={self.untied_steps}; call init for a new trajectory")
        if self.reset_state:
            if self._initial_state is None:
                raise RuntimeError("Call init before using reset_state")
            state = self._initial_state
        feedback = (torch.zeros_like(feature_tokens) if self.no_mask_feedback
                    else self.answer_tokens(answer))
        core = self.core[self._step_index] if self.untied_steps else self.core
        next_state = core(state + feature_tokens + feedback + self.position_encoding)
        correction = (unpatchify(self.output_head(next_state), self.image_size, self.image_size, self.patch)
                      if self.head_type == "patch_linear" else (self.output_head(next_state,self.grid,rgb,answer) if self.detail_head
                            else self.output_head(next_state, self.grid)))
        if self.untied_steps:
            self._step_index += 1
        return next_state, answer + correction

    def parameter_counts(self):
        # ModuleList.parameters() includes every independent core exactly once.
        names = ("core", "feature_projection", "answer_projection", "initializer", "output_head")
        counts = {name: sum(p.numel() for p in getattr(self, name).parameters()) for name in names}
        counts["trainable_total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if self.detail_head:
            counts['detail_added'] = sum(p.numel() for name,p in self.named_parameters()
                                        if name.startswith('output_head.detail_'))
            if counts['detail_added'] != 21680:
                raise AssertionError('Unexpected detail parameter count')
        if counts["core"] > 5_000_000 or counts["trainable_total"] > 10_000_000:
            raise ValueError(f"Refiner parameter budget exceeded: {counts}")
        return counts


def load_initial_refiner(model, state):
    """Strict old-key loading; only newly introduced detail keys may be absent."""
    if not getattr(model,'detail_head',False):
        return model.load_state_dict(state, strict=True)
    keys=set(model.state_dict())
    added={k for k in keys if k.startswith('output_head.detail_')}
    if set(state) != keys-added:
        raise ValueError('Initialization must contain every legacy key and no unexpected key')
    result=model.load_state_dict(state,strict=False)
    if set(result.missing_keys)!=added or result.unexpected_keys:
        raise AssertionError('Unexpected initialization incompatibility')
    return result
