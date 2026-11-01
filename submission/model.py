"""Student-owned adapter for the interface in project_a/cli/README.md:45-66.

The existing LoopRefiner math and state-dictionary names are retained through
inheritance. The runner alone owns K. No course starter implementation is used.
"""
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from looproad.models.refiner import LoopRefiner

DEFAULT_CONFIG = dict(dimension=256, layers=1, heads=8, mlp_ratio=4., patch=16,
                      image_size=1024, head_type='conv_upsample',
                      reset_state=False, no_mask_feedback=False)


class SubmissionRefiner(LoopRefiner):
    """Keep the projected image tokens alongside z in the runner-owned state."""
    def initialize(self, features, initial_logits):
        # Corresponds to predict.py:61; initialization never calls self.core.
        return super().init(features, initial_logits)

    def step(self, features, state, previous_logits):
        # features is invariant in the official loop; its projection is cached
        # in state. The previous_logits argument is BOTH feedback and residual
        # base, matching predict.py:67-68 and the residual warning in README:100.
        latent, feature_tokens = state
        next_latent, next_logits = super().step(latent, previous_logits, feature_tokens)
        return (next_latent, feature_tokens), next_logits


def refiner_from_config(feature_channels, config=None):
    """Restore architecture only; no loop count, paths or optimizer settings."""
    settings = dict(DEFAULT_CONFIG)
    if config is not None:
        if set(config) - set(settings):
            raise ValueError('Unexpected refiner architecture setting')
        settings.update(config)
    if settings['reset_state'] or settings['no_mask_feedback']:
        raise ValueError('Submit the unmodified e0 model; interventions belong to the runner')
    return SubmissionRefiner(in_channels=feature_channels, **settings)


def build_refiner(feature_channels):
    """Exact staff factory signature, constructing the selected v2 defaults."""
    return refiner_from_config(feature_channels)


def load_refiner(checkpoint, feature_channels, device):
    """Read tensor weights and architecture metadata without Python pickle."""
    with safe_open(str(checkpoint), framework='pt', device='cpu') as reader:
        metadata = reader.metadata() or {}
    config = json.loads(metadata['config']) if 'config' in metadata else None
    model = refiner_from_config(feature_channels, config).to(device=device, dtype=torch.float32)
    model.load_state_dict(load_file(str(checkpoint), device='cpu'), strict=True)
    return model.eval()
