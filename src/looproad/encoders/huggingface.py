"""Frozen Hugging Face vision encoders that preserve the 1024-pixel input grid."""
from functools import lru_cache

import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel


@lru_cache(maxsize=None)
def processor_statistics(repo_id):
    """Read normalization values only; never run the processor on an image."""
    processor = AutoImageProcessor.from_pretrained(repo_id, local_files_only=True)
    if processor.image_mean is None or processor.image_std is None:
        raise ValueError(f"No image_mean/image_std are configured for {repo_id}")
    return tuple(float(value) for value in processor.image_mean), tuple(
        float(value) for value in processor.image_std
    )


def patch_tokens_to_map(tokens, grid_height, grid_width):
    """Invert row-major flatten-and-transpose patch tokenization."""
    batch, count, channels = tokens.shape
    if count != grid_height * grid_width:
        raise ValueError(f"Expected {grid_height * grid_width} patch tokens, got {count}")
    return tokens.reshape(batch, grid_height, grid_width, channels).permute(0, 3, 1, 2).contiguous()


class FrozenHFSpatialEncoder(nn.Module):
    repo_id = None
    model_kwargs = {}
    out_channels = None
    stride = 16

    def __init__(self, model, image_mean, image_std, autocast_bf16=True):
        super().__init__()
        self.model = model
        self.register_buffer("image_mean", torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1))
        self.autocast_bf16 = bool(autocast_bf16)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.train(False)

    @classmethod
    def from_pretrained(cls, autocast_bf16=True):
        model = AutoModel.from_pretrained(
            cls.repo_id, local_files_only=True, **cls.model_kwargs
        )
        mean, std = processor_statistics(cls.repo_id)
        return cls(model, mean, std, autocast_bf16=autocast_bf16)

    @classmethod
    def from_config(cls, config, autocast_bf16=False):
        """Construct a random-weight model from a Transformers config for tests."""
        model = AutoModel.from_config(config, **cls.model_kwargs)
        mean, std = processor_statistics(cls.repo_id)
        return cls(model, mean, std, autocast_bf16=autocast_bf16)

    def train(self, mode=True):
        super().train(False)
        if hasattr(self, "model"):
            self.model.eval()
        return self

    def normalize(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images with shape [B, 3, H, W]")
        return (images.float() - self.image_mean) / self.image_std

    def extract(self, normalized):
        raise NotImplementedError

    def forward(self, images):
        normalized = self.normalize(images)
        use_bf16 = self.autocast_bf16 and images.device.type == "cuda"
        with torch.no_grad(), torch.autocast(
            device_type=images.device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            features = self.extract(normalized)
        expected = (images.shape[0], self.out_channels,
                    images.shape[-2] // self.stride, images.shape[-1] // self.stride)
        if tuple(features.shape) != expected:
            raise RuntimeError(f"{type(self).__name__} produced {tuple(features.shape)}, expected {expected}")
        return features.float()


class DinoV3ViTL16Encoder(FrozenHFSpatialEncoder):
    model_kwargs = {"attn_implementation": "sdpa"}

    def __init__(self, model, image_mean, image_std, autocast_bf16=True):
        super().__init__(model, image_mean, image_std, autocast_bf16)
        self.num_register_tokens = int(model.config.num_register_tokens)
        self.out_channels = int(model.config.hidden_size)
        self.patch_size = int(model.config.patch_size)

    def extract(self, normalized):
        output = self.model(pixel_values=normalized)
        grid_height = normalized.shape[-2] // self.patch_size
        grid_width = normalized.shape[-1] // self.patch_size
        patch_tokens = output.last_hidden_state[:, 1 + self.num_register_tokens:, :]
        return patch_tokens_to_map(patch_tokens, grid_height, grid_width)


class DinoV3SatViTL16Encoder(DinoV3ViTL16Encoder):
    repo_id = "facebook/dinov3-vitl16-pretrain-sat493m"


class DinoV3WebViTL16Encoder(DinoV3ViTL16Encoder):
    repo_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"


class DinoV3ConvNextSmallEncoder(FrozenHFSpatialEncoder):
    repo_id = "facebook/dinov3-convnext-small-pretrain-lvd1689m"

    def __init__(self, model, image_mean, image_std, autocast_bf16=True):
        super().__init__(model, image_mean, image_std, autocast_bf16)
        self.out_channels = int(model.config.hidden_sizes[2])

    def extract(self, normalized):
        output = self.model(pixel_values=normalized, output_hidden_states=True)
        # hidden_states[0] is the input; entries 1..4 follow stages 0..3.
        features = output.hidden_states[3]
        return features


class SamViTBaseEncoder(FrozenHFSpatialEncoder):
    repo_id = "facebook/sam-vit-base"

    def __init__(self, model, image_mean, image_std, autocast_bf16=True):
        super().__init__(model, image_mean, image_std, autocast_bf16)
        self.out_channels = int(model.config.vision_config.output_channels)

    def extract(self, normalized):
        return self.model.get_image_embeddings(pixel_values=normalized)


ENCODER_CLASSES = {
    "dinov3_vitl16_sat": DinoV3SatViTL16Encoder,
    "dinov3_vitl16_web": DinoV3WebViTL16Encoder,
    "dinov3_convnext_small": DinoV3ConvNextSmallEncoder,
    "sam_vitb": SamViTBaseEncoder,
}


def build_huggingface_encoder(name, **kwargs):
    try:
        encoder_class = ENCODER_CLASSES[name]
    except KeyError as error:
        raise ValueError(f"Unsupported Hugging Face encoder {name!r}") from error
    return encoder_class.from_pretrained(**kwargs)
