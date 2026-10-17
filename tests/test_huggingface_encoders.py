import json
import unittest

import torch
from huggingface_hub import try_to_load_from_cache
from transformers import AutoConfig, AutoModel

from looproad.encoders.huggingface import (
    DinoV3ConvNextSmallEncoder,
    DinoV3SatViTL16Encoder,
    DinoV3WebViTL16Encoder,
    SamViTBaseEncoder,
    ENCODER_CLASSES,
    patch_tokens_to_map,
    processor_statistics,
)


class HuggingFaceEncoderTests(unittest.TestCase):
    @staticmethod
    def tiny_encoder(encoder_class):
        config = AutoConfig.from_pretrained(encoder_class.repo_id, local_files_only=True)
        model_kwargs = encoder_class.model_kwargs
        if "dinov3_vit" in config.model_type:
            config.num_hidden_layers = 1
            config.hidden_size = 48
            config.num_attention_heads = 4
            config.intermediate_size = 96
            config.num_register_tokens = 4
        elif config.model_type == "dinov3_convnext":
            config.hidden_sizes = [8, 16, 24, 32]
            config.depths = [1, 1, 1, 1]
            config.drop_path_rate = 0.0
        elif config.model_type == "sam":
            vision = config.vision_config
            vision.image_size = 64
            vision.hidden_size = 32
            vision.output_channels = 16
            vision.num_hidden_layers = 1
            vision.num_attention_heads = 4
            vision.window_size = 2
            vision.global_attn_indexes = [0]
            vision.mlp_dim = 64
        model = AutoModel.from_config(config, **model_kwargs)
        mean, std = processor_statistics(encoder_class.repo_id)
        return encoder_class(model, mean, std, autocast_bf16=False)

    def test_all_models_return_frozen_stride16_features(self):
        classes = (DinoV3SatViTL16Encoder, DinoV3WebViTL16Encoder,
                   DinoV3ConvNextSmallEncoder, SamViTBaseEncoder)
        for encoder_class in classes:
            with self.subTest(encoder=encoder_class.repo_id):
                encoder = self.tiny_encoder(encoder_class)
                encoder.train()
                self.assertFalse(encoder.training)
                self.assertFalse(encoder.model.training)
                self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
                with torch.enable_grad():
                    output = encoder(torch.rand(1, 3, 64, 64))
                self.assertEqual(output.shape, (1, encoder.out_channels, 4, 4))
                self.assertEqual(output.dtype, torch.float32)
                self.assertFalse(output.requires_grad)
                self.assertEqual(encoder.stride, 16)

    def test_dinov3_patch_token_order_is_row_major_inverse(self):
        encoder = self.tiny_encoder(DinoV3SatViTL16Encoder)
        image = torch.arange(3 * 64 * 64, dtype=torch.float32).reshape(1, 3, 64, 64)
        patch_grid = encoder.model.embeddings.patch_embeddings(image)
        flattened = patch_grid.flatten(2).transpose(1, 2)
        restored = patch_tokens_to_map(flattened, 4, 4)
        torch.testing.assert_close(restored, patch_grid, rtol=0, atol=0)

    def test_normalization_values_match_cached_processor_configs(self):
        for name, encoder_class in ENCODER_CLASSES.items():
            with self.subTest(encoder=name):
                path = try_to_load_from_cache(encoder_class.repo_id, "preprocessor_config.json")
                self.assertIsNotNone(path, f"Missing cached processor config for {name}")
                with open(path, encoding="utf-8") as handle:
                    config = json.load(handle)
                mean, std = processor_statistics(encoder_class.repo_id)
                self.assertEqual(mean, tuple(config["image_mean"]))
                self.assertEqual(std, tuple(config["image_std"]))


if __name__ == "__main__":
    unittest.main()
