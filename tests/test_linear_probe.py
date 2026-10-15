import numpy as np
import torch
import unittest

from looproad.data.road_dataset import map_raw_labels
from looproad.encoders.resnet import build_encoder
from looproad.models.linear_probe import masked_bce_with_logits


class LinearProbeTests(unittest.TestCase):
    def test_resnet50_layer3_shape_and_frozen_state(self):
        encoder = build_encoder("resnet50_layer3", weights=None)
        encoder.train()
        self.assertFalse(encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
        with torch.no_grad():
            output = encoder(torch.zeros(1, 3, 1024, 1024))
        self.assertEqual(output.shape, (1, 1024, 64, 64))
        self.assertEqual(output.dtype, torch.float32)

    def test_raw_label_mapping_preserves_ignore_mask(self):
        raw = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.uint8)
        target, valid = map_raw_labels(raw)
        np.testing.assert_array_equal(target, [[0, 0, 0, 1], [0, 0, 0, 0]])
        np.testing.assert_array_equal(valid, [[False, True, True, True], [True, True, True, True]])

    def test_masked_bce_does_not_count_ignored_pixels(self):
        logits = torch.tensor([[[[0.0, 2.0], [-2.0, 1.0]]]])
        target = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
        valid = torch.tensor([[[True, True], [False, False]]])
        actual = masked_bce_with_logits(logits, target, valid)
        # Only the first two flat pixel positions are valid.
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([0.0, 2.0]), torch.tensor([0.0, 1.0])).mean()
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
