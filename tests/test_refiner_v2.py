"""Backward compatibility, smooth decoding and non-learning control checks."""
import unittest
import numpy as np
import torch

from looproad.models.refiner import LoopRefiner
from scripts.train_refiner import is_new_best, selection_score
from analysis.morph_baseline import apply_operation


class RefinerV2Tests(unittest.TestCase):
    def make_model(self, **kwargs):
        return LoopRefiner(in_channels=8, dimension=16, heads=4, image_size=32, **kwargs)

    def test_conv_output_shape_and_zero_initialization(self):
        model = self.make_model(head_type="conv_upsample")
        draft = torch.randn(2, 1, 32, 32)
        state, features = model.init(torch.randn(2, 8, 2, 2), draft)
        for _ in range(4):
            state, answer = model.step(state, draft, features)
            self.assertEqual(answer.shape, draft.shape)
            self.assertTrue(torch.equal(answer, draft))

    def test_default_and_explicit_patch_heads_are_bitwise_identical(self):
        torch.manual_seed(17)
        default = self.make_model()
        torch.manual_seed(17)
        explicit = self.make_model(head_type="patch_linear")
        for key, tensor in default.state_dict().items():
            self.assertTrue(torch.equal(tensor, explicit.state_dict()[key]), key)
        features, draft = torch.randn(1, 8, 2, 2), torch.randn(1, 1, 32, 32)
        z1, f1 = default.init(features, draft)
        z2, f2 = explicit.init(features, draft)
        for _ in range(4):
            z1, y1 = default.step(z1, draft, f1)
            z2, y2 = explicit.step(z2, draft, f2)
            self.assertTrue(torch.equal(z1, z2))
            self.assertTrue(torch.equal(y1, y2))

    def test_v1_format_state_loads_strictly_without_head_config(self):
        old = self.make_model()
        checkpoint = {"state_dict": old.state_dict(), "config": {}, "epoch": 20}
        new = self.make_model(head_type=checkpoint['config'].get('head_type', 'patch_linear'))
        new.load_state_dict(checkpoint['state_dict'], strict=True)
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))

    def test_selection_uses_weighted_score_and_strict_improvement(self):
        self.assertAlmostEqual(selection_score({'1': .1, '2': .2, '4': .3}), .225)
        best, saved = None, []
        for epoch, score in enumerate((.2, .1, .2, .3, .29), 1):
            if is_new_best(score, best):
                best = score
                saved.append(epoch)
        self.assertEqual(saved, [1, 4])

    def test_zero_morphology_is_identity(self):
        mask = np.random.default_rng(0).random((32, 32)) > .5
        for kind in ('closing', 'remove_small'):
            self.assertTrue(np.array_equal(apply_operation(mask, kind, 0), mask))

    def test_closing_connects_two_pixel_gap(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[12:20, 4:28] = True
        mask[:, 15:17] = False
        closed = apply_operation(mask, 'closing', 2)
        self.assertTrue(closed[16, 15:17].all())


if __name__ == '__main__':
    unittest.main()
