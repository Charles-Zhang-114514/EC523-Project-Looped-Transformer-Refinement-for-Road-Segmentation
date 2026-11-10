"""Numerical/behavioral checks for the six-run interventions; no weights needed."""
import unittest
import numpy as np
import torch

from looproad.models.linear_probe import masked_bce_with_logits
from looproad.models.refiner_losses import (RefinerObjective, false_addition_loss,
                                          lovasz_hinge_loss, soft_cldice_loss)
from looproad.data.augmentation import D4TrainingDataset, d4_transform


class RefinerObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_false_add_targets_only_increases_on_eligible_background(self):
        draft = torch.tensor([[[[-2., -2., 2., -2., -2.]]]])
        logits = torch.tensor([[[[0., 0., 3., 0., -3.]]]], requires_grad=True)
        target = torch.tensor([[[0., 1., 0., 0., 0.]]])
        valid = torch.tensor([[[True, True, True, False, True]]])
        loss = false_addition_loss(logits, target, valid, draft)
        loss.backward()
        self.assertGreater(float(logits.grad.flatten()[0]), 0.)
        torch.testing.assert_close(logits.grad.flatten()[1:], torch.zeros(4))
        expected = (.5 - torch.sigmoid(torch.tensor(-2.))).square() / 2
        torch.testing.assert_close(loss, expected)

    def test_lovasz_known_examples_and_ignore(self):
        truth = torch.tensor([[[1., 0.]]])
        valid = torch.ones_like(truth, dtype=torch.bool)
        for values, expected in (([2., -2.], 0.), ([-2., 2.], 3.), ([0., 0.], 1.)):
            logits = torch.tensor([[ [values] ]], requires_grad=True)
            loss = lovasz_hinge_loss(logits, truth, valid)
            self.assertAlmostEqual(float(loss.detach()), expected)
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
        logits = torch.tensor([[[[0., 100.]]]], requires_grad=True)
        loss = lovasz_hinge_loss(logits, truth, torch.tensor([[[True, False]]]))
        loss.backward()
        self.assertEqual(float(logits.grad.flatten()[1]), 0.)

    def test_soft_cldice_penalizes_a_broken_road_and_excludes_ignore(self):
        target = torch.zeros(1, 17, 17)
        target[:, 7:10, 2:15] = 1
        valid = torch.ones_like(target, dtype=torch.bool)
        valid[:, :2] = False
        perfect = (target.unsqueeze(1) * 2 - 1) * 12
        broken = perfect.clone()
        broken[:, :, :, 8] = -12
        broken.requires_grad_()
        clean_loss = soft_cldice_loss(perfect, target, valid, iterations=5)
        broken_loss = soft_cldice_loss(broken, target, valid, iterations=5)
        self.assertLess(float(clean_loss), .001)
        self.assertGreater(float(broken_loss.detach()), float(clean_loss) + .01)
        broken_loss.backward()
        self.assertTrue(torch.isfinite(broken.grad).all())
        self.assertLess(float(broken.grad[0, 0, 8, 8]), 0.)
        self.assertEqual(float(broken.grad[:, :, :2].abs().sum()), 0.)

    def test_empty_valid_and_empty_roads_are_finite(self):
        target = torch.zeros(1, 5, 5)
        for kind in ('bce', 'false_add', 'lovasz', 'soft_cldice'):
            logits = torch.randn(1, 1, 5, 5, requires_grad=True)
            objective = RefinerObjective(kind, skeleton_iterations=3)
            loss, _, _ = objective([logits], target, torch.zeros_like(target, dtype=torch.bool), logits.detach(), 1)
            loss.backward()
            self.assertEqual(float(loss.detach()), 0.)
            self.assertEqual(float(logits.grad.abs().sum()), 0.)
        logits = torch.zeros(1, 1, 5, 5, requires_grad=True)
        loss = soft_cldice_loss(logits, target, torch.ones_like(target, dtype=torch.bool), 3)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)

    def test_calibration_waits_for_warmup_and_survives_restore(self):
        target = torch.tensor([[[0., 1.]]])
        valid = torch.ones_like(target, dtype=torch.bool)
        draft = torch.full((1, 1, 1, 2), -2.)
        objective = RefinerObjective('false_add', warmup_steps=1, calibration_batches=2)
        for step in (1, 2, 3):
            outputs = [torch.zeros_like(draft, requires_grad=True)]
            loss, _, terms = objective(outputs, target, valid, draft, step)
            self.assertEqual(terms['auxiliary_weight_used'], 0.)
            loss.backward()
        state = dict(objective.state)
        weight = state['effective_aux_weight']
        self.assertGreater(weight, 0.)
        self.assertAlmostEqual(weight * state['aux_gradient_sum'] / state['bce_gradient_sum'], .1)
        resumed = RefinerObjective('false_add', warmup_steps=1, calibration_batches=2, state=state)
        outputs = [torch.zeros_like(draft, requires_grad=True)]
        a, _, terms = resumed(outputs, target, valid, draft, 4)
        b = masked_bce_with_logits(outputs[0], target, valid, 4.)
        self.assertGreater(float(a.detach()), float(b.detach()))
        self.assertEqual(terms['auxiliary_weight_used'], weight)
        self.assertEqual(resumed.state, state)

    def test_zero_false_add_gradients_fail_instead_of_silent_noop(self):
        logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
        target = torch.zeros(1, 2, 2)
        objective = RefinerObjective('false_add', warmup_steps=0, calibration_batches=1)
        with self.assertRaisesRegex(RuntimeError, 'no usable gradients'):
            objective([logits], target, torch.ones_like(target, dtype=torch.bool), logits.detach(), 1)

    def test_default_objective_is_exact_original_bce(self):
        torch.manual_seed(12)
        outputs = [torch.randn(2, 1, 5, 5, requires_grad=True) for _ in range(4)]
        target = (torch.rand(2, 5, 5) > .7).float()
        valid = torch.rand(2, 5, 5) > .1
        loss, _, _ = RefinerObjective()(outputs, target, valid, outputs[0].detach(), 1)
        original = torch.stack([masked_bce_with_logits(x, target, valid, 4.) for x in outputs]).mean()
        torch.testing.assert_close(loss, original, rtol=0, atol=0)
        actual = torch.autograd.grad(loss, outputs, retain_graph=True)
        expected = torch.autograd.grad(original, outputs)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA smoke check runs on the allocated SCC GPU')
    def test_cuda_objectives_have_finite_forward_and_backward(self):
        target = torch.zeros(2, 32, 32, device='cuda')
        target[:, 8:12, 3:29] = 1
        valid = torch.ones_like(target, dtype=torch.bool)
        draft = torch.full((2, 1, 32, 32), -2., device='cuda')
        for kind in ('bce', 'false_add', 'lovasz', 'soft_cldice'):
            objective = RefinerObjective(kind, warmup_steps=0, calibration_batches=1)
            for step in (1, 2):
                logits = torch.zeros_like(draft, requires_grad=True)
                loss, _, _ = objective([logits], target, valid, draft, step)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(logits.grad).all())


class D4Tests(unittest.TestCase):
    def test_all_eight_transforms_preserve_pixel_alignment(self):
        raw = np.arange(25).reshape(5, 5)
        image = np.stack([raw, raw + 25, raw + 50])
        seen = set()
        for code in range(8):
            a, b = d4_transform(image, code), d4_transform(raw, code)
            np.testing.assert_array_equal(a[0], b)
            self.assertTrue(a.flags.c_contiguous and b.flags.c_contiguous)
            self.assertEqual(set(b.flat), set(raw.flat))
            seen.add(b.tobytes())
        self.assertEqual(len(seen), 8)

    def test_dataset_is_deterministic_by_epoch_and_does_not_change_global_rng(self):
        class TinyDataset:
            split = 'Train'
            def __len__(self): return 16
            def __getitem__(self, index):
                raw = np.arange(25).reshape(5, 5)
                return np.stack([raw] * 3), raw.copy(), raw % 2 == 0, str(index), raw.copy()
        wrapper = D4TrainingDataset(TinyDataset(), seed=7)
        np.random.seed(23)
        state = np.random.get_state()
        first = [wrapper[i][0] for i in range(16)]
        self.assertTrue(np.array_equal(state[1], np.random.get_state()[1]))
        for i, a in enumerate(first):
            row = wrapper[i]
            np.testing.assert_array_equal(a, row[0])
            np.testing.assert_array_equal(row[0][0], row[4])
            np.testing.assert_array_equal(row[2], row[4] % 2 == 0)
        wrapper.set_epoch(1)
        self.assertTrue(any(not np.array_equal(a, wrapper[i][0]) for i, a in enumerate(first)))
        TinyDataset.split = 'Val'
        with self.assertRaises(ValueError):
            D4TrainingDataset(TinyDataset())


if __name__ == '__main__':
    unittest.main()
