"""Small-grid behavioral tests for shared recurrence, patch ordering and freezing."""
import inspect
import io
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from looproad.models.refiner import LoopRefiner, patchify, unpatchify
from scripts.train_refiner import capture_rng, deep_supervision, lr_multiplier, refine_steps, restore_rng


class RefinerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        self.features = torch.randn(2, 8, 4, 4)
        self.draft = torch.randn(2, 1, 16, 16)

    @staticmethod
    def model(**kwargs):
        return LoopRefiner(in_channels=8, dimension=16, heads=4, patch=4, image_size=16, **kwargs)

    def test_patchify_unpatchify_are_exact_inverses(self):
        image = torch.arange(2 * 16 * 32, dtype=torch.float32).reshape(2, 1, 16, 32)
        tokens = patchify(image, patch=4)
        torch.testing.assert_close(tokens[0, 0], image[0, 0, :4, :4].flatten(), rtol=0, atol=0)
        self.assertTrue(torch.equal(image, unpatchify(tokens, 16, 32, patch=4)))

    def test_zero_output_head_preserves_draft_at_all_four_steps(self):
        model = self.model()
        outputs = refine_steps(model, self.features, self.draft, 4)
        for answer in outputs:
            self.assertEqual(answer.shape, self.draft.shape)
            self.assertTrue(torch.equal(answer, self.draft))
        state, tokens = model.init(self.features, self.draft)
        self.assertEqual(state.shape, (2, 16, 16))
        self.assertEqual(tokens.shape, state.shape)

    def test_no_mask_feedback_uses_zero_but_initializer_uses_draft(self):
        model = self.model(no_mask_feedback=True)
        state, tokens = model.init(self.features, self.draft)
        expected_state = model.initializer(tokens + model.answer_tokens(self.draft))
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        seen = []
        handle = model.core.register_forward_pre_hook(lambda _, inputs: seen.append(inputs[0].detach().clone()))
        first, _ = model.step(state, self.draft, tokens)
        second, _ = model.step(state, self.draft + 7, tokens)
        handle.remove()
        torch.testing.assert_close(seen[0], state + tokens + model.position_encoding)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_reset_state_reuses_initial_state_each_step(self):
        model = self.model(reset_state=True)
        state, tokens = model.init(self.features, self.draft)
        seen = []
        handle = model.core.register_forward_pre_hook(lambda _, inputs: seen.append(inputs[0].detach().clone()))
        model.step(state + 100, self.draft + 1, tokens)
        handle.remove()
        expected = state + tokens + model.answer_tokens(self.draft + 1) + model.position_encoding
        torch.testing.assert_close(seen[0], expected)

    def test_external_loop_equivalence_one_core_call_and_shared_parameters(self):
        model = self.model()
        torch.nn.init.normal_(model.output_head[-1].weight, std=.02)
        parameter_ids = {name: id(value) for name, value in model.named_parameters()}
        calls = []
        handle = model.core.register_forward_hook(lambda *_: calls.append(1))
        outputs = refine_steps(model, self.features, self.draft, 4)
        self.assertEqual(len(calls), 4)
        handle.remove()
        state, tokens = model.init(self.features, self.draft)
        answer = self.draft
        for index in range(4):
            state, answer = model.step(state, answer, tokens)
            torch.testing.assert_close(answer, outputs[index], rtol=0, atol=0)
        self.assertEqual(parameter_ids, {name: id(value) for name, value in model.named_parameters()})
        for method in (LoopRefiner.__init__, LoopRefiner.init, LoopRefiner.step):
            self.assertFalse({"K", "k", "k_train", "steps"} & inspect.signature(method).parameters.keys())

    def test_position_encoding_is_a_buffer_and_default_budget(self):
        model = LoopRefiner()
        self.assertIn("position_encoding", dict(model.named_buffers()))
        self.assertNotIn("position_encoding", dict(model.named_parameters()))
        self.assertFalse(model.position_encoding.requires_grad)
        self.assertEqual(model.parameter_counts()["trainable_total"], 1_250_048)
        self.assertLessEqual(model.parameter_counts()["core"], 5_000_000)

    def test_deep_supervision_backpropagates_through_recurrent_state(self):
        model = self.model()
        torch.nn.init.normal_(model.output_head[-1].weight, std=.02)
        outputs = refine_steps(model, self.features, self.draft, 3)
        model._initial_state.retain_grad()
        target = torch.zeros(2, 16, 16)
        valid = torch.ones_like(target, dtype=torch.bool)
        valid[:, :4] = False
        loss, losses = deep_supervision(outputs, target, valid)
        torch.testing.assert_close(loss, torch.stack(losses).mean())
        loss.backward()
        self.assertTrue(torch.isfinite(model._initial_state.grad).all())
        self.assertGreater(float(model._initial_state.grad.abs().sum()), 0.)
        self.assertGreater(float(model.core[0].qkv.weight.grad.abs().sum()), 0.)

    def test_resume_restores_optimizer_scheduler_and_all_cpu_rng_streams(self):
        random.seed(11)
        np.random.seed(12)
        loader = SimpleNamespace(generator=torch.Generator().manual_seed(13))

        def components():
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: lr_multiplier(step, 4, warmup=2))
            return model, optimizer, scheduler

        def step(model, optimizer, scheduler):
            order = torch.randperm(4, generator=loader.generator)
            image = torch.randn(4, 2)[order] * (random.random() + np.random.random())
            optimizer.zero_grad(set_to_none=True)
            model(image).square().mean().backward()
            optimizer.step()
            scheduler.step()

        model, optimizer, scheduler = components()
        step(model, optimizer, scheduler)
        buffer = io.BytesIO()
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "rng": capture_rng(loader)}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        step(model, optimizer, scheduler)
        expected = {key: value.clone() for key, value in model.state_dict().items()}
        resumed_model, resumed_optimizer, resumed_scheduler = components()
        resumed_model.load_state_dict(saved["state_dict"])
        resumed_optimizer.load_state_dict(saved["optimizer"])
        resumed_scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"], loader, torch.device("cpu"))
        step(resumed_model, resumed_optimizer, resumed_scheduler)
        for key, value in resumed_model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        self.assertEqual(resumed_scheduler.state_dict(), scheduler.state_dict())
        self.assertEqual(resumed_optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])


if __name__ == "__main__":
    unittest.main()
