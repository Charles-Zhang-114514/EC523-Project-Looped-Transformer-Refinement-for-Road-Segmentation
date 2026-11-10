"""Check the precision/recall loss intervention without a GPU or model weights."""
import contextlib
import io
import unittest
from unittest.mock import patch

import torch

from scripts.train_refiner import build_parser, deep_supervision, main


class RefinerLossTests(unittest.TestCase):
    def test_lower_weight_reduces_only_positive_pixel_gradient(self):
        target = torch.tensor([[[1., 0., 1.]]])
        valid = torch.tensor([[[True, True, False]]])
        gradients = {}
        for weight in (2., 4.):
            logits = torch.zeros(1, 1, 1, 3, requires_grad=True)
            loss, _ = deep_supervision([logits], target, valid, pos_weight=weight)
            loss.backward()
            gradients[weight] = logits.grad.flatten()
        # Two valid pixels: at logit zero, gradients are -weight/4 and +1/4.
        torch.testing.assert_close(gradients[2.], torch.tensor([-.5, .25, 0.]))
        torch.testing.assert_close(gradients[4.], torch.tensor([-1., .25, 0.]))

    def test_default_preserves_original_loss_at_every_step(self):
        target = torch.tensor([[[1., 0.]]])
        valid = torch.ones_like(target, dtype=torch.bool)
        outputs = [torch.tensor([[[[value, -value]]]]) for value in (.3, -.7)]
        default, steps = deep_supervision(outputs, target, valid)
        explicit, explicit_steps = deep_supervision(outputs, target, valid, pos_weight=4.)
        torch.testing.assert_close(default, explicit, rtol=0, atol=0)
        torch.testing.assert_close(default, (steps[0] + steps[1]) / 2)
        for actual, expected in zip(steps, explicit_steps):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(build_parser().parse_args([]).pos_weight, 4.)

    def test_invalid_weights_fail_before_loading_data(self):
        for value in ('0', '-1', 'nan', 'inf'):
            with self.subTest(value=value), patch('sys.argv', ['train_refiner.py', '--pos-weight', value]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)

    def test_checkpoint_weight_cannot_be_silently_changed_on_resume(self):
        saved = vars(build_parser().parse_args([]))
        saved['pos_weight'] = 2.
        checkpoint = {'state_dict': {}, 'config': saved, 'epoch': 3}
        errors = io.StringIO()
        with patch('sys.argv', ['train_refiner.py', '--resume', '--pos-weight', '4']), \
                patch('pathlib.Path.is_file', return_value=True), \
                patch('scripts.train_refiner.torch.load', return_value=checkpoint):
            with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn('Checkpoint setting pos_weight=2.0 conflicts', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
