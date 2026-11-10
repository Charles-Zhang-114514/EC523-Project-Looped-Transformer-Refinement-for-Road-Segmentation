"""Loss interventions for the six-run study; evaluation metrics stay unchanged.

Independent implementations of the algorithms described by:
Berman et al., CVPR 2018, https://arxiv.org/abs/1705.08790 (binary Lovasz hinge).
Shit et al., CVPR 2021, https://arxiv.org/abs/2003.07311 (soft skeleton/clDice).
The false-addition penalty and its calibration are project-specific experiments.
"""
import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .linear_probe import masked_bce_with_logits


def false_addition_loss(logits, target, valid, draft):
    """Penalize only probability increases on true/draft background pixels."""
    initial = draft.detach().sigmoid()
    eligible = valid.unsqueeze(1) & (target.unsqueeze(1) == 0) & (initial < .5)
    increase = (logits.sigmoid() - initial).clamp_min(0).square()
    return (increase * eligible).sum() / eligible.sum().clamp_min(1)


def lovasz_hinge_loss(logits, target, valid):
    """Foreground Jaccard surrogate, pooling valid pixels across the batch."""
    scores = logits.squeeze(1)[valid]
    truth = target[valid].to(dtype=scores.dtype)
    if scores.numel() == 0:
        return logits.sum() * 0.
    errors = 1. - scores * (2. * truth - 1.)
    ranked_errors, order = errors.sort(descending=True)
    ranked_truth = truth[order]
    positives = ranked_truth.sum()
    intersections = positives - ranked_truth.cumsum(0)
    unions = positives + (1. - ranked_truth).cumsum(0)
    jaccard_loss = 1. - intersections / unions
    increments = torch.cat((jaccard_loss[:1], jaccard_loss[1:] - jaccard_loss[:-1]))
    return (ranked_errors.clamp_min(0) * increments).sum()


def soft_erode(values):
    # Explicit zero padding treats the image exterior as background, rather
    # than max_pool's implicit infinity for a min-pool operation.
    vertical = -F.max_pool2d(-F.pad(values, (0, 0, 1, 1)), (3, 1), stride=1)
    horizontal = -F.max_pool2d(-F.pad(values, (1, 1, 0, 0)), (1, 3), stride=1)
    return torch.minimum(vertical, horizontal)


def soft_skeleton(values, iterations):
    opened = F.max_pool2d(soft_erode(values), 3, stride=1, padding=1)
    skeleton = (values - opened).clamp_min(0)
    for _ in range(iterations):
        values = soft_erode(values)
        opened = F.max_pool2d(soft_erode(values), 3, stride=1, padding=1)
        layer = (values - opened).clamp_min(0)
        skeleton = skeleton + (1. - skeleton) * layer
    return skeleton


def soft_cldice_loss(logits, target, valid, iterations=20, target_skeleton=None):
    """Batch-pooled soft-clDice on valid-clipped foreground, plus no BCE here.

All-background batches contribute zero auxiliary loss; BCE still supervises
them. Checkpointing the predicted skeleton bounds retained training memory.
"""
    truth = (target * valid).unsqueeze(1)
    if not bool(truth.any()):
        return logits.sum() * 0.
    if target_skeleton is None:
        with torch.no_grad():
            target_skeleton = soft_skeleton(truth, iterations)
    probability = logits.sigmoid() * valid.unsqueeze(1)
    if probability.requires_grad:
        predicted_skeleton = checkpoint(soft_skeleton, probability, iterations, use_reentrant=False)
    else:
        predicted_skeleton = soft_skeleton(probability, iterations)
    epsilon = 1e-6
    precision = ((predicted_skeleton * truth).sum() + epsilon) / (predicted_skeleton.sum() + epsilon)
    recall = ((target_skeleton * probability).sum() + epsilon) / (target_skeleton.sum() + epsilon)
    return 1. - 2. * precision * recall / (precision + recall).clamp_min(epsilon)


class RefinerObjective:
    """One intervention at a time, with checkpointable false-add calibration.

For false_add, the first warmup_steps optimizer updates use BCE only. The next
calibration_batches also use BCE only, measuring loss gradients with respect
to independent answer logits (not model-parameter gradients). The coefficient
is fixed thereafter so the ratio of summed auxiliary/BCE gradient norms is
fa_ratio on those training batches. No validation labels enter calibration.
"""
    def __init__(self, kind='bce', pos_weight=4., aux_weight=.1, skeleton_iterations=20,
                 fa_ratio=.1, warmup_steps=500, calibration_batches=8, state=None):
        if kind not in ('bce', 'false_add', 'lovasz', 'soft_cldice'):
            raise ValueError('Unknown refiner loss')
        self.kind, self.pos_weight, self.aux_weight = kind, pos_weight, aux_weight
        self.skeleton_iterations = skeleton_iterations
        self.fa_ratio, self.warmup_steps = fa_ratio, warmup_steps
        self.calibration_batches = calibration_batches
        self.state = dict(state or {})
        if kind == 'false_add':
            for key, value in {'batches': 0, 'bce_gradient_sum': 0., 'aux_gradient_sum': 0.,
                               'effective_aux_weight': None}.items():
                self.state.setdefault(key, value)

    def _calibrate(self, outputs, target, valid, draft):
        # Independent detached leaves measure only the objective's local
        # gradients. The live recurrent graph and optimizer are not touched.
        leaves = [value.detach().requires_grad_(True) for value in outputs]
        bce = torch.stack([masked_bce_with_logits(x, target, valid, self.pos_weight) for x in leaves]).mean()
        extra = torch.stack([false_addition_loss(x, target, valid, draft) for x in leaves]).mean()
        norms = []
        for loss in (bce, extra):
            gradients = torch.autograd.grad(loss, leaves)
            norms.append(float(torch.stack([g.square().sum() for g in gradients]).sum().sqrt()))
        self.state['batches'] += 1
        self.state['bce_gradient_sum'] += norms[0]
        self.state['aux_gradient_sum'] += norms[1]
        if self.state['batches'] == self.calibration_batches:
            if min(self.state['bce_gradient_sum'], self.state['aux_gradient_sum']) <= 0:
                raise RuntimeError('False-add calibration found no usable gradients; no meaningful intervention can be calibrated')
            weight = self.fa_ratio * self.state['bce_gradient_sum'] / self.state['aux_gradient_sum']
            if not math.isfinite(weight) or weight <= 0:
                raise RuntimeError('Invalid calibrated false-add weight')
            self.state['effective_aux_weight'] = weight

    def __call__(self, outputs, target, valid, draft, step):
        bce_steps = [masked_bce_with_logits(x, target, valid, self.pos_weight) for x in outputs]
        weight = self.aux_weight
        phase = 'active'
        if self.kind == 'bce':
            extra_steps = [x.sum() * 0. for x in outputs]
            weight = 0.
        elif self.kind == 'false_add':
            extra_steps = [false_addition_loss(x, target, valid, draft) for x in outputs]
            if self.state['effective_aux_weight'] is None:
                phase = 'warmup' if step <= self.warmup_steps else 'calibration'
                if step > self.warmup_steps:
                    self._calibrate(outputs, target, valid, draft)
                # Even the final calibration batch is trained with BCE only.
                weight = 0.
            else:
                weight = self.state['effective_aux_weight']
        elif self.kind == 'lovasz':
            extra_steps = [lovasz_hinge_loss(x, target, valid) for x in outputs]
        else:
            with torch.no_grad():
                target_skeleton = soft_skeleton((target * valid).unsqueeze(1), self.skeleton_iterations)
            extra_steps = [soft_cldice_loss(x, target, valid, self.skeleton_iterations, target_skeleton) for x in outputs]
        # Preserve the old BCE-only computation exactly, including its graph.
        losses = bce_steps if weight == 0 else [a + weight * b for a, b in zip(bce_steps, extra_steps)]
        loss = torch.stack(losses).mean()
        terms = {'bce': float(torch.stack(bce_steps).mean().detach()),
                 'auxiliary': float(torch.stack(extra_steps).mean().detach()),
                 'auxiliary_weight_used': weight, 'phase': phase}
        return loss, losses, terms
