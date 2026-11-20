"""Inference-only D4 recurrent ensembles; one shared core invocation per step."""
import time
import torch


def transform(x, code, inverse=False):
    if code not in range(8):
        raise ValueError('D4 code must be in 0..7')
    if inverse:
        return (x.flip((-1,)) if code >= 4 else x).rot90(-code % 4, (-2, -1))
    x = x.rot90(code % 4, (-2, -1))
    return x.flip((-1,)) if code >= 4 else x


@torch.no_grad()
def trajectory(model, features, draft, ks, mode='S5a', codes=tuple(range(8))):
    if mode not in ('S5a', 'S5b') or not codes or features.shape[0] != len(codes) or draft.shape[0] != 1:
        raise ValueError('Expected one image, one feature map per view and S5a/S5b')
    if model.untied_steps or model.reset_state or model.no_mask_feedback:
        raise ValueError('Self-ensemble requires the ordinary shared refiner')
    if not ks or min(ks) < 1:
        raise ValueError('Positive K required')
    device = draft.device
    def stamp():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        return time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    calls = 0
    def count(*unused):
        nonlocal calls
        calls += 1
    hook = model.core.register_forward_hook(count)
    try:
        start = stamp()
        answer = draft.float().clone()
        views = torch.cat([transform(answer, c) for c in codes])
        state, tokens = model.init(features.float().clone(), views.clone())
        for step in range(1, max(ks)+1):
            before = views
            state, updated = model.step(state, before.clone(), tokens)
            if len(codes) == 1 and codes[0] == 0:
                answer = updated
            elif mode == 'S5a':
                delta = torch.stack([transform(updated[i:i+1]-before[i:i+1], c, True)
                                     for i,c in enumerate(codes)]).mean(0)
                answer = answer + delta
            else:
                answer = torch.stack([transform(updated[i:i+1], c, True)
                                      for i,c in enumerate(codes)]).mean(0)
            views = (torch.cat([transform(answer,c) for c in codes]) if mode == 'S5a' else updated)
            elapsed = stamp()-start
            if calls != step:
                raise AssertionError(f'Expected {step} core calls, observed {calls}')
            if step in ks:
                yield str(step), answer.clone(), elapsed, (torch.cuda.max_memory_allocated(device)
                                                         if device.type == 'cuda' else None), calls
    finally:
        hook.remove()
