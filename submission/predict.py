#!/usr/bin/env python3
"""Offline image-only inference with an exact externally controlled loop.

Student-written implementation aligned to project_a/cli/predict.py:40-78
(loop auditing), :83-106 (CLI/IDs), and :129-154 (files/manifest). The only
input extension is direct single-file support; encoder-dir replaces the
unreleased staff baseline loader. No course implementation is imported.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
from submission.baseline import ENCODER_FILES, load_baseline
from submission.model import load_refiner

PROTOCOL = 'ec523-project-a-v1-draft'  # Public format identifier: common.py:8.
MODES = ('none', 'reset-state', 'no-mask-feedback')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash():
    """Include reused project modules as well as the submission adapters."""
    files = sorted([*Path(__file__).parent.glob('*.py'), *(ROOT/'src/looproad').rglob('*.py')])
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def input_files(path):
    """Return stable relative IDs; a direct file uses its own stem as ID."""
    path = Path(path)
    candidates = [path] if path.is_file() else sorted(path.rglob('*')) if path.is_dir() else []
    rows = [(p, (Path(p.name) if path.is_file() else p.relative_to(path)).with_suffix('').as_posix())
            for p in candidates if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    ids = [image_id for _, image_id in rows]
    if not rows:
        raise ValueError('Input contains no supported PNG/JPEG images')
    if len(set(ids)) != len(ids) or any('\\' in key or '..' in Path(key).parts for key in ids):
        raise ValueError('Image IDs must be unique canonical relative paths')
    return rows


def validate_logits(logits, shape):
    if tuple(logits.shape) != tuple(shape) or not torch.isfinite(logits).all():
        raise ValueError('Logits must be finite and match the original image resolution')


def predict_tensor(baseline, model, images, steps, ablation='none'):
    """Outer runner only: models never receive steps or a final-step signal."""
    if steps not in range(9) or ablation not in MODES:
        raise ValueError('Invalid loop count or intervention')
    signature = tuple((name, id(p), p.data_ptr(), tuple(p.shape)) for name, p in model.named_parameters())
    called = []
    handle = model.core.register_forward_hook(lambda *_: called.append(True))
    try:
        with torch.no_grad():
            features, original = baseline(images)
            shape = (images.shape[0], 1, *images.shape[-2:])
            validate_logits(original, shape)
            current = original.clone()
            trace = [current.sigmoid().float().cpu().numpy()]
            state = model.initialize(features.clone(), original.clone()) if steps else None
            if called:
                raise ValueError('Initializer must not execute the recurrent core')
            for index in range(steps):
                if ablation == 'reset-state':
                    state = model.initialize(features.clone(), original.clone())
                answer = (original if ablation == 'no-mask-feedback' else current).clone()
                state, current = model.step(features.clone(), state, answer)
                validate_logits(current, shape)
                now = tuple((name, id(p), p.data_ptr(), tuple(p.shape)) for name, p in model.named_parameters())
                if len(called) != index + 1 or now != signature:
                    raise ValueError('Each update must reuse the same parameters and call core exactly once')
                trace.append(current.sigmoid().float().cpu().numpy())
        return trace, len(called)
    finally:
        handle.remove()


def write_probability(path, probability):
    if probability.ndim != 2 or probability.dtype != np.float32 or not np.isfinite(probability).all():
        raise ValueError('Expected a finite float32 HxW probability map')
    if (probability < 0).any() or (probability > 1).any():
        raise ValueError('Probabilities must lie in [0,1]')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, probability, allow_pickle=False)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('checkpoint', 'baseline', 'encoder-dir', 'input', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--steps', type=int, choices=range(9), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--save-trace', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ablation', choices=MODES, default='none')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    files = input_files(args.input)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError('Output must be a new or empty directory')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    baseline = load_baseline(args.baseline, args.encoder_dir, device)
    model = load_refiner(args.checkpoint, baseline.feature_channels, device)
    encoder_hashes = {name: sha256(args.encoder_dir/name) for name in ENCODER_FILES}
    args.output.mkdir(parents=True, exist_ok=True)
    records, peaks = [], []
    for path, image_id in files:
        with Image.open(path) as source:
            rgb = np.asarray(source.convert('RGB'), dtype=np.float32) / 255.0
        images = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None].to(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        start = time.perf_counter()
        trace, core_calls = predict_tensor(baseline, model, images, args.steps, args.ablation)
        synchronize(device)
        seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        if peak is not None:
            peaks.append(peak)
        final = trace[-1][0, 0]
        write_probability(args.output/'probabilities'/f'{image_id}.npy', final)
        mask_path = args.output/'masks'/f'{image_id}.png'
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.where(final >= .5, 255, 0).astype(np.uint8)).save(mask_path)
        if args.save_trace:
            for step, probability in enumerate(trace):
                write_probability(args.output/'trace'/f'k{step}'/f'{image_id}.npy', probability[0, 0])
        records.append(dict(id=image_id, shape=list(rgb.shape[:2]), input_sha256=sha256(path),
                            core_calls=core_calls, seconds_with_trace_transfer=seconds,
                            peak_cuda_allocated_bytes=peak))
        print(f'PREDICTED {image_id} K={args.steps} seconds={seconds:.6f} peak_cuda_bytes={peak}', flush=True)
    report = dict(protocol=PROTOCOL, steps=args.steps, demo=False, ablation=args.ablation,
                  seed=args.seed, device=args.device, checkpoint_sha256=sha256(args.checkpoint),
                  baseline_sha256=sha256(args.baseline), submission_source_sha256=source_hash(),
                  parameters=sum(p.numel() for p in model.parameters()),
                  core_parameters=sum(p.numel() for p in model.core.parameters()),
                  peak_cuda_allocated_bytes=max(peaks) if peaks else None,
                  timing_note='Includes baseline, updates and every-step CPU trace transfer; excludes file writes/model loading; no warmup.',
                  images=records, number_of_images=len(records), encoder_files_sha256=encoder_hashes,
                  encoder_bundle_sha256=hashlib.sha256(json.dumps(encoder_hashes, sort_keys=True).encode()).hexdigest())
    (args.output/'run.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    main()
