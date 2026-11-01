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
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
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


def refine_tensor(model, features, original, steps, ablation='none'):
    """Outer runner only: models never receive steps or a final-step signal."""
    if steps not in range(9) or ablation not in MODES:
        raise ValueError('Invalid loop count or intervention')
    signature = tuple((name, id(p), p.data_ptr(), tuple(p.shape)) for name, p in model.named_parameters())
    called = []
    handle = model.core.register_forward_hook(lambda *_: called.append(True))
    try:
        with torch.no_grad():
            shape = tuple(original.shape)
            if original.ndim != 4 or original.shape[1] != 1:
                raise ValueError('Expected original logits with shape [B,1,H,W]')
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


def predict_tensor(baseline, model, images, steps, ablation='none'):
    """Retain the original image-at-a-time helper and intervention semantics."""
    with torch.no_grad():
        features, original = baseline(images)
        validate_logits(original, (images.shape[0], 1, *images.shape[-2:]))
        return refine_tensor(model, features, original, steps, ablation)


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


def allocated(device):
    return torch.cuda.memory_allocated(device) if device.type == 'cuda' else None


def reset_peak(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device):
    return torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None


def image_shape(path):
    """Fail before model loading/caching instead of resizing an invalid tile."""
    with Image.open(path) as source:
        width, height = source.size
    if (height, width) != (1024, 1024):
        raise ValueError(f'Input {path.name} has size {width}x{height}; required size is 1024x1024. No resizing is performed.')
    return height, width


def image_tensor(path, device):
    image_shape(path)
    with Image.open(path) as source:
        rgb = np.asarray(source.convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None].to(device)


def create_cache(parent, output, count):
    """Use a unique disposable directory, outside this source tree and output."""
    parent = Path(parent or os.environ.get('TMPDIR') or tempfile.gettempdir()).resolve()
    if parent.is_relative_to(ROOT) or parent.is_relative_to(output.resolve()):
        raise ValueError('Feature cache must be outside both the repository/package and output directory; choose --cache-dir or TMPDIR accordingly')
    parent.mkdir(parents=True, exist_ok=True)
    # One [1,1024,64,64] float32 tensor is 16 MiB; include NPY headers and margin.
    required = count * (1024 * 64 * 64 * 4 + 256) + 64 * 1024**2
    free = shutil.disk_usage(parent).free
    if free < required:
        raise OSError(f'Insufficient feature-cache space: need {required} bytes (~16 MiB/image plus margin), available {free} bytes')
    return Path(tempfile.mkdtemp(prefix='looproad-features-', dir=parent)), required


def release_encoder(baseline, device):
    """Detach the encoder before allocating any refiner weights on the GPU."""
    synchronize(device)
    before = allocated(device)
    reserved_before = torch.cuda.memory_reserved(device) if device.type == 'cuda' else None
    probe = baseline.probe
    del baseline.encoder
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    synchronize(device)
    release = dict(allocated_before_bytes=before, allocated_after_bytes=allocated(device),
                   reserved_before_bytes=reserved_before,
                   reserved_after_bytes=torch.cuda.memory_reserved(device) if device.type == 'cuda' else None)
    print('ENCODER_RELEASE ' + json.dumps(release), flush=True)
    return probe, release


def write_outputs(output, image_id, trace, expected_shape, save_trace):
    final = trace[-1][0, 0]
    if final.shape != tuple(expected_shape):
        raise ValueError('Prediction shape must match the original input image')
    write_probability(output/'probabilities'/f'{image_id}.npy', final)
    binary = (final >= .5).astype(np.uint8)
    for directory, mask in (('masks', binary * 255), ('road_masks', binary)):
        if mask.shape != tuple(expected_shape) or mask.dtype != np.uint8:
            raise ValueError('Mask must be uint8 and match input HxW')
        path = output/directory/f'{image_id}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(path)
    if save_trace:
        for step, probability in enumerate(trace):
            write_probability(output/'trace'/f'k{step}'/f'{image_id}.npy', probability[0, 0])


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('checkpoint', 'baseline', 'encoder-dir', 'input', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--steps', type=int, choices=range(9), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--save-trace', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ablation', choices=MODES, default='none')
    parser.add_argument('--cache-dir', type=Path, help='Parent for a unique temporary feature cache; must be outside package/output')
    parser.add_argument('--keep-cache', action='store_true', help='Keep the feature cache for debugging, including on failure')
    parser.add_argument('--single-pass', action='store_true', help='Legacy image-at-a-time path with encoder/refiner co-resident')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    files = input_files(args.input)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError('Output must be a new or empty directory')
    shapes = {image_id: image_shape(path) for path, image_id in files}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    encoder_hashes = {name: sha256(args.encoder_dir/name) for name in ENCODER_FILES}
    records, peaks, cache, required, cache_bytes = [], [], None, 0, 0
    phases = {name: dict(seconds=0., peak_cuda_allocated_bytes=None) for name in ('encoding', 'refinement')}
    release = None
    try:
        if not args.single_pass:
            cache, required = create_cache(args.cache_dir, args.output, len(files))
        args.output.mkdir(parents=True, exist_ok=True)
        reset_peak(device)
        synchronize(device)
        start = time.perf_counter()
        baseline = load_baseline(args.baseline, args.encoder_dir, device)
        synchronize(device)
        channels = baseline.feature_channels
        phases['encoding']['seconds'] += time.perf_counter() - start
        indexed = []
        if not args.single_pass:
            for index, (path, image_id) in enumerate(files):
                synchronize(device)
                start = time.perf_counter()
                images = image_tensor(path, device)
                synchronize(device)
                forward_start = time.perf_counter()
                with torch.no_grad():
                    features = baseline.encoder(images)
                synchronize(device)
                seconds = time.perf_counter() - forward_start
                if features.dtype != torch.float32 or features.shape != (1, channels, 64, 64) or not torch.isfinite(features).all():
                    raise ValueError('Feature cache requires finite original float32 [1,C,64,64] tensors')
                destination = cache/f'{index:08d}.npy'
                np.save(destination, features.detach().cpu().numpy(), allow_pickle=False)
                cache_bytes += destination.stat().st_size
                indexed.append((path, image_id, destination, seconds))
                del features, images
                phases['encoding']['seconds'] += time.perf_counter() - start
            phases['encoding']['peak_cuda_allocated_bytes'] = peak_memory(device)
            probe, release = release_encoder(baseline, device)
            del baseline
        else:
            indexed = [(path, image_id, None, 0.) for path, image_id in files]
        synchronize(device)
        start = time.perf_counter()
        if not args.single_pass:
            reset_peak(device)
        model = load_refiner(args.checkpoint, channels, device)
        synchronize(device)
        phases['refinement']['seconds'] += time.perf_counter() - start
        for path, image_id, feature_path, encoding_seconds in indexed:
            if args.single_pass:
                reset_peak(device)
                synchronize(device)
                start = time.perf_counter()
                images = image_tensor(path, device)
                synchronize(device)
                forward_start = time.perf_counter()
                with torch.no_grad():
                    features, original = baseline(images)
                synchronize(device)
                encoding_seconds = time.perf_counter() - forward_start
                phases['encoding']['seconds'] += time.perf_counter() - start
                value = peak_memory(device)
                if value is not None:
                    phases['encoding']['peak_cuda_allocated_bytes'] = max(phases['encoding']['peak_cuda_allocated_bytes'] or 0, value)
                del images
            synchronize(device)
            start = time.perf_counter()
            reset_peak(device)
            if not args.single_pass:
                array = np.load(feature_path, allow_pickle=False)
                if array.dtype != np.float32 or array.shape != (1, channels, 64, 64):
                    raise ValueError('Invalid cached feature tensor; no precision conversion is permitted')
                features = torch.from_numpy(array).to(device)
                synchronize(device)
                forward_start = time.perf_counter()
                with torch.no_grad():
                    original = probe(features).float()
            else:
                forward_start = time.perf_counter()
            validate_logits(original, (1, 1, *shapes[image_id]))
            trace, core_calls = refine_tensor(model, features, original, args.steps, args.ablation)
            synchronize(device)
            seconds = encoding_seconds + time.perf_counter() - forward_start
            value = peak_memory(device)
            if value is not None:
                peaks.append(value)
            write_outputs(args.output, image_id, trace, shapes[image_id], args.save_trace)
            records.append(dict(id=image_id, shape=list(shapes[image_id]), input_sha256=sha256(path),
                                core_calls=core_calls, seconds_with_trace_transfer=seconds,
                                peak_cuda_allocated_bytes=value))
            phases['refinement']['seconds'] += time.perf_counter() - start
            del features, original, trace
            if not args.single_pass:
                del array
            print(f'PREDICTED {image_id} K={args.steps} seconds={seconds:.6f} peak_cuda_bytes={value}', flush=True)
        phases['refinement']['peak_cuda_allocated_bytes'] = max(peaks) if peaks else None
    finally:
        if cache is not None:
            if args.keep_cache:
                print(f'KEPT_CACHE {cache}', flush=True)
            else:
                shutil.rmtree(cache)
    report = dict(protocol=PROTOCOL, steps=args.steps, demo=False, ablation=args.ablation,
                  seed=args.seed, device=args.device, checkpoint_sha256=sha256(args.checkpoint),
                  baseline_sha256=sha256(args.baseline), submission_source_sha256=source_hash(),
                  parameters=sum(p.numel() for p in model.parameters()),
                  core_parameters=sum(p.numel() for p in model.core.parameters()),
                  peak_cuda_allocated_bytes=max([v['peak_cuda_allocated_bytes'] for v in phases.values() if v['peak_cuda_allocated_bytes'] is not None], default=None),
                  timing_note='Per-image time sums feature extraction and probe/refiner/CPU trace transfer; phases include model loads and their file I/O. No warmup.',
                  images=records, number_of_images=len(records), encoder_files_sha256=encoder_hashes,
                  encoder_bundle_sha256=hashlib.sha256(json.dumps(encoder_hashes, sort_keys=True).encode()).hexdigest(),
                  mode='single-pass' if args.single_pass else 'two-phase', phases=phases,
                  encoder_release=release, cache_bytes=cache_bytes, cache_required_bytes=required,
                  cache_kept=bool(cache is not None and args.keep_cache), cache_removed=bool(cache is not None and not args.keep_cache))
    (args.output/'run.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    main()
