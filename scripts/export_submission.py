#!/usr/bin/env python3
"""Export trusted local training checkpoints into tensor-only inference files.

This preparation tool may read our own .pt checkpoints with weights_only=True.
The submission inference package never reads pickle, optimizers or RNG state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
from submission.model import DEFAULT_CONFIG, refiner_from_config
from looproad.models.linear_probe import LinearProbe


def file_hash(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def export_weights(refiner_checkpoint, probe_checkpoint, output_dir):
    output_dir = Path(output_dir)
    paths = [output_dir/'model.safetensors', output_dir/'baseline.safetensors']
    if any(path.exists() for path in paths):
        raise FileExistsError('Refusing to overwrite an existing inference export')
    saved = torch.load(refiner_checkpoint, map_location='cpu', weights_only=True)
    config = {key: saved['config'].get(key, value) for key, value in DEFAULT_CONFIG.items()}
    refiner = refiner_from_config(1024, config)
    refiner.load_state_dict(saved['state_dict'], strict=True)
    count = sum(p.numel() for p in refiner.parameters())
    if config['head_type'] != 'conv_upsample' or count != 1_193_025:
        raise ValueError('Expected the selected v2 conv_upsample architecture')
    probe_saved = torch.load(probe_checkpoint, map_location='cpu', weights_only=True)
    if (probe_saved.get('encoder'), probe_saved.get('head'), float(probe_saved.get('pos_weight', -1))) != (
            'dinov3_vitl16_web', 'bn_linear', 4.):
        raise ValueError('Expected the selected frozen web-encoder BN probe')
    probe = LinearProbe(1024, head_type='bn_linear')
    probe.load_state_dict(probe_saved['state_dict'], strict=True)
    states = [refiner.state_dict(), probe.state_dict()]
    if any(not torch.isfinite(tensor).all() for state in states for tensor in state.values()):
        raise ValueError('Cannot export nonfinite model tensors')
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = [{'config': json.dumps(config, sort_keys=True), 'source_epoch': str(saved['epoch']),
                 'source_seed': str(saved['config']['seed']), 'source_sha256': file_hash(refiner_checkpoint)},
                {'encoder': 'dinov3_vitl16_web', 'head': 'bn_linear', 'pos_weight': '4',
                 'source_sha256': file_hash(probe_checkpoint)}]
    reports = []
    for path, state, model, attributes in zip(paths, states, (refiner, probe), metadata):
        tensors = {name: value.detach().cpu().contiguous() for name, value in state.items()}
        save_file(tensors, str(path), metadata=attributes)
        row = {'file': str(path), 'parameters': sum(p.numel() for p in model.parameters()),
               'tensor_elements_including_buffers': sum(t.numel() for t in tensors.values()),
               'bytes': path.stat().st_size, 'sha256': file_hash(path)}
        reports.append(row)
        print(json.dumps(row), flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refiner', type=Path, default=ROOT/'runs/exp/e0_v2_s0/best.pt')
    parser.add_argument('--probe', type=Path, default=ROOT/'runs/linear_probe/checkpoints/dinov3_vitl16_web_bn_posw4.pt')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'submission')
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(ROOT):
        parser.error('Exports must stay inside the repository')
    export_weights(args.refiner, args.probe, args.output_dir)


if __name__ == '__main__':
    main()
