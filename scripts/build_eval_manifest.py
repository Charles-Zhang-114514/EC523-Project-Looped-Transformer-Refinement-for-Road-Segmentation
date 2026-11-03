#!/usr/bin/env python3
"""Build an ordinary Val-rep manifest for the unchanged official evaluator.

Format follows the starter README.md:123-157. Paths remain local runtime
metadata under runs/; raw 0..7 labels are referenced, never remapped/copied.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_manifest(split_path, data_dir=None, limit_per_domain=None):
    value = data_dir or os.environ.get('EC523_DATA_DIR')
    if not value:
        raise EnvironmentError('Set EC523_DATA_DIR to the read-only dataset')
    split = json.loads(Path(split_path).read_text())['Val-rep']
    rows = []
    for domain in ('Urban', 'Rural'):
        names = sorted(split[domain])
        if limit_per_domain is not None:
            names = names[:limit_per_domain]
        for name in names:
            if Path(name).name != name or Path(name).suffix.lower() != '.png':
                raise ValueError('Val-rep entries must be plain PNG filenames')
            base = Path(value)/'Val'/domain
            image = (base/'images_png'/name).resolve(strict=True)
            label = (base/'masks_png'/name).resolve(strict=True)
            rows.append(dict(id=f'{domain.lower()}/{Path(name).stem}', image=str(image),
                             label=str(label), domain=domain.lower(), condition='ordinary'))
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Manifest must contain unique, nonempty Val-rep IDs')
    return rows


def write_manifest(output, rows):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and json.loads(output.read_text()) != rows:
        raise ValueError('Refusing to overwrite a different evaluation manifest')
    output.write_text(json.dumps(rows, indent=2)+'\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--val-split', type=Path, default=ROOT/'configs/val_split_seed0.json')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/stress/ordinary.json')
    parser.add_argument('--limit-per-domain', type=int)
    args = parser.parse_args()
    if args.limit_per_domain is not None and args.limit_per_domain < 1:
        parser.error('--limit-per-domain must be positive')
    rows = build_manifest(args.val_split, limit_per_domain=args.limit_per_domain)
    write_manifest(args.output, rows)
    print(json.dumps({'images': len(rows), 'domains': {d: sum(r['domain']==d for r in rows)
                                                     for d in ('urban', 'rural')}}))


if __name__ == '__main__':
    main()
