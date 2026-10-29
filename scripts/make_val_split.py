#!/usr/bin/env python3
"""Create a reproducible stratified selection/report split using filenames only."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("Urban", "Rural")


def build_split(names_by_domain, seed=0):
    result = {"seed": seed, "algorithm": "sorted names -> per-domain default_rng(seed) permutation -> floor half selection -> sorted output",
              "Val-sel": {}, "Val-rep": {}, "counts": {"Val-sel": {}, "Val-rep": {}}}
    for domain in DOMAINS:
        names = sorted(names_by_domain[domain])
        if len(set(names)) != len(names) or any(Path(n).name != n or not n.endswith('.png') for n in names):
            raise ValueError("Expected unique PNG basenames")
        indices = np.random.default_rng(seed).permutation(len(names))
        half = len(names) // 2
        for part, selected in (("Val-sel", indices[:half]), ("Val-rep", indices[half:])):
            result[part][domain] = sorted(names[int(i)] for i in selected)
            result["counts"][part][domain] = len(selected)
    return result


def validate_split(split, names_by_domain):
    for domain in DOMAINS:
        selected, reported = split['Val-sel'][domain], split['Val-rep'][domain]
        if selected != sorted(set(selected)) or reported != sorted(set(reported)):
            raise ValueError("Split filenames must be unique and sorted")
        if set(selected) & set(reported) or set(selected) | set(reported) != set(names_by_domain[domain]):
            raise ValueError("Split must partition the complete Val domain")
        total = len(names_by_domain[domain])
        if len(selected) != total // 2 or len(reported) != total - total // 2:
            raise ValueError("Odd extra image must belong to Val-rep")
        for part in ('Val-sel', 'Val-rep'):
            if split['counts'][part][domain] != len(split[part][domain]):
                raise ValueError("Split count metadata does not match filenames")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT/'configs/val_split_seed0.json')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT) or args.output.exists():
        parser.error('Choose a new output file inside the repository')
    data = Path(os.environ['EC523_DATA_DIR'])
    names = {d: [p.name for p in (data/'Val'/d/'images_png').glob('*.png')] for d in DOMAINS}
    if any(not value for value in names.values()):
        raise ValueError('Both Val domains must contain PNG images')
    split = build_split(names, args.seed)
    validate_split(split, names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(split['counts']), flush=True)


if __name__ == '__main__':
    main()
