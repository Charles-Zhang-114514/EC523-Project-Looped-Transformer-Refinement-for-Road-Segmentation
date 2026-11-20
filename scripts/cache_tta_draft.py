#!/usr/bin/env python3
"""Build identity-checked TTA drafts and gate cached e6 inference on GPU."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]
from analysis import evaluate_draft_tta as tta
from looproad.data.draft_cache import DraftCache, sha256, write_draft
from scripts import train_refiner as runner

HEAD = ROOT/"runs/draft_head/A1-i/best.pt"
REFINER = ROOT/"runs/exp/e6_convdraft_s0/best.pt"


def build_cache(root, smoke=False, per_domain=1, training_entrypoints=False):
    root = Path(root)
    if not root.resolve().is_relative_to(ROOT/"runs/cache") or root.exists():
        raise ValueError("Choose a fresh cache directory under runs/cache")
    split = json.loads(tta.SPLIT.read_text())
    datasets = {"Train": {d: runner.probe.RoadDataset("Train", d, limit=per_domain if smoke else None,
                                                    seed=i if training_entrypoints else 0)
                           for i,d in enumerate(runner.DOMAINS)},
                "Val-rep": runner.protocol_datasets(split, "Val-rep", per_domain if smoke else None)}
    if not smoke or training_entrypoints:
        datasets["Val-sel"] = runner.protocol_datasets(split, "Val-sel", per_domain if smoke else None)
    count = sum(len(ds) for part in datasets.values() for ds in part.values())
    free = shutil.disk_usage(ROOT).free
    dtype = "float32" if free >= count*1024*1024*4 + 2*1024**3 else "float16"
    if free < count*1024*1024*np.dtype(dtype).itemsize + 1024**3:
        raise ValueError("Insufficient free space for draft cache")
    root.mkdir(parents=True)
    device = torch.device("cuda")
    runner.probe.initialize_data_worker(0)
    encoder, head, _ = runner.y0.load_frozen_probe(HEAD, device)
    model, _ = tta.previous.shared.load_model(REFINER, device, sha256(tta.SPLIT))
    manifest = dict(head_sha256=sha256(HEAD), split_sha256=sha256(tta.SPLIT), dtype=dtype,
                    smoke=smoke, entries={}, gates_passed=False, pixels_flipped={},
                    free_bytes_at_start=free, total_images=count)
    metrics = runner.probe.official_metrics()
    live_counts = {d: metrics.Counts() for d in runner.DOMAINS}
    stored_counts = {d: metrics.Counts() for d in runner.DOMAINS}
    for part, domains in datasets.items():
        manifest["pixels_flipped"][part] = {}
        for domain, dataset in domains.items():
            flips = 0
            for images, _, _, ids, labels in runner.make_loader(dataset, 1, 1):
                image = runner.probe.as_device_tensor(images, device)
                with torch.no_grad():
                    features, initial = runner.frozen_draft(encoder, head, image)
                    logits = tta.average_d4_logits(lambda view: head(encoder(view)), image, initial)
                identity = f"{dataset.split}/{domain}/{ids[0]}"
                manifest["entries"][identity] = write_draft(root, identity, logits[0].cpu().numpy(), dtype)
                stored = logits.to(dtype=torch.float16 if dtype == "float16" else torch.float32).float()
                live_mask, stored_mask = tta.binary_mask(logits), tta.binary_mask(stored)
                flips += int((live_mask != stored_mask).sum())
                if part == "Val-rep" and smoke:
                    raw = np.asarray(labels).squeeze(0).astype(np.uint8)
                    live_counts[domain] += metrics.image_counts(live_mask.astype(np.float32), raw)
                    stored_counts[domain] += metrics.image_counts(stored_mask.astype(np.float32), raw)
            manifest["pixels_flipped"][part][domain] = flips
    live_counts["Overall"] = live_counts["Urban"] + live_counts["Rural"]
    stored_counts["Overall"] = stored_counts["Urban"] + stored_counts["Rural"]
    (root/"pending.json").write_text(json.dumps(manifest, indent=2)+"\n")
    cache = DraftCache(root, HEAD, tta.SPLIT, pending=True)
    tolerance = 1e-5 if dtype == "float32" else 2e-4
    original = json.loads((ROOT/"results/draft_head_v3/metrics.json").read_text())
    baseline = next(row for row in original["training"] if row["arm"] == "A1-i" and row["seed"] == 0)
    checks = []
    print("CACHE " + json.dumps(dict(images=count, dtype=dtype, flips=manifest["pixels_flipped"])), flush=True)
    evaluated = runner.evaluate(encoder, head, model, datasets["Val-rep"], [1,2,4,8], device,
                                1, 6, geometry=False, draft_cache=cache)
    for domain in tta.DOMAINS:
        actual = metrics.summarize(stored_counts[domain]) if smoke else evaluated["groups"][domain]["y0_draft"]["scores"]
        expected = metrics.summarize(live_counts[domain]) if smoke else baseline["tta"]["Val-rep"]["groups"]["d4"][domain]["y0_draft"]["scores"]
        for key in tta.SCORE_KEYS:
            checks.append(dict(gate="RG1", domain=domain, metric=key, difference=actual[key]-expected[key]))
    if not smoke:
        archived = json.loads((ROOT/"results/wave1/ensemble_bias/metrics.json").read_text())["splits"]["Val-rep"]["groups"]["E0"]
        for domain in tta.DOMAINS:
            for k in ("1","2","4","8"):
                for key in tta.SCORE_KEYS:
                    checks.append(dict(gate="RG2", domain=domain, K=k, metric=key,
                                       difference=evaluated["groups"][domain][k]["scores"][key]-archived[domain][k]["scores"][key]))
    manifest["checks"] = checks
    manifest["tolerance"] = tolerance
    manifest["gates_passed"] = all(abs(row["difference"]) <= tolerance for row in checks)
    manifest["cache_bytes"] = sum(path.stat().st_size for path in root.rglob("*.npz"))
    (root/"pending.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    if not manifest["gates_passed"]:
        raise ValueError("Cached draft regression failed; manifest was not published")
    (root/"pending.json").rename(root/"manifest.json")
    cache.manifest = manifest
    print("CACHE_PASS " + json.dumps(dict(smoke=smoke, tolerance=tolerance, comparisons=len(checks))), flush=True)
    return encoder, head, model, cache, datasets["Train"], datasets["Val-rep"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT/"runs/cache/tta_draft_a1i_s0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--per-domain", type=int, default=1)
    parser.add_argument("--training-entrypoints", action="store_true")
    args = parser.parse_args()
    if args.per_domain < 1:
        parser.error("Smoke sample count must be positive")
    if args.smoke:
        build_cache(args.cache_dir, smoke=True, per_domain=args.per_domain,
                    training_entrypoints=args.training_entrypoints)
    else:
        build_cache(args.cache_dir)


if __name__ == "__main__":
    main()
