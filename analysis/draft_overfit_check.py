#!/usr/bin/env python3
"""Compare frozen drafts on seed-0 Train samples and recorded Val-rep results."""
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import train_refiner as runner
from scripts import eval_stress as shared
from looproad.models.draft_head import ConvUpDraftHead

DOMAINS = ("Urban", "Rural")
HEADS = ("linear", "A1-i")
SCORES = ("road_iou", "cldice", "boundary_iou", "A")
PROBES = {"linear": "runs/linear_probe/checkpoints/dinov3_vitl16_web_bn_posw4.pt",
          "A1-i": "runs/draft_head/A1-i/best.pt"}
RUNS = {"linear": "e0_v2_s0", "A1-i": "e6_convdraft_s0"}


def train_subset(per_domain=100, seed=0):
    """Reuse RoadDataset's split-aware sampler (same rule as Val selection).

    train_refiner.protocol_datasets:396-408 is Val-only. RoadDataset uses the
    same default_rng(seed).choice without replacement followed by np.sort;
    passing Train explicitly avoids modifying or monkeypatching that helper.
    """
    if per_domain < 1:
        raise ValueError("per_domain must be positive")
    return {d: runner.probe.RoadDataset("Train", d, limit=per_domain, seed=seed)
            for d in DOMAINS}


def gap_comparison(train, val):
    gaps = {h: train[h]["Overall"]["y0_draft"]["scores"]["A"] -
            val[h]["Overall"]["y0_draft"]["scores"]["A"] for h in HEADS}
    return {"gap": gaps, "gap_difference": gaps["A1-i"] - gaps["linear"]}


def score_predictions(domain, predictions, labels):
    """Delegate every per-image metric to the existing official worker path."""
    result = {}
    for head, masks in predictions.items():
        result[head] = {}
        for key, mask in masks.items():
            _, counts, _, _, _ = runner.probe.metric_counts_for_image(domain, mask, labels)
            result[head][key] = counts["0.5"]
    return domain, result


def load_pipeline(device):
    # Reuse the legacy frozen loader and exactly one encoder instance.
    encoder, linear, linear_config = runner.y0.load_frozen_probe(ROOT / PROBES["linear"], device)
    linear.requires_grad_(False).eval()
    config = torch.load(ROOT / PROBES["A1-i"], map_location="cpu", weights_only=True)
    if (config.get("head") != "conv_up" or config.get("draft_layers") != "last" or
            config.get("encoder") != linear_config["encoder"] or config.get("pos_weight") != 4):
        raise ValueError("Expected the A1-i last-layer conv_up checkpoint with the same frozen encoder")
    # Same head construction/load as draft_head.load_frozen_conv_draft:124-132,
    # without loading a second identical encoder. Tensor input is supported by
    # ConvUpDraftHead.forward and equals the last-layer wrapper's one-map tuple.
    conv = ConvUpDraftHead(encoder.out_channels, 1).to(device)
    conv.load_state_dict(config["state_dict"], strict=True)
    conv.requires_grad_(False).eval()
    split_hash = hashlib.sha256((ROOT / "configs/val_split_seed0.json").read_bytes()).hexdigest()
    models, epochs = {}, {}
    for head, run in RUNS.items():
        path = ROOT / "runs/exp" / run / "best.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["config"]["probe_ckpt"] != PROBES[head]:
            raise ValueError(f"Refiner {run} was trained with a different frozen draft")
        models[head], epochs[head] = shared.load_model(path, device, split_hash)
    return encoder, {"linear": linear, "A1-i": conv}, models, epochs


def frozen_pair(encoder, heads, images):
    """Encode once through frozen_draft; apply both frozen heads to that map."""
    features, linear = runner.frozen_draft(encoder, heads["linear"], images)
    with torch.no_grad():
        conv = heads["A1-i"](features).float()
    for value in (features, linear, conv):
        if not torch.isfinite(value).all():
            raise FloatingPointError("Nonfinite frozen features or draft logits")
    return features, {"linear": linear, "A1-i": conv}


def evaluate(encoder, heads, models, datasets, ks, device, loader_workers, metric_workers):
    """Bounded spawn pool; only compact binary predictions leave the parent."""
    metrics = runner.probe.official_metrics()
    keys = ["y0_draft", *map(str, ks)]
    counts = {h: {d: {k: metrics.Counts() for k in keys} for d in DOMAINS} for h in HEADS}
    pending = deque()
    started = time.perf_counter()
    forward_seconds = 0.
    files = {d: [] for d in DOMAINS}

    def consume():
        domain, values = pending.popleft().get()
        for head in HEADS:
            for key, value in values[head].items():
                counts[head][domain][key] = counts[head][domain][key] + metrics.Counts(**value)

    with mp.get_context("spawn").Pool(metric_workers, initializer=runner.initialize_eval_worker,
                                     initargs=(os.environ.get("EC523_STARTER_DIR"),)) as pool:
        for domain in DOMAINS:
            loader = runner.make_loader(datasets[domain], 1, loader_workers)
            for images, _, _, ids, labels in loader:
                images = runner.probe.as_device_tensor(images, device)
                phase = runner.stamp(device)
                features, drafts = frozen_pair(encoder, heads, images)
                predictions = {}
                for head in HEADS:
                    draft = drafts[head]
                    predictions[head] = {"y0_draft": (draft.sigmoid().squeeze() >= .5).cpu().numpy().astype(np.uint8)}
                    if ks:
                        for key, logits, _, _ in runner.inference_steps(models[head], features, draft, ks, device):
                            if not torch.isfinite(logits).all():
                                raise FloatingPointError("Nonfinite refiner logits")
                            predictions[head][key] = (logits.sigmoid().squeeze() >= .5).cpu().numpy().astype(np.uint8)
                forward_seconds += runner.stamp(device) - phase
                files[domain].append(ids[0] + ".png")
                pending.append(pool.apply_async(score_predictions,
                               (domain, predictions, np.asarray(labels).squeeze(0).astype(np.uint8))))
                if len(pending) >= 2 * metric_workers:
                    consume()
            del loader
        while pending:
            consume()
    groups = {}
    for head in HEADS:
        counts[head]["Overall"] = {k: counts[head]["Urban"][k] + counts[head]["Rural"][k] for k in keys}
        groups[head] = {d: {k: {"counts": asdict(c), "scores": metrics.summarize(c)}
                            for k, c in by_k.items()} for d, by_k in counts[head].items()}
        for domain in groups[head].values():
            for item in domain.values():
                item["scores"]["A"] = item["scores"]["quality"]
    return {"groups": groups, "files": files, "forward_seconds": forward_seconds,
            "wall_seconds": time.perf_counter() - started}


def markdown(report):
    lines = ["# Observed draft overfit check", "",
             "Observed Train samples use seed 0, without replacement, independently in each domain. "
             "Val-rep scores below are read from existing reports, not recomputed for this comparison. "
             "Train and Val distribution differences affect both heads; the registered statistic is the difference of gaps.", "",
             "| Head | Set | Domain | K | IoU | clDice | BIoU | A |",
             "|---|---|---|---|---:|---:|---:|---:|"]
    for head in HEADS:
        for split, groups in (("Train subset", report["train"]["groups"]), ("Val-rep (recorded)", report["val_recorded"])):
            for domain in ("Overall", *DOMAINS):
                for key in ("y0_draft", *map(str, report["config"]["ks"])):
                    scores = groups[head][domain][key]["scores"]
                    lines.append(f"| {head} | {split} | {domain} | {key} | " +
                                 " | ".join(f"{scores[m]:.9f}" for m in SCORES) + " |")
    lines += ["", "| Head | Overall Train y0 A minus Val-rep y0 A |", "|---|---:|"]
    lines += [f"| {h} | {report['comparison']['gap'][h]:+.9f} |" for h in HEADS]
    lines += [f"| A1-i minus linear gap | {report['comparison']['gap_difference']:+.9f} |", "",
              "| Head | Set | Domain | K | A minus own y0 A |", "|---|---|---|---|---:|"]
    for row in report["refiner_gains"]:
        lines.append(f"| {row['head']} | {row['set']} | {row['domain']} | {row['k']} | {row['gain']:+.9f} |")
    lines += ["", "## Registered predictions", "", f"Observed P1: {report['predictions']['P1']}.",
              f"Observed P2: {report['predictions']['P2']}.", f"Observed decision rule: {report['predictions']['decision']}.",
              "", "CPU or limited-sample runs verify execution only; they do not adjudicate full-data predictions.",
              "", "## Recorded Val provenance", ""]
    lines += [f"- {h}: {path}; JSON path evaluation.groups.<domain>.<K>.scores." for h, path in report["val_sources"].items()]
    if "val_regression" in report:
        lines += ["", "## Full Val-rep draft regression", ""]
        lines += [f"- {h}: {r['status']}; A={r['actual']:.12f}; delta={r['difference']:+.12g}."
                  for h, r in report["val_regression"]["checks"].items()]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-domain", type=int, default=100)
    parser.add_argument("--ks", type=runner.parse_ks, default="1,2,4")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--loader-workers", type=int, default=1)
    parser.add_argument("--metric-workers", type=int, default=6)
    parser.add_argument("--verify-val-drafts", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/draft_overfit")
    args = parser.parse_args()
    if args.per_domain < 1 or args.loader_workers < 0 or args.metric_workers < 1:
        parser.error("Expected positive sample/metric counts and nonnegative loader workers")
    if "NSLOTS" in os.environ and args.loader_workers + args.metric_workers + 1 > int(os.environ["NSLOTS"]):
        parser.error("Loader workers + metric workers + main exceed NSLOTS")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    torch.manual_seed(0)
    encoder, heads, models, epochs = load_pipeline(device)
    datasets = train_subset(args.per_domain, seed=0)
    train = evaluate(encoder, heads, models, datasets, args.ks, device, args.loader_workers, args.metric_workers)
    val, sources = {}, {}
    for head, run in RUNS.items():
        path = Path("results/exp") / run / "metrics.json"
        sources[head] = path.as_posix()
        val[head] = json.loads((ROOT / path).read_text())["evaluation"]["groups"]
    comparison = gap_comparison(train["groups"], val)
    gains = [{"head": h, "set": split, "domain": d, "k": k,
              "gain": groups[h][d][str(k)]["scores"]["A"] - groups[h][d]["y0_draft"]["scores"]["A"]}
             for h in HEADS for split, groups in (("Train subset", train["groups"]), ("Val-rep", val))
             for d in ("Overall", *DOMAINS) for k in args.ks]
    full = device.type == "cuda" and args.per_domain == 100 and all(len(d) == 100 for d in datasets.values()) and 4 in args.ks
    p1 = comparison["gap_difference"] >= .02 if full else None
    rural_gain = next((r["gain"] for r in gains if r["head"] == "A1-i" and r["set"] == "Train subset" and r["domain"] == "Rural" and r["k"] == 4), None)
    report = {"config": {"seed": 0, "per_domain": args.per_domain, "ks": args.ks, "device": str(device),
                          "probe_checkpoints": PROBES, "refiner_runs": RUNS},
              "loaded_refiner_epochs": epochs, "train": train, "val_recorded": val, "val_sources": sources,
              "comparison": comparison, "refiner_gains": gains,
              "predictions": {"P1": p1, "P2": rural_gain >= 0 if full else None,
                              "decision": "not adjudicated (smoke)" if not full else
                              "cross-fitted drafts" if p1 else "abandon overfitting explanation; random K and loss weights"}}
    if args.verify_val_drafts:
        split = json.loads((ROOT / "configs/val_split_seed0.json").read_text())
        verification = evaluate(encoder, heads, {}, runner.protocol_datasets(split, "Val-rep"), [],
                                device, args.loader_workers, args.metric_workers)
        checks = {}
        for h, reference in (("linear", .526050), ("A1-i", .538246)):
            actual = verification["groups"][h]["Overall"]["y0_draft"]["scores"]["A"]
            difference = actual - reference
            status = "PASS" if abs(difference) <= 1e-5 else "FAIL"
            checks[h] = dict(actual=actual, reference=reference, difference=difference, status=status)
            print(f"{status} Val-rep {h} y0 A={actual:.12f} delta={difference:+.12g} tolerance=1e-5", flush=True)
        report["val_regression"] = dict(checks=checks, evaluation=verification)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "summary.md").write_text(markdown(report))
    print("TRAIN_OVERALL_A " + json.dumps({h: {k: v["scores"]["A"] for k, v in train["groups"][h]["Overall"].items()} for h in HEADS}), flush=True)
    print(f"Observed Train forward={train['forward_seconds']:.3f}s wall={train['wall_seconds']:.3f}s; wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
