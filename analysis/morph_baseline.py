#!/usr/bin/env python3
"""Non-learning morphology controls for the frozen probe draft."""
import argparse
from collections import deque
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
from skimage.morphology import binary_closing, disk, remove_small_objects
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import refiner_diagnostics as diagnostics
from scripts import train_refiner as runner

_METRICS = None


def apply_operation(mask, kind, value):
    mask = np.asarray(mask, dtype=bool)
    if value < 0:
        raise ValueError("Morphology sizes must be nonnegative")
    if value == 0 or kind == "y0_draft":
        return mask.copy()
    if kind == "closing":
        return binary_closing(mask, footprint=disk(value))
    if kind == "remove_small":
        return remove_small_objects(mask, min_size=value, connectivity=2)
    raise ValueError("Unknown morphology operation")


def initialize_worker(starter):
    global _METRICS
    diagnostics.initialize_worker(starter)
    _METRICS = runner.probe.official_metrics(starter)


def analyze_image(domain, draft, labels, operations):
    valid, truth = labels != 0, labels == 3
    draft = draft.astype(bool) & valid
    gap, terminal, covered = diagnostics.baseline_skeleton_sets(truth & valid, draft)
    result = {}
    for name, kind, value in operations:
        predicted = apply_operation(draft, kind, value) & valid
        counts = _METRICS.image_counts(predicted.astype(np.float32), labels)
        analysis = runner.y0.analyze_masks(predicted, labels)
        result[name] = {"counts": asdict(counts),
            "changes": diagnostics.change_counts(draft, predicted, truth, valid),
            "tracking": {"gap_total": int(gap.sum()), "gap_covered": int((gap & predicted).sum()),
                         "terminal_total": int(terminal.sum()), "terminal_covered": int((terminal & predicted).sum()),
                         "covered_total": int(covered.sum()), "covered_lost": int((covered & ~predicted).sum())},
            "structure": {"fp_fragments": len(analysis["fragment_areas"]),
                          "predicted_components": analysis["predicted_components"]}}
    return domain, result


def aggregate(records, metrics, names):
    result = {}
    for name in names:
        counts = metrics.Counts()
        for record in records:
            counts += metrics.Counts(**record[name]["counts"])
        changes = {key: sum(row[name]["changes"][key] for row in records) for key in diagnostics.CHANGE_KEYS}
        correct = changes["added_correct"] + changes["deleted_correct"]
        changes.update(fractions={key: diagnostics.fraction(value, counts.valid) for key, value in changes.items()},
                       correct_fraction=diagnostics.fraction(correct, sum(changes.values())),
                       added_correct_fraction=diagnostics.fraction(changes["added_correct"], changes["added_correct"] + changes["added_wrong"]),
                       deleted_correct_fraction=diagnostics.fraction(changes["deleted_correct"], changes["deleted_correct"] + changes["deleted_wrong"]))
        tracking = {key: sum(row[name]["tracking"][key] for row in records) for key in records[0][name]["tracking"]}
        tracking.update(gap_repair_fraction=diagnostics.fraction(tracking["gap_covered"], tracking["gap_total"]),
                        terminal_coverage_fraction=diagnostics.fraction(tracking["terminal_covered"], tracking["terminal_total"]),
                        new_break_fraction=diagnostics.fraction(tracking["covered_lost"], tracking["covered_total"]))
        result[name] = {"scores": metrics.summarize(counts), "changes": changes, "tracking": tracking,
                        "structure": {key: sum(row[name]["structure"][key] for row in records) for key in ("fp_fragments", "predicted_components")}}
    return result


def summary(report, previous):
    lines = ["# Frozen-draft morphology controls", "", "Observed results only. Threshold is 0.5; all statistics exclude label 0.",
             "Morphology operates on the valid draft; outputs are clipped to valid pixels after the operation.",
             "Shared fixed-skeleton coverage and flip definitions are imported from refiner_diagnostics.", ""]
    for domain, operations in report["groups"].items():
        rows = []
        for name, item in operations.items():
            s, c, t, geometry = (item[key] for key in ("scores", "changes", "tracking", "structure"))
            rows.append([name, s["road_iou"], s["cldice"], s["boundary_iou"], s["quality"],
                         *[c[key] for key in diagnostics.CHANGE_KEYS],
                         *[c["fractions"][key] for key in diagnostics.CHANGE_KEYS], c["correct_fraction"],
                         t["gap_repair_fraction"], t["terminal_coverage_fraction"], t["new_break_fraction"],
                         geometry["fp_fragments"], geometry["predicted_components"]])
        lines += [f"## {domain}", "", *runner.y0.markdown_table(
            ["Operation", "IoU", "clDice", "BIoU", "A", *diagnostics.CHANGE_KEYS,
             *[key + "/valid" for key in diagnostics.CHANGE_KEYS], "Correct/changed", "Gap repair", "Terminal completion", "New break", "FP fragments", "Predicted components"], rows), ""]
    rows = []
    for name in report["groups"]["Overall"]:
        groups = report["groups"]
        item = groups["Overall"][name]
        rows.append([name, *[groups[d][name]["scores"]["quality"] for d in ("Overall", "Urban", "Rural")],
                     item["tracking"]["gap_repair_fraction"], item["tracking"]["new_break_fraction"],
                     item["changes"]["added_correct_fraction"], item["changes"]["deleted_correct_fraction"]])
    for key in ("1", "4"):
        groups = previous["groups"]["Val"]
        rows.append(["v1 K=" + key, *[groups[d]["scores"][key]["0.5"]["quality"] for d in ("Overall", "Urban", "Rural")],
                     groups["Overall"]["tracking"][key]["gap_repair_fraction"], groups["Overall"]["tracking"][key]["new_break_fraction"],
                     groups["Overall"]["changes"][key]["added_correct_fraction"], groups["Overall"]["changes"][key]["deleted_correct_fraction"]])
    lines += ["## Comparison with recorded full-Val v1 diagnostics", "",
              "If this run is limited, its morphology rows and the recorded full-Val v1 rows use different subsets.", "",
              *runner.y0.markdown_table(["Operation", "Overall A", "Urban A", "Rural A", "Gap repair", "New break", "Correct/added", "Correct/deleted"], rows), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--limit-per-domain", type=int)
    parser.add_argument("--radii", default="1,2,4,8")
    parser.add_argument("--min-areas", default="16,64,256")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/baselines/morph_y0")
    parser.add_argument("--metric-workers", type=int, default=6)
    parser.add_argument("--loader-workers", type=int, default=1)
    args = parser.parse_args()
    operations = [("y0_draft", "y0_draft", 0)]
    for prefix, kind, text in (("closing_r", "closing", args.radii), ("remove_small_", "remove_small", args.min_areas)):
        operations += [(prefix + str(value), kind, value) for value in sorted(set(map(int, text.split(','))))]
    if args.metric_workers < 1 or args.loader_workers < 0 or any(value < 0 for _, _, value in operations):
        parser.error("Worker counts and morphology sizes are invalid")
    slots = int(os.environ.get("NSLOTS", "0"))
    if slots and 1 + args.metric_workers + args.loader_workers > slots:
        parser.error("Main plus workers exceed NSLOTS")
    if not args.out_dir.resolve().is_relative_to(ROOT) or any((args.out_dir / n).exists() for n in ("metrics.json", "summary.md")):
        parser.error("Choose a fresh repository output directory")
    runner.probe.initialize_data_worker(0)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    probe_path = ROOT / "runs/linear_probe/checkpoints/dinov3_vitl16_web_bn_posw4.pt"
    encoder, head, _ = runner.y0.load_frozen_probe(probe_path, device)
    encoder.requires_grad_(False).eval()
    head.requires_grad_(False).eval()
    records, pending = [], deque()
    started = time.perf_counter()
    pool = mp.get_context("spawn").Pool(args.metric_workers, initializer=initialize_worker,
                                       initargs=(os.environ.get("EC523_STARTER_DIR"),))
    try:
        for index, domain in enumerate(runner.DOMAINS):
            data = runner.probe.RoadDataset("Val", domain, limit=args.limit_per_domain, seed=10 + index)
            for images, _, _, names, raw in runner.make_loader(data, 1, args.loader_workers):
                images = runner.probe.as_device_tensor(images, device)
                _, draft = runner.frozen_draft(encoder, head, images)
                binary = (draft.sigmoid().squeeze() >= .5).cpu().numpy()
                pending.append(pool.apply_async(analyze_image, (domain, binary, np.asarray(raw).squeeze(), operations)))
                print(f"INFERRED {domain} {names[0]}", flush=True)
                if len(pending) >= 2 * args.metric_workers:
                    records.append(pending.popleft().get())
        while pending:
            records.append(pending.popleft().get())
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    metrics = runner.probe.official_metrics()
    groups = {domain: aggregate([row for d, row in records if domain == "Overall" or d == domain], metrics,
                               [name for name, _, _ in operations]) for domain in (*runner.DOMAINS, "Overall")}
    report = {"groups": groups, "images": len(records), "limit_per_domain": args.limit_per_domain,
              "operations": operations, "wall_seconds": time.perf_counter() - started}
    previous = json.loads((ROOT / "results/refiner/v1_diagnostics/metrics.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(summary(report, previous), encoding="utf-8")
    print(f"DONE images={len(records)} seconds={report['wall_seconds']:.3f}", flush=True)


if __name__ == "__main__":
    main()
