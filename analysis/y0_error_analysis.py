#!/usr/bin/env python3
"""Analyze frozen-probe Val errors without training or saving predictions/features."""
import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.morphology import disk, skeletonize
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Reuse the existing runner without changing its training/evaluation behavior.
from scripts import train_linear_probe as probe
from looproad.data.road_dataset import map_raw_labels

ENCODER = "dinov3_vitl16_web"
DISTANCE_BINS = ("1-2", "3-4", "5-8", "9-16", ">16")
LENGTH_BINS = ("<=8", "9-16", "17-32", "33-64", ">64")
SEGMENT_TYPES = ("gap", "terminal_missing", "whole_missed")
GROUPS = ("Urban", "Rural", "Overall")
_METRICS = None


def distance_error_bins(errors, reference):
    """Bin continuous EDT values using upper bounds 2, 4, 8, 16, infinity.

    Bin names describe ceil(distance), so sqrt(5) belongs to the 3-4 bin.
    An empty reference sends every error pixel to >16, avoiding EDT's
    implicit outside-array background. Invalid pixels are never references.
    """
    counts = {name: 0 for name in DISTANCE_BINS}
    if not errors.any():
        return counts
    if not reference.any():
        counts[">16"] = int(errors.sum())
        return counts
    distances = ndi.distance_transform_edt(~reference)[errors]
    indices = np.searchsorted((2, 4, 8, 16), distances, side="left")
    for index, count in enumerate(np.bincount(indices, minlength=5)):
        counts[DISTANCE_BINS[index]] = int(count)
    return counts


def empty_segment_table():
    return {kind: {length: {"segments": 0, "skeleton_pixels": 0}
                   for length in LENGTH_BINS} for kind in SEGMENT_TYPES}


def classify_skeleton_segments(truth, predicted, return_gap_mask=False):
    """Share the original contact-region classifier with optional gap geometry."""
    skeleton = skeletonize(truth, method="zhang")
    covered = skeleton & predicted
    uncovered = skeleton & ~predicted
    uncovered_labels, _ = label(uncovered, connectivity=2, return_num=True)
    segments = empty_segment_table()
    gap_mask = np.zeros_like(truth, dtype=bool) if return_gap_mask else None
    for segment_id, bounds in enumerate(ndi.find_objects(uncovered_labels), start=1):
        if bounds is None:
            continue
        # Include one pixel around the segment, including diagonal contacts.
        region = tuple(slice(max(0, axis.start - 1), min(size, axis.stop + 1))
                       for axis, size in zip(bounds, truth.shape))
        segment = uncovered_labels[region] == segment_id
        length = int(segment.sum())
        contacts = ndi.binary_dilation(segment, structure=np.ones((3, 3), bool)) & covered[region]
        _, contact_count = label(contacts, connectivity=2, return_num=True)
        kind = "gap" if contact_count >= 2 else "terminal_missing" if contact_count == 1 else "whole_missed"
        length_bin = LENGTH_BINS[int(np.searchsorted((8, 16, 32, 64), length, side="left"))]
        segments[kind][length_bin]["segments"] += 1
        segments[kind][length_bin]["skeleton_pixels"] += length
        if return_gap_mask and kind == "gap":
            gap_mask[region] |= segment
    return skeleton, uncovered, segments, gap_mask


def analyze_masks(prediction, labels, return_gap_mask=False):
    """Return geometry statistics; optional gap pixels are never saved by default."""
    target, valid = map_raw_labels(labels)
    prediction = np.asarray(prediction, dtype=bool)
    if prediction.shape != valid.shape:
        raise ValueError("Prediction and label shapes must match")
    predicted = prediction & valid
    truth = target.astype(bool) & valid
    fp, fn = predicted & ~truth, truth & ~predicted
    fp_bins = distance_error_bins(fp, truth)
    fn_bins = distance_error_bins(fn, predicted)
    skeleton, uncovered, segments, gap_mask = classify_skeleton_segments(
        truth, predicted, return_gap_mask=return_gap_mask)

    _, truth_components = label(truth, connectivity=2, return_num=True)
    predicted_labels, predicted_components = label(predicted, connectivity=2, return_num=True)
    # A Euclidean radius-2 disk defines the fragment tolerance.
    near_truth = ndi.binary_dilation(truth, structure=disk(2)) & valid
    touches_truth = np.zeros(predicted_components + 1, dtype=bool)
    touches_truth[np.unique(predicted_labels[near_truth])] = True
    areas = np.bincount(predicted_labels.ravel(), minlength=predicted_components + 1)
    fragment_areas = areas[1:][~touches_truth[1:]].astype(np.int64).tolist()
    result = {
        "valid_pixels": int(valid.sum()), "truth_road_pixels": int(truth.sum()),
        "predicted_road_pixels": int(predicted.sum()), "fp_pixels": int(fp.sum()),
        "fn_pixels": int(fn.sum()), "fp_distance": fp_bins, "fn_distance": fn_bins,
        "skeleton_pixels": int(skeleton.sum()), "uncovered_skeleton_pixels": int(uncovered.sum()),
        "segments": segments, "fragment_areas": fragment_areas,
        "truth_components": int(truth_components), "predicted_components": int(predicted_components),
        "no_truth_road": not bool(truth.any()), "no_predicted_road": not bool(predicted.any()),
    }
    # Fail immediately on any lost/double-counted pixel or skeleton segment.
    if sum(fp_bins.values()) != result["fp_pixels"] or sum(fn_bins.values()) != result["fn_pixels"]:
        raise AssertionError("Distance bins do not partition the FP/FN pixels")
    segment_pixels = sum(item["skeleton_pixels"] for bins in segments.values() for item in bins.values())
    if segment_pixels != result["uncovered_skeleton_pixels"]:
        raise AssertionError("Segment types do not partition uncovered skeleton pixels")
    if sum(fragment_areas) > result["fp_pixels"]:
        raise AssertionError("False-positive fragment area exceeds total FP area")
    if return_gap_mask:
        result["gap_mask"] = gap_mask
    return result


def initialize_analysis_worker(starter_dir):
    global _METRICS
    probe.initialize_data_worker(0)
    _METRICS = probe.official_metrics(starter_dir)


def analyze_image(domain, filename, prediction, labels):
    started = time.perf_counter()
    metric_started = time.perf_counter()
    # Official metrics only: never average the resulting per-image scores.
    counts = _METRICS.image_counts(prediction.astype(np.float32), labels)
    scores = _METRICS.summarize(counts)
    metric_seconds = time.perf_counter() - metric_started
    result = analyze_masks(prediction, labels)
    result.update(domain=domain, filename=filename, counts=asdict(counts),
                  scores={key: scores[key] for key in ("road_iou", "cldice", "boundary_iou", "quality")},
                  metric_seconds=metric_seconds, analysis_seconds=time.perf_counter() - started)
    return result


def quantiles(values):
    if not values:
        return {key: None for key in ("min", "p10", "p25", "p50", "p75", "p90", "max")}
    return dict(zip(("min", "p10", "p25", "p50", "p75", "p90", "max"),
                    (float(value) for value in np.quantile(values, (0, .1, .25, .5, .75, .9, 1)))))


def fraction(numerator, denominator):
    """Descriptive fractions are undefined (JSON null) for an empty denominator."""
    return numerator / denominator if denominator else None


def aggregate(records, metrics):
    counts = metrics.Counts()
    for record in records:
        counts = counts + metrics.Counts(**record["counts"])
    scores = metrics.summarize(counts)
    totals = {key: sum(record[key] for record in records) for key in (
        "valid_pixels", "truth_road_pixels", "predicted_road_pixels", "fp_pixels", "fn_pixels",
        "skeleton_pixels", "uncovered_skeleton_pixels", "truth_components", "predicted_components")}
    distance = {}
    total_errors = totals["fp_pixels"] + totals["fn_pixels"]
    for name in DISTANCE_BINS:
        fp = sum(record["fp_distance"][name] for record in records)
        fn = sum(record["fn_distance"][name] for record in records)
        distance[name] = {"fp_pixels": fp, "fn_pixels": fn,
                          "fp_fraction_of_all_errors": fraction(fp, total_errors),
                          "fn_fraction_of_all_errors": fraction(fn, total_errors)}
    segments = empty_segment_table()
    for record in records:
        for kind in SEGMENT_TYPES:
            for length in LENGTH_BINS:
                for key in ("segments", "skeleton_pixels"):
                    segments[kind][length][key] += record["segments"][kind][length][key]
    fragment_areas = [area for record in records for area in record["fragment_areas"]]
    empty_truth_with_predictions = [record for record in records
                                    if record["no_truth_road"] and not record["no_predicted_road"]]
    worst = sorted(records, key=lambda row: (row["scores"]["quality"], row["filename"]))[:10]
    return {"images": len(records), "official": scores, "totals": totals,
            "distance": distance, "segments": segments,
            "uncovered_skeleton_fraction": fraction(totals["uncovered_skeleton_pixels"], totals["skeleton_pixels"]),
            "fragments": {"count": len(fragment_areas), "total_area": sum(fragment_areas),
                          "area_quantiles": quantiles(fragment_areas),
                          "fraction_of_fp_pixels": fraction(sum(fragment_areas), totals["fp_pixels"])},
            "image_A_quantiles": quantiles([row["scores"]["quality"] for row in records]),
            "no_truth_road_images": sum(row["no_truth_road"] for row in records),
            "no_predicted_road_images": sum(row["no_predicted_road"] for row in records),
            "no_truth_with_predicted_road_images": len(empty_truth_with_predictions),
            "no_truth_with_predicted_road_fp_pixels": sum(row["fp_pixels"] for row in empty_truth_with_predictions),
            "worst_images": [{"filename": row["filename"], **row["scores"]} for row in worst]}


def csv_row(record):
    row = {key: record[key] for key in ("filename", "domain")}
    row.update(record["scores"])
    row["A"] = row["quality"]
    for key in ("valid_pixels", "truth_road_pixels", "predicted_road_pixels", "fp_pixels", "fn_pixels",
                "skeleton_pixels", "uncovered_skeleton_pixels", "truth_components", "predicted_components",
                "no_truth_road", "no_predicted_road", "analysis_seconds", "metric_seconds", "forward_seconds"):
        row[key] = record[key]
    for prefix in ("fp", "fn"):
        for name, value in record[f"{prefix}_distance"].items():
            row[f"{prefix}_distance_{name}"] = value
    for kind, bins in record["segments"].items():
        row[f"{kind}_segments"] = sum(item["segments"] for item in bins.values())
        row[f"{kind}_skeleton_pixels"] = sum(item["skeleton_pixels"] for item in bins.values())
    row["fp_fragments"] = len(record["fragment_areas"])
    row["fp_fragment_area"] = sum(record["fragment_areas"])
    return row


def format_number(value):
    if value is None:
        return "NA"
    return str(value) if isinstance(value, int) else f"{value:.6f}"


def markdown_table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(format_number(value) if not isinstance(value, str) else value for value in row) + " |"
        for row in rows]


def render_summary(summary):
    groups = summary["groups"]
    lines = ["# Frozen-probe y0 error analysis", "",
             f"Encoder: `{ENCODER}`; head: `bn_linear`; checkpoint: `{summary['checkpoint']}`; threshold: 0.5.",
             f"Images: {summary['images']}; seed: {summary['seed']}; device: {summary['device']}.",
             f"Full-Val regression against A=0.526507: {summary['full_val_regression']['status']}.", "",
             "A limited run is a workflow check, not evidence about the full Val split.", "",
             "## Official pooled scores", ""]
    lines += markdown_table(["Group", "Images", "IoU", "clDice", "BIoU", "A"], [
        [group, groups[group]["images"]] + [groups[group]["official"][key]
        for key in ("road_iou", "cldice", "boundary_iou", "quality")] for group in GROUPS])
    lines += ["", "Per-image scores and pooled scores use official image_counts/summarize. Overall Counts = Urban + Rural.",
              "The official boundary metric additionally excludes the image frame and 3-pixel neighborhoods of ignored regions",
              "(metrics.py:44-49); the descriptive geometry statistics below include all valid pixels.",
              "", "## Pixel errors by Euclidean distance", "",
              "Fractions use FP+FN in the same group as denominator. Bin labels describe ceil(distance):",
              "(0,2], (2,4], (4,8], (8,16], (16,infinity). No rounding is applied to EDT values.",
              "With no reference road, all relevant error pixels enter >16. Ignored pixels are excluded",
              "as sources and observations; straight-line EDT distances can cross ignored regions.", ""]
    lines += markdown_table(["Group", "Distance", "FP pixels", "FP / all errors", "FN pixels", "FN / all errors"], [
        [group, name, item["fp_pixels"], item["fp_fraction_of_all_errors"], item["fn_pixels"], item["fn_fraction_of_all_errors"]]
        for group in GROUPS for name, item in groups[group]["distance"].items()])
    lines += ["", "## Uncovered skeleton segments", "",
              "Zhang skeletons and 8-connectivity are used. Dilate each uncovered segment by a 3x3 square;",
              "count 8-connected contact regions on covered skeleton: >=2 = gap, 1 = terminal_missing, 0 = whole_missed.",
              "Lengths count skeleton pixels, not Euclidean arc length. These are geometric proxies, not verified road identities.", ""]
    lines += markdown_table(["Group", "Type", "Length", "Segments", "Skeleton pixels"], [
        [group, kind, length, item["segments"], item["skeleton_pixels"]]
        for group in GROUPS for kind, bins in groups[group]["segments"].items() for length, item in bins.items()])
    lines += [""] + markdown_table(["Group", "GT skeleton pixels", "Uncovered pixels", "Uncovered fraction"], [
        [group, groups[group]["totals"]["skeleton_pixels"], groups[group]["totals"]["uncovered_skeleton_pixels"],
         groups[group]["uncovered_skeleton_fraction"]] for group in GROUPS])
    lines += ["", "## False-positive fragments", "",
              "A predicted 8-connected component is a fragment only if it never touches GT dilated by a radius-2 Euclidean disk.",
              "Dilation outputs are clipped to valid pixels. Area quantiles pool fragment components, not per-image averages.", ""]
    lines += markdown_table(["Group", "Fragments", "Area", "Area / FP", "Min", "P10", "P25", "P50", "P75", "P90", "Max"], [
        [group, groups[group]["fragments"]["count"], groups[group]["fragments"]["total_area"],
         groups[group]["fragments"]["fraction_of_fp_pixels"]] + list(groups[group]["fragments"]["area_quantiles"].values())
        for group in GROUPS])
    lines += ["", "## Component counts", "", "Counts sum tile-local components; they are not a stitched geographic graph.", ""]
    lines += markdown_table(["Group", "GT components", "Predicted components"], [
        [group, groups[group]["totals"]["truth_components"], groups[group]["totals"]["predicted_components"]] for group in GROUPS])
    lines += ["", "## Per-image A distribution", ""]
    lines += markdown_table(["Group", "Min", "P10", "P25", "P50", "P75", "P90", "Max"], [
        [group] + list(groups[group]["image_A_quantiles"].values()) for group in GROUPS])
    lines += ["", "## Empty-road cases", ""]
    lines += markdown_table(["Group", "No GT road", "No predicted road", "No GT but predicted road", "FP pixels in those images"], [
        [group] + [groups[group][key] for key in ("no_truth_road_images", "no_predicted_road_images",
          "no_truth_with_predicted_road_images", "no_truth_with_predicted_road_fp_pixels")] for group in GROUPS])
    lines += ["", "## Lowest-A images (up to 10 per domain)", ""]
    lines += markdown_table(["Group", "Filename", "IoU", "clDice", "BIoU", "A"], [
        [group, row["filename"], row["road_iou"], row["cldice"], row["boundary_iou"], row["quality"]]
        for group in GROUPS[:2] for row in groups[group]["worst_images"]])
    lines += ["", "## Timing and self-checks", ""]
    lines += markdown_table(["Quantity", "Seconds"], list(summary["timing"].items()))
    lines += ["", "All per-image distance-bin and uncovered-skeleton partition checks passed.",
              "NA/null denotes an empty denominator or distribution. Empty-road official scores follow the official definitions.",
              "Visualizations remain under ignored runs/; no probabilities or features are saved.",
              "Panels show the original on the left and an overlay on the right: TP green, FP red, FN blue, ignored gray.", ""]
    return "\n".join(lines)


def save_visualizations(worst, datasets, vis_dir):
    paths = []
    for domain in GROUPS[:2]:
        sources = {image.name: (image, mask) for image, mask in datasets[domain].pairs}
        for rank, (_, filename, predicted) in enumerate(worst[domain], start=1):
            image_path, mask_path = sources[filename]
            with Image.open(image_path) as image_file:
                image = np.array(image_file.convert("RGB"))
            with Image.open(mask_path) as mask_file:
                raw = np.array(mask_file)
            valid, truth = raw != 0, raw == 3
            predicted = predicted.astype(bool) & valid
            overlay = image.copy()
            for mask, color in ((truth & predicted, (0, 255, 0)),
                                (predicted & ~truth, (255, 0, 0)),
                                (truth & ~predicted, (0, 0, 255))):
                overlay[mask] = (.5 * image[mask] + .5 * np.asarray(color)).astype(np.uint8)
            overlay[~valid] = (128, 128, 128)
            canvas = Image.new("RGB", (image.shape[1] * 2, image.shape[0]))
            canvas.paste(Image.fromarray(image), (0, 0))
            canvas.paste(Image.fromarray(overlay), (image.shape[1], 0))
            path = vis_dir / f"{domain}_{rank:02d}_{filename}"
            canvas.save(path)
            paths.append(path.name)
    return paths


def load_frozen_probe(checkpoint_path, device):
    """Load the selected checkpoint exactly as in the original analysis runner."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("encoder") != ENCODER or checkpoint.get("head") != "bn_linear":
        raise ValueError("Expected the selected dinov3_vitl16_web + bn_linear checkpoint")
    if float(checkpoint.get("pos_weight", -1)) != 4:
        raise ValueError("Expected a checkpoint trained with pos_weight=4")
    encoder = probe.build_encoder(ENCODER).to(device).eval()
    head = probe.LinearProbe(encoder.out_channels, head_type=checkpoint["head"]).to(device).eval()
    head.load_state_dict(checkpoint["state_dict"])
    if encoder.training or head.training or any(p.requires_grad for p in encoder.parameters()):
        raise RuntimeError("The encoder must be frozen and both modules must be in eval mode")
    return encoder, head, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=ROOT / "runs/linear_probe/checkpoints/dinov3_vitl16_web_bn_posw4.pt")
    parser.add_argument("--limit-per-domain", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    slots = int(os.environ.get("NSLOTS", "0"))
    loader_default = 1 if slots >= 3 else 0
    parser.add_argument("--loader-workers", type=int, default=loader_default)
    parser.add_argument("--metric-workers", type=int, default=max(1, slots - loader_default - 1) if slots else 1)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/y0_analysis/dinov3_vitl16_web_bn_posw4")
    parser.add_argument("--vis-dir", type=Path, default=ROOT / "runs/y0_analysis/vis")
    args = parser.parse_args()
    if args.limit_per_domain is not None and args.limit_per_domain < 1:
        parser.error("--limit-per-domain must be positive")
    if args.metric_workers < 1 or args.loader_workers < 0:
        parser.error("Worker counts must be positive (loader workers may be zero)")
    if slots and args.metric_workers + args.loader_workers + 1 > slots:
        parser.error("Metric workers + loader workers + main must not exceed NSLOTS")
    for path in (args.out_dir, args.vis_dir):
        if not path.resolve().is_relative_to(ROOT):
            parser.error("Outputs must remain inside the repository")
    if not args.vis_dir.resolve().is_relative_to(ROOT / "runs"):
        parser.error("Visualizations must remain under ignored runs/")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("--out-dir must be empty; existing results will not be overwritten")
    probe.initialize_data_worker(0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    started = time.perf_counter()
    metrics = probe.official_metrics()
    encoder, head, checkpoint = load_frozen_probe(args.ckpt, device)
    datasets = {domain: probe.RoadDataset("Val", domain, limit=args.limit_per_domain,
                                         seed=args.seed + i + 10) for i, domain in enumerate(GROUPS[:2])}
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("Both Val domains must contain images")
    if args.limit_per_domain is None and sum(map(len, datasets.values())) != 1669:
        raise ValueError("Full Val must contain exactly 1669 images")
    records, pending = [], deque()
    worst = {domain: [] for domain in GROUPS[:2]}

    def consume():
        job, predicted, forward_seconds = pending.popleft()
        record = job.get()
        record["forward_seconds"] = forward_seconds
        records.append(record)
        candidates = worst[record["domain"]]
        candidates.append((record["scores"]["quality"], record["filename"], predicted))
        candidates.sort(key=lambda item: (item[0], item[1]))
        del candidates[10:]
        if len(records) % 100 == 0:
            print(f"Analyzed {len(records)} Val images", flush=True)

    pool = mp.get_context("spawn").Pool(args.metric_workers, initializer=initialize_analysis_worker,
                                        initargs=(os.environ.get("EC523_STARTER_DIR"),))
    try:
        for domain, dataset in datasets.items():
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.loader_workers,
                                pin_memory=True, persistent_workers=args.loader_workers > 0,
                                worker_init_fn=probe.initialize_data_worker if args.loader_workers else None,
                                collate_fn=probe.collate_numpy_batch)
            for image, _, _, sample_id, raw_labels in loader:
                forward_started = time.perf_counter()
                with torch.no_grad():
                    probability = torch.sigmoid(head(encoder(probe.as_device_tensor(image, device))))
                    predicted = (probability.squeeze() >= .5).cpu().numpy().astype(np.uint8)
                forward_seconds = time.perf_counter() - forward_started
                del probability
                raw = np.asarray(raw_labels).squeeze().astype(np.uint8, copy=False)
                job = pool.apply_async(analyze_image, (domain, sample_id[0] + ".png", predicted, raw))
                pending.append((job, predicted, forward_seconds))
                if len(pending) >= 2 * args.metric_workers:
                    consume()
            del loader
        while pending:
            consume()
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    groups = {group: aggregate([row for row in records if row["domain"] == group], metrics) for group in GROUPS[:2]}
    groups["Overall"] = aggregate(records, metrics)
    pooled_counts = metrics.Counts(**groups["Urban"]["official"]["counts"]) + metrics.Counts(**groups["Rural"]["official"]["counts"])
    if groups["Overall"]["official"] != metrics.summarize(pooled_counts):
        raise AssertionError("Overall Counts must equal Urban + Rural")
    score = groups["Overall"]["official"]["quality"]
    regression = {"expected_A": .526507, "observed_A": score,
                  "status": "not run: limited subset" if args.limit_per_domain is not None else "passed"}
    if args.limit_per_domain is None and abs(score - .526507) > .0000005:
        raise RuntimeError(f"Full-Val A regression failed: observed {score:.12f}, expected 0.526507 at six decimals")
    args.vis_dir.mkdir(parents=True, exist_ok=True)
    visualization_names = save_visualizations(worst, datasets, args.vis_dir)
    forward_total = sum(row["forward_seconds"] for row in records)
    analysis_total = sum(row["analysis_seconds"] for row in records)
    summary = {"encoder": ENCODER, "head": checkpoint["head"], "pos_weight": checkpoint["pos_weight"],
               "checkpoint": args.ckpt.name, "threshold": .5, "seed": args.seed, "device": str(device),
               "images": len(records), "limit_per_domain": args.limit_per_domain,
               "workers": {"main": 1, "loader": args.loader_workers, "analysis": args.metric_workers, "threads_each": 1},
               "full_val_regression": regression, "groups": groups,
               "self_checks": {"distance_partitions": True, "skeleton_partition": True, "pooled_counts": True},
               "visualizations": visualization_names,
               "definitions": {"distance_bin_upper_bounds": [2, 4, 8, 16, None],
                   "empty_reference": "all relevant errors enter >16",
                   "ignored": "excluded from predictions, truth and statistics; EDT is not geodesic",
                   "segment_contact_dilation": "3x3 square, 8-connected contact regions",
                   "fragment_tolerance": "Euclidean radius-2 disk",
                   "undefined_descriptive_fraction": None},
               "timing": {"forward_total": forward_total, "forward_mean_per_image": forward_total / len(records),
                   "analysis_worker_total": analysis_total, "analysis_mean_per_image": analysis_total / len(records),
                   "official_metric_worker_total": sum(row["metric_seconds"] for row in records),
                   "wall_including_loading_and_visualization": time.perf_counter() - started}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [csv_row(row) for row in records]
    with (args.out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(render_summary(summary), encoding="utf-8")
    print(f"Overall A={score:.12f}; analysis seconds/image={analysis_total / len(records):.6f}", flush=True)
    print(f"Wrote {args.out_dir}; visualizations: {args.vis_dir}", flush=True)


if __name__ == "__main__":
    main()
