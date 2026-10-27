#!/usr/bin/env python3
"""Diagnose the final refiner checkpoint without training or saving predictions."""
import argparse
from collections import deque
import csv
from dataclasses import asdict
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_refiner as runner
from analysis import y0_error_analysis as y0
from analysis import y0_visual_audit as visual

KEYS = ("y0_draft", "1", "2", "4", "8")
THRESHOLDS = tuple(round(i / 10, 1) for i in range(1, 10))
CHANGE_KEYS = ("added_correct", "added_wrong", "deleted_correct", "deleted_wrong")
_METRICS = None


def change_counts(draft, prediction, truth, valid):
    """Partition threshold-0.5 flips within the original valid region."""
    added = ~draft & prediction & valid
    deleted = draft & ~prediction & valid
    return {"added_correct": int((added & truth).sum()),
            "added_wrong": int((added & ~truth).sum()),
            "deleted_correct": int((deleted & ~truth).sum()),
            "deleted_wrong": int((deleted & truth).sum())}


def fragment_shapes(prediction, draft, truth, valid, patch=16):
    """Eight-connected components with no exact overlap with draft or truth."""
    components, count = y0.label(prediction & valid, connectivity=2, return_num=True)
    touching = np.zeros(count + 1, dtype=bool)
    touching[np.unique(components[(draft | truth) & valid])] = True
    areas = np.bincount(components.ravel(), minlength=count + 1)
    shapes = []
    for index, bounds in enumerate(ndi.find_objects(components), 1):
        if bounds is None or touching[index]:
            continue
        row, col = bounds
        # Bounding maxima are inclusive. Both ends must lie in the same cell.
        single = row.start // patch == (row.stop - 1) // patch and col.start // patch == (col.stop - 1) // patch
        edges = (row.start, row.stop - 1, col.start, col.stop - 1)
        aligned = [int(edge % patch in (0, patch - 1)) for edge in edges]
        shapes.append({"area": int(areas[index]), "height": row.stop - row.start,
                       "width": col.stop - col.start, "single_cell": int(single),
                       "aligned_edges": sum(aligned), "any_aligned_edge": int(any(aligned)),
                       "all_aligned_edges": int(all(aligned))})
    return shapes


def baseline_skeleton_sets(truth, draft):
    """Reuse the shared classifier; recover terminal geometry among non-gap U."""
    skeleton, uncovered, table, gaps = y0.classify_skeleton_segments(truth, draft, return_gap_mask=True)
    covered = skeleton & draft
    # Every remaining uncovered component has zero or one contact region:
    # the shared classifier has already removed every >=2-contact component.
    remaining, _ = y0.label(uncovered & ~gaps, connectivity=2, return_num=True)
    terminals = np.zeros_like(truth)
    for index, bounds in enumerate(ndi.find_objects(remaining), 1):
        if bounds is None:
            continue
        region = tuple(slice(max(0, axis.start - 1), min(size, axis.stop + 1))
                       for axis, size in zip(bounds, truth.shape))
        segment = remaining[region] == index
        if (ndi.binary_dilation(segment, structure=np.ones((3, 3), bool)) & covered[region]).any():
            terminals[region] |= segment
    assert int(gaps.sum()) == sum(v["skeleton_pixels"] for v in table["gap"].values())
    assert int(terminals.sum()) == sum(v["skeleton_pixels"] for v in table["terminal_missing"].values())
    return gaps, terminals, covered


def initialize_worker(starter):
    global _METRICS
    runner.probe.initialize_metric_worker(starter, (.5,))
    _METRICS = runner.probe.official_metrics(starter)


def process_image(split, domain, filename, probabilities, binary, labels):
    started = time.perf_counter()
    valid, truth = labels != 0, labels == 3
    predictions = {key: ((probabilities[key] >= .5) if key in probabilities else binary[key]) & valid for key in KEYS}
    draft = predictions["y0_draft"]
    result = {"split": split, "domain": domain, "filename": filename,
              "valid_pixels": int(valid.sum()), "gt_has_road": bool((truth & valid).any()),
              "counts": {}, "scores": {}, "changes": {}, "tracking": {}, "fragments": {}}
    for key in KEYS:
        thresholds = THRESHOLDS if split == "Val" and key in probabilities else (.5,)
        result["counts"][key], result["scores"][key] = {}, {}
        for threshold in thresholds:
            mask = probabilities[key] >= threshold if key in probabilities else binary[key]
            counts = _METRICS.image_counts(mask.astype(np.float32), labels)
            tag = str(threshold)
            result["counts"][key][tag] = asdict(counts)
            result["scores"][key][tag] = _METRICS.summarize(counts)
    if split == "Val":
        gaps, terminals, covered = baseline_skeleton_sets(truth & valid, draft)
        # Independently reuse the old runner's flip-count path as a self-check.
        _, _, old_values, _ = runner.score_image(domain, filename, predictions, labels, geometry=False)
        for key in KEYS[1:]:
            predicted = predictions[key]
            changes = change_counts(draft, predicted, truth, valid)
            assert changes["added_correct"] + changes["added_wrong"] == old_values[key]["added"]
            assert changes["deleted_correct"] + changes["deleted_wrong"] == old_values[key]["deleted"]
            assert result["counts"][key]["0.5"] == old_values[key]["counts"]
            result["changes"][key] = changes
            result["tracking"][key] = {
                "gap_total": int(gaps.sum()), "gap_covered": int((gaps & predicted).sum()),
                "terminal_total": int(terminals.sum()), "terminal_covered": int((terminals & predicted).sum()),
                "covered_total": int(covered.sum()), "covered_lost": int((covered & ~predicted).sum())}
        for key in ("1", "4"):
            result["fragments"][key] = fragment_shapes(predictions[key], draft, truth, valid)
        result["packed"] = {key: np.packbits(predictions[key]) for key in ("y0_draft", "1", "4")}
        result["shape"] = labels.shape
    result["worker_seconds"] = time.perf_counter() - started
    return result


def fraction(value, total):
    return value / total if total else None


def aggregate(records, metrics):
    valid = sum(row["valid_pixels"] for row in records)
    out = {"images": len(records), "valid_pixels": valid, "scores": {}, "changes": {},
           "tracking": {}, "fragments": {}, "best_thresholds": {}}
    for key in KEYS:
        tags = records[0]["counts"][key].keys()
        out["scores"][key] = {}
        for tag in tags:
            counts = metrics.Counts()
            for row in records:
                counts = counts + metrics.Counts(**row["counts"][key][tag])
            out["scores"][key][tag] = {"counts": asdict(counts), **metrics.summarize(counts)}
        best = max(out["scores"][key], key=lambda tag: out["scores"][key][tag]["quality"])
        out["best_thresholds"][key] = {"threshold": float(best), "A": out["scores"][key][best]["quality"]}
    if records[0]["split"] == "Val":
        for key in KEYS[1:]:
            changes = {name: sum(row["changes"][key][name] for row in records) for name in CHANGE_KEYS}
            changed = sum(changes.values())
            correct = changes["added_correct"] + changes["deleted_correct"]
            out["changes"][key] = {**changes, "fractions": {name: fraction(value, valid) for name, value in changes.items()},
                "correct_fraction": fraction(correct, changed),
                "added_correct_fraction": fraction(changes["added_correct"], changes["added_correct"] + changes["added_wrong"]),
                "deleted_correct_fraction": fraction(changes["deleted_correct"], changes["deleted_correct"] + changes["deleted_wrong"]),
                "added_fraction": fraction(changes["added_correct"] + changes["added_wrong"], valid),
                "deleted_fraction": fraction(changes["deleted_correct"] + changes["deleted_wrong"], valid)}
            totals = {name: sum(row["tracking"][key][name] for row in records) for name in records[0]["tracking"][key]}
            out["tracking"][key] = {**totals, "gap_repair_fraction": fraction(totals["gap_covered"], totals["gap_total"]),
                "terminal_coverage_fraction": fraction(totals["terminal_covered"], totals["terminal_total"]),
                "new_break_fraction": fraction(totals["covered_lost"], totals["covered_total"])}
        for key in ("1", "4"):
            shapes = [shape for row in records for shape in row["fragments"][key]]
            count = len(shapes)
            out["fragments"][key] = {"count": count,
                "area": y0.quantiles([shape["area"] for shape in shapes]),
                "width": y0.quantiles([shape["width"] for shape in shapes]),
                "height": y0.quantiles([shape["height"] for shape in shapes]),
                "single_cell_fraction": fraction(sum(s["single_cell"] for s in shapes), count),
                "aligned_edge_fraction": fraction(sum(s["aligned_edges"] for s in shapes), 4 * count),
                "any_aligned_edge_fraction": fraction(sum(s["any_aligned_edge"] for s in shapes), count),
                "all_aligned_edges_fraction": fraction(sum(s["all_aligned_edges"] for s in shapes), count)}
    return out


def csv_row(record):
    row = {key: record[key] for key in ("split", "domain", "filename", "gt_has_road", "valid_pixels")}
    row["empty_gt_score_note"] = "official definition; no GT road" if not record["gt_has_road"] else ""
    for key in ("y0_draft", "1", "4"):
        row[f"{key}_A"] = record["scores"][key]["0.5"]["quality"]
    row["delta_A_1"] = row["1_A"] - row["y0_draft_A"]
    for key in KEYS[1:]:
        for name in CHANGE_KEYS:
            row[f"K{key}_{name}"] = record["changes"].get(key, {}).get(name, "")
        tracking = record["tracking"].get(key, {})
        for name, numerator, denominator in (("gap_repair", "gap_covered", "gap_total"),
                                             ("terminal_coverage", "terminal_covered", "terminal_total"),
                                             ("new_break", "covered_lost", "covered_total")):
            row[f"K{key}_{name}"] = fraction(tracking[numerator], tracking[denominator]) if tracking else ""
    if record["changes"]:
        row["main_change_K1"] = max(record["changes"]["1"], key=record["changes"]["1"].get)
    else:
        row["main_change_K1"] = "not analyzed (Train score-only)"
    return row


def rankings(records):
    out = {}
    for domain in runner.DOMAINS:
        rows = sorted((csv_row(row) for row in records if row["split"] == "Val" and row["domain"] == domain),
                      key=lambda row: (row["delta_A_1"], row["filename"]))
        out[domain] = {"worst": rows[:10], "best": list(reversed(rows[-10:]))}
    return out


class MemoryPanel(io.BytesIO):
    """Let the existing Pillow renderer write into memory instead of a PNG file."""
    name = "panel.png"
    parent = SimpleNamespace(mkdir=lambda **kwargs: None)

    def exists(self):
        return False


def render_triptych(row, packed, paths, destination):
    image_path, mask_path = paths
    with Image.open(image_path) as handle:
        image = np.asarray(handle.convert("RGB"))
    with Image.open(mask_path) as handle:
        raw = np.asarray(handle)
    shape = raw.shape
    masks = {key: np.unpackbits(value, count=raw.size).reshape(shape).astype(bool) for key, value in packed.items()}
    draft, truth, valid = masks["y0_draft"], raw == 3, raw != 0
    buffer = MemoryPanel()
    # Reuse the old panel renderer for the base overlay and ignore-region paint.
    visual.draw_panel(image, raw, draft, np.zeros(shape, bool),
                      {"filename": row["filename"], "domain": row["domain"], "group": "diagnostic",
                       "A": row["y0_draft_A"], "gap_segments": 0}, buffer)
    buffer.seek(0)
    with Image.open(buffer) as panel:
        base = np.array(panel.crop((shape[1], 40, shape[1] * 2, shape[0] + 40)))
    # Unchanged FP has no requested diagnostic color: leave its satellite pixels visible.
    base[draft & ~truth & valid] = image[draft & ~truth & valid]
    canvas = Image.new("RGB", (3 * shape[1], shape[0] + 85), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    legend = "green=retained TP cyan=added TP red=added FP yellow=deleted FP magenta=deleted TP blue=miss gray=ignore; unchanged FP uncolored"
    draw.text((8, 8), f"{row['domain']} {row['filename']} delta A(K1)={row['delta_A_1']:.6f}", font=font, fill="white")
    draw.text((8, 30), legend, font=font, fill="white")
    for index, key in enumerate(("y0_draft", "1", "4")):
        predicted = masks[key]
        overlay = base.copy()
        colors = ((draft & predicted & truth, (0, 255, 0)),
                  (~draft & predicted & truth, (0, 255, 255)),
                  (~draft & predicted & ~truth, (255, 0, 0)),
                  (draft & ~predicted & ~truth, (255, 255, 0)),
                  (draft & ~predicted & truth, (255, 0, 255)),
                  (~draft & ~predicted & truth, (0, 0, 255)))
        for mask, color in colors:
            mask = mask & valid
            overlay[mask] = (.5 * image[mask] + .5 * np.asarray(color)).astype(np.uint8)
        canvas.paste(Image.fromarray(overlay), (index * shape[1], 85))
        draw.text((index * shape[1] + 8, 58), f"{key} A={row[f'{key}_A']:.6f}", font=font, fill="white")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def markdown(report):
    lines = ["# Refiner v1 diagnostics", "", "Observed results only; no training or model selection was performed.",
             "Train uses the first sorted filenames in each domain; Val uses the complete split unless limited.",
             "Train is not an independent test set. Differences alone do not establish overfitting or loss/metric mismatch.", ""]
    def table(title, headers, rows):
        lines.extend([f"## {title}", "", *y0.markdown_table(headers, rows), ""])
    rows = []
    for group in (*runner.DOMAINS, "Overall"):
        for key in KEYS:
            train = report["groups"]["Train"][group]["scores"][key]["0.5"]
            val = report["groups"]["Val"][group]["scores"][key]["0.5"]
            rows.append([group, key, *[train[name] for name in ("road_iou", "cldice", "boundary_iou", "quality")],
                         *[val[name] for name in ("road_iou", "cldice", "boundary_iou", "quality")]])
    table("Train subset vs Val, threshold 0.5", ("Group", "K/reference", "Train IoU", "Train clDice", "Train BIoU", "Train A", "Val IoU", "Val clDice", "Val BIoU", "Val A"), rows)
    rows, best_rows = [], []
    for group in (*runner.DOMAINS, "Overall"):
        for key in ("y0_draft", "1", "4"):
            data = report["groups"]["Val"][group]
            rows.extend([group, key, t, data["scores"][key][str(t)]["quality"]] for t in THRESHOLDS)
            best = data["best_thresholds"][key]
            best_rows.append([group, key, best["threshold"], best["A"]])
    table("Val threshold scan (selected on Val; optimistic)", ("Group", "K/reference", "Threshold", "A"), rows)
    table("Best thresholds (selected on Val; optimistic)", ("Group", "K/reference", "Threshold", "A"), best_rows)
    rows, track_rows, shape_rows = [], [], []
    for group in (*runner.DOMAINS, "Overall"):
        data = report["groups"]["Val"][group]
        for key in KEYS[1:]:
            change = data["changes"][key]
            rows.append([group, key, *[change[name] for name in CHANGE_KEYS],
                         *[change["fractions"][name] for name in CHANGE_KEYS], change["correct_fraction"],
                         change["added_correct_fraction"], change["deleted_correct_fraction"],
                         change["added_fraction"], change["deleted_fraction"]])
            track = data["tracking"][key]
            track_rows.append([group, key, track["gap_covered"], track["gap_total"], track["gap_repair_fraction"],
                               track["terminal_covered"], track["terminal_total"], track["terminal_coverage_fraction"],
                               track["covered_lost"], track["covered_total"], track["new_break_fraction"]])
        for key in ("1", "4"):
            shape = data["fragments"][key]
            shape_rows.append([group, key, shape["count"], *[shape["area"][p] for p in ("p25", "p50", "p75")],
                               *[shape["width"][p] for p in ("p25", "p50", "p75")],
                               *[shape["height"][p] for p in ("p25", "p50", "p75")],
                               shape["single_cell_fraction"], shape["aligned_edge_fraction"],
                               shape["any_aligned_edge_fraction"], shape["all_aligned_edges_fraction"]])
    table("Val flip decomposition", ("Group", "K", *CHANGE_KEYS, *[name + "/valid" for name in CHANGE_KEYS],
          "Correct/changed", "Correct/added", "Correct/deleted", "Added/valid", "Deleted/valid"), rows)
    table("Tracking fixed y0 skeleton pixels", ("Group", "K", "Gap covered", "Gap total", "Gap repair fraction", "Terminal covered", "Terminal total", "Terminal coverage fraction", "Covered lost", "Covered total", "New break fraction"), track_rows)
    table("New fragment shapes", ("Group", "K", "Count", "Area Q1", "Area median", "Area Q3", "Width Q1", "Width median", "Width Q3", "Height Q1", "Height median", "Height Q3", "Single cell fraction", "Aligned edges/all edges", "Any aligned edge/fragment", "All aligned edges/fragment"), shape_rows)
    for domain, ranks in report["rankings"].items():
        for category, rows in ranks.items():
            table(f"{domain}: {category} delta A(K1-y0)", ("Filename", "GT road", "y0 A", "K1 A", "Delta A", "Largest flip category"),
                  [[row["filename"], str(row["gt_has_road"]), row["y0_draft_A"], row["1_A"], row["delta_A_1"], row["main_change_K1"]] for row in rows])
    lines.extend(["Coverage is exact skeleton-pixel intersection, with no dilation or distance tolerance.",
                  "Skeletons and gap categories use the original shared Zhang/8-connectivity classifier; baseline sets stay fixed across K.",
                  "New fragments require zero exact overlap with y0 and GT (no two-pixel tolerance). Bounding maxima are inclusive.",
                  "A boundary coordinate is aligned when mod 16 is 0 or 15; both per-edge and per-fragment rates are reported.",
                  "Undefined descriptive fractions are null/NA. Empty-GT image scores follow official summarize without overrides.",
                  "Triptychs leave unchanged FP uncolored; each selected image is inferred only once, with packed masks held in bounded RAM.",
                  "Flip counts and official 0.5 Counts passed an independent comparison with train_refiner.score_image on every Val image.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=ROOT / "runs/refiner/v1/checkpoints/last.pt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/refiner/v1_diagnostics")
    parser.add_argument("--vis-dir", type=Path, default=ROOT / "runs/refiner/v1_diagnostics/vis")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit-per-domain", type=int)
    parser.add_argument("--loader-workers", type=int, default=1)
    parser.add_argument("--metric-workers", type=int, default=6)
    args = parser.parse_args()
    if args.metric_workers < 1 or args.loader_workers < 0 or (args.limit_per_domain is not None and args.limit_per_domain < 1):
        parser.error("Worker counts and limits are invalid")
    if int(os.environ.get("NSLOTS", "0")) and args.metric_workers + args.loader_workers + 1 > int(os.environ["NSLOTS"]):
        parser.error("Main + loader + metric workers exceed NSLOTS")
    if not args.out_dir.resolve().is_relative_to(ROOT) or not args.vis_dir.resolve().is_relative_to(ROOT / "runs"):
        parser.error("Reports must stay in the repo and visualizations under runs/")
    if any((args.out_dir / name).exists() for name in ("summary.md", "metrics.json", "per_image.csv")):
        parser.error("Refusing to overwrite existing reports")
    runner.probe.initialize_data_worker(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    saved = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    config = saved["config"]
    encoder, head, _ = y0.load_frozen_probe(ROOT / config["probe_ckpt"], device)
    encoder.requires_grad_(False).eval()
    head.requires_grad_(False).eval()
    model = runner.LoopRefiner(encoder.out_channels, config["dimension"], config["layers"], config["heads"],
                               config["mlp_ratio"], config["patch"], no_mask_feedback=config["no_mask_feedback"],
                               reset_state=config["reset_state"]).to(device=device, dtype=torch.float32)
    model.load_state_dict(saved["state_dict"])
    model.requires_grad_(False).eval()
    print(f"LOADED epoch={saved['epoch']} device={device}; no training", flush=True)
    metrics = runner.probe.official_metrics()
    records, candidates, paths, pending = [], {}, {}, deque()
    started = time.perf_counter()
    inference_seconds = 0.
    pool = mp.get_context("spawn").Pool(args.metric_workers, initializer=initialize_worker,
                                        initargs=(os.environ.get("EC523_STARTER_DIR"),))

    def consume():
        result = pending.popleft().get()
        packed = result.pop("packed", None)
        records.append(result)
        if packed is not None:
            identity = (result["domain"], result["filename"])
            candidates[identity] = packed
            selected = {(domain, row["filename"]) for domain, groups in rankings(records).items()
                        for rows in groups.values() for row in rows}
            for old in list(candidates):
                if old not in selected:
                    del candidates[old]
        print(f"ANALYZED {result['split']} {result['domain']} {result['filename']} worker_seconds={result['worker_seconds']:.3f}", flush=True)

    try:
        for split in ("Train", "Val"):
            for index, domain in enumerate(runner.DOMAINS):
                # Train: deterministic first sorted 100. Limited Val: match the existing eval-only sampler.
                dataset = runner.probe.RoadDataset(split, domain,
                    limit=args.limit_per_domain if split == "Val" else None, seed=10 + index)
                if split == "Train":
                    dataset.pairs = dataset.pairs[:min(100, args.limit_per_domain or 100)]
                paths.update({(domain, a.name): (a, b) for a, b in dataset.pairs} if split == "Val" else {})
                loader = runner.make_loader(dataset, 1, args.loader_workers)
                for images, _, _, image_ids, raw in loader:
                    images = runner.probe.as_device_tensor(images, device)
                    phase = runner.stamp(device)
                    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                        features, draft = runner.frozen_draft(encoder, head, images)
                        outputs = runner.refine_steps(model, features.float(), draft.float(), 8)
                    duration = runner.stamp(device) - phase
                    inference_seconds += duration
                    print(f"INFERRED {split} {domain} {image_ids[0]} seconds={duration:.3f}", flush=True)
                    # Float32 avoids threshold flips caused by float16 probability quantization.
                    probabilities = {"y0_draft": draft.sigmoid().squeeze().cpu().numpy(),
                                     "1": outputs[0].sigmoid().squeeze().cpu().numpy(),
                                     "4": outputs[3].sigmoid().squeeze().cpu().numpy()}
                    binary = {str(k): (outputs[k - 1].sigmoid().squeeze() >= .5).cpu().numpy() for k in (2, 8)}
                    labels = np.asarray(raw).squeeze().astype(np.uint8, copy=False)
                    pending.append(pool.apply_async(process_image, (split, domain, image_ids[0] + ".png", probabilities, binary, labels)))
                    del features, draft, outputs
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
    groups = {}
    for split in ("Train", "Val"):
        groups[split] = {domain: aggregate([row for row in records if row["split"] == split and (domain == "Overall" or row["domain"] == domain)], metrics)
                         for domain in (*runner.DOMAINS, "Overall")}
    ranks = rankings(records)
    for domain, categories in ranks.items():
        for category, rows in categories.items():
            for rank, row in enumerate(rows, 1):
                render_triptych(row, candidates[(domain, row["filename"])], paths[(domain, row["filename"])],
                                args.vis_dir / f"{domain}_{category}_{rank:02d}_{row['filename']}")
    report = {"checkpoint": str(args.ckpt.resolve().relative_to(ROOT)), "epoch": saved["epoch"],
              "device": str(device), "limit_per_domain": args.limit_per_domain, "thresholds": THRESHOLDS,
              "groups": groups, "rankings": ranks, "inference_seconds": inference_seconds,
              "worker_seconds": sum(row["worker_seconds"] for row in records),
              "wall_seconds": time.perf_counter() - started,
              "self_checks": "All Val flips and 0.5 Counts equal train_refiner.score_image; baseline geometry totals match shared classifier"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(markdown(report), encoding="utf-8")
    rows = [csv_row(record) for record in records]
    with (args.out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"DONE images={len(records)} wall_seconds={report['wall_seconds']:.3f} inference_seconds={inference_seconds:.3f}", flush=True)


if __name__ == "__main__":
    main()
