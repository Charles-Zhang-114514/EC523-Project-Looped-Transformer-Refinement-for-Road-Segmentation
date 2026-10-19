#!/usr/bin/env python3
"""Select y0 audit tiles and render selected frozen-probe predictions only."""
import argparse
import csv
import io
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULT_DIR = ROOT / "results/y0_analysis/dinov3_vitl16_web_bn_posw4"
SELECTION_FIELDS = (
    "filename", "domain", "group", "rank", "road_iou", "cldice", "boundary_iou", "A",
    "truth_road_fraction", "predicted_road_fraction", "fp_pixels", "fn_pixels",
    "gap_segments", "gap_skeleton_pixels", "also_in_group")
AUDIT_FIELDS = ("filename", "domain", "fp_pixels", "judgement", "note")


def read_records(path):
    """Selection reads stored statistics; only missing GT counts need mask reads."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"filename", "domain", "road_iou", "cldice", "boundary_iou",
                "predicted_road_pixels", "fp_pixels", "fn_pixels", "gap_segments", "gap_skeleton_pixels"}
    if not rows or not required.issubset(rows[0]) or not ({"A", "quality"} & rows[0].keys()):
        raise ValueError("per_image.csv lacks required prediction/metric fields; these cannot be inferred from labels")
    seen = set()
    for row in rows:
        key = row["domain"], row["filename"]
        if key in seen or row["domain"] not in ("Urban", "Rural") or Path(row["filename"]).name != row["filename"]:
            raise ValueError(f"Invalid or duplicate image identifier: {key}")
        seen.add(key)
        if not row.get("valid_pixels") or not row.get("truth_road_pixels"):
            data_dir = os.environ.get("EC523_DATA_DIR")
            if not data_dir:
                raise EnvironmentError("Set EC523_DATA_DIR to read missing GT statistics")
            mask_path = Path(data_dir) / "Val" / row["domain"] / "masks_png" / row["filename"]
            with Image.open(mask_path) as mask_file:
                raw = np.asarray(mask_file)
            row["valid_pixels"] = str(int((raw != 0).sum()))
            row["truth_road_pixels"] = str(int((raw == 3).sum()))
        if int(row["valid_pixels"]) <= 0:
            raise ValueError(f"No valid pixels: {key}")
        row["A"] = row.get("A") or row["quality"]
    return sorted(rows, key=lambda row: (row["domain"], row["filename"]))


def select_records(records, seed=0):
    """Use stable filename tie breaks and a seed-0 generator over sorted C pools."""
    rng = np.random.default_rng(seed)
    selected, overview = [], {}
    for domain in ("Urban", "Rural"):
        pool = [row for row in records if row["domain"] == domain]
        eligible_a = [row for row in pool
                      if 1000 * int(row["truth_road_pixels"]) >= 5 * int(row["valid_pixels"])]
        eligible_c = [row for row in pool if int(row["truth_road_pixels"]) == 0
                      and int(row["predicted_road_pixels"]) > 0]
        if len(eligible_a) < 10 or len(pool) < 10 or len(eligible_c) < 15:
            raise ValueError(f"Insufficient {domain} candidates: A={len(eligible_a)}, B={len(pool)}, C={len(eligible_c)}")
        groups = {
            "A": sorted(eligible_a, key=lambda row: (float(row["A"]), row["filename"]))[:10],
            "B": sorted(pool, key=lambda row: (-int(row["gap_skeleton_pixels"]), row["filename"]))[:10],
            "C": [eligible_c[int(index)] for index in rng.choice(len(eligible_c), size=15, replace=False)],
        }
        memberships = {}
        for group, chosen in groups.items():
            for row in chosen:
                memberships.setdefault(row["filename"], []).append(group)
        for group, chosen in groups.items():
            for rank, source in enumerate(chosen, start=1):
                row = {key: source[key] for key in
                       ("filename", "domain", "road_iou", "cldice", "boundary_iou", "A",
                        "fp_pixels", "fn_pixels", "gap_segments", "gap_skeleton_pixels")}
                valid = int(source["valid_pixels"])
                row.update(group=group, rank=rank,
                           truth_road_fraction=int(source["truth_road_pixels"]) / valid,
                           predicted_road_fraction=int(source["predicted_road_pixels"]) / valid,
                           also_in_group=";".join(value for value in memberships[source["filename"]] if value != group))
                selected.append(row)
        overview[domain] = {"A_candidates": len(eligible_a), "B_candidates": len(pool),
                            "C_candidates": len(eligible_c),
                            "A_B_overlap": len({row["filename"] for row in groups["A"]} &
                                               {row["filename"] for row in groups["B"]})}
    return selected, overview


def csv_text(fields, rows):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def save_selection(selected, output_dir):
    """Create the two CSVs once; subsequent jobs preserve existing annotations."""
    selection_path = output_dir / "visual_audit_selection.csv"
    audit_path = output_dir / "no_gt_audit.csv"
    selection = csv_text(SELECTION_FIELDS, selected)
    audit_rows = [{key: row[key] for key in ("filename", "domain", "fp_pixels")} |
                  {"judgement": "", "note": ""} for row in selected if row["group"] == "C"]
    if selection_path.exists() and selection_path.read_bytes() != selection.encode("utf-8"):
        raise ValueError("Existing selection CSV differs; refusing to modify an existing result")
    if audit_path.exists():
        with audit_path.open(newline="", encoding="utf-8-sig") as handle:
            existing = list(csv.DictReader(handle))
        identity = lambda rows: [(row["filename"], row["domain"], str(row["fp_pixels"])) for row in rows]
        if identity(existing) != identity(audit_rows):
            raise ValueError("Existing manual-audit CSV contains a different selection; refusing to overwrite it")
    else:
        with audit_path.open("x", newline="", encoding="utf-8") as handle:
            handle.write(csv_text(AUDIT_FIELDS, audit_rows))
    if not selection_path.exists():
        with selection_path.open("x", newline="", encoding="utf-8") as handle:
            handle.write(selection)


def check_consistency(result, source):
    """Describe all count deltas without rejecting cross-device predictions."""
    actual = {"fp_pixels": result["fp_pixels"], "fn_pixels": result["fn_pixels"],
              "gap_segments": sum(item["segments"] for item in result["segments"]["gap"].values()),
              "gap_skeleton_pixels": sum(item["skeleton_pixels"] for item in result["segments"]["gap"].values())}
    comparisons = {key: {"pr6": int(source[key]), "current": value, "delta": value - int(source[key])}
                   for key, value in actual.items()}
    return actual, comparisons


def relative_count_difference(comparison):
    """Absolute delta / PR #6 count; zero/zero is 0, nonzero/zero is infinity."""
    reference, delta = comparison["pr6"], abs(comparison["delta"])
    return delta / reference if reference else (float("inf") if delta else 0.)


def draw_panel(image, raw, predicted, gaps, row, output_path):
    """Yellow shared-classifier gap pixels are painted last, above the overlay."""
    from analysis import y0_error_analysis as y0
    valid, truth = raw != 0, raw == 3
    predicted = predicted.astype(bool) & valid
    overlay = image.copy()
    for mask, color in ((truth & predicted, (0, 255, 0)), (predicted & ~truth, (255, 0, 0)),
                        (truth & ~predicted, (0, 0, 255))):
        overlay[mask] = (.5 * image[mask] + .5 * np.asarray(color)).astype(np.uint8)
    overlay[~valid] = (128, 128, 128)
    if row["group"] in ("A", "B"):
        yellow = y0.ndi.binary_dilation(gaps, structure=y0.disk(2)) & valid
        overlay[yellow] = (255, 255, 0)
    header_height = 40
    canvas = Image.new("RGB", (image.shape[1] * 2, image.shape[0] + header_height), (20, 20, 20))
    canvas.paste(Image.fromarray(image), (0, header_height))
    canvas.paste(Image.fromarray(overlay), (image.shape[1], header_height))
    text = (f"{row['filename']} | {row['domain']} | Group {row['group']} | A={float(row['A']):.6f} | "
            f"gaps={row['gap_segments']} | TP=green FP=red FN=blue ignore=gray gap=yellow")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    draw.text((10, 9), text, font=font, fill="white")
    if draw.textbbox((10, 9), text, font=font)[2] > canvas.width - 10:
        raise ValueError("Visualization header does not fit on a single line")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing visualization: {output_path}")
    canvas.save(output_path)


def render_selected(selected, records, args):
    import torch
    from analysis import y0_error_analysis as y0
    y0.probe.initialize_data_worker(0)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    started = time.perf_counter()
    encoder, head, _ = y0.load_frozen_probe(args.ckpt, device)
    sources = {(row["domain"], row["filename"]): row for row in records}
    tasks = {}
    for row in selected:
        if (row["domain"] in args.domains and row["group"] in args.groups
                and (args.limit_per_group is None or row["rank"] <= args.limit_per_group)):
            tasks.setdefault((row["domain"], row["filename"]), []).append(row)
    datasets = {domain: y0.probe.RoadDataset("Val", domain) for domain in args.domains}
    indices = {domain: {image.name: index for index, (image, _) in enumerate(dataset.pairs)}
               for domain, dataset in datasets.items()}
    forward_seconds, analysis_seconds, drawing_seconds, panels = 0., 0., 0., 0
    matched, mismatched = 0, 0
    max_fp_relative_difference, max_fn_relative_difference = 0., 0.
    for (domain, filename), rows in tasks.items():
        dataset = datasets[domain]
        index = indices[domain][filename]
        image, _, _, _, raw = dataset[index]
        phase = time.perf_counter()
        with torch.no_grad():
            probability = torch.sigmoid(head(encoder(y0.probe.as_device_tensor(image[None], device))))
            predicted = (probability.squeeze() >= .5).cpu().numpy().astype(np.uint8)
        del probability
        forward_seconds += time.perf_counter() - phase
        phase = time.perf_counter()
        result = y0.analyze_masks(predicted, raw, return_gap_mask=True)
        actual, comparisons = check_consistency(result, sources[(domain, filename)])
        differs = any(item["delta"] != 0 for item in comparisons.values())
        if differs:
            mismatched += 1
            max_fp_relative_difference = max(max_fp_relative_difference,
                                             relative_count_difference(comparisons["fp_pixels"]))
            max_fn_relative_difference = max(max_fn_relative_difference,
                                             relative_count_difference(comparisons["fn_pixels"]))
            values = " ".join(f"{key}(pr6={item['pr6']},current={item['current']},delta={item['delta']:+d})"
                              for key, item in comparisons.items())
            print(f"DIFF filename={filename} domain={domain} groups={','.join(row['group'] for row in rows)} {values}", flush=True)
        else:
            matched += 1
        analysis_seconds += time.perf_counter() - phase
        phase = time.perf_counter()
        with Image.open(dataset.pairs[index][0]) as image_file:
            original = np.array(image_file.convert("RGB"))
        for row in rows:
            category = "vis_no_gt" if row["group"] == "C" else "vis_gt_road"
            path = args.vis_root / category / f"{domain}_{row['group']}_{row['rank']:02d}_{filename}"
            panel_row = dict(row, gap_segments=actual["gap_segments"])
            draw_panel(original, raw, predicted, result["gap_mask"], panel_row, path)
            panels += 1
        drawing_seconds += time.perf_counter() - phase
        status = "DIFF_RENDERED" if differs else "PASS"
        print(f"{status} {domain}/{filename}: {actual}; panels={len(rows)}", flush=True)
    count = len(tasks)
    print(f"CONSISTENCY_SUMMARY matched={matched} mismatched={mismatched} "
          f"max_fp_relative_difference={max_fp_relative_difference:.9f} "
          f"max_fn_relative_difference={max_fn_relative_difference:.9f} "
          "(absolute_delta/PR6; zero_reference: 0_if_unchanged_else_inf; unique_images)", flush=True)
    print(f"Completed {count} unique images, {panels} panels; "
          f"forward={forward_seconds:.3f}s analysis={analysis_seconds:.3f}s "
          f"drawing={drawing_seconds:.3f}s wall={time.perf_counter() - started:.3f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-image-csv", type=Path, default=RESULT_DIR / "per_image.csv")
    parser.add_argument("--ckpt", type=Path, default=ROOT / "runs/linear_probe/checkpoints/dinov3_vitl16_web_bn_posw4.pt")
    parser.add_argument("--select-only", action="store_true", help="Write selection/audit CSVs without inference")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--groups", default="A,B,C", help="Comma-separated groups to visualize")
    parser.add_argument("--domains", nargs="+", choices=("Urban", "Rural"), default=("Urban", "Rural"),
                        help="Domains to visualize; selection CSVs always retain both domains")
    parser.add_argument("--limit-per-group", type=int, help="Limit panels per domain and group, not selection CSV rows")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--vis-root", type=Path, default=ROOT / "runs/y0_analysis")
    args = parser.parse_args()
    args.groups = tuple(args.groups.split(","))
    if not args.groups or any(group not in ("A", "B", "C") for group in args.groups):
        parser.error("--groups must contain A, B and/or C")
    if args.limit_per_group is not None and args.limit_per_group < 1:
        parser.error("--limit-per-group must be positive")
    if not args.vis_root.resolve().is_relative_to(ROOT / "runs"):
        parser.error("Visualizations must remain under ignored runs/")
    if not args.per_image_csv.resolve().parent.is_relative_to(ROOT):
        parser.error("Result CSVs must remain inside the repository")
    records = read_records(args.per_image_csv)
    selected, overview = select_records(records, args.seed)
    save_selection(selected, args.per_image_csv.parent)
    for domain, stats in overview.items():
        print(f"{domain}: selected A=10 B=10 C=15; {stats}", flush=True)
    print(f"Selection rows={len(selected)}; unique images="
          f"{len({(row['domain'], row['filename']) for row in selected})}", flush=True)
    if not args.select_only:
        render_selected(selected, records, args)


if __name__ == "__main__":
    main()
