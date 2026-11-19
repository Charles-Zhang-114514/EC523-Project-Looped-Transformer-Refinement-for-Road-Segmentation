#!/usr/bin/env python3
"""Evaluate frozen draft heads with D4 TTA and the e6 seed-0 refiner."""
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from analysis import draft_overfit_check as previous
from scripts import train_refiner as runner


DOMAINS = ("Overall", *runner.DOMAINS)
KS = (1, 2, 4, 8)
SCORE_KEYS = ("road_iou", "cldice", "boundary_iou", "quality")
BASELINE = ("A1-i", 0)
ARMS = ("A1-i", "v3a", "v3b")
MATRIX = [(arm, seed) for arm in ARMS for seed in (0, 1)]
SPLIT = ROOT / "configs/val_split_seed0.json"
E6_REPORT = ROOT / "results/exp/e6_convdraft_s0/metrics.json"


def d4(array, code, inverse=False):
    """Match training D4: rotate first, then optionally flip the width axis."""
    if code not in range(8):
        raise ValueError("D4 code must be in 0..7")
    if inverse:
        if code >= 4:
            array = array.flip((-1,))
        return array.rot90(-(code % 4), (-2, -1))
    rotated = array.rot90(code % 4, (-2, -1))
    return rotated.flip((-1,)) if code >= 4 else rotated


@torch.no_grad()
def average_d4_logits(predict, image, identity):
    """Reuse the original-view logits and accumulate aligned float32 logits."""
    total = identity.float().clone()
    for code in range(1, 8):
        logits = predict(d4(image, code).contiguous()).float()
        total.add_(d4(logits, code, inverse=True))
    return total.div_(8)


def binary_mask(logits):
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Nonfinite inference logits")
    return (logits.sigmoid()[0, 0] >= .5).cpu().numpy().astype(np.uint8)


def evaluate(encoder, head, model, datasets, ks, device, loader_workers, metric_workers, tta):
    """Stream binary masks to official metrics; keep feature maps batch-local."""
    metrics = runner.probe.official_metrics()
    modes = ("none", "d4") if tta == "d4" else ("none",)
    keys = ("y0_draft", *map(str, ks))
    counts = {mode: {domain: {key: metrics.Counts() for key in keys}
                     for domain in runner.DOMAINS} for mode in modes}
    pending, images = deque(), []

    def consume():
        domain, values = pending.popleft().get()
        for mode, by_step in values.items():
            for key, value in by_step.items():
                counts[mode][domain][key] += metrics.Counts(**value)

    with mp.get_context("spawn").Pool(metric_workers, initializer=runner.initialize_eval_worker,
                                     initargs=(os.environ["EC523_STARTER_DIR"],)) as pool:
        for domain in runner.DOMAINS:
            loader = runner.make_loader(datasets[domain], 1, loader_workers)
            for images_np, _, _, ids, labels in loader:
                image = runner.probe.as_device_tensor(images_np, device)
                started = runner.stamp(device)
                features, draft = runner.frozen_draft(encoder, head, image)
                single_seconds = runner.stamp(device) - started
                logits = {"none": draft}
                times = {"none": single_seconds}
                if tta == "d4":
                    started = runner.stamp(device)
                    logits["d4"] = average_d4_logits(lambda view: head(encoder(view)), image, draft)
                    times["d4"] = single_seconds + runner.stamp(device) - started

                predictions = {}
                refiner_times = {}
                for mode, answer in logits.items():
                    predictions[mode] = {"y0_draft": binary_mask(answer)}
                    if ks:
                        refiner_times[mode] = {}
                        for key, refined, seconds, _ in runner.inference_steps(model, features, answer, ks, device):
                            predictions[mode][key] = binary_mask(refined)
                            refiner_times[mode][key] = seconds

                images.append(dict(domain=domain, image_id=ids[0], draft_seconds=times,
                                   refiner_seconds=refiner_times))
                raw = np.asarray(labels).squeeze(0).astype(np.uint8, copy=False)
                pending.append(pool.apply_async(previous.score_predictions, (domain, predictions, raw)))
                if len(pending) >= 2 * metric_workers:
                    consume()
            del loader
        while pending:
            consume()

    for by_domain in counts.values():
        by_domain["Overall"] = {key: by_domain["Urban"][key] + by_domain["Rural"][key] for key in keys}
    return {
        "images": images,
        "groups": {mode: {domain: {key: {"counts": asdict(value), "scores": metrics.summarize(value)}
                                   for key, value in by_key.items()}
                          for domain, by_key in by_domain.items()}
                   for mode, by_domain in counts.items()},
    }


def paths(arm, seed):
    if (arm, seed) == BASELINE:
        return ROOT / "results/draft_head/A1-i", ROOT / previous.PROBES["A1-i"]
    run = f"{arm}_s{seed}"
    return ROOT / "runs/draft_head_v3" / run / "report", ROOT / "runs/draft_head_v3" / run / "checkpoints/best.pt"


def selected_names(dataset):
    return {domain: [image.name for image, _ in dataset[domain].pairs] for domain in runner.DOMAINS}


def load_training(datasets, split_hash):
    reports = {}
    selection_files = selected_names(datasets["Val-sel"])
    report_files = selected_names(datasets["Val-rep"])
    for arm, seed in MATRIX:
        report_dir, checkpoint = paths(arm, seed)
        required = [report_dir / "summary.md", checkpoint]
        if (arm, seed) != BASELINE:
            required.append(checkpoint.with_name("last.pt"))
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed training artifact: {path.relative_to(ROOT)}")
        raw = json.loads((report_dir / "metrics.json").read_text())
        config = raw["config"]
        expected = dict(draft_layers="last", head="conv_up", seed=seed, epochs=10, batch_size=8,
                        lr=1e-3, pos_weight=4., weight_decay=0., select_per_domain=100,
                        limit_per_domain=None, threshold=.5, val_split_sha256=split_hash)
        if (any(config.get(k) != v for k, v in expected.items())
                or config.get("lr_schedule", "constant") != ("constant" if arm == "A1-i" else "cosine")
                or config.get("d4_aug", False) != (arm == "v3b")):
            raise ValueError(f"Unexpected training configuration for {arm} seed {seed}")
        epochs = raw["epochs"]
        if ([entry["epoch"] for entry in epochs] != list(range(1, 11))
                or any(entry["selection"]["files"] != selection_files for entry in epochs)
                or raw["evaluation"]["files"] != report_files):
            raise ValueError(f"Unexpected validation protocol for {arm} seed {seed}")
        best = max(epochs, key=lambda entry: entry["selection_A"])
        if (raw["best_epoch"], raw["best_selection_A"]) != (best["epoch"], best["selection_A"]):
            raise ValueError(f"Unexpected checkpoint selection for {arm} seed {seed}")
        reports[arm, seed] = dict(arm=arm, seed=seed, checkpoint=checkpoint.relative_to(ROOT).as_posix(),
                                  best_epoch=raw["best_epoch"], best_selection_A=raw["best_selection_A"],
                                  val_sel_curve=[entry["selection_A"] for entry in epochs],
                                  val_rep=raw["evaluation"]["groups"])
    return reports


def write_report(out, report, markdown):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (out / "summary.md").open("x") as stream:
        stream.write(markdown)


def compare_refiner(datasets, device, args):
    """Compare the non-TTA evaluator to the unchanged evaluation entry point."""
    _, checkpoint = paths(*BASELINE)
    encoder, head, _ = runner.y0.load_frozen_probe(checkpoint, device)
    refiner, _ = previous.shared.load_model(ROOT / "runs/exp/e6_convdraft_s0/best.pt", device,
                                            hashlib.sha256(SPLIT.read_bytes()).hexdigest())
    ks = (1, 2, 4)
    current = evaluate(encoder, head, refiner, datasets, ks, device, args.loader_workers,
                       args.metric_workers, args.tta)
    legacy = runner.evaluate(encoder, head, refiner, datasets, ks, device, args.loader_workers,
                             args.metric_workers, geometry=False)
    if [{"domain": row["domain"], "image_id": row["image_id"]}
            for row in current["images"]] != legacy["images"]:
        raise ValueError("Refiner comparison used different images")
    rows = []
    for domain in DOMAINS:
        for key in ("y0_draft", *map(str, ks)):
            left = current["groups"]["none"][domain][key]
            right = legacy["groups"][domain][key]
            differences = {name: left["scores"][name] - right["scores"][name] for name in SCORE_KEYS}
            rows.append(dict(domain=domain, step=key, differences=differences,
                             counts_equal=left["counts"] == right["scores"]["counts"]))
    maximum = max(abs(value) for row in rows for value in row["differences"].values())
    report = dict(smoke=True, maximum_absolute_difference=maximum, comparisons=rows,
                  images=current["images"], current=current["groups"], legacy=legacy["groups"])
    write_report(args.out_dir, report, "# Observed CPU refiner regression\n\n"
                 f"Maximum absolute score difference across {len(rows)} rows: {maximum:.12g}.\n")
    if maximum > 1e-5:
        raise ValueError(f"Refiner regression difference {maximum:.12g} exceeds 1e-5")
    print(f"PASS refiner regression: {len(rows)} rows, max absolute score difference {maximum:.12g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tta", choices=("none", "d4"), default="none")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--loader-workers", type=int, default=1)
    parser.add_argument("--metric-workers", type=int, default=6)
    parser.add_argument("--refiner-regression", type=int, metavar="IMAGES_PER_DOMAIN")
    args = parser.parse_args()
    slots = int(os.environ.get("NSLOTS", "0"))
    if args.loader_workers < 0 or args.metric_workers < 1 or (slots and 1 + args.loader_workers + args.metric_workers > slots):
        parser.error("Invalid loader/metric worker budget")
    if not args.out_dir.resolve().is_relative_to(ROOT) or (args.out_dir.exists() and any(args.out_dir.iterdir())):
        parser.error("Choose a fresh output directory inside the repository")
    if args.refiner_regression is not None and (
        args.refiner_regression < 1 or args.device != "cpu" or not args.out_dir.resolve().is_relative_to(ROOT / "runs")
    ):
        parser.error("The regression requires a positive image limit, CPU and output under runs")
    if args.refiner_regression is None and args.tta != "d4":
        parser.error("The full comparison requires --tta d4")

    split_bytes = SPLIT.read_bytes()
    split, split_hash = json.loads(split_bytes), hashlib.sha256(split_bytes).hexdigest()
    runner.probe.initialize_data_worker(0)
    device = torch.device(args.device)
    if args.refiner_regression is not None:
        return compare_refiner(runner.protocol_datasets(split, "Val-rep", args.refiner_regression), device, args)
    from analysis import summarize_draft_head_v3 as summary
    datasets = {part: runner.protocol_datasets(split, part, limit)
                for part, limit in (("Val-sel", 100), ("Val-rep", None))}
    training = load_training(datasets, split_hash)
    reference = json.loads(E6_REPORT.read_text())
    if (reference["config"]["val_split_sha256"] != split_hash or reference["config"]["seed"] != 0
            or reference["config"]["probe_ckpt"] != previous.PROBES["A1-i"]):
        raise ValueError("Archived e6 report does not match the expected seed, draft or split")
    model, epoch = previous.shared.load_model(ROOT / "runs/exp/e6_convdraft_s0/best.pt", device, split_hash)
    if epoch != reference["best_epoch"]:
        raise ValueError("e6 checkpoint epoch differs from the archived report")

    # All six heads use the same frozen encoder. Each measurement runs a
    # standalone draft pipeline to retain real per-image timings.
    encoder = None
    for arm, seed in MATRIX:
        _, checkpoint = paths(arm, seed)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        row = training[arm, seed]
        if (saved["encoder"] != runner.y0.ENCODER or saved["head"] != "conv_up"
                or saved["draft_layers"] != "last" or saved["config"]["seed"] != seed
                or saved["epoch"] != row["best_epoch"]):
            raise ValueError(f"Unexpected best checkpoint for {arm} seed {seed}")
        if encoder is None:
            encoder, head, _ = runner.y0.load_frozen_probe(checkpoint, device)
        else:
            head.load_state_dict(saved["state_dict"], strict=True)
        row["tta"] = {}
        for part, dataset in datasets.items():
            use_refiner = (arm, seed) == BASELINE and part == "Val-rep"
            row["tta"][part] = evaluate(encoder, head, model if use_refiner else None, dataset,
                                         KS if use_refiner else (), device, args.loader_workers,
                                         args.metric_workers, args.tta)
            print(f"OBSERVED {arm} seed {seed} {part}: {len(row['tta'][part]['images'])} images", flush=True)
    report = summary.build_report(training, reference["evaluation"]["groups"], split_hash)
    write_report(args.out_dir, report, summary.markdown(report))


if __name__ == "__main__":
    main()
