#!/usr/bin/env python3
"""Train and evaluate a frozen-encoder RoadSeg linear probe."""
import argparse
import importlib.util
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
import torch
from torch.utils.data import DataLoader, ConcatDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from looproad.data.road_dataset import RoadDataset
from looproad.encoders.resnet import build_encoder
from looproad.models.linear_probe import LinearProbe, masked_bce_with_logits

_WORKER_METRICS = None
_WORKER_THRESHOLDS = None


class NumpyBatch:
    """Batch container that avoids Torch shared-memory IPC from loader workers."""
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def pin_memory(self):
        values = list(self.values)
        for index in (0, 1, 2, 4):
            values[index] = torch.from_numpy(np.ascontiguousarray(values[index])).pin_memory()
        return NumpyBatch(tuple(values))


def collate_numpy_batch(rows):
    columns = list(zip(*rows))
    values = [np.stack([np.asarray(value) for value in columns[index]])
              for index in (0, 1, 2, 4)]
    values.insert(3, list(columns[3]))
    return NumpyBatch(tuple(values))


def as_device_tensor(value, device):
    if not torch.is_tensor(value):
        value = torch.from_numpy(np.ascontiguousarray(value))
    return value.to(device, non_blocking=device.type == "cuda")


def official_metrics(starter=None):
    starter = starter or os.environ.get("EC523_STARTER_DIR")
    if not starter:
        raise EnvironmentError("Set EC523_STARTER_DIR to the official starter package")
    sys.path.insert(0, starter)
    path = Path(starter) / "metrics.py"
    spec = importlib.util.spec_from_file_location("ec523_official_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_group(metrics, records, threshold):
    counts = metrics.Counts()
    for probability, labels in records:
        hard = (probability >= threshold).astype(np.float32)
        counts = counts + metrics.image_counts(hard, labels)
    return metrics.summarize(counts)


def initialize_metric_worker(starter_dir, thresholds):
    """Initialize an isolated, single-threaded process for official metric work."""
    global _WORKER_METRICS, _WORKER_THRESHOLDS
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    _WORKER_METRICS = official_metrics(starter_dir)
    _WORKER_THRESHOLDS = thresholds


def metric_counts_for_image(domain, probability, labels):
    """Return official Counts for one image at each requested threshold."""
    started = time.perf_counter()
    probability = np.asarray(probability, dtype=np.float32)
    result = {}
    for threshold in _WORKER_THRESHOLDS:
        hard = (probability >= threshold).astype(np.float32)
        counts = _WORKER_METRICS.image_counts(hard, labels)
        result[str(threshold)] = asdict(counts)
    finished = time.perf_counter()
    return domain, result, finished - started, started, finished


def compact_probability_for_thresholds(probability):
    """Compress to float16 while preserving decisions at all requested thresholds."""
    probability = np.asarray(probability, dtype=np.float32)
    compact = probability.astype(np.float16)
    exact_bucket = sum((probability >= threshold).astype(np.uint8)
                       for threshold in _PARENT_THRESHOLDS)
    compact_bucket = sum((compact >= threshold).astype(np.uint8)
                         for threshold in _PARENT_THRESHOLDS)
    changed = exact_bucket != compact_bucket
    bounds = (0.0,) + tuple(_PARENT_THRESHOLDS) + (1.0,)
    representatives = np.asarray([(bounds[i] + bounds[i + 1]) / 2
                                   for i in range(len(bounds) - 1)], dtype=np.float16)
    compact[changed] = representatives[exact_bucket[changed]]
    return compact


_PARENT_THRESHOLDS = (0.3, 0.5, 0.7)


def initialize_data_worker(_worker_id):
    """Keep each image-loader subprocess single-threaded."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)


def report_score(metrics, counts):
    return metrics.summarize(counts)["quality"]


def parse_thresholds(value):
    try:
        thresholds = tuple(round(float(part.strip()), 2) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from error
    if not thresholds or any(not 0 < value < 1 for value in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be non-empty values between 0 and 1")
    if len(set(thresholds)) != len(thresholds):
        raise argparse.ArgumentTypeError("thresholds must be unique after rounding to two decimals")
    if 0.5 not in thresholds:
        raise argparse.ArgumentTypeError("thresholds must include 0.5 for the primary report")
    return tuple(sorted(thresholds))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="resnet50_layer3")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-per-domain", type=int)
    eval_loader_default = (1 if int(os.environ.get("NSLOTS", "0")) >= 3 else 0)
    metric_worker_default = max(1, int(os.environ["NSLOTS"]) - eval_loader_default - 1) if "NSLOTS" in os.environ else 1
    parser.add_argument("--metric-workers", type=int, default=metric_worker_default)
    parser.add_argument("--loader-workers", type=int,
                        default=max(0, int(os.environ["NSLOTS"]) - 1) if "NSLOTS" in os.environ else 0)
    parser.add_argument("--eval-loader-workers", type=int,
                        default=eval_loader_default)
    parser.add_argument("--thresholds", type=parse_thresholds,
                        default=parse_thresholds(",".join(str(round(i * 0.05, 2)) for i in range(1, 20))))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt")
    return parser


def main():
    args = build_parser().parse_args()
    if args.eval_only and not args.ckpt:
        raise SystemExit("--eval-only requires --ckpt")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    slots_value = os.environ.get("NSLOTS")
    if args.metric_workers < 1:
        raise ValueError("--metric-workers must be at least 1")
    if slots_value and args.metric_workers > int(slots_value):
        raise ValueError(f"--metric-workers ({args.metric_workers}) exceeds NSLOTS ({slots_value})")
    data_loader_workers = args.loader_workers
    if data_loader_workers < 0 or args.eval_loader_workers < 0:
        raise ValueError("DataLoader worker counts cannot be negative")
    if slots_value:
        slots = int(slots_value)
        if data_loader_workers + 1 > slots:
            raise ValueError("Training DataLoader workers plus main process exceed NSLOTS")
        if args.eval_loader_workers + args.metric_workers + 1 > slots:
            raise ValueError("Evaluation DataLoader, metric workers, and main process exceed NSLOTS")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    global _PARENT_THRESHOLDS
    _PARENT_THRESHOLDS = args.thresholds
    metrics = official_metrics()
    limit = args.limit_per_domain
    train_sets = ([] if args.eval_only else
                  [RoadDataset("Train", domain, limit=limit, seed=args.seed + i)
                   for i, domain in enumerate(("Urban", "Rural"))])
    val_sets = [RoadDataset("Val", domain, limit=limit, seed=args.seed + i + 10)
                for i, domain in enumerate(("Urban", "Rural"))]
    train_data = ConcatDataset(train_sets) if train_sets else None
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=data_loader_workers,
                              generator=torch.Generator().manual_seed(args.seed),
                              pin_memory=True, persistent_workers=data_loader_workers > 0,
                              worker_init_fn=initialize_data_worker if data_loader_workers > 0 else None,
                              collate_fn=collate_numpy_batch) if train_data else None
    encoder = build_encoder(args.encoder).to(device).eval()
    head = LinearProbe(encoder.out_channels).to(device)
    encoder_params = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    if trainable != 1025 or any(p.requires_grad for p in encoder.parameters()):
        raise RuntimeError("Encoder freeze or linear-head parameter invariant failed")

    background_pixels = road_pixels = 0
    if not args.eval_only:
        for dataset in train_sets:
            for i in range(len(dataset)):
                _, target, valid, _, _ = dataset[i]
                valid = np.asarray(valid, dtype=bool)
                target = np.asarray(target)
                road_pixels += int((valid & (target == 1)).sum())
                background_pixels += int((valid & (target == 0)).sum())
        if road_pixels == 0:
            raise ValueError("Training sample has no road pixels")
    pos_weight = (background_pixels / road_pixels if args.pos_weight == "auto" and road_pixels
                  else float(args.pos_weight) if args.pos_weight != "auto" else None)
    if pos_weight is not None and pos_weight <= 0:
        raise ValueError("pos_weight must be positive")
    print(f"pos_weight={pos_weight if pos_weight is None else format(pos_weight, '.8g')} valid_background={background_pixels} road={road_pixels}", flush=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0) if not args.eval_only else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = out_dir.name
    checkpoint_path = ROOT / "runs" / "linear_probe" / "checkpoints" / f"{run_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    epoch_logs = []
    total_gpu_seconds = 0.0
    if args.eval_only:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        head.load_state_dict(state)
    else:
        for epoch in range(args.epochs):
            encoder.eval()
            head.train()
            started = time.perf_counter()
            total_loss = 0.0
            seen = 0
            for images, targets, valid, _, _ in train_loader:
                images, targets, valid = (as_device_tensor(value, device)
                                          for value in (images, targets, valid))
                optimizer.zero_grad(set_to_none=True)
                features = encoder(images)
                logits = head(features)
                loss = masked_bce_with_logits(logits, targets, valid, pos_weight)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * images.shape[0]
                seen += images.shape[0]
            elapsed = time.perf_counter() - started
            total_gpu_seconds += elapsed if device.type == "cuda" else 0.0
            epoch_logs.append({"epoch": epoch + 1, "train_loss": total_loss / seen,
                               "seconds": elapsed, "seconds_per_image": elapsed / seen})
            print(f"epoch={epoch + 1} loss={total_loss / seen:.6f} seconds={elapsed:.2f}", flush=True)
        torch.save(head.state_dict(), checkpoint_path)
    del train_loader
    del train_data, train_sets

    # Stream every forward result to a bounded queue of metric worker processes.
    clean_checks = []
    metric_counts = {domain: {str(threshold): metrics.Counts()
                              for threshold in args.thresholds}
                     for domain in ("Urban", "Rural")}
    per_image_seconds = []
    image_timings = []
    metric_worker_seconds = 0.0
    metric_worker_started = None
    metric_worker_finished = None
    pending = deque()
    max_pending = max(1, 2 * args.metric_workers)

    def consume_oldest():
        nonlocal metric_worker_seconds, metric_worker_started, metric_worker_finished
        domain, values_by_threshold, elapsed, started, finished = pending.popleft().get()
        metric_worker_seconds += elapsed
        metric_worker_started = started if metric_worker_started is None else min(metric_worker_started, started)
        metric_worker_finished = finished if metric_worker_finished is None else max(metric_worker_finished, finished)
        for threshold, values in values_by_threshold.items():
            metric_counts[domain][threshold] = (
                metric_counts[domain][threshold] + metrics.Counts(**values))

    metric_started = time.perf_counter()
    pool = mp.get_context("spawn").Pool(
        processes=args.metric_workers, initializer=initialize_metric_worker,
        initargs=(os.environ.get("EC523_STARTER_DIR"), args.thresholds))
    try:
        for domain, dataset in zip(("Urban", "Rural"), val_sets):
            loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=args.eval_loader_workers, pin_memory=True,
                                persistent_workers=args.eval_loader_workers > 0,
                                worker_init_fn=initialize_data_worker if args.eval_loader_workers > 0 else None,
                                collate_fn=collate_numpy_batch)
            for image, target, valid, sample_id, raw_labels in loader:
                started = time.perf_counter()
                with torch.no_grad():
                    probability = torch.sigmoid(head(encoder(as_device_tensor(image, device)))).squeeze().cpu().numpy()
                elapsed = time.perf_counter() - started
                per_image_seconds.append(elapsed)
                image_timings.append({"domain": domain, "image_id": sample_id[0], "seconds": elapsed})
                raw = np.asarray(raw_labels).squeeze().astype(np.uint8, copy=False)
                if len(clean_checks) < 5:
                    clean_probability = (raw == 3).astype(np.float32)
                    clean_checks.append((clean_probability, raw))
                compact_probability = compact_probability_for_thresholds(probability)
                pending.append(pool.apply_async(
                    metric_counts_for_image, (domain, compact_probability, raw)))
                if len(pending) >= max_pending:
                    consume_oldest()
            del loader
        while pending:
            consume_oldest()
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    metric_wall_seconds = (metric_worker_finished - metric_worker_started
                           if metric_worker_started is not None else 0.0)
    val_pipeline_wall_seconds = time.perf_counter() - metric_started
    val_forward_seconds = float(sum(per_image_seconds))
    if device.type == "cuda":
        total_gpu_seconds += val_forward_seconds
    clean_score = score_group(metrics, clean_checks, 0.5)
    if any(abs(clean_score[key] - 1.0) > 1e-12
           for key in ("road_iou", "cldice", "boundary_iou", "quality")):
        raise RuntimeError(f"GT self-score sanity check failed: {clean_score}")

    overall_counts = {str(threshold): (
        metric_counts["Urban"][str(threshold)] + metric_counts["Rural"][str(threshold)])
        for threshold in args.thresholds}
    best_threshold = max(args.thresholds, key=lambda threshold: report_score(
        metrics, metric_counts["Urban"][str(threshold)] + metric_counts["Rural"][str(threshold)]))
    report = {"config": vars(args), "device": str(device), "pos_weight": pos_weight,
              "train_pixels": {"valid_background": background_pixels, "road": road_pixels},
              "encoder_parameters_frozen": encoder_params, "trainable_parameters": trainable,
              "epochs": epoch_logs,
              "train_seconds": float(sum(entry["seconds"] for entry in epoch_logs)),
              "val_forward_seconds": val_forward_seconds,
              "metric_seconds": metric_worker_seconds,
              "metric_wall_seconds": metric_wall_seconds,
              "val_pipeline_wall_seconds": val_pipeline_wall_seconds,
              "total_gpu_seconds": total_gpu_seconds,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "mean_val_seconds_per_image": float(np.mean(per_image_seconds)) if per_image_seconds else None,
              "val_image_seconds": image_timings,
              "sanity_gt_self_5_images": clean_score, "val": {}}
    report["val"] = {domain: {threshold: metrics.summarize(counts)
                              for threshold, counts in metric_counts[domain].items()}
                     for domain in ("Urban", "Rural")}
    report["val"]["Overall"] = {threshold: metrics.summarize(counts)
                                 for threshold, counts in overall_counts.items()}
    for group_scores in report["val"].values():
        for score in group_scores.values():
            score["A"] = score["quality"]
    report["best_overall_a"] = {"threshold": str(best_threshold),
                                "score": report["val"]["Overall"][str(best_threshold)]["quality"]}
    report["checkpoint"] = str(checkpoint_path.relative_to(ROOT)) if not args.eval_only else args.ckpt
    with (out_dir / "metrics.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    lines = ["# ResNet-50 Layer3 Linear Probe", "",
             f"Device: `{device}`; pos_weight: `{pos_weight if pos_weight is None else format(pos_weight, '.8g')}`.",
             "", "Official metrics are computed by summing per-image `Counts` before `summarize`.",
             "", "## Validation at threshold 0.5", "", "| Group | Road IoU | clDice | Boundary IoU | A |", "|---|---:|---:|---:|---:|"]
    for group in ("Urban", "Rural", "Overall"):
        score = report["val"][group]["0.5"]
        lines.append(f"| {group} | {score['road_iou']:.6f} | {score['cldice']:.6f} | {score['boundary_iou']:.6f} | {score['quality']:.6f} |")
    best_threshold_text = str(best_threshold)
    lines += ["", f"## Validation at best Overall A threshold ({best_threshold_text})",
              "", "Threshold selected on Val; results are optimistically biased.",
              "", "| Group | Road IoU | clDice | Boundary IoU | A |", "|---|---:|---:|---:|---:|"]
    for group in ("Urban", "Rural", "Overall"):
        score = report["val"][group][best_threshold_text]
        lines.append(f"| {group} | {score['road_iou']:.6f} | {score['cldice']:.6f} | {score['boundary_iou']:.6f} | {score['quality']:.6f} |")
    lines += ["", "Overall A by threshold: " + ", ".join(
        f"{threshold}={report['val']['Overall'][str(threshold)]['quality']:.6f}"
        for threshold in args.thresholds), "", f"GT self-score sanity check (5 images): `{clean_score}`.",
              f"Mean validation seconds per image: `{report['mean_val_seconds_per_image']}`.",
              f"Val forward seconds: `{val_forward_seconds}`; metric worker CPU seconds: `{metric_worker_seconds}`; metric wall seconds: `{metric_wall_seconds}`; pipeline wall seconds: `{val_pipeline_wall_seconds}`.",
              f"Checkpoint: `{report['checkpoint']}`.", ""]
    (out_dir / "summary.md").write_text("\n".join(lines))
    print(f"Wrote {out_dir / 'metrics.json'} and {out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
