#!/usr/bin/env python3
"""Train convolutional draft heads, select on Val-sel, report on Val-rep."""
import argparse
from collections import deque
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from looproad.models.draft_head import ENCODER, build_draft, layer_text, parse_draft_layers
from scripts import train_linear_probe as probe
from scripts.train_refiner import protocol_datasets, stamp


DOMAINS = ("Urban", "Rural")


def evaluate(encoder, head, datasets, device, loader_workers=1, metric_workers=6):
    """Stream uint8 masks to the existing single-thread spawn metric workers."""
    metrics = probe.official_metrics()
    counts = {domain: metrics.Counts() for domain in DOMAINS}
    pending, files = deque(), {domain: [] for domain in DOMAINS}
    encoder.eval()
    head.eval()
    started, forward_seconds, worker_seconds = time.perf_counter(), 0., 0.

    def consume():
        nonlocal worker_seconds
        domain, values, elapsed, _, _ = pending.popleft().get()
        counts[domain] = counts[domain] + metrics.Counts(**values["0.5"])
        worker_seconds += elapsed

    with mp.get_context("spawn").Pool(metric_workers, initializer=probe.initialize_metric_worker,
                                     initargs=(os.environ.get("EC523_STARTER_DIR"), (0.5,))) as pool:
        for domain in DOMAINS:
            loader = DataLoader(datasets[domain], batch_size=1, num_workers=loader_workers,
                                pin_memory=True, collate_fn=probe.collate_numpy_batch,
                                worker_init_fn=probe.initialize_data_worker if loader_workers else None)
            for images, _, _, ids, raw in loader:
                images = probe.as_device_tensor(images, device)
                phase = stamp(device)
                with torch.no_grad():
                    logits = head(encoder(images))
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError("Nonfinite draft logits during evaluation")
                    predicted = (logits.sigmoid() >= .5).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
                forward_seconds += stamp(device) - phase
                labels = np.asarray(raw).squeeze(0).astype(np.uint8, copy=False)
                files[domain].append(ids[0] + ".png")
                pending.append(pool.apply_async(probe.metric_counts_for_image, (domain, predicted, labels)))
                if len(pending) >= 2 * metric_workers:
                    consume()
            del loader
        while pending:
            consume()
    counts["Overall"] = counts["Urban"] + counts["Rural"]
    return {"groups": {domain: metrics.summarize(value) for domain, value in counts.items()},
            "files": files, "forward_seconds": forward_seconds,
            "metric_worker_seconds": worker_seconds, "wall_seconds": time.perf_counter() - started}


def summary(report):
    lines = ["# Observed convolutional draft-head results", "",
             f"Variant: {report['variant']}; layers: {report['config']['draft_layers']}; "
             f"trainable parameters: {report['trainable_parameters']}; best epoch: {report['best_epoch']}.", "",
             "Observed selection uses only fixed seed-0 Val-sel samples and A at threshold 0.5. "
             "The best head is evaluated once on Val-rep. Overall sums Urban and Rural Counts before official summarize. "
             "CPU smoke limits apply independently to Train, Val-sel and Val-rep; they do not adjudicate preregistered predictions.", "",
             "| Domain | IoU | clDice | BIoU | A |", "|---|---:|---:|---:|---:|"]
    for domain in (*DOMAINS, "Overall"):
        row = report["evaluation"]["groups"][domain]
        lines.append(f"| {domain} | " + " | ".join(f"{row[k]:.9f}" for k in
                     ("road_iou", "cldice", "boundary_iou", "quality")) + " |")
    lines += ["", "| Epoch | Train loss | Train seconds | Val-sel A | New best |",
              "|---|---:|---:|---:|---|"]
    for row in report["epochs"]:
        lines.append(f"| {row['epoch']} | {row['loss']:.9f} | {row['seconds']:.6f} | "
                     f"{row['selection_A']:.9f} | {row['new_best']} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-layers", type=parse_draft_layers, default="last")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--select-per-domain", type=int, default=100)
    parser.add_argument("--limit-per-domain", type=int)
    parser.add_argument("--val-split", type=Path, default=ROOT / "configs/val_split_seed0.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--loader-workers", type=int, default=7)
    parser.add_argument("--eval-loader-workers", type=int, default=1)
    parser.add_argument("--metric-workers", type=int, default=6)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--ckpt-dir", type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.select_per_domain, args.metric_workers) < 1:
        parser.error("Epochs, batch size, selection count and metric workers must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0 or min(args.loader_workers, args.eval_loader_workers) < 0:
        parser.error("Learning rate must be finite and positive; loader workers must be nonnegative")
    if args.limit_per_domain is not None and args.limit_per_domain < 1:
        parser.error("The smoke limit must be positive")
    slots = int(os.environ.get("NSLOTS", "0"))
    if slots and max(1 + args.loader_workers, 1 + args.eval_loader_workers + args.metric_workers) > slots:
        parser.error("Main plus active loader/metric workers exceeds NSLOTS")
    variant = "A1-i" if args.draft_layers == "last" else "A1-ii"
    args.out_dir = args.out_dir or ROOT / "results/draft_head" / variant
    args.ckpt_dir = args.ckpt_dir or ROOT / "runs/draft_head" / variant
    if not args.out_dir.resolve().is_relative_to(ROOT) or not args.ckpt_dir.resolve().is_relative_to(ROOT / "runs"):
        parser.error("Reports must be inside the repository; checkpoints must be inside runs")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Refusing to overwrite a nonempty report directory")
    checkpoint_path = args.ckpt_dir / "best.pt"
    if checkpoint_path.exists():
        parser.error("Refusing to overwrite an existing checkpoint")
    split_bytes = args.val_split.read_bytes()
    split = json.loads(split_bytes)
    selection_limit = min(args.select_per_domain, args.limit_per_domain) if args.limit_per_domain else args.select_per_domain
    selection = protocol_datasets(split, "Val-sel", selection_limit)
    evaluation = protocol_datasets(split, "Val-rep", args.limit_per_domain)
    assert all(set(selection[d].pairs).isdisjoint(evaluation[d].pairs) for d in DOMAINS)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    probe.initialize_data_worker(0)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    encoder, head = build_draft(args.draft_layers)
    encoder, head = encoder.to(device).eval(), head.to(device)
    assert not any(p.requires_grad for p in encoder.parameters()) and not encoder.training
    parameters = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print("PARAMETERS " + json.dumps({"variant": variant, "trainable": parameters,
          "encoder_frozen": sum(p.numel() for p in encoder.parameters())}), flush=True)
    train_sets = [probe.RoadDataset("Train", d, limit=args.limit_per_domain, seed=args.seed + i)
                  for i, d in enumerate(DOMAINS)]
    loader = DataLoader(ConcatDataset(train_sets), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.loader_workers, pin_memory=True,
                        # Selection runs after every epoch: release all seven
                        # training workers before starting 1+6 eval workers.
                        persistent_workers=False,
                        generator=torch.Generator().manual_seed(args.seed),
                        worker_init_fn=probe.initialize_data_worker if args.loader_workers else None,
                        collate_fn=probe.collate_numpy_batch)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0)
    config = {key: (value.resolve().relative_to(ROOT).as_posix() if value.resolve().is_relative_to(ROOT)
                   else value.name) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(draft_layers=layer_text(args.draft_layers), encoder=ENCODER, head="conv_up",
                  pos_weight=4., weight_decay=0., threshold=.5, feature_normalization="frozen_final_layernorm",
                  val_split_sha256=hashlib.sha256(split_bytes).hexdigest())
    logs, best_score, best_epoch = [], None, None
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        encoder.eval()
        head.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started, batches, seen, total = stamp(device), [], 0, 0.
        for index, (images, targets, valid, _, _) in enumerate(loader, 1):
            images, targets, valid = (probe.as_device_tensor(v, device) for v in (images, targets, valid))
            optimizer.zero_grad(set_to_none=True)
            phase = stamp(device)
            features = encoder(images)
            encoder_seconds = stamp(device) - phase
            phase = stamp(device)
            logits = head(features)
            loss = probe.masked_bce_with_logits(logits, targets, valid, pos_weight=4.)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite draft-head loss; hyperparameters were not changed")
            loss.backward()
            optimizer.step()
            head_seconds = stamp(device) - phase
            samples = len(images)
            value = float(loss.detach())
            row = dict(batch=index, samples=samples, loss=value, encoder_seconds=encoder_seconds,
                       head_forward_backward_seconds=head_seconds)
            batches.append(row)
            print("BATCH " + json.dumps(row), flush=True)
            total += value * samples
            seen += samples
            del features, logits, loss
        elapsed = stamp(device) - started
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        selected = evaluate(encoder, head, selection, device, args.eval_loader_workers, args.metric_workers)
        score = selected["groups"]["Overall"]["quality"]
        improved = best_score is None or score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            torch.save(dict(encoder=ENCODER, head="conv_up", draft_layers=layer_text(args.draft_layers),
                            pos_weight=4., state_dict=head.state_dict(), epoch=epoch,
                            config=config, trainable_parameters=parameters), checkpoint_path)
        record = dict(epoch=epoch, loss=total / seen, seconds=elapsed, batches=batches,
                      peak_cuda_bytes=peak, selection_A=score, new_best=improved,
                      selection=selected)
        logs.append(record)
        print("EPOCH " + json.dumps({k: v for k, v in record.items() if k not in ("batches", "selection")}), flush=True)
    del loader, train_sets, optimizer
    best = torch.load(checkpoint_path, map_location=device, weights_only=True)
    head.load_state_dict(best["state_dict"])
    observed = evaluate(encoder, head, evaluation, device, args.eval_loader_workers, args.metric_workers)
    report = dict(variant=variant, config=config, trainable_parameters=parameters,
                  encoder_parameters=sum(p.numel() for p in encoder.parameters()),
                  epochs=logs, best_epoch=best_epoch, best_selection_A=best_score, evaluation=observed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "summary.md").write_text(summary(report))
    print("FINAL " + json.dumps(observed["groups"]), flush=True)


if __name__ == "__main__":
    main()
