#!/usr/bin/env python3
"""Train only a shared float32 refiner on frozen RoadSeg probe drafts."""
import argparse
from collections import deque
import json
import math
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
from analysis import y0_error_analysis as y0
from scripts import train_linear_probe as probe
from looproad.models.refiner import LoopRefiner

DOMAINS = ("Urban", "Rural")
STRUCTURE_KEYS = ("gap_skeleton_pixels", "terminal_missing_skeleton_pixels",
                  "whole_missed_skeleton_pixels", "fp_fragments", "predicted_components")


def stamp(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def frozen_draft(encoder, head, images):
    """Use the exact existing encoder.forward -> LinearProbe.forward path.

    Normalization/bf16 live in the encoder; projection/interpolation live in
    LinearProbe.forward. Neither is duplicated here or changed for this run.
    """
    encoder.eval()
    head.eval()
    with torch.no_grad():
        features = encoder(images)
        draft = head(features).float()
    return features, draft


def refine_steps(model, features, draft, steps):
    """The caller owns the loop bound; preserve the entire recurrent graph."""
    state, feature_tokens = model.init(features, draft)
    answer, outputs = draft, []
    for _ in range(steps):
        state, answer = model.step(state, answer, feature_tokens)
        outputs.append(answer)
    return outputs


def deep_supervision(outputs, targets, valid):
    losses = [probe.masked_bce_with_logits(answer, targets, valid, pos_weight=4.) for answer in outputs]
    return torch.stack(losses).mean(), losses


def initialize_eval_worker(starter):
    probe.initialize_metric_worker(starter, (.5,))


def score_image(domain, image_id, predictions, labels, geometry):
    started = time.perf_counter()
    valid = labels != 0
    draft = predictions["y0_draft"].astype(bool)
    outputs, cache = {}, []
    for key, predicted in predictions.items():
        cached = next((item for mask, item in cache if np.array_equal(mask, predicted)), None)
        if cached is None:
            _, counts, _, _, _ = probe.metric_counts_for_image(domain, predicted, labels)
            structure = None
            if geometry:
                analysis = y0.analyze_masks(predicted, labels)
                structure = {kind + "_skeleton_pixels": sum(
                    item["skeleton_pixels"] for item in analysis["segments"][kind].values())
                    for kind in ("gap", "terminal_missing", "whole_missed")}
                structure.update(fp_fragments=len(analysis["fragment_areas"]),
                                 predicted_components=analysis["predicted_components"])
            cached = {"counts": counts["0.5"], "structure": structure}
            cache.append((predicted, cached))
        mask = predicted.astype(bool)
        outputs[key] = {**cached, "added": int((mask & ~draft & valid).sum()),
                        "deleted": int((~mask & draft & valid).sum())}
    return domain, image_id, outputs, time.perf_counter() - started


def make_loader(dataset, batch_size, workers, shuffle=False, seed=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      generator=torch.Generator().manual_seed(seed), pin_memory=True,
                      persistent_workers=False, collate_fn=probe.collate_numpy_batch,
                      worker_init_fn=probe.initialize_data_worker if workers else None)


def evaluate(encoder, head, model, datasets, ks, device, loader_workers=1, metric_workers=6, geometry=True):
    """One streamed trajectory to max(ks); workers receive only uint8 masks."""
    started = time.perf_counter()
    model.eval()
    metrics = probe.official_metrics()
    keys = ["y0_draft"] + [str(k) for k in ks]
    counts = {domain: {key: metrics.Counts() for key in keys} for domain in DOMAINS}
    sums = {domain: {key: {name: 0 for name in (*STRUCTURE_KEYS, "added", "deleted")} for key in keys}
            for domain in DOMAINS}
    times = {domain: {key: [] for key in keys} for domain in DOMAINS}
    peaks = {domain: {key: None for key in keys} for domain in DOMAINS}
    pending, images = deque(), []
    worker_seconds, encoder_seconds = 0., 0.

    def consume():
        nonlocal worker_seconds
        domain, image_id, values, seconds = pending.popleft().get()
        worker_seconds += seconds
        images.append({"domain": domain, "image_id": image_id})
        for key, item in values.items():
            counts[domain][key] = counts[domain][key] + metrics.Counts(**item["counts"])
            for name in ("added", "deleted"):
                sums[domain][key][name] += item[name]
            if geometry:
                for name in STRUCTURE_KEYS:
                    sums[domain][key][name] += item["structure"][name]

    pool = mp.get_context("spawn").Pool(metric_workers, initializer=initialize_eval_worker,
                                        initargs=(os.environ.get("EC523_STARTER_DIR"),))
    try:
        for domain in DOMAINS:
            loader = make_loader(datasets[domain], 1, loader_workers)
            for image, _, _, image_id, raw in loader:
                image = probe.as_device_tensor(image, device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                phase = stamp(device)
                features, draft = frozen_draft(encoder, head, image)
                encoder_seconds += stamp(device) - phase
                predictions = {"y0_draft": (draft.sigmoid().squeeze() >= .5).cpu().numpy().astype(np.uint8)}
                times[domain]["y0_draft"].append(0.)
                if device.type == "cuda":
                    value = torch.cuda.max_memory_allocated(device)
                    peaks[domain]["y0_draft"] = max(peaks[domain]["y0_draft"] or 0, value)
                    torch.cuda.reset_peak_memory_stats(device)
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                    phase = stamp(device)
                    state, feature_tokens = model.init(features.float(), draft.float())
                    elapsed = stamp(device) - phase
                    answer = draft
                    for step in range(1, max(ks) + 1):
                        phase = stamp(device)
                        state, answer = model.step(state, answer, feature_tokens)
                        elapsed += stamp(device) - phase
                        if step in ks:
                            key = str(step)
                            times[domain][key].append(elapsed)
                            if device.type == "cuda":
                                peaks[domain][key] = max(peaks[domain][key] or 0,
                                                         torch.cuda.max_memory_allocated(device))
                            predictions[key] = (answer.sigmoid().squeeze() >= .5).cpu().numpy().astype(np.uint8)
                labels = np.asarray(raw).squeeze().astype(np.uint8, copy=False)
                pending.append(pool.apply_async(score_image, (domain, image_id[0], predictions, labels, geometry)))
                if len(pending) >= 2 * metric_workers:
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
    report = {domain: {} for domain in (*DOMAINS, "Overall")}
    for domain in report:
        for key in keys:
            domains = DOMAINS if domain == "Overall" else (domain,)
            pooled = metrics.Counts()
            for item in domains:
                pooled = pooled + counts[item][key]
            totals = {name: sum(sums[item][key][name] for item in domains) for name in sums[DOMAINS[0]][key]}
            durations = [value for item in domains for value in times[item][key]]
            memory = [peaks[item][key] for item in domains if peaks[item][key] is not None]
            score = metrics.summarize(pooled)
            score["A"] = score["quality"]
            report[domain][key] = {"scores": score,
                "structure": {name: totals[name] for name in STRUCTURE_KEYS} if geometry else None,
                "added_road_pixels": totals["added"], "deleted_road_pixels": totals["deleted"],
                "added_road_fraction": totals["added"] / pooled.valid,
                "deleted_road_fraction": totals["deleted"] / pooled.valid,
                "refiner_seconds": sum(durations),
                "refiner_seconds_per_image": sum(durations) / len(durations),
                "peak_cuda_bytes": max(memory) if memory else None}
    return {"groups": report, "images": images, "encoder_seconds": encoder_seconds,
            "metric_geometry_worker_seconds": worker_seconds, "wall_seconds": time.perf_counter() - started}


def parse_ks(text):
    try:
        values = sorted(set(int(value) for value in text.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--eval-k must be comma-separated positive integers") from error
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("y0_draft is a reference, not K=0")
    return values


def lr_multiplier(completed, total_steps, warmup=500):
    if completed < warmup:
        return (completed + 1) / warmup
    progress = min(1., (completed - warmup) / max(1, total_steps - warmup))
    return .5 * (1. + math.cos(math.pi * progress))


def capture_rng(loader):
    """Use tensors and Python primitives so weights_only=True can load states."""
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            "loader": loader.generator.get_state(),
            "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": numpy_state[2], "has_gauss": numpy_state[3],
                      "cached_gaussian": numpy_state[4]}}


def restore_rng(state, loader, device):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    loader.generator.set_state(state["loader"].cpu())
    if device.type == "cuda" and state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def summary_markdown(report):
    lines = [*("WARNING: " + warning for warning in report["warnings"]),
             "# Looped refiner v1", "", "Observed results for the final epoch; no Val model selection was performed.",
             "The y0_draft row is a reference, not a legal loop count. Threshold is fixed at 0.5.", "",
             "## Parameter counts", "", "| Part | Parameters |", "|---|---:|"]
    for name, count in report["parameters"].items():
        lines.append(f"| {name} | {count} |")
    lines += ["", "## K curves", "", "| Group | K/reference | IoU | clDice | BIoU | A | Added/valid | Deleted/valid | Refiner s/image | Peak CUDA bytes |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for domain, variants in report["evaluation"]["groups"].items():
        for key, item in variants.items():
            score = item["scores"]
            lines.append(f"| {domain} | {key} | {score['road_iou']:.6f} | {score['cldice']:.6f} | {score['boundary_iou']:.6f} | {score['A']:.6f} | {item['added_road_fraction']:.6f} | {item['deleted_road_fraction']:.6f} | {item['refiner_seconds_per_image']:.6f} | {item['peak_cuda_bytes']} |")
    lines += ["", "## Structural counts", "", "| Group | K/reference | Gap skeleton pixels | Terminal missing pixels | Whole missed pixels | FP fragments | Predicted components |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for domain, variants in report["evaluation"]["groups"].items():
        for key, item in variants.items():
            values = " | ".join(str(item["structure"][name]) for name in STRUCTURE_KEYS)
            lines.append(f"| {domain} | {key} | {values} |")
    step_headers = " | ".join(f"Loss t={step}" for step in range(1, report["config"]["k_train"] + 1))
    lines += ["", "## Training epochs", "", f"| Epoch | Loss | {step_headers} | Wall s | Encoder+probe s | Refiner forward+backward s | Peak CUDA bytes |",
              "|" + "---|" * (6 + report["config"]["k_train"])]
    for epoch in report["epochs"]:
        losses = " | ".join(f"{value:.6f}" for value in epoch["step_losses"])
        lines.append(f"| {epoch['epoch']} | {epoch['loss']:.6f} | {losses} | {epoch['seconds']:.3f} | {epoch['encoder_seconds']:.3f} | {epoch['refiner_forward_backward_seconds']:.3f} | {epoch['peak_cuda_bytes']} |")
    lines += ["", "## Fixed-subset monitoring (not used for selection)", "",
              "| Epoch | y0_draft A | K=1 A | K=4 A | Seconds |", "|---|---:|---:|---:|---:|"]
    for row in report["train_curve"]:
        lines.append(f"| {row['epoch']} | {row['y0_draft']:.6f} | {row['1']:.6f} | {row['4']:.6f} | {row['seconds']:.3f} |")
    lines += ["", "Observed timings synchronize CUDA at both boundaries. Refiner timings include initialization but exclude encoder/probe, snapshot copies and metric workers.",
              "Epoch losses are weighted by batch sample count; every batch loss averages valid pixels and supervised steps.",
              "Geometry uses the existing y0 analysis definitions. Counts are summed before official summarize; Overall = Urban + Rural.",
              "CPU peak CUDA memory is reported as None. Fixed positional buffers have zero trainable parameters.", ""]
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20, help="Zero evaluates an untrained zero-output refiner")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--k-train", type=int, default=4)
    parser.add_argument("--eval-k", type=parse_ks, default=parse_ks("1,2,4,6,8"))
    parser.add_argument("--limit-per-domain", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/refiner/v1")
    parser.add_argument("--ckpt-dir", type=Path, default=ROOT / "runs/refiner/v1/checkpoints")
    parser.add_argument("--probe-ckpt", type=Path, default=ROOT / "runs/linear_probe/checkpoints/dinov3_vitl16_web_bn_posw4.pt")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eval-only", action="store_true", help="Load --ckpt and evaluate without Train access")
    mode.add_argument("--resume", action="store_true", help="Continue ckpt-dir/last.pt to the total --epochs target")
    parser.add_argument("--ckpt", type=Path, help="Refiner checkpoint for --eval-only")
    parser.add_argument("--monitor-every", type=int, default=5)
    parser.add_argument("--monitor-per-domain", type=int, default=50)
    parser.add_argument("--loader-workers", type=int, default=7)
    parser.add_argument("--eval-loader-workers", type=int, default=1)
    parser.add_argument("--metric-workers", type=int, default=6)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.)
    parser.add_argument("--patch", type=int, default=16)
    parser.add_argument("--no-mask-feedback", action="store_true")
    parser.add_argument("--reset-state", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.eval_only and args.ckpt is None:
        parser.error("--eval-only requires --ckpt")
    if args.ckpt is not None and not args.eval_only:
        parser.error("--ckpt is used with --eval-only; --resume uses ckpt-dir/last.pt")
    restored = None
    if args.resume or args.eval_only:
        source = args.ckpt if args.eval_only else args.ckpt_dir / "last.pt"
        if not source.is_file():
            parser.error(f"Checkpoint not found: {source}")
        restored = torch.load(source, map_location="cpu", weights_only=True)
        if not {"state_dict", "config", "epoch"}.issubset(restored):
            parser.error("Checkpoint lacks refiner weights/config/epoch")
        explicit = {value.split("=", 1)[0] for value in sys.argv[1:] if value.startswith("--")}
        settings = ["dimension", "layers", "heads", "mlp_ratio", "patch", "no_mask_feedback", "reset_state", "k_train"]
        if args.resume:
            settings += ["batch_size", "lr", "seed", "limit_per_domain", "monitor_every", "monitor_per_domain"]
        for name in settings:
            saved = restored["config"][name]
            if "--" + name.replace("_", "-") in explicit and getattr(args, name) != saved:
                parser.error(f"Checkpoint setting {name}={saved!r} conflicts with the requested value")
            setattr(args, name, saved)
        if args.eval_only:
            args.epochs = restored["config"]["epochs"]
            if "--seed" not in explicit:
                args.seed = restored["config"]["seed"]
        if restored["config"]["probe_ckpt"] != str(args.probe_ckpt.resolve().relative_to(ROOT)):
            parser.error("The frozen probe checkpoint must match the training checkpoint config")
    if args.epochs < 0 or min(args.batch_size, args.k_train, args.monitor_every, args.monitor_per_domain, args.metric_workers) < 1:
        parser.error("Epochs must be nonnegative; batch/loop/monitor/metric settings must be positive")
    if args.resume and args.epochs < int(restored["epoch"]):
        parser.error("--epochs is the total target and cannot precede the saved epoch")
    if args.lr <= 0 or min(args.loader_workers, args.eval_loader_workers) < 0:
        parser.error("Learning rate must be positive and loader workers nonnegative")
    if args.limit_per_domain is not None and args.limit_per_domain < 1:
        parser.error("--limit-per-domain must be positive")
    slots = int(os.environ.get("NSLOTS", "0"))
    if slots and max(args.loader_workers + 1, args.eval_loader_workers + args.metric_workers + 1) > slots:
        parser.error("Main plus active loader/metric workers exceed NSLOTS")
    if not args.out_dir.resolve().is_relative_to(ROOT) or not args.ckpt_dir.resolve().is_relative_to(ROOT / "runs"):
        parser.error("Reports must stay in the repository; checkpoints must stay under runs/")
    if (args.out_dir / "metrics.json").exists() or (args.out_dir / "summary.md").exists():
        parser.error("Refusing to overwrite existing reports; choose a fresh --out-dir")
    if args.epochs and not args.resume and not args.eval_only and (args.ckpt_dir / "last.pt").exists():
        parser.error("Refusing to overwrite a checkpoint from an earlier run")
    probe.initialize_data_worker(0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    encoder, head, checkpoint = y0.load_frozen_probe(args.probe_ckpt, device)
    head.requires_grad_(False).eval()
    encoder.requires_grad_(False).eval()
    model = LoopRefiner(encoder.out_channels, args.dimension, args.layers, args.heads,
                        args.mlp_ratio, args.patch, no_mask_feedback=args.no_mask_feedback,
                        reset_state=args.reset_state).to(device=device, dtype=torch.float32)
    if restored is not None:
        model.load_state_dict(restored["state_dict"])
    parameters = model.parameter_counts()
    print("PARAMETERS " + json.dumps(parameters), flush=True)
    frozen_parameters = {"encoder": sum(p.numel() for p in encoder.parameters()),
                         "probe": sum(p.numel() for p in head.parameters())}
    if any(p.requires_grad for p in (*encoder.parameters(), *head.parameters())):
        raise AssertionError("Encoder and probe must be completely frozen")
    config = {key: (str(value.resolve().relative_to(ROOT)) if value.resolve().is_relative_to(ROOT) else value.name)
              if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(weight_decay=.01, warmup_steps=500, clip_grad_norm=1., pos_weight=4.,
                  refiner_dtype="float32", encoder=y0.ENCODER, probe_head=checkpoint["head"])
    train_sets = ([probe.RoadDataset("Train", domain, limit=args.limit_per_domain, seed=args.seed + i)
                   for i, domain in enumerate(DOMAINS)] if args.epochs and not args.eval_only else [])
    loader = (make_loader(ConcatDataset(train_sets), args.batch_size, args.loader_workers, True, args.seed)
              if train_sets else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01) if loader else None
    scheduler = (torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, len(loader) * args.epochs))
                 if loader else None)
    start_epoch = int(restored["epoch"]) if restored is not None else 0
    epochs = list(restored.get("epochs", [])) if restored is not None else []
    curve = list(restored.get("train_curve", [])) if restored is not None else []
    config["scheduler_total_steps"] = (len(loader) * args.epochs if loader else
                                        restored["config"].get("scheduler_total_steps") if restored else None)
    if args.resume:
        if loader is None:
            parser.error("--resume requires a positive total --epochs target")
        if not {"optimizer", "scheduler", "rng"}.issubset(restored):
            parser.error("Resume requires optimizer, scheduler and RNG states")
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        restore_rng(restored["rng"], loader, device)
        print(f"RESUME loaded_epoch={start_epoch} next_epoch={start_epoch + 1} target_epochs={args.epochs} "
              f"scheduler_last_epoch={scheduler.last_epoch} lr={optimizer.param_groups[0]['lr']:.12g} "
              "restored=refiner,optimizer,scheduler,python_rng,numpy_rng,torch_rng,loader_rng "
              f"cuda_rng_states={len(restored['rng']['cuda'])}", flush=True)
        old_horizon = restored["config"].get("scheduler_total_steps")
        if old_horizon != config["scheduler_total_steps"]:
            print(f"RESUME scheduler_total_steps={old_horizon}->{config['scheduler_total_steps']} "
                  "(changed total epoch target; completed steps and current LR preserved)", flush=True)
    if args.eval_only:
        print(f"EVAL_ONLY loaded_epoch={start_epoch}; Train is not constructed", flush=True)
    for epoch in range(start_epoch, args.epochs) if not args.eval_only else ():
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = stamp(device)
        batches, seen = [], 0
        for batch_index, (images, targets, valid, _, _) in enumerate(loader):
            images, targets, valid = (probe.as_device_tensor(value, device) for value in (images, targets, valid))
            optimizer.zero_grad(set_to_none=True)
            phase = stamp(device)
            features, draft = frozen_draft(encoder, head, images)
            encoder_time = stamp(device) - phase
            phase = stamp(device)
            with torch.autocast(device_type=device.type, enabled=False):
                logits = refine_steps(model, features.float(), draft.float(), args.k_train)
                loss, losses = deep_supervision(logits, targets, valid)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
            refiner_time = stamp(device) - phase
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            learning_rate = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            batch = {"batch": batch_index + 1, "samples": images.shape[0], "loss": float(loss.detach()),
                     "step_losses": [float(value.detach()) for value in losses],
                     "encoder_seconds": encoder_time, "refiner_forward_backward_seconds": refiner_time,
                     "lr": learning_rate, "gradient_norm_before_clip": float(gradient_norm)}
            batches.append(batch)
            seen += images.shape[0]
            print(f"BATCH epoch={epoch + 1} " + json.dumps(batch), flush=True)
            del logits, loss, losses, features, draft
        record = {"epoch": epoch + 1, "loss": sum(row["loss"] * row["samples"] for row in batches) / seen,
                  "step_losses": [sum(row["step_losses"][step] * row["samples"] for row in batches) / seen
                                  for step in range(args.k_train)],
                  "seconds": stamp(device) - epoch_started,
                  "encoder_seconds": sum(row["encoder_seconds"] for row in batches),
                  "refiner_forward_backward_seconds": sum(row["refiner_forward_backward_seconds"] for row in batches),
                  "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  "batches": batches}
        epochs.append(record)
        print("EPOCH " + json.dumps({key: value for key, value in record.items() if key != "batches"}), flush=True)
        if epoch == 0 or (epoch + 1) % args.monitor_every == 0:
            monitor = {domain: probe.RoadDataset("Val", domain) for domain in DOMAINS}
            for dataset in monitor.values():
                dataset.pairs = dataset.pairs[:args.monitor_per_domain]
            observed = evaluate(encoder, head, model, monitor, [1, 4], device,
                                args.eval_loader_workers, args.metric_workers, geometry=False)
            point = {"epoch": epoch + 1, "seconds": observed["wall_seconds"],
                     "images_per_domain": {domain: len(dataset) for domain, dataset in monitor.items()},
                     "selection": "first sorted filenames; monitoring only"}
            point.update({key: observed["groups"]["Overall"][key]["scores"]["A"] for key in ("y0_draft", "1", "4")})
            curve.append(point)
            print("MONITOR " + json.dumps(point), flush=True)
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "config": config,
                    "epochs": epochs, "train_curve": curve, "rng": capture_rng(loader)}, args.ckpt_dir / "last.pt")
    del loader, train_sets
    datasets = {domain: probe.RoadDataset("Val", domain, limit=args.limit_per_domain, seed=args.seed + i + 10)
                for i, domain in enumerate(DOMAINS)}
    if args.limit_per_domain is None and sum(map(len, datasets.values())) != 1669:
        raise ValueError("Full Val must contain 1669 images")
    observed = evaluate(encoder, head, model, datasets, args.eval_k, device,
                        args.eval_loader_workers, args.metric_workers)
    warnings = []
    draft_a = observed["groups"]["Overall"]["y0_draft"]["scores"]["A"]
    regression = "not checked: CPU or limited subset"
    if device.type == "cuda" and args.limit_per_domain is None:
        regression = "passed" if f"{draft_a:.6f}" == "0.526507" else "warning"
        if regression == "warning":
            warnings.append(f"GPU y0_draft Overall A@0.5={draft_a:.6f} (actual {draft_a:.12f}), expected 0.526507.")
    for warning in warnings:
        print("WARNING: " + warning, flush=True)
    report = {"config": config, "parameters": parameters, "frozen_parameters": frozen_parameters,
              "epochs": epochs, "train_curve": curve, "evaluation": observed,
              "warnings": warnings, "y0_gpu_regression": regression,
              "loaded_epoch": start_epoch if restored is not None else None,
              "evaluated_epoch": epochs[-1]["epoch"] if epochs else start_epoch,
              "model_selection": "loaded checkpoint" if args.eval_only else "final epoch only; monitor scores never select weights"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
    print("FINAL_OVERALL_A " + json.dumps({key: row["scores"]["A"] for key, row in observed["groups"]["Overall"].items()}), flush=True)
    print(f"Wrote {args.out_dir / 'metrics.json'} and summary.md", flush=True)


if __name__ == "__main__":
    main()
