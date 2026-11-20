#!/usr/bin/env python3
"""Offline DINOv3 LoRA training and short GPU resource measurement."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import ConcatDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from analysis import evaluate_draft_tta as tta
from looproad.data.augmentation import D4TrainingDataset
from looproad.models.lora import LoRAEncoder
from scripts import train_draft_head as draft
from scripts import train_refiner as runner


def set_training(encoder, head, freeze_head=False):
    encoder.train()
    head.train(not freeze_head)


def update_lr(step, total, peak, warmup):
    if warmup == 300:
        return draft.cosine_lr(step, total, peak)
    if total == 1:
        return peak
    warmup = min(warmup, total-1)
    if step < warmup:
        return peak*(step+1)/warmup
    return peak*.5*(1+math.cos(math.pi*(step-warmup+1)/(total-warmup)))


def train_updates(encoder, head, loader, optimizer, steps, total, offset=0,
                  freeze_head=False, lr=1e-4, warmup=300, callback=None):
    set_training(encoder, head, freeze_head)
    device = next(head.parameters()).device
    rows = []
    iterator = iter(loader)
    for index in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        images, target, valid, _, _ = batch
        images, target, valid = (runner.probe.as_device_tensor(x, device)
                                 for x in (images, target, valid))
        rate = update_lr(offset + index, total, lr, warmup)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        start = runner.stamp(device)
        logits = head(encoder(images))
        loss = runner.probe.masked_bce_with_logits(logits, target, valid, pos_weight=4.)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite LoRA loss")
        loss.backward()
        optimizer.step()
        seconds = runner.stamp(device) - start
        row = dict(step=offset+index, seconds=seconds, loss=float(loss.detach()), lr=rate,
                   peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None)
        rows.append(row)
        print("UPDATE " + json.dumps(row), flush=True)
        if callback is not None:
            callback(offset+index+1)
            set_training(encoder, head, freeze_head)
    return rows


def save(path, encoder, head, epoch, config, global_step=None):
    payload = dict(lora=encoder.adapter_state() if hasattr(encoder, "adapter_state") else {},
                   head=head.state_dict(), epoch=epoch, config=config)
    if global_step is not None:
        payload["global_step"] = global_step
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--freeze-head", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--max-steps", type=int, help="Bound total updates, cycling a small dataset if needed")
    parser.add_argument("--limit-per-domain", type=int)
    parser.add_argument("--loader-workers", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--estimate", type=Path, help="Print full-job h_rt from completed smoke JSON and exit")
    args = parser.parse_args()
    if min(args.epochs, args.warmup_steps) < 1 or args.eval_every_steps < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Invalid training duration, evaluation interval or learning rate")
    if args.freeze_head and args.no_lora:
        parser.error("Freezing the head without LoRA leaves no trainable parameters")
    if args.max_steps is not None and (args.max_steps < 1 or args.smoke or args.estimate):
        parser.error("--max-steps requires a positive fresh training run")
    if args.limit_per_domain is not None and args.limit_per_domain < 1:
        parser.error("Sample limit must be positive")
    if args.loader_workers < 0:
        parser.error("Loader workers must be nonnegative")
    if args.estimate:
        report = json.loads(args.estimate.read_text())
        if not report["L0"]["passed"] or report["batch4_peak_bytes"] > 40 * 1024**3:
            raise ValueError("Smoke gate or batch-4 memory bound failed")
        rate = max(row["seconds"] for row in report["batches"]["4"])
        selection_count = sum(len(files) for files in report["selection"]["files"].values())
        inference_rate = report["selection"]["forward_seconds"] / selection_count
        # 5*ceil(2522/4) updates; five full selection passes, report 9 views.
        seconds = math.ceil(1.5 * (3155 * rate + (5*834 + 9*835)*(inference_rate+.125)) + 900)
        print(f"{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}")
        return
    if not args.out_dir.resolve().is_relative_to(ROOT / "runs") or args.out_dir.exists():
        parser.error("Choose a fresh output directory under runs")
    if not torch.cuda.is_available():
        parser.error("This experiment requires an allocated GPU")
    args.out_dir.mkdir(parents=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    runner.probe.initialize_data_worker(0)
    device = torch.device("cuda")
    frozen, head, _ = runner.y0.load_frozen_probe(ROOT / tta.previous.PROBES["A1-i"], device)
    dataset = runner.probe.RoadDataset("Train", "Urban", limit=2, seed=0)
    image = runner.probe.as_device_tensor(np.stack([dataset[i][0] for i in range(2)]), device)
    with torch.no_grad():
        reference = head(frozen(image)).float()
    encoder = frozen if args.no_lora else LoRAEncoder(frozen.encoder).to(device).eval()
    with torch.no_grad():
        actual = head(encoder(image)).float()
    difference = float((reference-actual).abs().max())
    l0 = dict(passed=difference <= 1e-5, maximum_pixel_difference=difference)
    print("L0 " + json.dumps(l0), flush=True)
    if not l0["passed"]:
        raise ValueError("L0 initialization regression failed")
    del image, reference, actual, frozen
    head.requires_grad_(not args.freeze_head)
    frozen_head_before = ({key:value.detach().cpu().clone() for key,value in head.state_dict().items()}
                          if args.freeze_head else None)
    lora_parameters = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    head_parameters = sum(p.numel() for p in head.parameters() if p.requires_grad)
    counts = dict(lora=lora_parameters, head=head_parameters, refiner=1193025)
    assert sum(counts.values()) <= 10_000_000
    print("PARAMETERS " + json.dumps(counts), flush=True)
    config = dict(seed=0, epochs=args.epochs, batch_size=4, lr=args.lr, weight_decay=0, rank=16, alpha=16,
                  warmup=args.warmup_steps, d4_aug=True, selection_per_domain=None,
                  selection_images=834, gradient_checkpointing=not args.no_lora)
    if args.freeze_head or args.no_lora or args.eval_every_steps:
        config.update(freeze_head=args.freeze_head, no_lora=args.no_lora, eval_every_steps=args.eval_every_steps)
    if args.max_steps is not None or args.limit_per_domain is not None:
        config.update(max_steps=args.max_steps, limit_per_domain=args.limit_per_domain)
    limit = 2 if args.smoke else args.limit_per_domain
    split = json.loads(tta.SPLIT.read_text())
    sets = [D4TrainingDataset(runner.probe.RoadDataset("Train", domain, limit=limit), i)
            for i, domain in enumerate(runner.DOMAINS)]
    selection = runner.protocol_datasets(split, "Val-sel", limit)
    if args.limit_per_domain is not None:
        config["selection_images"] = sum(len(dataset) for dataset in selection.values())
    parameters = [p for p in encoder.parameters() if p.requires_grad] + [p for p in head.parameters() if p.requires_grad]
    if args.smoke:
        rows = {}
        for batch_size in (2, 4):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            loader = runner.make_loader(ConcatDataset(sets), batch_size, 0, shuffle=True)
            optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0)
            rows[str(batch_size)] = train_updates(encoder, head, loader, optimizer, 20, 3155,
                                                 freeze_head=args.freeze_head, lr=args.lr, warmup=args.warmup_steps)
            del optimizer, loader
        selected = draft.evaluate(encoder, head, selection, device, 1, 6)
        report = dict(config=config, L0=l0, parameters=counts, batches=rows, selection=selected,
                      batch4_peak_bytes=max(row["peak_cuda_bytes"] for row in rows["4"]))
        (args.out_dir / "smoke.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print("SMOKE_COMPLETE", flush=True)
        return
    loader = runner.make_loader(ConcatDataset(sets), 4, 0 if args.eval_every_steps else args.loader_workers, shuffle=True, seed=0)
    total = args.max_steps if args.max_steps is not None else args.epochs * len(loader)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0)
    epochs, best_score, best_epoch = [], -1., None
    step_selection, best_step = [], None
    for epoch in range(1, args.epochs+1):
        for dataset in sets:
            dataset.set_epoch(epoch)
        torch.cuda.reset_peak_memory_stats()
        def select_step(step):
            nonlocal best_score, best_epoch, best_step
            if step_selection and step_selection[-1]["step"] == step:
                return
            selected = draft.evaluate(encoder, head, selection, device, 1, 6)
            score = selected["groups"]["Overall"]["quality"]
            step_selection.append(dict(step=step, epoch=epoch, selection_A=score, selection=selected))
            if score > best_score:
                best_score, best_epoch, best_step = score, epoch, step
                save(args.out_dir/"best.pt", encoder, head, epoch, config, step)

        def maybe_select(step):
            if step % args.eval_every_steps == 0:
                select_step(step)

        epoch_steps = total if args.max_steps is not None else len(loader)
        rows = train_updates(encoder, head, loader, optimizer, epoch_steps, total, (epoch-1)*len(loader),
                             freeze_head=args.freeze_head, lr=args.lr, warmup=args.warmup_steps,
                             callback=maybe_select if args.eval_every_steps else None)
        if args.eval_every_steps:
            select_step(total if args.max_steps is not None else epoch*len(loader))
            selected = step_selection[-1]["selection"]
            score = selected["groups"]["Overall"]["quality"]
        else:
            selected = draft.evaluate(encoder, head, selection, device, 1, 6)
            score = selected["groups"]["Overall"]["quality"]
            if score > best_score:
                best_score, best_epoch = score, epoch
                save(args.out_dir/"best.pt", encoder, head, epoch, config)
        epochs.append(dict(epoch=epoch, batches=rows, selection=selected, selection_A=score))
        save(args.out_dir/"last.pt", encoder, head, epoch, config)
        if args.max_steps is not None:
            break
    frozen_head_check = None
    if frozen_head_before is not None:
        frozen_head_check = all(torch.equal(value, head.state_dict()[key].detach().cpu())
                                for key,value in frozen_head_before.items())
        if not frozen_head_check or any(p.grad is not None for p in head.parameters()):
            raise ValueError("Frozen head parameters, BN statistics or gradients changed")
        print("FROZEN_HEAD_PASS", flush=True)
    best = torch.load(args.out_dir/"best.pt", map_location="cpu", weights_only=True)
    if best["lora"]:
        encoder.load_adapters(best["lora"])
    head.load_state_dict(best["head"], strict=True)
    evaluation = tta.evaluate(encoder, head, None, runner.protocol_datasets(split, "Val-rep", args.limit_per_domain), (),
                              device, 1, 6, "d4")
    report = dict(config=config, L0=l0, parameters=counts, epochs=epochs, best_epoch=best_epoch,
                  best_selection_A=best_score, evaluation=evaluation)
    if args.eval_every_steps:
        report.update(step_selection=step_selection, best_step=best_step)
    if frozen_head_check is not None:
        report["frozen_head_unchanged"] = frozen_head_check
    baseline = json.loads((ROOT/"results/draft_head/A1-i/metrics.json").read_text())["evaluation"]["groups"]
    actual = evaluation["groups"]["none"]
    gain = actual["Overall"]["y0_draft"]["scores"]["quality"] - .538246
    iou_gain = actual["Overall"]["y0_draft"]["scores"]["road_iou"] - baseline["Overall"]["road_iou"]
    cldice_gain = actual["Overall"]["y0_draft"]["scores"]["cldice"] - baseline["Overall"]["cldice"]
    peak = max(row["peak_cuda_bytes"] for epoch in epochs for row in epoch["batches"])
    domains_pass = all(actual[d]["y0_draft"]["scores"]["quality"] >= baseline[d]["quality"] - .002
                       for d in runner.DOMAINS)
    report["observations"] = dict(A_gain=gain, IoU_gain=iou_gain, clDice_gain=cldice_gain,
                                  B_P1=.005 <= gain <= .020 and iou_gain > cldice_gain,
                                  B_P2=peak <= 40*1024**3, train_peak_cuda_bytes=peak,
                                  B_R1=gain >= .005 and domains_pass)
    (args.out_dir/"metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    lines = ["# Observed LoRA draft results", "", f"Observed best epoch {best_epoch}.", "",
             "| TTA | Domain | IoU | clDice | BIoU | A |", "|---|---|---:|---:|---:|---:|"]
    for mode, groups in evaluation["groups"].items():
        for domain, steps in groups.items():
            values = " | ".join(f"{steps['y0_draft']['scores'][key]:.9f}" for key in tta.SCORE_KEYS)
            lines.append(f"| {mode} | {domain} | {values} |")
    lines += ["", "| Epoch | Val-sel A |", "|---:|---:|"]
    lines += [f"| {epoch['epoch']} | {epoch['selection_A']:.9f} |" for epoch in epochs]
    (args.out_dir/"summary.md").write_text("\n".join(lines)+"\n")
    print("LORA_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
