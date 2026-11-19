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


def train_updates(encoder, head, loader, optimizer, steps, total, offset=0):
    encoder.train()
    head.train()
    rows = []
    iterator = iter(loader)
    for index in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        images, target, valid, _, _ = batch
        images, target, valid = (runner.probe.as_device_tensor(x, torch.device("cuda"))
                                 for x in (images, target, valid))
        lr = draft.cosine_lr(offset + index, total, 1e-4)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        start = runner.stamp(torch.device("cuda"))
        logits = head(encoder(images))
        loss = runner.probe.masked_bce_with_logits(logits, target, valid, pos_weight=4.)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite LoRA loss")
        loss.backward()
        optimizer.step()
        seconds = runner.stamp(torch.device("cuda")) - start
        row = dict(step=offset+index, seconds=seconds, loss=float(loss.detach()), lr=lr,
                   peak_cuda_bytes=torch.cuda.max_memory_allocated())
        rows.append(row)
        print("UPDATE " + json.dumps(row), flush=True)
    return rows


def save(path, encoder, head, epoch, config):
    torch.save(dict(lora=encoder.adapter_state(), head=head.state_dict(), epoch=epoch, config=config), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--estimate", type=Path, help="Print full-job h_rt from completed smoke JSON and exit")
    args = parser.parse_args()
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
    encoder = LoRAEncoder(frozen.encoder).to(device).eval()
    with torch.no_grad():
        actual = head(encoder(image)).float()
    difference = float((reference-actual).abs().max())
    l0 = dict(passed=difference <= 1e-5, maximum_pixel_difference=difference)
    print("L0 " + json.dumps(l0), flush=True)
    if not l0["passed"]:
        raise ValueError("L0 initialization regression failed")
    del image, reference, actual, frozen
    head.requires_grad_(True)
    lora_parameters = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    head_parameters = sum(p.numel() for p in head.parameters())
    counts = dict(lora=lora_parameters, head=head_parameters, refiner=1193025)
    assert sum(counts.values()) <= 10_000_000
    print("PARAMETERS " + json.dumps(counts), flush=True)
    config = dict(seed=0, epochs=5, batch_size=4, lr=1e-4, weight_decay=0, rank=16, alpha=16,
                  warmup=300, d4_aug=True, selection_per_domain=None,
                  selection_images=834, gradient_checkpointing=True)
    split = json.loads(tta.SPLIT.read_text())
    sets = [D4TrainingDataset(runner.probe.RoadDataset("Train", domain, limit=2 if args.smoke else None), i)
            for i, domain in enumerate(runner.DOMAINS)]
    selection = runner.protocol_datasets(split, "Val-sel", 2 if args.smoke else None)
    parameters = [p for p in encoder.parameters() if p.requires_grad] + list(head.parameters())
    if args.smoke:
        rows = {}
        for batch_size in (2, 4):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            loader = runner.make_loader(ConcatDataset(sets), batch_size, 0, shuffle=True)
            optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0)
            rows[str(batch_size)] = train_updates(encoder, head, loader, optimizer, 20, 3155)
            del optimizer, loader
        selected = draft.evaluate(encoder, head, selection, device, 1, 6)
        report = dict(config=config, L0=l0, parameters=counts, batches=rows, selection=selected,
                      batch4_peak_bytes=max(row["peak_cuda_bytes"] for row in rows["4"]))
        (args.out_dir / "smoke.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print("SMOKE_COMPLETE", flush=True)
        return
    loader = runner.make_loader(ConcatDataset(sets), 4, 7, shuffle=True, seed=0)
    total = 5 * len(loader)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0)
    epochs, best_score, best_epoch = [], -1., None
    for epoch in range(1, 6):
        for dataset in sets:
            dataset.set_epoch(epoch)
        torch.cuda.reset_peak_memory_stats()
        rows = train_updates(encoder, head, loader, optimizer, len(loader), total, (epoch-1)*len(loader))
        selected = draft.evaluate(encoder, head, selection, device, 1, 6)
        score = selected["groups"]["Overall"]["quality"]
        epochs.append(dict(epoch=epoch, batches=rows, selection=selected, selection_A=score))
        if score > best_score:
            best_score, best_epoch = score, epoch
            save(args.out_dir/"best.pt", encoder, head, epoch, config)
        save(args.out_dir/"last.pt", encoder, head, epoch, config)
    best = torch.load(args.out_dir/"best.pt", map_location="cpu", weights_only=True)
    encoder.load_adapters(best["lora"])
    head.load_state_dict(best["head"], strict=True)
    evaluation = tta.evaluate(encoder, head, None, runner.protocol_datasets(split, "Val-rep"), (),
                              device, 1, 6, "d4")
    report = dict(config=config, L0=l0, parameters=counts, epochs=epochs, best_epoch=best_epoch,
                  best_selection_A=best_score, evaluation=evaluation)
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
