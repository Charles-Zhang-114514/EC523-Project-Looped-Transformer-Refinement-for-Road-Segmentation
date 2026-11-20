#!/usr/bin/env python3
"""Run four independent short GPU checks through the production training CLIs."""
import argparse
from collections import deque
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]
from scripts import cache_tta_draft as build


def stage_commands(cache_dir, out):
    commands = {}
    for arm in ("RS", "RW"):
        directory = out/arm
        command = [sys.executable, "-u", "scripts/train_refiner.py",
                   "--head-type", "conv_upsample", "--epochs", "10" if arm=="RS" else "3",
                   "--batch-size", "4", "--k-train", "4", "--seed", "0", "--device", "cuda",
                   "--lr", "3e-4" if arm=="RS" else "1e-4", "--warmup-steps", "500" if arm=="RS" else "100",
                   "--max-steps", "20", "--limit-per-domain", "2", "--select-per-domain", "2",
                   "--select-every", "1", "--selection-ks", "1:0.2,2:0.2,4:0.6",
                   "--val-split", "configs/val_split_seed0.json", "--eval-k", "1,2,4,8",
                   "--loader-workers", "0", "--eval-loader-workers", "1", "--metric-workers", "6",
                   "--probe-ckpt", "runs/draft_head/A1-i/best.pt", "--draft-cache", str(cache_dir),
                   "--out-dir", str(directory/"report"), "--ckpt-dir", str(directory/"checkpoints")]
        if arm=="RW":
            command += ["--init-refiner", "runs/exp/e6_convdraft_s0/best.pt"]
        commands[arm] = command
    for arm in ("LA", "LC"):
        command = [sys.executable, "-u", "scripts/train_lora_draft.py",
                   "--max-steps", "20", "--limit-per-domain", "2", "--loader-workers", "0",
                   "--out-dir", str(out/arm)]
        if arm=="LA":
            command += ["--freeze-head", "--lr", "2e-5", "--warmup-steps", "100",
                        "--epochs", "2", "--eval-every-steps", "200"]
        else:
            command += ["--no-lora"]
        commands[arm] = command
    return commands


def run_stage(arm, command, out):
    log = out/(arm+".log")
    record = dict(command=command[1:], log=log.name, passed=False, exit_code=None)
    try:
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
        record["exit_code"] = result.returncode
        with log.open() as stream:
            record["log_tail"] = list(deque(stream, maxlen=12))
        if result.returncode:
            return record
        path = out/arm/("report/metrics.json" if arm in ("RS","RW") else "metrics.json")
        data = json.loads(path.read_text())
        updates = sum(len(epoch["batches"]) for epoch in data["epochs"])
        if updates != 20:
            raise ValueError(f"Expected 20 optimizer updates, observed {updates}")
        evaluation = data["evaluation"]
        if arm in ("RS","RW"):
            images = evaluation["images"]
            counts = {d:sum(row["domain"]==d for row in images) for d in ("Urban","Rural")}
            finite = all(torch.isfinite(torch.tensor(epoch["loss"])) for epoch in data["epochs"])
        else:
            images = evaluation["images"]
            counts = {d:sum(row["domain"]==d for row in images) for d in ("Urban","Rural")}
            finite = all(torch.isfinite(torch.tensor(batch["loss"])) for epoch in data["epochs"] for batch in epoch["batches"])
        unique = {(row["domain"],row["image_id"]) for row in images}
        if counts != dict(Urban=2,Rural=2) or len(unique) != 4 or not finite:
            raise ValueError("Missing finite training losses or two scored images per domain")
        record.update(updates=updates, scored_images=counts, evaluation=evaluation)
        if arm=="LA":
            record["head_and_BN_unchanged"] = data.get("frozen_head_unchanged",False)
            if not record["head_and_BN_unchanged"]:
                raise ValueError("LA frozen head/BN check did not pass")
        record["passed"] = True
    except Exception:
        record["error"] = traceback.format_exc()
    return record


def run_stages(commands, out, report):
    for arm, command in commands.items():
        report[arm] = run_stage(arm, command, out)
        (out/"metrics.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        print("STAGE " + json.dumps(dict(arm=arm,passed=report[arm]["passed"],
                                         exit_code=report[arm]["exit_code"])),flush=True)
    return report


def finish(report):
    for arm in ("RS","RW","LA","LC"):
        record=report[arm]
        print(f"{arm}: {'PASS' if record['passed'] else 'FAIL'} exit_code={record['exit_code']}",flush=True)
    passed = report.get("cache",{}).get("gates_passed",False) and all(report[arm]["passed"] for arm in ("RS","RW","LA","LC"))
    print("WAVE2_SMOKE_COMPLETE" if passed else "WAVE2_SMOKE_FAILED",flush=True)
    return passed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=ROOT/"runs/wave2"/("smoke_"+os.environ.get("JOB_ID",str(time.time_ns()))))
    args=parser.parse_args()
    out=args.out_dir.resolve()
    if not out.is_relative_to(ROOT/"runs") or out.exists():
        parser.error("Choose a fresh output directory under runs")
    if not torch.cuda.is_available():
        parser.error("Run on an allocated GPU")
    out.mkdir(parents=True)
    print("SMOKE_OUTPUT " + out.relative_to(ROOT).as_posix(),flush=True)
    (ROOT/"runs/cache").mkdir(parents=True,exist_ok=True)
    report={}
    with tempfile.TemporaryDirectory(prefix="wave2-smoke-",dir=ROOT/"runs/cache") as directory:
        cache_dir=Path(directory)/"draft"
        try:
            built=build.build_cache(cache_dir,smoke=True,per_domain=2,training_entrypoints=True)
            report["cache"]=built[3].manifest
            del built
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            report["cache"]=dict(gates_passed=False,error=traceback.format_exc())
            print("CACHE_FAILED " + report["cache"]["error"],flush=True)
        run_stages(stage_commands(cache_dir,out),out,report)
    if not finish(report):
        raise SystemExit(1)


if __name__=="__main__":
    main()
