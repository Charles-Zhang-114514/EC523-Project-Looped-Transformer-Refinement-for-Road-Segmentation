#!/bin/bash
# Run manually from a clean checkout; this script submits six real jobs.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ ! -d runs/logs ]]; then
    echo "Create runs/logs before submission." >&2
    exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "The working tree must be clean before submission." >&2
    exit 1
fi
if [[ ! -f configs/val_split_seed0.json ]]; then
    echo "Missing configs/val_split_seed0.json." >&2
    exit 1
fi

submit() {
    printf 'qsub'
    printf ' %s' "$@"
    printf '\n'
    qsub "$@"
}

submit -N ec523-e0-s0 jobs/exp_refiner.qsub --head-type conv_upsample --epochs 10 --batch-size 4 --lr 3e-4 --k-train 4 --seed 0 --device cuda --select-every 1 --select-per-domain 100 --selection-ks 1:0.25,2:0.25,4:0.5 --val-split configs/val_split_seed0.json --eval-extra --eval-k 1,2,4,6,8 --out-dir results/exp/e0_v2_s0 --ckpt-dir runs/exp/e0_v2_s0
submit -N ec523-e0-s1 jobs/exp_refiner.qsub --head-type conv_upsample --epochs 10 --batch-size 4 --lr 3e-4 --k-train 4 --seed 1 --device cuda --select-every 1 --select-per-domain 100 --selection-ks 1:0.25,2:0.25,4:0.5 --val-split configs/val_split_seed0.json --eval-extra --eval-k 1,2,4,6,8 --out-dir results/exp/e0_v2_s1 --ckpt-dir runs/exp/e0_v2_s1
submit -N ec523-e0-s2 jobs/exp_refiner.qsub --head-type conv_upsample --epochs 10 --batch-size 4 --lr 3e-4 --k-train 4 --seed 2 --device cuda --select-every 1 --select-per-domain 100 --selection-ks 1:0.25,2:0.25,4:0.5 --val-split configs/val_split_seed0.json --eval-extra --eval-k 1,2,4,6,8 --out-dir results/exp/e0_v2_s2 --ckpt-dir runs/exp/e0_v2_s2
submit -N ec523-e1-s0 jobs/exp_refiner.qsub --head-type conv_upsample --epochs 10 --batch-size 4 --lr 3e-4 --k-train 4 --seed 0 --device cuda --select-every 1 --select-per-domain 100 --selection-ks 1:0.25,2:0.25,4:0.5 --val-split configs/val_split_seed0.json --eval-extra --eval-k 1,2,4,6,8 --reset-state --out-dir results/exp/e1_reset_s0 --ckpt-dir runs/exp/e1_reset_s0
submit -N ec523-e2-s0 jobs/exp_refiner.qsub --head-type conv_upsample --epochs 10 --batch-size 4 --lr 3e-4 --k-train 4 --seed 0 --device cuda --select-every 1 --select-per-domain 100 --selection-ks 1:0.25,2:0.25,4:0.5 --val-split configs/val_split_seed0.json --eval-extra --eval-k 1,2,4,6,8 --no-mask-feedback --out-dir results/exp/e2_nofb_s0 --ckpt-dir runs/exp/e2_nofb_s0
submit -N ec523-e3-s0 jobs/exp_refiner.qsub --head-type conv_upsample --epochs 10 --batch-size 4 --lr 3e-4 --k-train 1 --seed 0 --device cuda --select-every 1 --select-per-domain 100 --selection-ks 1:1.0 --val-split configs/val_split_seed0.json --eval-extra --eval-k 1,2,4,6,8 --out-dir results/exp/e3_single_s0 --ckpt-dir runs/exp/e3_single_s0
