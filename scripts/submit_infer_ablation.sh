#!/bin/bash
# Run manually from a clean checkout; this script submits three real jobs.
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
for seed in 0 1 2; do
    if [[ ! -f "runs/exp/e0_v2_s${seed}/best.pt" ]]; then
        echo "Missing e0_v2 seed ${seed} best checkpoint." >&2
        exit 1
    fi
done

submit() {
    printf 'qsub'
    printf ' %s' "$@"
    printf '\n'
    qsub "$@"
}

submit -N ec523-infer-s0 jobs/infer_ablation.qsub --eval-only --ckpt runs/exp/e0_v2_s0/best.pt --seed 0 --device cuda --val-split configs/val_split_seed0.json --eval-extra --eval-k 1,2,4,6,8 --infer-ablations none,reset-state,no-mask-feedback --out-dir results/exp_infer/e0_v2_s0/
submit -N ec523-infer-s1 jobs/infer_ablation.qsub --eval-only --ckpt runs/exp/e0_v2_s1/best.pt --seed 1 --device cuda --val-split configs/val_split_seed0.json --eval-extra --eval-k 1,2,4,6,8 --infer-ablations none,reset-state,no-mask-feedback --out-dir results/exp_infer/e0_v2_s1/
submit -N ec523-infer-s2 jobs/infer_ablation.qsub --eval-only --ckpt runs/exp/e0_v2_s2/best.pt --seed 2 --device cuda --val-split configs/val_split_seed0.json --eval-extra --eval-k 1,2,4,6,8 --infer-ablations none,reset-state,no-mask-feedback --out-dir results/exp_infer/e0_v2_s2/
