#!/usr/bin/env python3
"""Detail training/reporting in one CUDA PID, with smoke-measured time budgets."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def estimate(directory,arm):
    report=json.loads((directory/'metrics.json').read_text())
    required={'DRG1','base-RS','detail-RS','score-RS','base-RW','detail-RW','score-RW'}
    if set(report)!=required or not all(r['passed']for r in report.values()):
        raise ValueError('A complete passed wave4 smoke is required')
    name='RS'if arm=='S'else'RW'
    def mean(kind):
        rows=report[kind+'-'+name]['batches'][1:]
        if len(rows)!=19:raise ValueError('Expected 20 smoke updates; discard only the first')
        return statistics.mean(row['update_seconds']for row in rows)
    base,detail=mean('base'),mean('detail');extra=max(0.,detail-base)
    epochs=10 if arm=='S'else 3
    train_updates=631*epochs
    # Conservative inference overhead: charge one quarter of the measured
    # four-step forward/backward/optimizer increment for each inference step.
    inference_steps=epochs*834*4 + 2*835*8 + 835*4
    if arm=='W':inference_steps+=835*8  # Additional full pre-update gate trajectory.
    gate_seconds=835*(.04+8*.012+4*.125) if arm=='W'else 0.
    baseline_seconds=(60 if arm=='S'else 25)*60
    raw=baseline_seconds+train_updates*extra+inference_steps*extra/4+gate_seconds
    seconds=math.ceil((raw*1.3+300)/60)*60
    return dict(arm=arm,base_mean_update_seconds=base,detail_mean_update_seconds=detail,
                added_seconds_per_update=extra,train_updates=train_updates,inference_steps=inference_steps,
                full_DRG1_allowance_seconds=gate_seconds,baseline_seconds=baseline_seconds,
                formula='ceil_to_minute(1.3*(baseline + train_updates*extra + inference_steps*extra/4 + full_gate) + 300)',
                seconds=seconds,h_rt=f'{seconds//3600:02}:{seconds%3600//60:02}:{seconds%60:02}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--estimate',type=Path);p.add_argument('--smoke-dir',type=Path)
    p.add_argument('--arm',choices=['S','W']);p.add_argument('--seed',type=int,choices=range(3));a=p.parse_args()
    if a.estimate:
        directory=a.estimate.resolve();relative=directory.relative_to(ROOT)
        for arm in ([a.arm]if a.arm else ['S','W']):
            budget=estimate(directory,arm);print(json.dumps(budget))
            for seed in range(3):print(f'qsub -l gpu_type=L40S,h_rt={budget["h_rt"]} -v ARM={arm},SEED={seed},SMOKE_DIR={relative} jobs/refiner_dr.qsub')
        return
    if a.smoke_dir is None or a.arm is None or a.seed is None:p.error('Supply arm, seed and passed smoke directory')
    budget=estimate(a.smoke_dir,a.arm)
    from scripts import train_refiner as training
    from scripts import evaluate_refiner_focus as evaluation
    from scripts import detail_gate
    out=ROOT/'runs/detail_refiner'/f'DR_{a.arm}_s{a.seed}'
    if a.arm=='W'and a.seed==0:
        # Gate before any training update; exception/failed comparison exits.
        detail_gate.run(out/'pretraining_DRG1',full=True)
    rs=a.arm=='S'
    sys.argv=['train_refiner.py','--detail-head','--d4-aug','--draft-cache','runs/cache/tta_draft_a1i_s0',
              '--probe-ckpt','runs/draft_head/A1-i/best.pt','--head-type','conv_upsample',
              '--epochs','10'if rs else'3','--lr','3e-4'if rs else'1e-4','--warmup-steps','500'if rs else'100',
              '--batch-size','4','--k-train','4','--pos-weight','4','--seed',str(a.seed),'--device','cuda',
              '--select-per-domain','all','--select-every','1','--selection-ks','1:0.2,2:0.2,4:0.6',
              '--val-split','configs/val_split_seed0.json','--eval-k','1,2,4,8',
              '--out-dir',str(out/'report'),'--ckpt-dir',str(out/'checkpoints')]
    if not rs:sys.argv+=['--init-refiner','runs/exp/e6_convdraft_s0/best.pt']
    print('SMOKE_BUDGET '+json.dumps(budget),flush=True)
    training.main()
    sys.argv=['evaluate_refiner_focus.py','--phase','trained','--checkpoint',str(out/'checkpoints/best.pt'),'--out-dir',str(out/'evaluation')]
    evaluation.main()


if __name__=='__main__':main()
