#!/usr/bin/env python3
"""CD/BS/HO formal entries and GPU-smoke-derived time budgets; one CUDA PID."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def training_command(arm,seed,out,smoke=False):
    cmd=[sys.executable,'-u','scripts/train_refiner.py','--head-type','conv_upsample',
         '--epochs','10','--batch-size','4','--lr','3e-4','--warmup-steps','500',
         '--k-train','4','--pos-weight','4','--seed',str(seed),'--device','cuda',
         '--d4-aug','--record-selection-domains','--select-every','1',
         '--select-per-domain','2'if smoke else'all','--selection-ks','1:0.2,2:0.2,4:0.6',
         '--val-split','configs/val_split_seed0.json','--eval-k','1,2,4,8',
         '--probe-ckpt','runs/draft_head/A1-i/best.pt','--draft-cache','runs/cache/tta_draft_a1i_s0',
         '--out-dir',str(out/'report'),'--ckpt-dir',str(out/'checkpoints')]
    if arm=='CD':cmd+=['--draft-corrupt','.5']
    elif arm=='BS':cmd+=['--delta-bound','1.0','--step-weights','0.1,0.2,0.3,0.4']
    elif arm=='HO':
        if seed!=0:raise ValueError('HO uses seed 0 only')
        cmd+=['--train-holdout-per-domain','2'if smoke else'100']
    else:raise ValueError('Unknown arm')
    if smoke:
        cmd+=['--max-steps','20','--limit-per-domain','40'if arm=='CD'else'2',
              '--loader-workers','2'if arm=='CD'else'0','--eval-loader-workers','1','--metric-workers','6']
    return cmd


def score_command(checkpoint,out,smoke=False):
    cmd=[sys.executable,'-u','scripts/evaluate_refiner_focus.py','--phase','trained',
         '--audit-training-controls','--checkpoint',str(checkpoint),'--out-dir',str(out)]
    if smoke:cmd+=['--limit-per-domain','2']
    return cmd


def estimate(directory,arm):
    report=json.loads((directory/'metrics.json').read_text());required={a for arm in ['CD','BS','HO']for a in [arm,'score-'+arm]}
    if set(report)!=required or not all(r['passed']for r in report.values()):raise ValueError('Complete passed wave5 smoke required')
    samples=report['CD']['corruption']
    cpu_mean=statistics.mean(r['corruption_seconds']for r in samples)
    if not math.isfinite(cpu_mean) or cpu_mean<0:raise ValueError('Invalid measured CPU corruption time')
    baseline=3600.
    added_cpu=2522*10*cpu_mean if arm=='CD'else 0.
    # Four metric groups (draft, K1/K2/K4), with the observed six-worker rate.
    added_holdout=10*200*(.04+4*.012+4*.125) if arm=='HO'else 0.
    seconds=math.ceil((1.3*(baseline+added_cpu+added_holdout)+300)/60)*60
    return dict(arm=arm,baseline_seconds=baseline,mean_CPU_corruption_seconds=cpu_mean,
                added_CPU_seconds=added_cpu,added_holdout_seconds=added_holdout,
                CPU_allowance='Conservative sum over all 25220 Train presentations; no parallelism/overlap discount, excluding cache read time.',
                formula='ceil_to_minute(1.3*(baseline+added_CPU+added_holdout)+300)',seconds=seconds,
                h_rt=f'{seconds//3600:02}:{seconds%3600//60:02}:{seconds%60:02}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['CD','BS','HO']);p.add_argument('--seed',type=int,choices=range(3));p.add_argument('--estimate',type=Path);p.add_argument('--smoke-dir',type=Path);a=p.parse_args()
    if a.estimate:
        relative=a.estimate.resolve().relative_to(ROOT)
        for arm in ([a.arm]if a.arm else['CD','BS','HO']):
            b=estimate(a.estimate,arm);print(json.dumps(b))
            for seed in ([0]if arm=='HO'else range(3)):
                print(f'qsub -l gpu_type=L40S,h_rt={b["h_rt"]} -v ARM={arm},SEED={seed},SMOKE_DIR={relative} jobs/refiner_r5.qsub')
        return
    if a.arm is None or a.seed is None or a.smoke_dir is None:p.error('Supply arm, seed and passed smoke directory')
    b=estimate(a.smoke_dir,a.arm)
    out=ROOT/'runs/refiner_r5'/f'{a.arm}_s{a.seed}'
    if out.exists():raise FileExistsError('Refusing to overwrite an existing run')
    from scripts import train_refiner as training
    from scripts import evaluate_refiner_focus as scoring
    print('SMOKE_BUDGET '+json.dumps(b),flush=True)
    sys.argv=training_command(a.arm,a.seed,out)[2:];training.main()
    sys.argv=score_command(out/'checkpoints/best.pt',out/'evaluation')[2:];scoring.main()


if __name__=='__main__':main()
