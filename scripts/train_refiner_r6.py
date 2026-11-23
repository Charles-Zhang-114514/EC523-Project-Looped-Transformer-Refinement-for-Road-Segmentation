#!/usr/bin/env python3
"""Wave 6 formal commands and L40S budgets; one CUDA PID."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def training_command(arm,seed,out,smoke=False):
    from scripts.train_refiner_r5 import training_command as previous
    cmd=previous('CD',seed,out,smoke=smoke)
    def setting(key,value):cmd[cmd.index(key)+1]=str(value)
    setting('--draft-corrupt',.8 if arm=='P8' else .5)
    if arm=='W':
        setting('--epochs',3);setting('--lr',1e-4);setting('--warmup-steps',100)
        cmd+=['--init-refiner','runs/exp/e6_convdraft_s0/best.pt']
    elif arm=='TVS':
        cmd+=['--train-include-valsel','--fixed-epoch','2']
        if smoke:setting('--epochs',2)
    elif arm!='P8':raise ValueError('Unknown wave6 arm')
    if smoke:setting('--limit-per-domain',2)
    return cmd


def checkpoint_for(arm,out):
    return out/'checkpoints'/('epoch_002.pt' if arm=='TVS' else 'best.pt')


def score_command(arm,out,destination,smoke=False):
    from scripts.train_refiner_r5 import score_command as previous
    return previous(checkpoint_for(arm,out),destination,smoke)


def require_smoke(directory):
    data=json.loads((directory/'metrics.json').read_text())
    expected={name for arm in ('P8','W','TVS') for name in (arm,'score-'+arm)}
    if set(data)!=expected or not all(row['passed'] for row in data.values()):
        raise ValueError('Complete passed wave6 smoke required')


def estimate(arm):
    # Actual L40S observations; no login-node throughput extrapolation.
    prior=json.loads((ROOT/'runs/wave5/smoke_8821468/metrics.json').read_text())
    cpu=statistics.mean(row['corruption_seconds'] for row in prior['CD']['corruption'])
    if not math.isfinite(cpu) or cpu<0:raise ValueError('Invalid archived CPU timing')
    if arm=='P8':
        baseline,extra=61*60.,2522*10*cpu*(.8/.5-1)
        explanation='61 min CD plus conservative extra CPU work for P=.8 instead of .5'
    elif arm=='W':
        baseline,extra=25*60.,2522*3*cpu
        explanation='25 min RW-D4 plus conservative CPU corruption work for three epochs'
    elif arm=='TVS':
        baseline=61*60.*(3356/2522)
        extra=10*835*(.04+4*.012+4*.125)
        explanation=('Scale the entire CD job by 3356/2522, add all ten descriptive Val-rep '
                     'passes including draft/K1/K2/K4, and conservatively retain the old selection allowance.')
    else:raise ValueError('Unknown arm')
    seconds=math.ceil((1.3*(baseline+extra)+300)/60)*60
    return dict(arm=arm,baseline_seconds=baseline,added_seconds=extra,
                mean_CPU_corruption_seconds=cpu,explanation=explanation,
                formula='ceil_to_minute(1.3*(baseline+added)+300)',seconds=seconds,
                h_rt=f'{seconds//3600:02}:{seconds%3600//60:02}:{seconds%60:02}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['P8','W','TVS']);p.add_argument('--seed',type=int,choices=range(3))
    p.add_argument('--smoke-dir',type=Path);p.add_argument('--estimate',action='store_true');a=p.parse_args()
    if a.estimate:
        for arm in ([a.arm] if a.arm else ['P8','W','TVS']):print(json.dumps(estimate(arm)),flush=True)
        return
    if a.arm is None or a.seed is None or a.smoke_dir is None:p.error('Supply arm, seed and passed wave6 smoke directory')
    require_smoke(a.smoke_dir)
    out=ROOT/'runs/refiner_r6'/f'{a.arm}_s{a.seed}'
    if out.exists():raise FileExistsError('Refusing to overwrite an existing run')
    print('GPU_BUDGET '+json.dumps(estimate(a.arm)),flush=True)
    from scripts import train_refiner as training
    from scripts import evaluate_refiner_focus as scoring
    sys.argv=training_command(a.arm,a.seed,out)[2:];training.main()
    sys.argv=score_command(a.arm,out,out/'evaluation')[2:];scoring.main()


if __name__=='__main__':main()
