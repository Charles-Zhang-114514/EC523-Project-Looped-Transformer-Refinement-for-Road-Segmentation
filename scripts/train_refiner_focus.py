#!/usr/bin/env python3
"""Run formal D4 training then clean/stress reporting in the same CUDA process."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['RS','RW'],required=True);p.add_argument('--seed',type=int,choices=range(3),required=True);a=p.parse_args()
    from scripts import train_refiner as training
    from scripts import evaluate_refiner_focus as evaluation
    out=ROOT/'runs/refiner_focus'/f'{a.arm}_D4_s{a.seed}'
    rs=a.arm=='RS'
    sys.argv=['train_refiner.py','--d4-aug','--draft-cache','runs/cache/tta_draft_a1i_s0',
              '--probe-ckpt','runs/draft_head/A1-i/best.pt','--head-type','conv_upsample',
              '--epochs','10'if rs else'3','--lr','3e-4'if rs else'1e-4',
              '--warmup-steps','500'if rs else'100','--batch-size','4','--k-train','4',
              '--pos-weight','4','--seed',str(a.seed),'--device','cuda','--select-per-domain','all',
              '--select-every','1','--selection-ks','1:0.2,2:0.2,4:0.6',
              '--val-split','configs/val_split_seed0.json','--eval-k','1,2,4,8',
              '--out-dir',str(out/'report'),'--ckpt-dir',str(out/'checkpoints')]
    if not rs:sys.argv+=['--init-refiner','runs/exp/e6_convdraft_s0/best.pt']
    training.main()
    sys.argv=['evaluate_refiner_focus.py','--phase','trained','--checkpoint',str(out/'checkpoints/best.pt'),'--out-dir',str(out/'evaluation')]
    evaluation.main()


if __name__=='__main__':main()
