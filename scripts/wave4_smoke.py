#!/usr/bin/env python3
"""CUDA-free serial supervisor for detail gates, formal training and stress."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def commands(out):
    from scripts.wave2_smoke import stage_commands
    result={'DRG1':[sys.executable,'-u','scripts/detail_gate.py','--out-dir',str(out/'DRG1')]}
    for arm,cmd in stage_commands(ROOT/'runs/cache/tta_draft_a1i_s0',out/'base').items():
        if arm not in ('RS','RW'):continue
        base=cmd+['--d4-aug'];result['base-'+arm]=base
        detailed=[x.replace(str(out/'base'),str(out/'detail'))for x in base]+['--detail-head']
        result['detail-'+arm]=detailed
        result['score-'+arm]=[sys.executable,'-u','scripts/evaluate_refiner_focus.py','--phase','trained','--limit-per-domain','2','--checkpoint',str(out/'detail'/arm/'checkpoints/best.pt'),'--out-dir',str(out/('score-'+arm))]
    return result


def run_stages(cmds,out):
    report={}
    for name,cmd in cmds.items():
        row=dict(exit_code=None,passed=False)
        try:
            with (out/(name+'.log')).open('w')as stream:
                done=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
            row['exit_code']=done.returncode
            if done.returncode:raise ValueError('Child failed; inspect its stage log')
            if name=='DRG1':
                data=json.loads((out/'DRG1/metrics.json').read_text());assert data['DRG1']['passed']
            elif name.startswith('score-'):
                data=json.loads((out/name/'metrics.json').read_text())
                for part,result in data['splits'].items():
                    assert len(result['images'])==4
                    for image in result['images']:
                        for counts in image['core_calls'].values():assert all(int(k)==v for k,v in counts.items())
            else:
                group,arm=name.split('-');data=json.loads((out/group/arm/'report/metrics.json').read_text());batches=[b for e in data['epochs']for b in e['batches']]
                assert len(batches)==20 and all(math.isfinite(b["loss"]) for b in batches)
                row.update(updates=len(batches),batches=batches,peak_cuda_bytes=max(e['peak_cuda_bytes']for e in data['epochs']))
                assert data['config'].get('detail_head',False)==(group=='detail')
            row['passed']=True
        except Exception as error:row['error']=str(error)
        report[name]=row;(out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('STAGE '+json.dumps(dict(stage=name,exit_code=row['exit_code'],passed=row['passed'])),flush=True)
    passed=all(r['passed']for r in report.values())
    print('WAVE4_SMOKE_COMPLETE'if passed else'WAVE4_SMOKE_FAILED',flush=True)
    return passed


def main():
    import os
    p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,default=ROOT/'runs/wave4'/('smoke_'+os.environ.get('JOB_ID','manual')));a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=False)
    if not run_stages(commands(a.out_dir),a.out_dir):raise SystemExit(1)


if __name__=='__main__':main()
