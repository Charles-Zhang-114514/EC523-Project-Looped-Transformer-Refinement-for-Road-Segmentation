#!/usr/bin/env python3
"""CUDA-free supervisor; six independent, serial production-entry children."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def commands(out):
    from scripts.train_refiner_r5 import training_command,score_command
    result={}
    for arm in ['CD','BS','HO']:
        result[arm]=training_command(arm,0,out/arm,smoke=True)
        result['score-'+arm]=score_command(out/arm/'checkpoints/best.pt',out/('score-'+arm),smoke=True)
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
            if name.startswith('score-'):
                d=json.loads((out/name/'metrics.json').read_text());assert d['evaluation_corruption_applied']is False
                for part,result in d['splits'].items():
                    assert len(result['images'])==4
                    for image in result['images']:
                        for calls in image['core_calls'].values():assert all(int(k)==v for k,v in calls.items())
                        for maxima in image.get('delta_maxima',{}).values():assert all(v<=1. for v in maxima)
                row['evaluation_corruption_applied']=False
            else:
                d=json.loads((out/name/'report/metrics.json').read_text());bs=[b for e in d['epochs']for b in e['batches']]
                assert len(bs)==20 and all(math.isfinite(b['loss'])for b in bs)
                assert all(set(p['L_by_domain'])=={'Overall','Urban','Rural'}for p in d['selection_curve'])
                row.update(updates=20,loss_range=[min(b['loss']for b in bs),max(b['loss']for b in bs)])
                if name=='CD':
                    rows=[r for b in bs for r in b['corruption']];assert len(rows)==80
                    assert d['evaluation_corruption_applied']is False
                    assert 0<sum(r['selected']for r in rows)<len(rows) and sum(r['changed_pixels']for r in rows)>0
                    row.update(corruption=rows,samples=len(rows),selected=sum(r['selected']for r in rows),
                               changed_pixels=sum(r['changed_pixels']for r in rows),selected_fraction=sum(r['selected']for r in rows)/len(rows),
                               gap_count=sum(r['gap_count']for r in rows),false_block_count=sum(r['false_block_count']for r in rows),evaluation_corruption_applied=False)
                elif name=='BS':
                    maxima=[v for b in bs for v in b['delta_maxima']];assert len(maxima)==80 and max(maxima)<=1.
                    row.update(maximum_absolute_delta=max(maxima),delta_checks=len(maxima))
                else:
                    h=d['holdout'];train={n for ns in h['train_ids'].values()for n in ns};held={n for ns in h['holdout_ids'].values()for n in ns}
                    assert not train&held and len(held)==4 and len(h['curve'])==1
                    row.update(train_images=len(train),holdout_images=len(held),disjoint=True)
            row['passed']=True
        except Exception as error:row['error']=str(error)
        report[name]=row;(out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('STAGE '+json.dumps({k:v for k,v in dict(stage=name,**row).items()if k!='corruption'}),flush=True)
    passed=all(r['passed']for r in report.values())
    print('WAVE5_SMOKE_COMPLETE'if passed else'WAVE5_SMOKE_FAILED',flush=True)
    return passed


def main():
    import os
    p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,default=ROOT/'runs/wave5'/('smoke_'+os.environ.get('JOB_ID','manual')));a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=False)
    if not run_stages(commands(a.out_dir),a.out_dir):raise SystemExit(1)


if __name__=='__main__':main()
