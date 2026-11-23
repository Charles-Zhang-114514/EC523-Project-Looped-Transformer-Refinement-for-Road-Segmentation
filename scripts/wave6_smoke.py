#!/usr/bin/env python3
"""CUDA-free supervisor; independent serial production-entry children."""
import argparse
import json
import math
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]


def commands(out):
    from scripts.train_refiner_r6 import training_command,score_command
    result={}
    for arm in ('P8','W','TVS'):
        result[arm]=training_command(arm,0,out/arm,smoke=True)
        result['score-'+arm]=score_command(arm,out/arm,out/('score-'+arm),smoke=True)
    return result


def validate_training(arm,directory):
    d=json.loads((directory/'report/metrics.json').read_text())
    batches=[b for ep in d['epochs'] for b in ep['batches']]
    if len(batches)!=20 or not all(math.isfinite(b['loss']) for b in batches):
        raise ValueError('Twenty finite updates required')
    if d['evaluation_corruption_applied'] is not False:raise ValueError('CG1 failed')
    row=dict(updates=len(batches),loss_range=[min(b['loss'] for b in batches),max(b['loss'] for b in batches)],
             evaluation_corruption_applied=False)
    rows=[r for b in batches for r in b['corruption']]
    row.update(corruption_selected=sum(r['selected'] for r in rows),samples=len(rows),
               mean_CPU_corruption_seconds=sum(r['corruption_seconds'] for r in rows)/len(rows))
    if arm=='TVS':
        info=d['training_protocol']
        if (not info['passed'] or set(info['training_ids'])&set(info['valrep_ids'])
                or set(info['training_ids'])!=set(info['train_ids'])|set(info['valsel_ids'])):
            raise ValueError('TG1 pool identity assertion failed')
        if (d['evaluated_epoch']!=2 or d['checkpoint_source']!='epoch_002.pt'
                or d['selection_curve'] or d['selection_files']):
            raise ValueError('TVS did not adopt the predeclared checkpoint')
        if len(d['descriptive_curve'])!=2 or any(not x['descriptive_only'] or x['evaluation_corruption_applied'] for x in d['descriptive_curve']):
            raise ValueError('Two uncorrupted descriptive epochs required')
        for epoch in (1,2):
            if not (directory/'checkpoints'/f'epoch_{epoch:03d}.pt').is_file():raise ValueError('Missing epoch checkpoint')
        row.update(TG1=True,training_count=info['training_count'],valrep_intersection=0,fixed_epoch=2,
                   saved_epochs=[1,2],updates_by_epoch=[len(e['batches']) for e in d['epochs']],
                   descriptive_epochs=2,selection_disabled=True)
    else:
        if not d['selection_curve'] or any(set(x['L_by_domain'])!={'Overall','Urban','Rural'} for x in d['selection_curve']):
            raise ValueError('Per-domain Val-sel selection required')
        row['best_epoch']=d['best_epoch']
    return row


def validate_score(directory):
    d=json.loads((directory/'metrics.json').read_text());endpoints=0
    if d['evaluation_corruption_applied'] is not False:raise ValueError('CG1 scoring failed')
    for result in d['splits'].values():
        if len(result['images'])!=4:raise ValueError('Four images per split required')
        for image in result['images']:
            for calls in image['core_calls'].values():
                if any(int(k)!=v for k,v in calls.items()):raise ValueError('KG core count differs from K')
                endpoints+=len(calls)
        for domains in result['groups'].values():
            for group in domains.values():
                for item in group.values():
                    if not all(math.isfinite(item['scores'][k]) for k in ('road_iou','cldice','boundary_iou','quality')):
                        raise ValueError('Nonfinite official score')
    return dict(core_count_checks=endpoints,clean_images=4,stress_images=4,
                evaluation_corruption_applied=False,checkpoint=d['checkpoint'])


def run_stages(cmds,out):
    report={}
    for name,cmd in cmds.items():
        row=dict(exit_code=None,passed=False)
        try:
            with (out/(name+'.log')).open('w') as stream:
                done=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
            row['exit_code']=done.returncode
            if done.returncode:raise ValueError('Child failed; inspect its stage log')
            if name.startswith('score-'):
                row.update(validate_score(out/name))
                if name=='score-TVS' and row['checkpoint']!='epoch_002.pt':raise ValueError('Scoring bypassed fixed checkpoint')
            else:row.update(validate_training(name,out/name))
            row['passed']=True
        except Exception as error:row['error']=str(error)
        report[name]=row;(out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('STAGE '+json.dumps(dict(stage=name,**row)),flush=True)
    passed=all(r['passed'] for r in report.values())
    print('WAVE6_SMOKE_COMPLETE' if passed else 'WAVE6_SMOKE_FAILED',flush=True)
    return passed


def main():
    import os
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,default=ROOT/'runs/wave6'/('smoke_'+os.environ.get('JOB_ID','manual')))
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=False)
    if not run_stages(commands(a.out_dir),a.out_dir):raise SystemExit(1)


if __name__=='__main__':main()
