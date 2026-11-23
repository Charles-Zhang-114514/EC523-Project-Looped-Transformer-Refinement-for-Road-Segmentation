#!/usr/bin/env python3
"""CPU-only coordinator; serial, independent allocated-GPU child stages."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def commands(out):
    result={}
    for phase in ['sg1','S5a','S5b','dg1']:
        result[phase]=[sys.executable,'-u','scripts/evaluate_refiner_focus.py','--phase',phase,
                       '--limit-per-domain','2','--out-dir',str(out/phase)]
    # Reuse the production CLI arguments from the repaired wave-2 coordinator.
    from scripts.wave2_smoke import stage_commands
    for arm,cmd in stage_commands(ROOT/'runs/cache/tta_draft_a1i_s0',out).items():
        if arm not in ('RS','RW'):continue
        result[arm+'-D4']=cmd+['--d4-aug']
    result['stress']=[sys.executable,'-u','scripts/evaluate_refiner_focus.py','--phase','stress',
                      '--limit-per-domain','2','--out-dir',str(out/'stress')]
    return result


def run_stages(cmds,out):
    report={}
    for name,cmd in cmds.items():
        row=dict(exit_code=None,passed=False)
        try:
            with (out/(name+'.log')).open('w') as stream:
                done=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
            row=dict(exit_code=done.returncode,passed=done.returncode==0)
            if row['passed']:
                if name in ('RS-D4','RW-D4'):
                    data=json.loads((out/name[:2]/'report/metrics.json').read_text())
                    row['updates']=sum(len(e['batches'])for e in data['epochs'])
                    assert row['updates']==20 and data['config']['d4_aug']
                    assert len(data['evaluation']['images'])==4
                else:
                    data=json.loads((out/name/'metrics.json').read_text())
                    if name=='dg1':assert data['DG1']['passed'] and len(data['DG1']['checks'])==32
                    else:
                        images=data['splits']['stress'if name=='stress'else'Val-rep']['images']
                        assert len(images)==4
                        for image in images:
                            for calls in image['core_calls'].values():
                                assert all(int(k)==v for k,v in calls.items())
                        if name=='sg1':assert data['SG1']['passed']
        except Exception as error:
            row=dict(exit_code=row.get('exit_code'),passed=False,error=str(error))
        report[name]=row
        print('STAGE '+json.dumps(dict(stage=name,**row)),flush=True)
        (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    passed=all(r['passed']for r in report.values())
    print('WAVE3_SMOKE_COMPLETE'if passed else'WAVE3_SMOKE_FAILED',flush=True)
    return passed


def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,default=ROOT/'runs/wave3'/('smoke_'+__import__('os').environ.get('JOB_ID','manual')));a=p.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=False)
    if not run_stages(commands(a.out_dir),a.out_dir):raise SystemExit(1)


if __name__=='__main__':
    main()
