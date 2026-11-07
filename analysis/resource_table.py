#!/usr/bin/env python3
"""Collect measured resources without inference; retain per-value provenance."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.domain_gating_bound import table

FAMILIES=('e0_v2','e1_reset','e2_nofb','e3_single','e4_untied','e5_noadd')


def json_line(path,keys):
    """Locate a pretty-printed object path for reviewable JSON line citations."""
    lines=Path(path).read_text().splitlines();start,stop=0,len(lines)
    for key in keys:
        marker=json.dumps(str(key))+':'
        index=next(i for i in range(start,stop) if lines[i].lstrip().startswith(marker))
        indent=len(lines[index])-len(lines[index].lstrip())
        end=next((i for i in range(index+1,stop) if lines[i].strip() and len(lines[i])-len(lines[i].lstrip())<=indent),stop)
        start,stop=index+1,end
    return index+1


def collect(accounting=None):
    report=dict(parameters=[],training=[],inference=[],gpu_hours=[],notes=[
        'GPU hours mean allocated one-GPU job wallclock, not GPU utilization. Training GPU hours use summed epoch wallclock.',
        'No time averages combine GPU types. Submission CLI phase times include feature I/O/output; stress refiner times are synchronized init+step time.',
        'CUDA peaks are allocator peaks, not nvidia-smi process memory; existing training peaks include encoder/probe/refiner/optimizer allocations.',
        'Missing measurements remain null/NA. No interpolation supplies K6.',
        'Inference measurements describe the submitted e0_v2 seed-0 model. Training totals cover all controlled experiment seeds.'])
    frozen=ROOT/'results/exp/e0_v2_s0/metrics.json'
    frozen_line=next(i for i,line in enumerate(frozen.read_text().splitlines(),1) if '"frozen_parameters":' in line)
    report['frozen_parameters']=dict(values=json.loads(frozen.read_text())['frozen_parameters'],
                                     source=f'results/exp/e0_v2_s0/metrics.json:{frozen_line} frozen_parameters')
    job_ids={}
    for family in FAMILIES:
        for seed in (*range(3), *(['0_rerun'] if family=='e0_v2' else [])):
            name=f'{family}_s{seed}'
            path=ROOT/f'results/exp/{name}/metrics.json'
            data=json.loads(path.read_text())
            log_seed='0-rerun' if seed=='0_rerun' else str(seed)
            files=sorted((ROOT/'runs/logs').glob(f'ec523-e{FAMILIES.index(family)}-s{log_seed}.o*'),key=lambda p:p.stat().st_mtime)
            log=files[-1] if files else None
            lines=log.read_text().splitlines() if log else []
            relative=str(log.relative_to(ROOT)) if log else 'NA'
            gpu=next(((line[5:],i) for i,line in enumerate(lines,1) if line.startswith('GPU: ')),('NA',None))
            epochs=[(i,json.loads(line[6:])) for i,line in enumerate(lines,1) if line.startswith('EPOCH ')]
            assert len(epochs)==len(data['epochs']),name
            for (_,a),b in zip(epochs,data['epochs']):
                assert a['seconds']==b['seconds'] and a['peak_cuda_bytes']==b['peak_cuda_bytes']
            job=log.name.rsplit('.o',1)[1] if log else None
            row=dict(run=name,experiment=family,seed=seed,rerun=seed=='0_rerun',job=job,gpu=gpu[0],
                     train_seconds=sum(e['seconds'] for _,e in epochs),
                     peak_train_cuda_bytes=max(e['peak_cuda_bytes'] for _,e in epochs),
                     job_seconds=None,allocated_gpu_hours=None,
                     train_gpu_hours=sum(e['seconds'] for _,e in epochs)/3600,
                     sources=dict(gpu=f'{relative}:{gpu[1]}',epochs=[f'{relative}:{i}' for i,_ in epochs],
                                  json=f'results/exp/{name}/metrics.json: epochs[*].seconds / peak_cuda_bytes',job='NA'))
            report['training'].append(row)
            job_ids[job]=row
            if seed==0:
                param_line=next((i for i,line in enumerate(lines,1) if line.startswith('PARAMETERS ')),None)
                if param_line is not None:assert json.loads(lines[param_line-1][11:])==data['parameters']
                report['parameters'].append(dict(experiment=family,values=data['parameters'],
                    source=f'{relative}:{param_line}',json_source=f'results/exp/{name}/metrics.json: parameters'))
    # Accounting is a scheduler log, read-only. A single streaming pass finds
    # exact line numbers without copying other users' records into the report.
    if accounting and accounting.is_file():
        wanted={j.encode() for j in job_ids if j}
        with accounting.open('rb') as handle:
            for line_no,line in enumerate(handle,1):
                prefix=line.split(b':',6)
                if len(prefix)<7 or prefix[5] not in wanted:continue
                fields=line.decode(errors='replace').rstrip().split(':')
                if len(fields)>13 and fields[5] in job_ids:
                    row=job_ids[fields[5]]
                    row.update(job_seconds=float(fields[13]),allocated_gpu_hours=float(fields[13])/3600,
                               exit_status=int(fields[12]),failed=int(fields[11]))
                    row['sources']['job']=f'SGE accounting:{line_no} job={fields[5]} fields wallclock/exit_status/failed'
    for family in FAMILIES:
        rows=[r for r in report['training'] if r['experiment']==family and not r['rerun']]
        all_rows=[r for r in report['training'] if r['experiment']==family]
        report['gpu_hours'].append(dict(experiment=family,train_gpu_hours=sum(r['train_gpu_hours'] for r in rows),
            allocated_gpu_hours=sum(r['allocated_gpu_hours'] for r in rows) if all(r['allocated_gpu_hours'] is not None for r in rows) else None,
            allocated_gpu_hours_including_rerun=sum(r['allocated_gpu_hours'] for r in all_rows) if all(r['allocated_gpu_hours'] is not None for r in all_rows) else None,
            sources=[r['sources'] for r in rows]))
    cli=ROOT/'runs/submission_check/8386872/job.log'
    for i,line in enumerate(cli.read_text().splitlines(),1):
        match=re.match(r'PHASES none (\d+) (\{.*\}) RELEASE ',line)
        if not match:continue
        k=int(match[1]);phases=json.loads(match[2]);source=f'runs/submission_check/8386872/job.log:{i}'
        run=json.loads((cli.parent/f'none_k{k}_run.json').read_text())
        images=len(run['images'])
        for phase,values in phases.items():
            report['inference'].append(dict(gpu='NVIDIA A40',source=source+'; GPU job.log:1; image count '+f'none_k{k}_run.json:images',phase=phase,K=k,images=images,
                seconds_per_image=values['seconds']/images,peak_cuda_bytes=values['peak_cuda_allocated_bytes'],
                measurement='two-phase CLI phase wall time including I/O, not pure model latency'))
    stress=json.loads((ROOT/'results/stress/v2/metrics.json').read_text())
    stress_log=sorted((ROOT/'runs/logs').glob('ec523-stress-eval.o*'))[-1]
    gpu_line=next((i,line[5:]) for i,line in enumerate(stress_log.read_text().splitlines(),1) if line.startswith('GPU: '))
    for dataset,v in stress['datasets'].items():
        if dataset!='ordinary':continue
        n=v['models']['0']['groups']['Overall']['y0_draft']['images']
        enc=v['encoding']
        report['inference'].append(dict(gpu=gpu_line[1],phase='encoding',K=None,images=n,
            seconds_per_image=enc['seconds']/n,peak_cuda_bytes=enc['peak_cuda_bytes'],
            source=f'results/stress/v2/metrics.json:{json_line(ROOT/"results/stress/v2/metrics.json",("datasets",dataset,"encoding"))} datasets.{dataset}.encoding; {stress_log.relative_to(ROOT)}:{gpu_line[0]}',
            measurement='encoding wall time includes loading and float32 cache I/O'))
        for seed,m in v['models'].items():
            if seed!='0':continue
            for k in (1,2,4,6,8):
                report['inference'].append(dict(gpu=gpu_line[1],phase='refiner',K=k,images=n,
                    seconds_per_image=m['refiner_seconds'].get(str(k))/n if str(k) in m['refiner_seconds'] else None,
                    peak_cuda_bytes=m['peak_cuda_bytes'].get(str(k)),
                    source=f'results/stress/v2/metrics.json:{json_line(ROOT/"results/stress/v2/metrics.json",("datasets",dataset,"models",seed,"refiner_seconds"))} / {json_line(ROOT/"results/stress/v2/metrics.json",("datasets",dataset,"models",seed,"peak_cuda_bytes"))} datasets.{dataset}.models.{seed}.refiner_seconds / peak_cuda_bytes; {stress_log.relative_to(ROOT)}:{gpu_line[0]}',
                    measurement='cumulative synchronized init+steps, excludes encoder/metrics/output writing'))
    return report


def fmt(x):return 'NA' if x is None else f'{x:.6f}'


def markdown(p):
    lines=['# Measured resources','',*['Observed '+s for s in p['notes']],'','## Trainable parameters','',
        table(['Experiment','Core','Feature projection','Answer projection','Initializer','Output head','Total','Source'],[
            [r['experiment'],*[r['values'][k] for k in ('core','feature_projection','answer_projection','initializer','output_head','trainable_total')],r['source']] for r in p['parameters']]),'',
        'Observed frozen parameter counts: '+str(p['frozen_parameters']['values'])+'; source '+p['frozen_parameters']['source']+'.','',
        '## Training jobs','','Observed training duration is sum of epoch seconds, excluding selection/evaluation. Each job reserves one GPU. E0 rerun is listed separately; totals explicitly distinguish the three-seed budget from the budget including rerun.','',
        table(['Run','GPU','Training seconds','Peak GiB','Whole job seconds','Allocated GPU hours','Training GPU hours','Source'],[
            [r['run'],r['gpu'],fmt(r['train_seconds']),fmt(r['peak_train_cuda_bytes']/1024**3),fmt(r['job_seconds']),fmt(r['allocated_gpu_hours']),fmt(r['train_gpu_hours']),
             r['sources']['gpu']+'; epochs '+','.join(s.rsplit(':',1)[1] for s in r['sources']['epochs'])+'; '+r['sources']['job']] for r in p['training']]),'',
        table(['Experiment','Three-seed training GPU hours','Three-seed allocated GPU hours','Allocated GPU hours including rerun'],[[r['experiment'],fmt(r['train_gpu_hours']),fmt(r['allocated_gpu_hours']),fmt(r['allocated_gpu_hours_including_rerun'])] for r in p['gpu_hours']]),'',
        '## Inference','','Observed submission-check phases and stress synchronized refinement are distinct timing scopes; no mixed-scope/GPU mean is reported. K6 was not measured in these preferred sources and remains NA.','',
        table(['GPU','Phase','K','Seconds/image','Peak GiB','Measurement scope','Source'],[
            [r['gpu'],r['phase'],r['K'] if r['K'] is not None else 'NA',fmt(r['seconds_per_image']),fmt(r['peak_cuda_bytes']/1024**3) if r['peak_cuda_bytes'] is not None else 'NA',r['measurement'],r['source']] for r in p['inference']])]
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'results/resources')
    parser.add_argument('--accounting',type=Path,default=Path(os.environ.get('SGE_ROOT','/usr/local/sge/sge_root'))/'default/common/accounting')
    args=parser.parse_args()
    if not args.out_dir.resolve().is_relative_to(ROOT):parser.error('Reports must stay inside the repository')
    report=collect(args.accounting)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for name,text in [('metrics.json',json.dumps(report,indent=2,allow_nan=False)+'\n'),('summary.md',markdown(report))]:
        with (args.out_dir/name).open('x') as f:f.write(text)
    print('COMPLETE read-only resource collection')


if __name__=='__main__':main()
