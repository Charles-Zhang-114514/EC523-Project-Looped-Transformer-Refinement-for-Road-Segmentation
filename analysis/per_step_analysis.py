#!/usr/bin/env python3
"""Frozen, consecutive-update diagnostics with official pooled metrics.

GT boundary means inner four-neighbor boundary pixels, including the image
edge (erosion border=0). Distances are Euclidean distances between pixel
centers. Ignore pixels never contribute to any denominator. A repeated
addition after deletion counts again: this audit measures update events.
"""
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import statistics
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import eval_stress as shared
from scripts import train_refiner as runner
from analysis import refiner_diagnostics as diagnostic
from analysis.domain_gating_bound import table
from submission import predict as inference

REGIONS=('R1','R2','R3','R4')
FIELDS=('pixels','added','deleted','sum_abs_probability_change','change_over_005',
        'no_flip_pixels','no_flip_sum_abs','no_flip_over_005',
        'below_threshold_pixels','below_threshold_sum_abs','below_threshold_over_005')
CHANGES=('added_correct','added_wrong','deleted_wrong','deleted_correct')
TRACK=('gap_total','gap_covered','covered_total','covered_lost')
_METRICS=None


def assign_regions(labels):
    """Disjoint/exhaustive valid-pixel regions; empty GT has no phantom edge."""
    valid=labels!=0;truth=labels==3
    regions=np.zeros(labels.shape,np.uint8)
    if not truth.any():
        regions[valid]=4
        return regions
    boundary=truth & ~ndi.binary_erosion(truth,structure=ndi.generate_binary_structure(2,1),border_value=0)
    to_boundary=ndi.distance_transform_edt(~boundary)
    to_road=ndi.distance_transform_edt(~truth)
    masks=(valid & (to_boundary<=3),valid & truth & (to_boundary>3),
           valid & ~truth & (to_road>3) & (to_road<=16),valid & ~truth & (to_road>16))
    coverage=np.zeros(labels.shape,np.uint8)
    for i,mask in enumerate(masks,1):regions[mask]=i;coverage+=mask
    assert np.array_equal(coverage,valid.astype(np.uint8))
    return regions


def step_changes(previous,current,labels,regions):
    """Raw micro-statistics for consecutive probabilities; do not quantize."""
    valid=labels!=0;truth=labels==3
    before=previous>=.5;after=current>=.5
    added=~before & after & valid;deleted=before & ~after & valid
    delta=np.abs(current-previous)
    changes=diagnostic.change_counts(before,after,truth,valid)
    output=dict(**changes,added=int(added.sum()),deleted=int(deleted.sum()),
                no_gt_added=int(added.sum()) if not truth.any() else 0,regions={})
    def count(mask):return np.bincount(regions[mask],minlength=5)
    def weighted(mask):return np.bincount(regions[mask],weights=delta[mask],minlength=5)
    no_flip=valid & (before==after)
    below=valid & ~before & ~after
    counts={'pixels':count(valid),'added':count(added),'deleted':count(deleted),
            'sum_abs_probability_change':weighted(valid),'change_over_005':count(valid & (delta>np.float32(.05))),
            'no_flip_pixels':count(no_flip),'no_flip_sum_abs':weighted(no_flip),
            'no_flip_over_005':count(no_flip & (delta>np.float32(.05))),
            'below_threshold_pixels':count(below),'below_threshold_sum_abs':weighted(below),
            'below_threshold_over_005':count(below & (delta>np.float32(.05)))}
    for i,name in enumerate(REGIONS,1):
        output['regions'][name]={f:(float(value[i]) if 'sum_abs' in f else int(value[i])) for f,value in counts.items()}
    assert sum(v['added'] for v in output['regions'].values())==output['added']
    assert sum(v['deleted'] for v in output['regions'].values())==output['deleted']
    assert sum(v['pixels'] for v in output['regions'].values())==int(valid.sum())
    return output


def initialize_worker(starter):
    global _METRICS
    shared.initialize_worker(starter)
    _METRICS=runner.probe.official_metrics(starter)


def process_image(row,draft,answers):
    """One label read/region partition/skeleton definition for all model seeds."""
    started=time.perf_counter()
    with Image.open(row['label']) as source:labels=np.array(source,copy=True)
    phase=time.perf_counter();regions=assign_regions(labels);region_seconds=time.perf_counter()-phase
    valid=labels!=0;truth=labels==3;base=(draft>=.5)&valid
    phase=time.perf_counter();gaps,_,covered=diagnostic.baseline_skeleton_sets(truth,base)
    skeleton_seconds=time.perf_counter()-phase
    phase=time.perf_counter();baseline=asdict(_METRICS.image_counts(draft,labels));metric_seconds=time.perf_counter()-phase
    results={};change_seconds=0.
    for seed,sequence in answers.items():
        previous=draft;steps={}
        for t,current in enumerate(sequence,1):
            phase=time.perf_counter();values=step_changes(previous,current,labels,regions)
            mask=(current>=.5)&valid
            values['tracking']=dict(gap_total=int(gaps.sum()),gap_covered=int((gaps & mask).sum()),
                                    covered_total=int(covered.sum()),covered_lost=int((covered & ~mask).sum()))
            change_seconds+=time.perf_counter()-phase
            phase=time.perf_counter();values['counts']=asdict(_METRICS.image_counts(current,labels))
            metric_seconds+=time.perf_counter()-phase
            steps[str(t)]=values;previous=current
        results[seed]=dict(y0_counts=baseline,steps=steps)
    timing=dict(region_seconds=region_seconds,skeleton_seconds=skeleton_seconds,change_seconds=change_seconds,
                metric_seconds=metric_seconds,worker_seconds=time.perf_counter()-started)
    return row['domain'].title(),int(valid.sum()),int(not truth.any()),results,timing


def empty_bucket(metrics,steps):
    return dict(images=0,valid_pixels=0,no_gt_images=0,y0_counts=asdict(metrics.Counts()),steps={str(t):dict(
        **{f:0 for f in (*CHANGES,'added','deleted','no_gt_added')},counts=asdict(metrics.Counts()),
        tracking=dict.fromkeys(TRACK,0),regions={name:dict.fromkeys(FIELDS,0) for name in REGIONS}) for t in range(1,steps+1)})


def add_bucket(target,source):
    """Sum raw sufficient statistics in submission order."""
    for f in ('images','valid_pixels','no_gt_images'):target[f]+=source[f]
    for f,n in source['y0_counts'].items():target['y0_counts'][f]+=n
    for t,step in source['steps'].items():
        out=target['steps'][t]
        for f in (*CHANGES,'added','deleted','no_gt_added'):out[f]+=step[f]
        for section in ('counts','tracking'):
            for f,n in step[section].items():out[section][f]+=n
        for name,region in step['regions'].items():
            for f,n in region.items():out['regions'][name][f]+=n


def finalize(bucket,metrics):
    bucket['y0_scores']=metrics.summarize(metrics.Counts(**bucket['y0_counts']))
    previous=bucket['y0_scores']
    for step in bucket['steps'].values():
        step['scores']=metrics.summarize(metrics.Counts(**step['counts']))
        step['metric_changes']={f:step['scores'][f]-previous[f] for f in ('road_iou','cldice','boundary_iou','quality')}
        previous=step['scores']
        step['added_valid']=diagnostic.fraction(step['added'],bucket['valid_pixels'])
        step['deleted_valid']=diagnostic.fraction(step['deleted'],bucket['valid_pixels'])
        tracking=step['tracking'];step['gap_repair']=diagnostic.fraction(tracking['gap_covered'],tracking['gap_total'])
        step['new_break']=diagnostic.fraction(tracking['covered_lost'],tracking['covered_total'])
        for region in step['regions'].values():
            region['mean_abs_probability_change']=diagnostic.fraction(region['sum_abs_probability_change'],region['pixels'])
            region['fraction_over_005']=diagnostic.fraction(region['change_over_005'],region['pixels'])
            region['no_flip_mean_abs']=diagnostic.fraction(region['no_flip_sum_abs'],region['no_flip_pixels'])
            region['no_flip_fraction_over_005']=diagnostic.fraction(region['no_flip_over_005'],region['no_flip_pixels'])
            region['below_threshold_mean_abs']=diagnostic.fraction(region['below_threshold_sum_abs'],region['below_threshold_pixels'])
            region['below_threshold_fraction_over_005']=diagnostic.fraction(region['below_threshold_over_005'],region['below_threshold_pixels'])
        assert sum(r['added'] for r in step['regions'].values())==step['added']
        assert sum(r['deleted'] for r in step['regions'].values())==step['deleted']
    return bucket


def preregistration():
    text=(ROOT/'notes/worklog.md').read_text()
    block=text.split('## 2026-11-09 - Per-step analysis preregistration',1)[1].split('\n## ',1)[0].strip()
    stamp=re.search(r'Recorded at ([^,]+),',block)[1]
    return dict(recorded_at=stamp,sha256=hashlib.sha256(block.encode()).hexdigest(),source='notes/worklog.md')


def prediction_checks(report):
    full=report['config']['limit_per_domain'] is None and report['config']['steps']==8
    result={}
    for seed,groups in report['seeds'].items():
        steps=groups['Overall']['steps'];first=steps['1'];total=sum(s['added'] for s in steps.values())
        p1=diagnostic.fraction(first['added'],total)
        p2=diagnostic.fraction(first['regions']['R1']['added'],first['added'])
        maximum=max((abs(s['metric_changes']['boundary_iou']) for t,s in steps.items() if int(t)>=2),default=None)
        result[seed]=dict(scope='full Val-rep' if full else 'limited smoke: full-data predictions not adjudicated',
            first_step_added_share=p1,first_step_R1_added_share=p2,max_later_abs_delta_BIoU=maximum,
            P1=(p1>=.8 if p1 is not None else None) if full else None,
            P2=(p2>=.4 if p2 is not None else None) if full else None,
            P3=(maximum<.001 if maximum is not None else None) if full else None,
            first_step_region_added_fractions={r:diagnostic.fraction(v['added'],first['added']) for r,v in first['regions'].items()},
            later_boundary_below_threshold={t:{f:s['regions']['R1'][f] for f in
                ('below_threshold_pixels','below_threshold_mean_abs','below_threshold_fraction_over_005')}
                for t,s in steps.items() if int(t)>=2})
        result[seed]['P1_and_P3']=(result[seed]['P1'] and result[seed]['P3']) if full and p1 is not None and maximum is not None else None
    return result


def mean_sd(values,scale=1):
    vals=[float(v)*scale for v in values if v is not None]
    if not vals:return '--'
    return f'{statistics.mean(vals):.6f} +/- '+(f'{statistics.stdev(vals):.6f}' if len(vals)>1 else '--')


def markdown(report):
    lines=['# Per-step v2 analysis','',
      'Observed scope: '+json.dumps(report['config'])+'.','',
      'Observed region definition: inner four-neighbor GT-road boundary, erosion border=0, Euclidean pixel-center distances. '
      'R1 distance<=3 inside/outside; R2 road interior >3; R3 background 3<distance-to-road<=16; R4 background >16. '
      'Empty GT assigns every valid pixel to R4. Ignore regions do not contribute to any denominator.', '',
      'Observed additions/deletions are consecutive-step events, not cumulative flips from y0. Gap repair/new break use fixed y0 skeleton sets. '
      'Official Counts are summed before summarize; metric deltas subtract the previous aggregate scores. '
      'Seed means/sample deviations use ddof=1, omitting undefined values; singleton deviation is --.', '',
      'Observed probability changes below are micro-averaged over every valid pixel in the region, including flips. '
      'Separate no-flip and both-probabilities-below-0.5 statistics are also reported. No probability maps or per-image files are saved.','']
    for domain in ('Overall','Urban','Rural'):
        groups=[r[domain] for r in report['seeds'].values()]
        lines += ['## Observed '+domain,'',
                  'Observed cells: addition/deletion count then valid-pixel percent; region flip cells add/delete; probability cells mean absolute change then percent >0.05.','']
        rows=[];score_rows=[];sub_rows=[]
        score_rows.append(['y0',*[mean_sd([g['y0_scores'][f] for g in groups]) for f in ('road_iou','cldice','boundary_iou','quality')],'--','--','--','--'])
        for t in range(1,report['config']['steps']+1):
            values=[g['steps'][str(t)] for g in groups]
            row=[t,mean_sd([s['added'] for s in values])+'; '+mean_sd([s['added_valid'] for s in values],100),
                 mean_sd([s['deleted'] for s in values])+'; '+mean_sd([s['deleted_valid'] for s in values],100)]
            row += [mean_sd([s['regions'][r]['added'] for s in values])+' / '+mean_sd([s['regions'][r]['deleted'] for s in values]) for r in REGIONS]
            row += [mean_sd([s['regions'][r]['mean_abs_probability_change'] for s in values])+' / '+mean_sd([s['regions'][r]['fraction_over_005'] for s in values],100) for r in REGIONS]
            row += [mean_sd([s['metric_changes'][f] for s in values]) for f in ('road_iou','cldice','boundary_iou','quality')]
            row += [mean_sd([s['gap_repair'] for s in values],100),mean_sd([s['new_break'] for s in values],100),mean_sd([s['no_gt_added'] for s in values])]
            rows.append(row)
            score_rows.append([t,*[mean_sd([s['scores'][f] for s in values]) for f in ('road_iou','cldice','boundary_iou','quality')],
                               *[mean_sd([s[f] for s in values]) for f in CHANGES]])
            sub_rows.append([t,*[mean_sd([s['regions'][r]['no_flip_mean_abs'] for s in values])+' / '+mean_sd([s['regions'][r]['no_flip_fraction_over_005'] for s in values],100) for r in REGIONS],
                              *[mean_sd([s['regions'][r]['below_threshold_mean_abs'] for s in values])+' / '+mean_sd([s['regions'][r]['below_threshold_fraction_over_005'] for s in values],100) for r in REGIONS]])
        lines += [table(['t','Added; %valid','Deleted; %valid',*[r+' add/delete' for r in REGIONS],*[r+' mean|dp| / >.05%' for r in REGIONS],
                        'dIoU','dclDice','dBIoU','dA','Gap repair %','New break %','No-GT additions'],rows),'',
                  table(['t','IoU','clDice','BIoU','A','Added correct','Added wrong','Deleted wrong','Deleted correct'],score_rows),'',
                  table(['t',*[r+' no-flip mean / >.05%' for r in REGIONS],*[r+' both below .5 mean / >.05%' for r in REGIONS]],sub_rows),'']
    lines += ['## Observed preregistered P1-P3 checks','',
      'Observed preregistration recorded at '+report['preregistration']['recorded_at']+' before the full run. '
      'P1: first-step share>=80%; P2: first-step R1 share>=40%; P3: every later |delta BIoU|<0.001. '
      'Each trained seed is checked separately. Limited smoke outputs do not adjudicate these full-data predictions.','',
      table(['Seed','First-step added share','P1','First-step R1 share','P2','Max later |dBIoU|','P3','P1 and P3','Scope'],[
          [s,v['first_step_added_share'],v['P1'],v['first_step_R1_added_share'],v['P2'],v['max_later_abs_delta_BIoU'],v['P3'],v['P1_and_P3'],v['scope']] for s,v in report['predictions'].items()]),'',
      'Observed actual step-1 addition distribution by region:', '',
      table(['Seed',*REGIONS],[[seed,*[v['first_step_region_added_fractions'][r] for r in REGIONS]] for seed,v in report['predictions'].items()]),'',
      'Observed the supplied decision rule uses P1 and P3 jointly and asks for boundary-loss priority if later boundary sub-threshold activity is obvious. '
      'The word obvious has no preregistered population cutoff; the measured below-threshold R1 statistics are provided without an invented binary activity claim. '
      'No causal or training recommendation is inferred from this report.']
    for seed,v in report['predictions'].items():
        if v['P1'] is None and v['P2'] is None and v['P3'] is None:
            lines += ['',f'Observed seed {seed}: P1, P2 and P3 remain unadjudicated on this limited smoke scope.']
        else:
            lines += ['',f"Observed P1, seed {seed}: first-step added-event share={v['first_step_added_share']}; criterion >=0.80; result={v['P1']}.",
                      f"Observed P2, seed {seed}: step-1 R1 addition share={v['first_step_R1_added_share']}; criterion >=0.40; result={v['P2']}. The actual R1-R4 distribution is shown above.",
                      f"Observed P3, seed {seed}: maximum later absolute Overall BIoU change={v['max_later_abs_delta_BIoU']}; criterion <0.001; result={v['P3']}.",
                      f"Observed preregistered joint P1-and-P3 condition, seed {seed}: {v['P1_and_P3']}."]
    return '\n'.join(lines)+'\n'


def reference_checks(report):
    checks=[]
    if report['config']['limit_per_domain'] is not None or '0' not in report['seeds']:
        print('SKIP full-Val-rep historical regression for limited scope',flush=True);return checks
    old=json.loads((ROOT/'results/exp/e0_v2_s0/metrics.json').read_text())['evaluation']['groups']['Overall']
    for t in (1,2,4,8):
        if t>report['config']['steps']:continue
        value=report['seeds']['0']['Overall']['steps'][str(t)]['scores']['quality'];reference=old[str(t)]['scores']['quality']
        passed=abs(value-reference)<=1e-5
        checks.append(dict(t=t,actual=value,reference=reference,difference=value-reference,passed=passed))
        print(f'{"PASS" if passed else "FAIL"} seed0 t={t} Overall_A={value:.12f} reference={reference:.12f} delta={value-reference:+.3e}',flush=True)
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'results/per_step/v2')
    parser.add_argument('--val-split',type=Path,default=ROOT/'configs/val_split_seed0.json')
    parser.add_argument('--encoder-dir',type=Path,default=ROOT/'submission/encoder')
    parser.add_argument('--probe',type=Path,default=ROOT/'submission/baseline.safetensors')
    parser.add_argument('--model-seeds',default='0,1,2')
    parser.add_argument('--steps',type=int,default=8)
    parser.add_argument('--limit-per-domain',type=int)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--loader-workers',type=int,default=1)
    parser.add_argument('--metric-workers',type=int,default=6)
    args=parser.parse_args();seed_ids=args.model_seeds.split(',')
    if not seed_ids or len(set(seed_ids))!=len(seed_ids) or any(s not in ('0','1','2') for s in seed_ids):parser.error('Use unique seeds 0,1,2')
    if not 1<=args.steps<=8:parser.error('Steps must be 1..8')
    if args.limit_per_domain is not None and args.limit_per_domain<1:parser.error('Limit must be positive')
    if not args.out_dir.resolve().is_relative_to(ROOT):parser.error('Reports must stay in the repository')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):parser.error('Output must be absent or empty')
    slots=int(os.environ.get('NSLOTS',0))
    if args.loader_workers<0 or args.metric_workers<1 or (slots and 1+args.loader_workers+args.metric_workers>slots):parser.error('Loader + metrics + main exceeds slots')
    pre=preregistration();runner.probe.initialize_data_worker(0);device=torch.device(args.device)
    starter=os.environ['EC523_STARTER_DIR'];metrics=runner.probe.official_metrics(starter)
    rows=shared.build_manifest(args.val_split,limit_per_domain=args.limit_per_domain)
    inference.verify_encoder(args.encoder_dir);baseline=inference.load_baseline(args.probe,args.encoder_dir,device)
    models={};checkpoints={}
    for seed in seed_ids:
        path=ROOT/f'runs/exp/e0_v2_s{seed}/best.pt';model,epoch=shared.load_model(path,device,inference.sha256(args.val_split))
        if epoch!=json.loads((ROOT/f'results/exp/e0_v2_s{seed}/metrics.json').read_text())['best_epoch']:raise ValueError('Checkpoint epoch mismatch')
        models[seed]=model;checkpoints[seed]=dict(epoch=epoch,sha256=inference.sha256(path),path=f'runs/exp/e0_v2_s{seed}/best.pt')
    buckets={s:{d:empty_bucket(metrics,args.steps) for d in ('Urban','Rural')} for s in seed_ids}
    timing=dict(encoder_probe_seconds=0.,refiner_seconds={s:{str(t):0. for t in range(1,args.steps+1)} for s in seed_ids},
                region_seconds=0.,skeleton_seconds=0.,change_seconds=0.,metric_seconds=0.,worker_seconds=0.)
    kwargs=dict(batch_size=1,num_workers=args.loader_workers,collate_fn=shared.image_batch,
                worker_init_fn=runner.probe.initialize_data_worker if args.loader_workers else None)
    if args.loader_workers:kwargs['multiprocessing_context']='spawn'
    loader=DataLoader(shared.ImageInputs(rows),**kwargs);pending=deque();started=time.perf_counter()
    def consume():
        domain,valid,no_gt,values,times=pending.popleft().get()
        for seed,result in values.items():add_bucket(buckets[seed][domain],dict(images=1,valid_pixels=valid,no_gt_images=no_gt,**result))
        for field,value in times.items():timing[field]+=value
    with mp.get_context('spawn').Pool(args.metric_workers,initializer=initialize_worker,initargs=(starter,)) as pool:
        for indices,images in loader:
            index=indices[0];phase=runner.stamp(device)
            features,draft=runner.frozen_draft(baseline.encoder,baseline.probe,torch.from_numpy(images).to(device))
            timing['encoder_probe_seconds']+=runner.stamp(device)-phase
            inference.validate_logits(draft,(1,1,1024,1024));probability=draft.sigmoid()[0,0].float().cpu().numpy()
            answers={}
            for seed,model in models.items():
                answers[seed]=[]
                for t,logits,seconds,_ in runner.inference_steps(model,features,draft,list(range(1,args.steps+1)),device):
                    inference.validate_logits(logits,draft.shape)
                    answers[seed].append(logits.sigmoid()[0,0].float().cpu().numpy())
                    timing['refiner_seconds'][seed][t]+=seconds
            pending.append(pool.apply_async(process_image,(rows[index],probability,answers)))
            if len(pending)>=args.metric_workers:consume()
            print(f'FORWARD {index+1}/{len(rows)} seeds={len(models)} steps={args.steps}',flush=True)
            del features,draft,probability,answers,logits
        while pending:consume()
    seeds={}
    for seed,domains in buckets.items():
        overall=empty_bucket(metrics,args.steps)
        for bucket in domains.values():add_bucket(overall,bucket)
        seeds[seed]={d:finalize(v,metrics) for d,v in dict(domains,Overall=overall).items()}
    timing['wall_seconds']=time.perf_counter()-started
    report=dict(config=dict(seeds=seed_ids,steps=args.steps,device=args.device,limit_per_domain=args.limit_per_domain,
                            images=len(rows),loader_workers=args.loader_workers,metric_workers=args.metric_workers,
                            val_split_sha256=inference.sha256(args.val_split)),preregistration=pre,checkpoints=checkpoints,seeds=seeds,timing=timing)
    report['predictions']=prediction_checks(report);report['regression']=reference_checks(report)
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if re.search(r'/(?:project[a-z]+|usr\d+|home)/',text):raise ValueError('Private absolute path in report')
    args.out_dir.mkdir(parents=True,exist_ok=True);(args.out_dir/'metrics.json').write_text(text)
    (args.out_dir/'summary.md').write_text(markdown(report))
    print('COMPLETE per-step analysis '+json.dumps(timing),flush=True)


if __name__=='__main__':main()
