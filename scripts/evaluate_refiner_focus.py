#!/usr/bin/env python3
"""Stream official clean/stress scores for D4 recurrent ensembles and refiners."""
import argparse
from collections import deque
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from analysis import evaluate_draft_tta as tta
from scripts import train_refiner as runner
from looproad.data.draft_cache import DraftCache, sha256
from looproad.models.d4_refiner import trajectory, transform
from looproad.models.detail_features import DetailFeatures

HEAD=ROOT/'runs/draft_head/A1-i/best.pt'
CACHE=ROOT/'runs/cache/tta_draft_a1i_s0'
BASE=ROOT/'runs/exp/e6_convdraft_s0/best.pt'
STRESS=ROOT/'runs/stress/seed523'


def score_masks(domain, masks, labels):
    metrics=runner.probe._WORKER_METRICS
    return domain,{mode:{k:asdict(metrics.image_counts(mask.astype(np.float32),labels)) for k,mask in values.items()}
                   for mode,values in masks.items()}


def stress_rows(limit=None):
    generation=json.loads((STRESS/'generation.json').read_text())
    rows=json.loads((STRESS/'manifest.json').read_text())
    split=json.loads(tta.SPLIT.read_text())
    assert generation['seed']==523 and len(rows)==835
    assert {r['id'] for r in rows}=={d.lower()+'/'+Path(n).stem for d,ns in split['Val-rep'].items() for n in ns}
    for d in runner.DOMAINS:
        group=[r for r in rows if r['domain']==d.lower()]
        for row in group[:limit] if limit else group:
            image=Path(row['image']);image=image if image.is_absolute() else STRESS/image
            assert image.is_file() and Path(row['label']).is_file()
            yield d,row['id'],image,Path(row['label'])


def clean_rows(part,limit=None):
    sets=runner.protocol_datasets(json.loads(tta.SPLIT.read_text()),part,limit)
    for domain,dataset in sets.items():
        for image,label in dataset.pairs:
            yield domain,f'Val/{domain}/{image.stem}',image,label


def read_image(path,device):
    with Image.open(path) as im:
        x=np.array(im,dtype=np.uint8)
    normalized=np.ascontiguousarray(x.transpose(2,0,1),dtype=np.float32)/255.0
    return torch.from_numpy(normalized).unsqueeze(0).to(device)


@torch.no_grad()
def feature_views(encoder,head,image,codes):
    features=[];drafts=[]
    for c in codes:
        raw=encoder(transform(image,c).contiguous())
        features.append(head.refiner_features(raw) if hasattr(head,'refiner_features') else raw)
        drafts.append(head(raw).float())
    return torch.cat(features),drafts


@torch.no_grad()
def live_draft(encoder,head,image):
    identity=head(encoder(image))
    return tta.average_d4_logits(lambda x:head(encoder(x)),image,identity)


def compare_groups(actual,expected,tolerance=1e-5):
    diffs=[]
    for domain in tta.DOMAINS:
        for k in ('1','2','4','8'):
            for key in tta.SCORE_KEYS:
                diffs.append(abs(actual[domain][k]['scores'][key]-expected[domain][k]['scores'][key]))
    maximum=max(diffs)
    if maximum>tolerance:
        raise AssertionError(f'SG1 metric maximum difference {maximum} > {tolerance}')
    return dict(passed=True,comparisons=len(diffs),maximum_difference=maximum,tolerance=tolerance)


def evaluate(encoder,head,model,rows,ks,device,modes,cache,stress=False,regression=False):
    metrics=runner.probe.official_metrics();counts={m:{d:{str(k):metrics.Counts() for k in ks}for d in runner.DOMAINS}for m in modes};images=[];pending=deque()
    def consume():
        domain,result=pending.popleft().get()
        for mode,values in result.items():
            for k,v in values.items():counts[mode][domain][k]+=metrics.Counts(**v)
    with mp.get_context('spawn').Pool(6,initializer=runner.initialize_eval_worker,initargs=(os.environ['EC523_STARTER_DIR'],)) as pool:
        for domain,identity,path,label in rows:
            image=read_image(path,device)
            codes=tuple(range(8)) if any(m!='E0'for m in modes) else (0,)
            features,_=feature_views(encoder,head,image,codes)
            draft=live_draft(encoder,head,image) if stress else cache.load([identity],device)
            masks={};timing={};peaks={};calls={};regression_max=0.0;bound_maxima={}
            for mode in modes:
                masks[mode]={};timing[mode]={};peaks[mode]={};calls[mode]={}
                fs=features[:1] if mode=='E0' else features
                if getattr(model,'detail_head',False):
                    if mode!='E0':raise ValueError('Detail evaluation currently supports identity-view refinement only')
                    fs=DetailFeatures(fs,image)
                cs=(0,) if mode=='E0' else codes
                originals=({key:y.clone() for key,y,_,_ in runner.inference_steps(model,fs,draft,ks,device)}
                           if regression and mode=='E0' else {})
                for k,answer,seconds,peak,count in trajectory(model,fs,draft,ks,'S5b'if mode=='S5b'else'S5a',cs):
                    masks[mode][k]=tta.binary_mask(answer);timing[mode][k]=seconds;peaks[mode][k]=peak;calls[mode][k]=count
                    if regression and mode=='E0':
                        original=originals[k]
                        diff=(answer-original).abs().max().item()
                        regression_max=max(regression_max,diff)
                        if diff>1e-5 or not np.array_equal(masks[mode][k],tta.binary_mask(original)):
                            raise AssertionError(f'Identity-view SG1 failed: {diff}')
                if getattr(model,'delta_bound',None) is not None:
                    bound_maxima[mode]=[float(x) for x in model._delta_maxima]
            with Image.open(label) as im:labels=np.array(im,dtype=np.uint8)
            pending.append(pool.apply_async(score_masks,(domain,masks,labels)))
            if len(pending)>=12:consume()
            images.append(dict(domain=domain,image_id=identity,refiner_seconds=timing,peak_cuda_bytes=peaks,core_calls=calls,identity_regression_max_difference=regression_max))
            if bound_maxima:images[-1]['delta_maxima']=bound_maxima
            print('SCORED '+json.dumps(dict(id=identity,calls=calls)),flush=True)
        while pending:consume()
    groups={}
    for mode,ds in counts.items():
        ds['Overall']={k:ds['Urban'][k]+ds['Rural'][k]for k in ds['Urban']}
        groups[mode]={d:{k:dict(counts=asdict(v),scores=metrics.summarize(v))for k,v in byk.items()}for d,byk in ds.items()}
    return dict(groups=groups,images=images)


@torch.no_grad()
def dg1(encoder,head,device,cache):
    results=[]
    for domain,identity,path,_ in clean_rows('Val-rep',2):
        image=read_image(path,device);draft=cache.load([identity],device)
        for code in range(8):
            actual=live_draft(encoder,head,transform(image,code).contiguous());expected=transform(draft,code)
            difference=(actual-expected).abs().max().item();flips=int(((actual.sigmoid()>=.5)!=(expected.sigmoid()>=.5)).sum())
            results.append(dict(image_id=identity,code=code,maximum_difference=difference,flipped_pixels=flips))
    report=dict(checks=results,maximum_difference=max(x['maximum_difference']for x in results),flipped_pixels=sum(x['flipped_pixels']for x in results))
    report['passed']=report['maximum_difference']<=1e-4 and report['flipped_pixels']==0
    print('DG1 '+json.dumps(report),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['full','sg1','S5a','S5b','dg1','stress','trained'],default='full')
    p.add_argument('--limit-per-domain',type=int)
    p.add_argument('--audit-training-controls',action='store_true')
    p.add_argument('--checkpoint',type=Path,default=BASE)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda');cache=DraftCache(CACHE,HEAD,tta.SPLIT)
    encoder,head,_=runner.y0.load_frozen_probe(HEAD,device)
    model,_=tta.previous.shared.load_model(args.checkpoint,device,sha256(tta.SPLIT));model.eval()
    report=dict(checkpoint=args.checkpoint.name,manifest_sha256=sha256(CACHE/'manifest.json'),splits={})
    if args.audit_training_controls:
        report['evaluation_corruption_applied']=False
        report['delta_bound']=getattr(model,'delta_bound',None)
    if args.phase=='dg1':
        report['DG1']=dg1(encoder,head,device,cache)
    else:
        modes=['E0','S5a','S5b'] if args.phase=='full' else [args.phase] if args.phase in ['S5a','S5b'] else ['E0']
        parts=['Val-sel','Val-rep','stress']if args.phase=='full'else ['Val-rep','stress']if args.phase=='trained'else ['stress']if args.phase=='stress'else['Val-rep']
        for part in parts:
            stress=part=='stress';ks=[4]if stress else [1,2,4,8]if args.phase in ['full','trained','sg1']else[1,2,4]
            rows=stress_rows(args.limit_per_domain)if stress else clean_rows(part,args.limit_per_domain)
            result=evaluate(encoder,head,model,rows,ks,device,modes,cache,stress,args.phase=='sg1')
            report['splits'][part]=result
            if args.phase in ['full','sg1']and part=='Val-rep':
                if args.limit_per_domain is None:
                    reference=json.loads((ROOT/'results/wave1/ensemble_bias/metrics.json').read_text())['splits']['Val-rep']['groups']['E0']
                    report['SG1']=compare_groups(result['groups']['E0'],reference)
                else:report['SG1']=dict(passed=True,basis='Per-image logits/masks compared with production inference',images=len(result['images']))
        if args.phase in ['full','trained']:
            report['L']={part:{mode:{d:sum(w*gs[d][k]['scores']['quality']for k,w in [('1',.2),('2',.2),('4',.6)])for d in tta.DOMAINS}for mode,gs in result['groups'].items()}for part,result in report['splits'].items()if part!='stress'}
            report['LB']={mode:.2*gs['Overall']['1']['scores']['quality']+.2*gs['Overall']['2']['scores']['quality']+.4*gs['Overall']['4']['scores']['quality']+.2*report['splits']['stress']['groups'][mode]['Overall']['4']['scores']['quality']for mode,gs in report['splits']['Val-rep']['groups'].items()}
    (args.out_dir/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if args.phase=='dg1'and not report['DG1']['passed']:raise SystemExit('DG1_FAILED')
    print('FOCUS_EVAL_COMPLETE',flush=True)


if __name__=='__main__':
    main()
