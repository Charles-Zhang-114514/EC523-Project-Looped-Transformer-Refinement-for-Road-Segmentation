#!/usr/bin/env python3
"""Allocated-GPU pre-update detail initialization gate; no training."""
import argparse
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from scripts import evaluate_refiner_focus as focus
from scripts import train_refiner as runner
from looproad.models.refiner import LoopRefiner,load_initial_refiner
from looproad.models.detail_features import DetailFeatures


def load_detail(device):
    saved=torch.load(focus.BASE,map_location='cpu',weights_only=True)
    c=saved['config']
    model=LoopRefiner(1024,**{k:c[k]for k in ['dimension','layers','heads','mlp_ratio','patch','head_type']},detail_head=True).to(device)
    load_initial_refiner(model,saved['state_dict'])
    return model.eval()


def run(out,full=False):
    out.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda')
    encoder,head,_=runner.y0.load_frozen_probe(focus.HEAD,device)
    cache=focus.DraftCache(focus.CACHE,focus.HEAD,focus.tta.SPLIT)
    model=load_detail(device)
    if full:
        result=focus.evaluate(encoder,head,model,focus.clean_rows('Val-rep'),[1,2,4,8],device,['E0'],cache)
        reference=json.loads((ROOT/'results/wave1/ensemble_bias/metrics.json').read_text())['splits']['Val-rep']['groups']['E0']
        try:
            gate=focus.compare_groups(result['groups']['E0'],reference)
        except AssertionError as error:
            gate=dict(passed=False,reason=str(error),tolerance=1e-5)
        gate.update(images=len(result['images']),basis='Full Val-rep official-score comparison with archived E0')
        if gate['images']!=835:
            gate.update(passed=False,reason='Expected 835 Val-rep images')
    else:
        old,_=focus.tta.previous.shared.load_model(focus.BASE,device,focus.sha256(focus.tta.SPLIT))
        rows=[]
        for domain,identity,path,_ in focus.clean_rows('Val-rep',2):
            image=focus.read_image(path,device)
            features,_=focus.feature_views(encoder,head,image,(0,));draft=cache.load([identity],device)
            expected={k:y.clone()for k,y,_,_ in runner.inference_steps(old,features,draft,[1,2,4,8],device)}
            values=list(focus.trajectory(model,DetailFeatures(features,image),draft,[1,2,4,8],codes=(0,)))
            for k,y,_,_,calls in values:
                rows.append(dict(image_id=identity,K=int(k),core_calls=calls,maximum_difference=float((y-expected[k]).abs().max()),flipped_pixels=int(((y.sigmoid()>=.5)!=(expected[k].sigmoid()>=.5)).sum())))
        gate=dict(images=4,checks=rows,maximum_difference=max(x['maximum_difference']for x in rows),flipped_pixels=sum(x['flipped_pixels']for x in rows))
        gate['passed']=gate['maximum_difference']<=1e-5 and gate['flipped_pixels']==0 and all(x['core_calls']==x['K']for x in rows)
    (out/'metrics.json').write_text(json.dumps(dict(DRG1=gate),indent=2,allow_nan=False)+'\n')
    print('DRG1 '+json.dumps(gate),flush=True)
    if not gate['passed']:raise SystemExit('DRG1_FAILED: detail initialization differs from E0; training must not start')
    print('DETAIL_GATE_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--full',action='store_true');a=p.parse_args();run(a.out_dir,a.full)


if __name__=='__main__':main()
