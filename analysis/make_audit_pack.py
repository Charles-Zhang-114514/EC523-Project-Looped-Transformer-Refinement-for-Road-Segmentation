#!/usr/bin/env python3
"""Create a blinded, seed-0 manual audit pack from stored failure statistics.

Share panels/, examples/ and audit_sheet.csv with the reviewer. Keep key.csv
private: its delta, added count and precision are the stored GPU references,
not recomputed CPU statistics. No upload or network operation is performed.
"""
import argparse
import csv
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from analysis import failure_analysis as failure
from scripts import eval_stress as shared
from scripts import train_refiner as runner
from submission import predict as inference

AUDIT_FIELDS=('audit_no','red_main','real_road_pct','blue_notable','wrong_connection','confidence','note')
KEY_FIELDS=('audit_no','id','delta_A4_y0','added_pixels','added_precision')
EXAMPLES=('urban/3738','urban/3663')


def select_records(path,seed=0,count=30):
    """Preserve stored CSV order; draw order defines the anonymous audit number."""
    with Path(path).open(newline='') as handle:
        reader=csv.DictReader(handle)
        required={'seed','condition','domain','id','delta_A4_y0','4_added','4_added_precision'}
        if not required.issubset(reader.fieldnames or []):raise ValueError('Failure CSV lacks required columns')
        eligible=[]
        for row in reader:
            if row['seed']!='0' or row['condition']!='ordinary' or row['domain']!='Urban':continue
            if not row['delta_A4_y0']:continue
            delta=float(row['delta_A4_y0'])
            if not math.isfinite(delta):raise ValueError('Nonfinite stored delta')
            if delta<=-1e-4:eligible.append(row)
    if len({row['id'] for row in eligible})!=len(eligible):raise ValueError('Duplicate candidate IDs')
    if count<1 or len(eligible)<count:raise ValueError(f'Need {count} candidates, found {len(eligible)}')
    indices=np.random.default_rng(seed).choice(len(eligible),count,replace=False)
    return [eligible[int(i)] for i in indices],len(eligible)


def panel_name(audit_no):
    if not 1<=audit_no<=30:raise ValueError('Audit number must be between 1 and 30')
    return f'audit_{audit_no:02d}.png'


def write_sheets(selected,out_dir):
    """Only audit numbers appear in the reviewer sheet; original GPU values stay private."""
    for filename,fields,rows in (
        ('audit_sheet.csv',AUDIT_FIELDS,[dict(audit_no=i,**{f:'' for f in AUDIT_FIELDS[1:]}) for i in range(1,len(selected)+1)]),
        ('key.csv',KEY_FIELDS,[dict(audit_no=i,id=r['id'],delta_A4_y0=r['delta_A4_y0'],
                                   added_pixels=r['4_added'],added_precision=r['4_added_precision'])
                                  for i,r in enumerate(selected,1)])):
        with (out_dir/filename).open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def render_anonymous(row,masks,audit_no,path):
    # failure_analysis.py:210-227 reads the actual image/label from row, but
    # draws only the sanitized record/category. Neither ID nor A is passed
    # into that displayed metadata. Fresh Pillow canvases carry no source EXIF.
    displayed=dict(id=f'Audit {audit_no:02d}',condition='',kind='',rectangle=None)
    packed={key:np.packbits(value) for key,value in masks.items()}
    failure.render_case(row,displayed,packed,'',path)


def example_sources(vis_dir):
    with (vis_dir/'index.csv').open(newline='') as handle:rows=list(csv.DictReader(handle))
    sources={}
    for image_id in EXAMPLES:
        matches=sorted(r['file'] for r in rows if r['id']==image_id and r['condition']=='ordinary')
        if not matches or Path(matches[0]).name!=matches[0]:raise ValueError('Missing or invalid requested example')
        source=vis_dir/matches[0]
        if not source.is_file():raise FileNotFoundError(source.name)
        sources[image_id]=source
    return sources


def validate_resume(selected,out_dir,examples):
    """Keep reviewer annotations and completed images; reject a different key."""
    with (out_dir/'key.csv').open(newline='') as handle:actual=list(csv.DictReader(handle))
    expected=[dict(audit_no=str(i),id=r['id'],delta_A4_y0=r['delta_A4_y0'],
                   added_pixels=r['4_added'],added_precision=r['4_added_precision']) for i,r in enumerate(selected,1)]
    if actual!=expected:raise ValueError('Existing key does not match the fixed sample/source statistics')
    with (out_dir/'audit_sheet.csv').open(newline='') as handle:
        reader=csv.DictReader(handle);sheet=list(reader)
        if tuple(reader.fieldnames or [])!=AUDIT_FIELDS or [r['audit_no'] for r in sheet]!=[str(i) for i in range(1,31)]:
            raise ValueError('Existing audit sheet has different fields/numbers')
    expected_top={'panels','examples','audit_sheet.csv','key.csv'}
    if {p.name for p in out_dir.iterdir()}!=expected_top:raise ValueError('Unexpected existing audit artifacts')
    expected_names={panel_name(i) for i in range(1,31)}
    done=set()
    for path in (out_dir/'panels').iterdir():
        if path.name not in expected_names:raise ValueError('Unexpected panel filename')
        with Image.open(path) as image:
            if image.size!=(5120,1064):raise ValueError('Existing panel has an unexpected shape')
            image.verify()
        done.add(path.name)
    for image_id,source in examples.items():
        copy=out_dir/'examples'/f'{image_id.replace("/","_")}.png'
        if not copy.is_file() or inference.sha256(copy)!=inference.sha256(source):
            raise ValueError('Existing example differs from its original visualization')
    return done


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/failure/per_image.csv')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'runs/exp/e0_v2_s0/best.pt')
    parser.add_argument('--val-split',type=Path,default=ROOT/'configs/val_split_seed0.json')
    parser.add_argument('--encoder-dir',type=Path,default=ROOT/'submission/encoder')
    parser.add_argument('--probe',type=Path,default=ROOT/'submission/baseline.safetensors')
    parser.add_argument('--vis-dir',type=Path,default=ROOT/'runs/failure_vis')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'runs/manual_audit')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--resume',action='store_true',help='Validate the original sample and skip existing complete panels')
    args=parser.parse_args()
    if not args.out_dir.resolve().is_relative_to(ROOT/'runs'):parser.error('Audit artifacts must stay under ignored runs/')
    nonempty=args.out_dir.exists() and any(args.out_dir.iterdir())
    if nonempty and not args.resume:parser.error('Output directory must be absent or empty, or use --resume')
    if args.resume and not nonempty:parser.error('--resume requires the existing partial audit pack')
    selected,total=select_records(args.input)
    lookup={r['id']:r for r in shared.build_manifest(args.val_split)}
    if any(r['id'] not in lookup for r in selected):raise ValueError('Selected ID is not in Val-rep')
    examples=example_sources(args.vis_dir)
    completed=validate_resume(selected,args.out_dir,examples) if args.resume else set()
    if not args.checkpoint.is_file():raise FileNotFoundError('Missing seed-0 best checkpoint')
    runner.probe.initialize_data_worker(0)
    device=torch.device(args.device)
    inference.verify_encoder(args.encoder_dir)
    baseline=inference.load_baseline(args.probe,args.encoder_dir,device)
    model,epoch=shared.load_model(args.checkpoint,device,inference.sha256(args.val_split))
    reference=json_load_reference()
    if epoch!=reference['best_epoch']:raise ValueError('Checkpoint epoch differs from selected seed-0 result')
    if not args.resume:
        args.out_dir.mkdir(parents=True,exist_ok=True)
        (args.out_dir/'panels').mkdir()
        (args.out_dir/'examples').mkdir()
        write_sheets(selected,args.out_dir)
        for image_id,source in examples.items():
            shutil.copyfile(source,args.out_dir/'examples'/f'{image_id.replace("/","_")}.png')
    print(f'SELECTED candidates={total} audit_images={len(selected)} seed=0 checkpoint_epoch={epoch}',flush=True)
    differences=[];started=time.perf_counter()
    for number,record in enumerate(selected,1):
        if panel_name(number) in completed:
            print(f'SKIP audit {number:02d}: validated existing panel',flush=True)
            continue
        phase=time.perf_counter();row=lookup[record['id']]
        image=inference.image_tensor(row['image'],device)
        features,draft=runner.frozen_draft(baseline.encoder,baseline.probe,image)
        inference.validate_logits(draft,(1,1,1024,1024))
        output=None
        for _,answer,_,_ in runner.inference_steps(model,features,draft,[4],device):output=answer
        inference.validate_logits(output,draft.shape)
        with Image.open(row['label']) as source:labels=np.array(source,copy=True)
        valid,truth=labels!=0,labels==3
        masks={'y0_draft':(draft.sigmoid()[0,0]>=.5).cpu().numpy() & valid,
               '4':(output.sigmoid()[0,0]>=.5).cpu().numpy() & valid}
        changes=failure.diagnostic.change_counts(masks['y0_draft'],masks['4'],truth,valid)
        added=changes['added_correct']+changes['added_wrong']
        precision=changes['added_correct']/added if added else None
        gpu_added=int(record['4_added']);delta=added-gpu_added;differences.append(delta)
        render_anonymous(row,masks,number,args.out_dir/'panels'/panel_name(number))
        print(f'AUDIT {number:02d} seconds={time.perf_counter()-phase:.3f} GPU_added={gpu_added} CPU_or_current_added={added} delta={delta:+d} GPU_precision={record["4_added_precision"]} current_precision={precision}',flush=True)
        del image,features,draft,output,answer,masks
    print(f'COMPLETE panels=30 examples=2 sheets=2 resumed_panels={len(completed)} seconds={time.perf_counter()-started:.3f} changed_added_counts_current_pass={sum(d!=0 for d in differences)} max_abs_added_difference_current_pass={max(map(abs,differences),default=0)}',flush=True)
    print('SHARE panels/, examples/ and audit_sheet.csv only; keep key.csv private. No upload was performed.',flush=True)


def json_load_reference():
    import json
    return json.loads((ROOT/'results/exp/e0_v2_s0/metrics.json').read_text())


if __name__=='__main__':main()
