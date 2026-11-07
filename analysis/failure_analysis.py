#!/usr/bin/env python3
"""Frozen v2 failure audit with reproducible, explicitly specified case selection.

Connection events are an eight-connected geometric proxy, not a topological
ground-truth annotation. A predicted component overlapping two or more distinct
draft components is one merge. Its valid added pixels determine correctness:
<50% GT-road is false, >=50% is correct. Ignore pixels never connect components.
"""
import argparse
from collections import deque
import csv
from dataclasses import asdict
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import eval_stress as shared
from scripts import train_refiner as runner
from analysis import refiner_diagnostics as diagnostic
from analysis.domain_gating_bound import table
from submission import predict as inference

_METRICS=None


def connections(draft,prediction,truth,valid):
    """One event per predicted 8-component, even when it merges >2 draft parts."""
    structure=np.ones((3,3),bool)
    base,_=ndi.label(draft & valid,structure)
    current,_=ndi.label(prediction & valid,structure)
    added=prediction & ~draft & valid
    result=dict(false_connection=0,correct_connection=0)
    for index,bounds in enumerate(ndi.find_objects(current),1):
        if bounds is None:continue
        inside=current[bounds]==index
        ids=np.unique(base[bounds][inside]);ids=ids[ids!=0]
        if len(ids)<2:continue
        changes=inside & added[bounds]
        total=int(changes.sum())
        if total==0:raise AssertionError('A new merge must include new valid pixels')
        correct=int((changes & truth[bounds]).sum())
        result['false_connection' if 2*correct<total else 'correct_connection']+=1
    return result


def difference_colors(draft,prediction,truth,valid):
    """Solid audit colors: new TP green, new FP red, deleted TP blue, FP cyan."""
    out=np.full((*valid.shape,3),128,np.uint8)
    added=~draft & prediction & valid;deleted=draft & ~prediction & valid
    for mask,color in ((added & truth,(0,255,0)),(added & ~truth,(255,0,0)),
                       (deleted & truth,(0,0,255)),(deleted & ~truth,(0,255,255))):out[mask]=color
    return out


def initialize_worker(starter):
    global _METRICS
    shared.initialize_worker(starter)
    _METRICS=runner.probe.official_metrics(starter)


def process_image(row,predictions,seed):
    """Reuse official scores and existing shared skeleton/change definitions."""
    started=time.perf_counter()
    with Image.open(row['label']) as source:labels=np.array(source,copy=True)
    valid,truth=labels!=0,labels==3
    predictions={k:v.astype(bool)&valid for k,v in predictions.items()}
    base=predictions['y0_draft'];has_road=bool((truth&valid).any())
    shared_result=shared.analyze_image(row,predictions)
    # Independently compare the pre-existing runner's per-image metric path.
    direct=runner.score_image(row['domain'].title(),row['id'],predictions,labels,geometry=False)[2]
    out=dict(id=row['id'],domain=row['domain'].title(),condition=row['condition'],
             kind=row.get('corruption',{}).get('kind','ordinary'),seed=seed,
             rectangle=row.get('corruption',{}).get('rectangle_xywh'),gt_has_road=has_road,
             valid_pixels=int(valid.sum()),gt_road_pixels=int(truth.sum()),counts={},scores={},A={},steps={})
    for k,mask in predictions.items():
        value=shared_result[2][k];assert value['counts']==direct[k]['counts']
        out['counts'][k]=value['counts']
        scores=_METRICS.summarize(_METRICS.Counts(**value['counts']))
        out['scores'][k]=scores
        out['A'][k]=scores['quality'] if has_road else None
        c=value['changes'];added=c['added_correct']+c['added_wrong'];deleted=c['deleted_correct']+c['deleted_wrong']
        out['steps'][k]=dict(**c,added=added,deleted=deleted,
            added_precision=diagnostic.fraction(c['added_correct'],added),
            deleted_precision=diagnostic.fraction(c['deleted_correct'],deleted),
            disappearing_road_pixels=c['deleted_wrong'],**value['tracking'],
            **connections(base,mask,truth,valid),
            new_fp_fragments=len(diagnostic.fragment_shapes(mask,base,truth,valid)))
    out['delta_A4_y0']=out['A']['4']-out['A']['y0_draft'] if has_road else None
    out['delta_A4_K1']=out['A']['4']-out['A']['1'] if has_road else None
    out['delta_A8_K4']=out['A']['8']-out['A']['4'] if has_road and '8' in out['A'] else None
    # Packed masks are transient parent-process memory only, never report data.
    packed={k:np.packbits(predictions[k]) for k in ('y0_draft','4')} if seed==0 else None
    return out,shared_result,packed,time.perf_counter()-started


def aggregate(records,metrics):
    """Per-seed counts sum before summarize; empty-GT A is excluded from distributions."""
    keys=records[0]['counts']
    result=dict(images=len(records),gt_empty_images=sum(not r['gt_has_road'] for r in records),groups={})
    for key in keys:
        counts=metrics.Counts()
        for r in records:counts=counts+metrics.Counts(**r['counts'][key])
        totals={name:sum(r['steps'][key][name] for r in records) for name in
                ('false_connection','correct_connection','disappearing_road_pixels','new_fp_fragments','added','deleted',
                 'added_correct','added_wrong','deleted_correct','deleted_wrong','gap_total','gap_covered','covered_total','covered_lost')}
        totals['false_connection_per_image']=totals['false_connection']/len(records)
        totals['correct_connection_per_image']=totals['correct_connection']/len(records)
        totals['no_gt_added_pixels']=sum(r['steps'][key]['added'] for r in records if not r['gt_has_road'])
        result['groups'][key]=dict(counts=asdict(counts),scores=metrics.summarize(counts),**totals)
    vals=[r['delta_A4_y0'] for r in records if r['delta_A4_y0'] is not None]
    result['delta_A4_y0']=dict(applicable_images=len(vals),
        better=diagnostic.fraction(sum(x>=1e-4 for x in vals),len(vals)),
        worse=diagnostic.fraction(sum(x<=-1e-4 for x in vals),len(vals)),
        unchanged=diagnostic.fraction(sum(abs(x)<1e-4 for x in vals),len(vals)),
        quantiles={str(q):float(np.quantile(vals,q)) if vals else None for q in (.05,.25,.5,.75,.95)})
    later=[r['delta_A8_K4'] for r in records if r['delta_A8_K4'] is not None]
    result['K8_worse_than_K4_by_001']=dict(applicable_images=len(later),
        images=sum(v<-.01 for v in later),fraction=diagnostic.fraction(sum(v<-.01 for v in later),len(later)))
    return result


def grouped(records,metrics):
    result={}
    for seed in sorted({r['seed'] for r in records}):
        for condition in ('ordinary','stress'):
            subset=[r for r in records if r['seed']==seed and r['condition']==condition]
            if not subset:continue
            groups={'Overall':subset}
            for d in ('Urban','Rural'):groups[d]=[r for r in subset if r['domain']==d]
            if condition=='stress':
                for kind in shared.KINDS:
                    groups[kind]=[r for r in subset if r['kind']==kind]
                    for d in ('Urban','Rural'):groups[d+'/'+kind]=[r for r in subset if r['domain']==d and r['kind']==kind]
            result[f'{condition}/s{seed}']={g:aggregate(rows,metrics) for g,rows in groups.items() if rows}
    return result


def seed_summary(groups):
    """Summarize trained seeds, without pooling their distinct predictions."""
    pools={}
    for run,values in groups.items():
        condition=run.split('/')[0]
        for group,value in values.items():pools.setdefault(condition+'/'+group,[]).append(value)
    result={}
    for name,values in pools.items():
        fields={}
        for stat in ('better','worse','unchanged'):
            fields[stat+'_fraction']=[v['delta_A4_y0'][stat] for v in values]
        for q in (.05,.25,.5,.75,.95):
            fields['delta_quantile_'+str(q)]=[v['delta_A4_y0']['quantiles'][str(q)] for v in values]
        for stat in ('false_connection','false_connection_per_image','correct_connection','correct_connection_per_image',
                     'disappearing_road_pixels','no_gt_added_pixels'):
            fields[stat]=[v['groups']['4'][stat] for v in values]
        fields['later_degradation_fraction']=[v['K8_worse_than_K4_by_001']['fraction'] for v in values]
        fields['gt_empty_images']=[v['gt_empty_images'] for v in values]
        result[name]={}
        for field,items in fields.items():
            valid=[x for x in items if x is not None]
            result[name][field]=dict(mean=statistics.mean(valid) if valid else None,
                sample_std=statistics.stdev(valid) if len(valid)>1 else None,applicable_seeds=len(valid))
    return result


def select_cases(records):
    """Sorted pools, RNG seed 0, three random examples plus explicit extrema."""
    records=sorted((r for r in records if r['seed']==0),key=lambda r:(r['condition'],r['domain'],r['id']))
    pools={}
    for d in ('Urban','Rural'):
        for sign in ('positive','negative'):
            pools[f'clean_{d}_{sign}']=[r for r in records if r['condition']=='ordinary' and r['domain']==d and
                r['delta_A4_y0'] is not None and (r['delta_A4_y0']>0 if sign=='positive' else r['delta_A4_y0']<0)]
    pools['false_connection']=[r for r in records if r['steps']['4']['false_connection']>0]
    pools['correct_connection']=[r for r in records if r['steps']['4']['correct_connection']>0]
    pools['later_degradation']=[r for r in records if r['delta_A8_K4'] is not None and r['delta_A8_K4']<-.01]
    for kind in shared.KINDS:
        pools['stress_'+kind]=[r for r in records if r['condition']=='stress' and r['kind']==kind and r['delta_A4_y0'] is not None and r['delta_A4_y0']<0]
    pools['no_gt_additions']=[r for r in records if not r['gt_has_road'] and r['steps']['4']['added']>0]
    rng=np.random.default_rng(0);selections=[];overview={}
    for category,pool in pools.items():
        chosen=[]
        for index in rng.choice(len(pool),min(3,len(pool)),replace=False):chosen.append((pool[int(index)],'random'))
        applicable=[r for r in pool if r['delta_A4_y0'] is not None]
        if applicable:
            chosen += [(min(applicable,key=lambda r:(r['delta_A4_y0'],r['id'])),'minimum_delta'),
                       (max(applicable,key=lambda r:(r['delta_A4_y0'],r['id'])),'maximum_delta')]
        if category=='no_gt_additions' and pool:
            chosen.append((max(pool,key=lambda r:(r['steps']['4']['added'],r['id'])),'maximum_added'))
        unique={}
        for r,reason in chosen:
            key=(r['condition'],r['id']);unique.setdefault(key,[r,[]])[1].append(reason)
        for r,reasons in unique.values():selections.append(dict(category=category,record=r,reasons=reasons))
        overview[category]=dict(candidates=len(pool),selected=len(unique))
    return selections,overview


def render_case(row,record,packed,category,path):
    with Image.open(row['image']) as src:rgb=np.array(src.convert('RGB'),copy=True)
    with Image.open(row['label']) as src:labels=np.array(src,copy=True)
    shape=labels.shape;valid=labels!=0;truth=labels==3
    masks={k:np.unpackbits(v,count=labels.size).reshape(shape).astype(bool) for k,v in packed.items()}
    def binary(mask):
        value=np.repeat((mask.astype(np.uint8)*255)[...,None],3,axis=2);value[~valid]=128;return value
    panels=[rgb,binary(truth),binary(masks['y0_draft']),binary(masks['4']),
            difference_colors(masks['y0_draft'],masks['4'],truth,valid)]
    h,w=shape;canvas=Image.new('RGB',(5*w,h+40),(20,20,20))
    for i,panel in enumerate(panels):canvas.paste(Image.fromarray(panel),(i*w,40))
    draw=ImageDraw.Draw(canvas);font=ImageFont.load_default(size=18)
    draw.text((8,8),f"{record['id']} {record['condition']} {record['kind']} {category} | RGB / GT / y0 / K4 / flips; green=added TP red=added FP blue=deleted TP cyan=deleted FP gray=other",font=font,fill='white')
    if record['rectangle']:
        x,y,bw,bh=record['rectangle'];draw.rectangle((x,y+40,x+bw-1,y+bh+39),outline='yellow',width=3)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path.name)
    canvas.save(path)


def evaluate(rows,cache,head,seed,args,device):
    model,epoch=shared.load_model(ROOT/f'runs/exp/e0_v2_s{seed}/best.pt',device,inference.sha256(args.val_split))
    metrics=runner.probe.official_metrics(str(args.starter));pending=deque();records=[];packed={}
    direct=shared.empty_buckets(['y0_draft',*map(str,args.ks)],metrics)
    times={str(k):0. for k in args.ks};worker_seconds=0.;started=time.perf_counter()
    def consume():
        nonlocal worker_seconds
        record,old,masks,seconds=pending.popleft().get();records.append(record);shared.add_result(direct,old)
        if masks is not None:packed[record['id']]=masks
        worker_seconds+=seconds
    with mp.get_context('spawn').Pool(args.metric_workers,initializer=initialize_worker,initargs=(str(args.starter),)) as pool:
        for i,row in enumerate(rows):
            feature=torch.from_numpy(np.load(cache/f'{i:08d}.npy',allow_pickle=False)).to(device)
            with torch.no_grad():draft=head(feature).float()
            inference.validate_logits(draft,(1,1,1024,1024))
            predictions={'y0_draft':(draft.sigmoid()[0,0]>=.5).cpu().numpy().astype(np.uint8)}
            for k,answer,seconds,_ in runner.inference_steps(model,feature,draft,args.ks,device):
                inference.validate_logits(answer,draft.shape)
                predictions[k]=(answer.sigmoid()[0,0]>=.5).cpu().numpy().astype(np.uint8);times[k]+=seconds
            pending.append(pool.apply_async(process_image,(row,predictions,seed)))
            if len(pending)>=2*args.metric_workers:consume()
        while pending:consume()
    aggregate_records=aggregate(records,metrics)
    direct=shared.finalize(direct,metrics,rows[0]['condition']=='stress')
    for k in aggregate_records['groups']:
        assert aggregate_records['groups'][k]['counts']==direct['Overall'][k]['counts']
        assert aggregate_records['groups'][k]['scores']==direct['Overall'][k]['scores']
    print(f'PASS per-image summed Counts and official scores equal shared direct evaluation {rows[0]["condition"]} seed={seed}',flush=True)
    del model;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return records,packed,dict(epoch=epoch,checkpoint=f'runs/exp/e0_v2_s{seed}/best.pt',
                               checkpoint_sha256=inference.sha256(ROOT/f'runs/exp/e0_v2_s{seed}/best.pt'),
                               refiner_seconds=times,worker_seconds=worker_seconds,wall_seconds=time.perf_counter()-started,
                               direct_count_regression=True)


def markdown(report):
    lines=['# Failure analysis','',
        'Observed only frozen inference is used. Valid pixels have raw label !=0; GT road is label 3. '
        'Official per-image scores are retained, but analysis A/delta-A is not applicable for images with no GT road. '
        'These images are counted separately, including their added pixels; aggregate official Counts still include them. '
        'An empty road annotation is not evidence that the image has no visible road; prior manual audits do not alter labels here.', '',
        'Observed connection proxy: one predicted eight-connected component overlapping >=2 distinct y0 components is one merge, '
        'regardless of how many components it joins. Added pixels in that component determine its label: <50% GT-road is false, >=50% is correct. '
        'Ignore regions are excluded before connectivity. This is a geometric approximation, not proof of a road-network connection.', '',
        'Observed disappearing-road pixels are GT road covered by y0 but deleted at K; new-break pixels are the existing fixed-y0 GT skeleton subset lost at K. '
        'Gap sets and new-break sets reuse refiner_diagnostics.baseline_skeleton_sets through eval_stress.analyze_image.', '',
        'Observed case selection: only model seed 0; sorted candidate pools and RNG seed 0; up to three random examples per category plus each category\'s '
        'minimum/maximum applicable K4-y0 delta. Duplicates within a category are combined and listed in the visualization index. '
        'The no-GT category additionally includes maximum K4 added pixels; delta extrema are inapplicable there. '
        'Named positive/negative and connection/stress categories are predeclared; explicit extrema are not random examples.', '',
        'Observed each cell below is for one trained seed; domain/corruption groups overlap and must not be summed indiscriminately.', '']
    rows=[]
    for run,groups in report['groups'].items():
        for group,r in groups.items():
            delta=r['delta_A4_y0'];k=r['groups']['4'];late=r['K8_worse_than_K4_by_001']
            rows.append([run,group,r['images'],r['gt_empty_images'],delta['better'],delta['worse'],delta['unchanged'],
                         k['false_connection'],k['false_connection_per_image'],k['correct_connection'],k['correct_connection_per_image'],
                         k['disappearing_road_pixels'],late['fraction'],k['no_gt_added_pixels']])
    lines += [table(['Run','Group','Images','No GT','Better fraction','Worse fraction','Unchanged fraction','False merges','False/image',
                     'Correct merges','Correct/image','Disappearing road pixels','K8-K4 < -0.01 fraction','No-GT added pixels'],rows),'']
    lines += [table(['Run','Group','q05','q25','q50','q75','q95'],[
        [run,g,*[v['delta_A4_y0']['quantiles'][str(q)] for q in (.05,.25,.5,.75,.95)]] for run,groups in report['groups'].items() for g,v in groups.items()]),'',
        'Observed all detailed per-K official Counts, official scores, per-seed grouped totals and per-image data are in metrics.json/per_image.csv. '
        'No-GT A is blank in CSV; *_official_A retains the official definition. Confidence/probability maps and features are not report artifacts.', '',
        'Observed visualization pools and selected counts:', '',json.dumps(report['selection'],indent=2)]
    def display(value):
        return '--' if value['mean'] is None else f"{value['mean']:.6f} +/- "+(f"{value['sample_std']:.6f}" if value['sample_std'] is not None else '--')
    lines += ['', 'Observed model-seed means and sample standard deviations (ddof=1; no averaging across domains/categories):', '',
        table(['Condition/group','Better','Worse','Unchanged','False merges','Correct merges','Disappearing pixels',
               'Later degradation fraction','No-GT added pixels'],[
            [name,*[display(v[k]) for k in ('better_fraction','worse_fraction','unchanged_fraction','false_connection',
                'correct_connection','disappearing_road_pixels','later_degradation_fraction','no_gt_added_pixels')]]
            for name,v in report['seed_summary'].items()])]
    return '\n'.join(lines)+'\n'


def export_csv(records,path):
    rows=[]
    for r in records:
        row={k:r[k] for k in ('id','domain','condition','kind','seed','gt_has_road','valid_pixels','gt_road_pixels','delta_A4_y0','delta_A4_K1','delta_A8_K4')}
        for i,key in enumerate(('rectangle_x','rectangle_y','rectangle_width','rectangle_height')):row[key]=r['rectangle'][i] if r['rectangle'] else None
        for k in r['counts']:
            row[k+'_A']=r['A'][k];row[k+'_official_A']=r['scores'][k]['quality']
            for f in ('road_iou','cldice','boundary_iou'):row[k+'_'+f]=r['scores'][k][f]
            for field,value in r['counts'][k].items():row[k+'_count_'+field]=value
            for field,value in r['steps'][k].items():row[k+'_'+field]=value
        rows.append(row)
    with path.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'results/failure')
    parser.add_argument('--vis-dir',type=Path,default=ROOT/'runs/failure_vis')
    parser.add_argument('--val-split',type=Path,default=ROOT/'configs/val_split_seed0.json')
    parser.add_argument('--stress-manifest',type=Path,default=ROOT/'runs/stress/seed523/manifest.json')
    parser.add_argument('--model-seeds',default='0,1,2')
    parser.add_argument('--eval-k',type=runner.parse_ks,default=[1,2,4,8])
    parser.add_argument('--limit-per-domain',type=int)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--loader-workers',type=int,default=1)
    parser.add_argument('--metric-workers',type=int,default=6)
    parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--encoder-dir',type=Path,default=ROOT/'submission/encoder')
    parser.add_argument('--probe',type=Path,default=ROOT/'submission/baseline.safetensors')
    args=parser.parse_args();args.ks=args.eval_k;args.starter=Path(os.environ['EC523_STARTER_DIR'])
    if not args.out_dir.resolve().is_relative_to(ROOT):parser.error('Reports must stay inside the repository')
    seeds=[int(s) for s in args.model_seeds.split(',')]
    if not seeds or len(set(seeds))!=len(seeds) or any(s not in (0,1,2) for s in seeds):parser.error('Use unique seeds 0,1,2')
    if not {1,4}.issubset(args.ks) or max(args.ks)>8:parser.error('K must include 1 and 4 and be <=8')
    if args.limit_per_domain is not None and args.limit_per_domain<1:parser.error('Limit must be positive')
    slots=int(os.environ.get('NSLOTS',0))
    if args.loader_workers<0 or args.metric_workers<1 or (slots and max(args.loader_workers+1,args.metric_workers+1)>slots):parser.error('Process allocation exceeds slots')
    for directory in (args.out_dir,args.vis_dir):
        if directory.exists() and any(directory.iterdir()):parser.error('Output directories must be absent or empty')
    if not args.vis_dir.resolve().is_relative_to(ROOT/'runs'):parser.error('Visualizations must stay in ignored runs')
    runner.probe.initialize_data_worker(0);device=torch.device(args.device)
    ordinary=shared.build_manifest(args.val_split,limit_per_domain=args.limit_per_domain)
    common=shared.official_module('common',args.starter);stress=common.load_manifest(args.stress_manifest)
    wanted={r['id'] for r in ordinary};stress=[r for r in stress if r['id'] in wanted]
    if {r['id'] for r in stress}!=wanted:raise ValueError('Stress seed523 IDs do not match ordinary IDs')
    metrics=runner.probe.official_metrics(str(args.starter));records=[];index=[];timings={}
    transient_masks={};image_rows={}
    for condition,rows in (('ordinary',ordinary),('stress',stress)):
        cache,_=inference.create_cache(args.cache_dir,args.out_dir,len(rows))
        try:
            head,encoding=shared.encode_images(rows,cache,args,device);timings[condition]=dict(encoding=encoding,models={})
            for seed in seeds:
                out,packed,metadata=evaluate(rows,cache,head,seed,args,device);records.extend(out)
                timings[condition]['models'][str(seed)]=metadata
                if seed==0:
                    transient_masks.update({(condition,key):value for key,value in packed.items()})
                    image_rows.update({(condition,row['id']):row for row in rows})
                del packed
            del head
        finally:
            shutil.rmtree(cache);print('CACHE_REMOVED '+condition,flush=True)
    chosen,selection=select_cases(records)
    for i,item in enumerate(chosen,1):
        r=item['record'];condition=r['condition'];key=(condition,r['id'])
        filename=f'{condition}_{item["category"]}_{i:02d}_{r["id"].replace("/","_")}.png'
        render_case(image_rows[key],r,transient_masks[key],item['category'],args.vis_dir/filename)
        index.append(dict(file=filename,category=item['category'],id=r['id'],condition=condition,
                          reasons=';'.join(item['reasons']),delta_A4_y0=r['delta_A4_y0'],
                          delta_A8_K4=r['delta_A8_K4'],added_K4=r['steps']['4']['added'],
                          false_connections=r['steps']['4']['false_connection'],correct_connections=r['steps']['4']['correct_connection']))
    del transient_masks
    groups=grouped(records,metrics)
    report=dict(config=dict(seeds=seeds,ks=args.ks,limit_per_domain=args.limit_per_domain,device=args.device,
                            val_split_sha256=inference.sha256(args.val_split),stress_manifest_sha256=inference.sha256(args.stress_manifest)),
                groups=groups,seed_summary=seed_summary(groups),
                timing=timings,selection=selection,per_image=records)
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if re.search(r'/(?:project[a-z]+|usr\d+|home)/',text):raise ValueError('Personal absolute path in portable results')
    args.out_dir.mkdir(parents=True,exist_ok=True);(args.out_dir/'metrics.json').write_text(text)
    (args.out_dir/'summary.md').write_text(markdown(report));export_csv(records,args.out_dir/'per_image.csv')
    args.vis_dir.mkdir(parents=True,exist_ok=True)
    with (args.vis_dir/'index.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(index[0]) if index else ['file','category','id']);writer.writeheader();writer.writerows(index)
    print(f'COMPLETE records={len(records)} visualizations={len(index)}',flush=True)


if __name__=='__main__':main()
