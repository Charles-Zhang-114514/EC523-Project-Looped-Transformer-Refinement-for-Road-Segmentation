#!/usr/bin/env python3
"""Offline, inference-only ordinary/stress evaluation of three frozen v2 seeds.

Use the original staff generator and metrics, shared submission feature I/O,
the training runner's exact recurrent inference, and existing diagnostics.
Only runtime manifests, generated stress PNGs and one evaluator cross-check
contain local paths; portable reports contain hashes and relative provenance.
"""
import argparse
from collections import deque
from dataclasses import asdict
import gc
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
from scripts.build_eval_manifest import build_manifest, write_manifest
from scripts import train_refiner as runner
from analysis import refiner_diagnostics as diagnostics
from submission import predict as inference

KINDS = ('shadow', 'occlusion', 'blur')
DOMAINS = ('Urban', 'Rural')
GROUPS = ('Overall', *DOMAINS, *KINDS, *(f'{d}/{k}' for d in DOMAINS for k in KINDS))
CHANGE_KEYS = ('added_correct', 'added_wrong', 'deleted_correct', 'deleted_wrong')
TRACK_KEYS = ('gap_total', 'gap_covered', 'terminal_total', 'terminal_covered', 'covered_total', 'covered_lost')
RECT_KEYS = ('inside_added', 'inside_correct', 'outside_added', 'outside_correct')
_METRICS = None


def official_module(name, starter):
    sys.path.insert(0, str(starter))
    spec = importlib.util.spec_from_file_location('ec523_stress_'+name, Path(starter)/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def distribution(rows):
    return {d: {k: sum(r['domain']==d.lower() and r.get('corruption', {}).get('kind')==k for r in rows)
                for k in KINDS} for d in DOMAINS}


def generate_stress(manifest, output, seed, starter):
    """Call the unchanged official generate(); refuse unverified old outputs."""
    signature = dict(seed=seed, source_sha256=inference.sha256(manifest),
                     generator_sha256=inference.sha256(Path(starter)/'make_stress.py'))
    output = Path(output)
    stamp = output/'generation.json'
    common = official_module('common', starter)
    if output.exists() and any(output.iterdir()):
        recorded = json.loads(stamp.read_text())
        if any(recorded[key] != value for key, value in signature.items()):
            raise ValueError('Existing stress generation does not match seed/source/generator')
        rows = common.load_manifest(output/'manifest.json')
        if {r['id'] for r in rows} != {r['id'] for r in common.load_manifest(manifest)} or any(not Path(r['image']).is_file() for r in rows):
            raise ValueError('Existing stress output is incomplete')
        return output/'manifest.json', recorded
    generator = official_module('make_stress', starter)
    started = time.perf_counter()
    result = generator.generate(Path(manifest), output, seed)
    elapsed = time.perf_counter()-started
    rows = common.load_manifest(result)
    report = dict(**signature, images=len(rows), seconds=elapsed,
                  seconds_per_image=elapsed/len(rows), by_domain_corruption=distribution(rows))
    stamp.write_text(json.dumps(report, indent=2)+'\n')
    print('GENERATED '+json.dumps(report), flush=True)
    return result, report


def rectangle_changes(draft, prediction, truth, valid, rectangle):
    """Partition valid added pixels by the official half-open [x,y,w,h] box."""
    x, y, width, height = map(int, rectangle)
    if min(x, y, width, height) < 0 or x+width > valid.shape[1] or y+height > valid.shape[0]:
        raise ValueError('Corruption rectangle is outside the original image')
    inside = np.zeros_like(valid, dtype=bool)
    inside[y:y+height, x:x+width] = True
    added = ~draft & prediction & valid
    result = {f'{name}_added': int((added & region).sum()) for name, region in (('inside', inside), ('outside', ~inside))}
    result.update({f'{name}_correct': int((added & region & truth).sum()) for name, region in (('inside', inside), ('outside', ~inside))})
    assert result['inside_added']+result['outside_added'] == int(added.sum())
    return result


def initialize_worker(starter):
    global _METRICS
    runner.probe.initialize_metric_worker(str(starter), (.5,))
    _METRICS = runner.probe.official_metrics(str(starter))


def analyze_image(row, predictions):
    started = time.perf_counter()
    with Image.open(row['label']) as source:
        labels = np.array(source, copy=True)
    valid, truth = labels != 0, labels == 3
    masks = {key: value.astype(bool) & valid for key, value in predictions.items()}
    draft = masks['y0_draft']
    gap, terminal, covered = diagnostics.baseline_skeleton_sets(truth, draft)
    result = {}
    for key, mask in masks.items():
        counts = _METRICS.image_counts(mask.astype(np.float32), labels)
        changes = diagnostics.change_counts(draft, mask, truth, valid)
        tracking = dict(gap_total=int(gap.sum()), gap_covered=int((gap & mask).sum()),
                        terminal_total=int(terminal.sum()), terminal_covered=int((terminal & mask).sum()),
                        covered_total=int(covered.sum()), covered_lost=int((covered & ~mask).sum()))
        rectangle = None
        if key == '4' and row.get('corruption', {}).get('kind') in ('shadow', 'occlusion'):
            rectangle = rectangle_changes(draft, mask, truth, valid, row['corruption']['rectangle_xywh'])
            assert rectangle['inside_added']+rectangle['outside_added'] == changes['added_correct']+changes['added_wrong']
        result[key] = dict(counts=asdict(counts), changes=changes, tracking=tracking, rectangle=rectangle)
    return row['domain'].title(), row.get('corruption', {}).get('kind'), result, time.perf_counter()-started, row.get('id')


def empty_buckets(keys, metrics):
    return {g: {k: dict(images=0, counts=asdict(metrics.Counts()),
                       changes=dict.fromkeys(CHANGE_KEYS, 0), tracking=dict.fromkeys(TRACK_KEYS, 0),
                       rectangle=dict.fromkeys(RECT_KEYS, 0), rectangle_images=0) for k in keys} for g in GROUPS}


def add_result(buckets, result):
    domain, kind, outputs, seconds = result[:4]
    groups = ['Overall', domain] + ([kind, f'{domain}/{kind}'] if kind else [])
    for group in groups:
        for key, value in outputs.items():
            target = buckets[group][key]
            target['images'] += 1
            for section in ('counts', 'changes', 'tracking'):
                for name, count in value[section].items():
                    target[section][name] += count
            if value['rectangle'] is not None:
                target['rectangle_images'] += 1
                for name, count in value['rectangle'].items():
                    target['rectangle'][name] += count
    return seconds


def add_paired_clean(paired, assignments, result):
    """Reuse one ordinary metric result in each stress-seed tile assignment."""
    domain, _, outputs, seconds, image_id = result
    for name, assignment in assignments.items():
        add_result(paired[name], (domain, assignment[image_id], outputs, seconds))


def finalize(buckets, metrics, stress=False):
    for group, values in buckets.items():
        for key, value in values.items():
            counts = metrics.Counts(**value['counts'])
            value['scores'] = metrics.summarize(counts) if counts.valid else None
            changes, tracking = value['changes'], value['tracking']
            added, deleted = changes['added_correct']+changes['added_wrong'], changes['deleted_correct']+changes['deleted_wrong']
            f = diagnostics.fraction
            value['diagnostics'] = dict(added_valid=f(added, counts.valid), deleted_valid=f(deleted, counts.valid),
                                       added_accuracy=f(changes['added_correct'], added), deleted_accuracy=f(changes['deleted_correct'], deleted),
                                       gap_repair=f(tracking['gap_covered'], tracking['gap_total']),
                                       terminal_completion=f(tracking['terminal_covered'], tracking['terminal_total']),
                                       new_break=f(tracking['covered_lost'], tracking['covered_total']))
            rect = value['rectangle']
            rect['inside_accuracy'] = f(rect['inside_correct'], rect['inside_added'])
            rect['outside_accuracy'] = f(rect['outside_correct'], rect['outside_added'])
        draft_score = values['y0_draft']['scores']
        for value in values.values():
            value['delta_quality'] = value['scores']['quality']-draft_score['quality'] if value['scores'] and draft_score else None
    for key in buckets['Overall']:
        overall = buckets['Overall'][key]['counts']
        assert all(overall[n] == sum(buckets[d][key]['counts'][n] for d in DOMAINS) for n in overall)
        if stress:
            assert all(overall[n] == sum(buckets[k][key]['counts'][n] for k in KINDS) for n in overall)
    return buckets


class ImageInputs(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return index, inference.image_tensor(self.rows[index]['image'], torch.device('cpu'))[0].numpy()


def image_batch(rows):
    return [row[0] for row in rows], np.stack([row[1] for row in rows])


def encode_images(rows, cache, args, device):
    phase = runner.stamp(device)
    inference.reset_peak(device)
    hashes = inference.verify_encoder(args.encoder_dir)
    baseline = inference.load_baseline(args.probe, args.encoder_dir, device)
    kwargs = dict(batch_size=1, num_workers=args.loader_workers, collate_fn=image_batch,
                  worker_init_fn=runner.probe.initialize_data_worker if args.loader_workers else None)
    if args.loader_workers:
        kwargs['multiprocessing_context'] = 'spawn'
    loader = DataLoader(ImageInputs(rows), **kwargs)
    size = 0
    with torch.no_grad():
        for indices, values in loader:
            images = torch.from_numpy(values).to(device)
            features = baseline.encoder(images)
            if features.dtype != torch.float32 or features.shape != (1, 1024, 64, 64) or not torch.isfinite(features).all():
                raise ValueError('Expected finite original float32 DINOv3 feature tensors')
            path = cache/f'{indices[0]:08d}.npy'
            np.save(path, features.cpu().numpy(), allow_pickle=False)
            size += path.stat().st_size
            del features, images
    peak = inference.peak_memory(device)
    head, release = inference.release_encoder(baseline, device)
    del baseline, loader
    return head, dict(seconds=runner.stamp(device)-phase, peak_cuda_bytes=peak, cache_bytes=size, release=release,
                      encoder_files_sha256=hashes, probe_sha256=inference.sha256(args.probe))


def load_model(path, device, split_hash):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if config.get('val_split_sha256') != split_hash or config['head_type'] != 'conv_upsample':
        raise ValueError('Checkpoint does not match the controlled v2 Val split/head')
    if config['reset_state'] or config['no_mask_feedback']:
        raise ValueError('Expected an unmodified e0_v2 checkpoint')
    model = runner.LoopRefiner(1024, **{key: config[key] for key in ('dimension', 'layers', 'heads', 'mlp_ratio', 'patch', 'head_type')})
    model.load_state_dict(checkpoint['state_dict'])
    model.requires_grad_(False).to(device).eval()
    return model, checkpoint['epoch']


def evaluate_model(rows, cache, head, checkpoint, args, device, official_root=None, paired_assignments=None):
    metrics = runner.probe.official_metrics(str(args.starter))
    keys = ['y0_draft', *map(str, args.ks)]
    buckets = empty_buckets(keys, metrics)
    paired_assignments = paired_assignments or {}
    paired = {name: empty_buckets(keys, metrics) for name in paired_assignments}
    def consume(result):
        seconds = add_result(buckets, result)
        if paired:
            add_paired_clean(paired, paired_assignments, result)
        return seconds
    elapsed = dict.fromkeys(map(str, args.ks), 0.)
    peaks = dict.fromkeys(map(str, args.ks), None)
    model, epoch = load_model(checkpoint, device, inference.sha256(args.val_split))
    pending, worker_seconds, images = deque(), 0., []
    phase = time.perf_counter()
    with mp.get_context('spawn').Pool(args.metric_workers, initializer=initialize_worker, initargs=(str(args.starter),)) as pool:
        for index, row in enumerate(rows):
            feature = np.load(cache/f'{index:08d}.npy', allow_pickle=False)
            if feature.dtype != np.float32:
                raise ValueError('Feature cache precision changed')
            features = torch.from_numpy(feature).to(device)
            with torch.no_grad():
                draft = head(features).float()
            inference.validate_logits(draft, (1,1,1024,1024))
            predictions = {'y0_draft': (draft.sigmoid()[0,0] >= .5).cpu().numpy().astype(np.uint8)}
            for key, answer, seconds, peak in runner.inference_steps(model, features, draft, args.ks, device):
                inference.validate_logits(answer, draft.shape)
                probability = answer.sigmoid()[0,0].float().cpu().numpy()
                predictions[key] = (probability >= .5).astype(np.uint8)
                elapsed[key] += seconds
                if peak is not None:
                    peaks[key] = max(peaks[key] or 0, peak)
                if official_root is not None and int(key) == args.official_check_k:
                    inference.write_probability(official_root/'probabilities'/f"{row['id']}.npy", probability)
            if official_root is not None:
                images.append(dict(id=row['id'], core_calls=args.official_check_k,
                                   source_core_calls=max(args.ks), input_sha256=inference.sha256(row['image'])))
            pending.append(pool.apply_async(analyze_image, (row, predictions)))
            if len(pending) >= 2*args.metric_workers:
                worker_seconds += consume(pending.popleft().get())
            del features, draft, predictions, feature, answer, probability
        while pending:
            worker_seconds += consume(pending.popleft().get())
    report = dict(epoch=epoch, checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=inference.sha256(checkpoint),
                  groups=finalize(buckets, metrics, rows[0]['condition']=='stress'),
                  refiner_seconds=elapsed, peak_cuda_bytes=peaks,
                  worker_seconds=worker_seconds, wall_seconds=time.perf_counter()-phase)
    if paired:
        report['paired_clean_groups'] = {name: finalize(values, metrics, True) for name, values in paired.items()}
    if official_root is not None:
        run = dict(protocol=inference.PROTOCOL, steps=args.official_check_k, demo=False, ablation='none',
                   source_loop_steps=max(args.ks),
                   checkpoint_sha256=report['checkpoint_sha256'], baseline_sha256=inference.sha256(args.probe),
                   submission_source_sha256=inference.source_hash(), images=images)
        (official_root/'run.json').write_text(json.dumps(run, indent=2)+'\n')
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return report


def regression(report, seed, args):
    reference = json.loads((ROOT/f'results/exp/e0_v2_s{seed}/metrics.json').read_text())['evaluation']['groups']
    checks = []
    for group in ('Overall', *DOMAINS):
        for key in report['groups'][group]:
            if key not in reference[group]:
                continue
            actual = report['groups'][group][key]['scores']['quality']
            expected = reference[group][key]['scores']['quality']
            item = dict(group=group, key=key, actual=actual, expected=expected,
                        absolute_difference=abs(actual-expected), passed=abs(actual-expected)<=1e-5)
            checks.append(item)
            print(('PASS' if item['passed'] else 'FAIL')+' ordinary regression '+json.dumps(item), flush=True)
    return checks


def pooled_models(datasets, seeds, metrics):
    """Pool counts across corruption instances within each model seed first."""
    pooled = {}
    for seed in seeds:
        sources = [d['models'][str(seed)]['groups'] for d in datasets]
        keys = list(sources[0]['Overall'])
        out = empty_buckets(keys, metrics)
        for group in GROUPS:
            for key in keys:
                target = out[group][key]
                for source in sources:
                    value = source[group][key]
                    target['images'] += value['images']
                    target['rectangle_images'] += value['rectangle_images']
                    for section, names in (('counts', target['counts']), ('changes', CHANGE_KEYS), ('tracking', TRACK_KEYS), ('rectangle', RECT_KEYS)):
                        for name in names:
                            target[section][name] += value[section][name]
        pooled[str(seed)] = dict(groups=finalize(out, metrics, True))
    return dict(models=pooled)


def mean_std(values):
    values = [v for v in values if v is not None]
    return dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)) if len(values)>1 else None, n=len(values)) if values else dict(mean=None, std=None, n=0)


def format_values(values):
    value = mean_std(values)
    if value['mean'] is None:
        return '--'
    return f"{value['mean']:.6f}" + (f" +/- {value['std']:.6f}" if value['std'] is not None else ' (n=1)')


def aggregate_model_seeds(report):
    datasets = dict(report['datasets'], stress_pooled=report['combined_stress'])
    out = {}
    for name, dataset in datasets.items():
        out[name] = {}
        for group in GROUPS:
            out[name][group] = {}
            for key in next(iter(dataset['models'].values()))['groups'][group]:
                rows = [m['groups'][group][key] for m in dataset['models'].values()]
                out[name][group][key] = {
                    'delta_quality': mean_std([r['delta_quality'] for r in rows]),
                    'scores': {metric: mean_std([r['scores'][metric] if r['scores'] else None for r in rows])
                               for metric in ('road_iou','cldice','boundary_iou','quality')},
                    'diagnostics': {metric: mean_std([r['diagnostics'][metric] for r in rows]) for metric in rows[0]['diagnostics']}}
    return out


def summary(report):
    lines = ['# Frozen v2 ordinary/stress evaluation', '',
             'Observed results only: no training or checkpoint selection was performed.',
             'Official Counts are pooled before summarize. Sample standard deviations use model seeds (ddof=1).',
             'Pooled stress results first pool all stress instances within each model seed; repeated source tiles are not independent model seeds.',
             'Empty subsets and zero-denominator diagnostics are reported as --.', '']
    sets = dict(report['datasets'])
    if report.get('combined_stress'):
        sets['stress_pooled'] = report['combined_stress']

    def table(title, headers, rows):
        lines.extend(['## '+title, '', '| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'])
        lines.extend('| '+' | '.join(map(str,row))+' |' for row in rows)
        lines.append('')

    keys = [str(k) for k in report['config']['ks']]
    def values(dataset, group, key, section, name):
        return [(m['groups'][group][key][section] or {}).get(name) for m in dataset['models'].values()]
    table('Ordinary vs stress A (model-seed mean +/- sample std)', ['Set', 'y0_draft', *('K='+k for k in keys)],
          [[name, *[format_values(values(d,'Overall',k,'scores','quality')) for k in ['y0_draft',*keys]]] for name,d in sets.items()])
    table('Official scores by domain and corruption', ['Set','Group','K/reference','IoU','clDice','BIoU','A'],
          [[name,group,key,*[format_values(values(dataset,group,key,'scores',metric))
                             for metric in ('road_iou','cldice','boundary_iou','quality')]]
           for name,dataset in sets.items() for group in GROUPS for key in ['y0_draft',*keys]])
    gain_rows, diagnostic_rows, rectangles = [], [], []
    for name, dataset in sets.items():
        for group in GROUPS:
            for key in keys:
                groups = [m['groups'][group] for m in dataset['models'].values()]
                gains = [(g[key]['scores']['quality']-g['y0_draft']['scores']['quality']) if g[key]['scores'] else None for g in groups]
                gain_rows.append([name,group,key,format_values(values(dataset,group,'y0_draft','scores','quality')),
                                  format_values(values(dataset,group,key,'scores','quality')),format_values(gains)])
                diagnostic_rows.append([name,group,key,*[format_values(values(dataset,group,key,'diagnostics',metric))
                                                         for metric in ('added_valid','deleted_valid','added_accuracy','deleted_accuracy','gap_repair','new_break')]])
            if '4' in keys and any(m['groups'][group]['4']['rectangle_images'] for m in dataset['models'].values()):
                for metric in ('inside_added','inside_correct','inside_accuracy','outside_added','outside_correct','outside_accuracy'):
                    rectangles.append([name,group,metric,format_values(values(dataset,group,'4','rectangle',metric))])
    table('Gain over the same corrupted-image y0 draft', ['Set','Group','K','y0 A','Refined A','Delta A'], gain_rows)
    paired_rows = []
    for name, clean in report['paired_clean'].items():
        corrupted = sets[name]
        for group in GROUPS:
            for key in keys:
                paired_rows.append([name, group, key,
                                    format_values(values(clean,group,'y0_draft','scores','quality')),
                                    format_values(values(clean,group,key,'scores','quality')),
                                    format_values([m['groups'][group][key]['delta_quality'] for m in clean['models'].values()]),
                                    format_values(values(corrupted,group,'y0_draft','scores','quality')),
                                    format_values(values(corrupted,group,key,'scores','quality')),
                                    format_values([m['groups'][group][key]['delta_quality'] for m in corrupted['models'].values()])])
    table('Paired clean/stress gains on the same assigned tiles',
          ['Stress set','Group','K','Clean y0 A','Clean refined A','Clean gain','Stress y0 A','Stress refined A','Stress gain'],paired_rows)
    table('Valid-pixel flips and exact fixed-y0 skeleton tracking', ['Set','Group','K','Added/valid','Deleted/valid','Correct/added','Correct/deleted','Gap repair','New break'], diagnostic_rows)
    table('K4 added pixels inside/outside shadow and occlusion rectangles', ['Set','Group','Quantity','Model-seed mean +/- std'], rectangles)
    lines += ['Rectangle totals include only shadow/occlusion images; blur and ordinary have no rectangle.',
              'Coverage is exact intersection with the shared y0 skeleton sets, without dilation tolerance.',
              'Every per-domain/per-corruption official metric and its integer Counts is retained in metrics.json.', '']
    return '\n'.join(lines)


def parse_ints(value):
    result = sorted(set(map(int, value.split(','))))
    if not result:
        raise ValueError('Expected nonempty comma-separated integers')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--val-split', type=Path, default=ROOT/'configs/val_split_seed0.json')
    parser.add_argument('--work-dir', type=Path, default=ROOT/'runs/stress')
    parser.add_argument('--out-dir', type=Path, default=ROOT/'results/stress/v2')
    parser.add_argument('--encoder-dir', type=Path, default=ROOT/'submission/encoder')
    parser.add_argument('--probe', type=Path, default=ROOT/'submission/baseline.safetensors')
    parser.add_argument('--checkpoint-template', default='runs/exp/e0_v2_s{seed}/best.pt')
    parser.add_argument('--stress-seeds', type=parse_ints, default=[523,524,525])
    parser.add_argument('--model-seeds', type=parse_ints, default=[0,1,2])
    parser.add_argument('--ks', type=parse_ints, default=[1,2,4,8])
    parser.add_argument('--limit-per-domain', type=int)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--loader-workers', type=int, default=1)
    parser.add_argument('--metric-workers', type=int, default=6)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--cleanup-stress', action='store_true', help='Remove generated pilot PNG directories after evaluation')
    parser.add_argument('--official-check-k', type=int, default=4)
    args = parser.parse_args()
    args.starter = Path(os.environ['EC523_STARTER_DIR'])
    if min(args.ks)<1 or max(args.ks)>8 or args.metric_workers<1 or args.loader_workers<0:
        parser.error('Use K1..8 and positive metric/nonnegative loader workers')
    if args.limit_per_domain is not None and args.limit_per_domain<1:
        parser.error('--limit-per-domain must be positive')
    if int(os.environ.get('NSLOTS','0')) and max(1+args.metric_workers,1+args.loader_workers)>int(os.environ['NSLOTS']):
        parser.error('Active workers plus main exceed NSLOTS; encoding/scoring phases do not overlap')
    for path in (args.work_dir,args.out_dir):
        if not path.resolve().is_relative_to(ROOT):
            parser.error('Work and report directories must remain in the repository')
    if not args.generate_only and any((args.out_dir/name).exists() for name in ('metrics.json','summary.md')):
        parser.error('Refusing to overwrite existing reports')
    common = official_module('common', args.starter)
    ordinary = write_manifest(args.work_dir/'ordinary.json', build_manifest(args.val_split, limit_per_domain=args.limit_per_domain))
    manifests, generations = {'ordinary':ordinary}, {}
    generated_paths = []
    for seed in args.stress_seeds:
        path, record = generate_stress(ordinary,args.work_dir/f'seed{seed}',seed,args.starter)
        manifests[f'stress_{seed}'], generations[str(seed)] = path, record
        generated_paths.append(path.parent)
    if args.generate_only:
        print(json.dumps(generations,indent=2),flush=True)
        return
    device = torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    runner.probe.initialize_data_worker(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    report = dict(config=dict(ks=args.ks,model_seeds=args.model_seeds,stress_seeds=args.stress_seeds,
                              limit_per_domain=args.limit_per_domain,device=args.device,threshold=.5,
                              loader_workers=args.loader_workers,metric_workers=args.metric_workers,
                              val_split_sha256=inference.sha256(args.val_split)),
                  generation=generations,datasets={},ordinary_regressions=[],official_evaluator_check=None)
    assignments = {name: {r['id']: r['corruption']['kind'] for r in common.load_manifest(manifest)}
                   for name, manifest in manifests.items() if name != 'ordinary'}
    try:
        for name, manifest in manifests.items():
            rows = common.load_manifest(manifest)
            cache, required = inference.create_cache(args.cache_dir,args.out_dir,len(rows))
            dataset = dict(images=len(rows),by_domain_corruption=distribution(rows),manifest_sha256=inference.sha256(manifest),models={})
            try:
                head, dataset['encoding'] = encode_images(rows,cache,args,device)
                for seed in args.model_seeds:
                    path = (ROOT/args.checkpoint_template.format(seed=seed)).resolve(strict=True)
                    official_root = None
                    if name=='stress_523' and seed==0 and args.official_check_k in args.ks:
                        official_root=args.work_dir/'official_check'
                        common.fresh_directory(official_root)
                    model_report = evaluate_model(rows,cache,head,path,args,device,official_root,
                                                  paired_assignments=assignments if name=='ordinary' else None)
                    dataset['models'][str(seed)] = model_report
                    print('EVALUATED '+name+' model_seed='+str(seed)+' Overall='+json.dumps({k:v['scores']['quality'] if v['scores'] else None for k,v in model_report['groups']['Overall'].items()}),flush=True)
                    if name=='ordinary' and device.type=='cuda' and args.limit_per_domain is None:
                        report['ordinary_regressions'] += regression(model_report,seed,args)
                    if official_root is not None:
                        output = args.work_dir/'official_evaluator.json'
                        subprocess.run([sys.executable,str(args.starter/'evaluate.py'),'--manifest',str(manifest),
                                        '--predictions',str(official_root),'--output',str(output)],check=True)
                        actual=json.loads(output.read_text())['overall']['quality']
                        expected=model_report['groups']['Overall'][str(args.official_check_k)]['scores']['quality']
                        report['official_evaluator_check']=dict(k=args.official_check_k,actual=actual,expected=expected,
                                                               absolute_difference=abs(actual-expected),passed=abs(actual-expected)<=1e-6)
                        shutil.rmtree(official_root)
                        print(('PASS' if abs(actual-expected)<=1e-6 else 'FAIL')+' official evaluator '+json.dumps(report['official_evaluator_check']),flush=True)
                        if abs(actual-expected)>1e-6:
                            raise AssertionError('Official evaluate.py cross-check failed')
                del head
            finally:
                shutil.rmtree(cache)
                print('CACHE_REMOVED '+name,flush=True)
            first=next(iter(dataset['models'].values()))
            assert all(m['groups']['Overall']['y0_draft']['counts']==first['groups']['Overall']['y0_draft']['counts'] for m in dataset['models'].values())
            report['datasets'][name]=dataset
        stress=[d for name,d in report['datasets'].items() if name.startswith('stress_')]
        metrics=runner.probe.official_metrics(str(args.starter))
        report['combined_stress']=pooled_models(stress,args.model_seeds,metrics)
        ordinary_models=report['datasets']['ordinary']['models']
        report['paired_clean']={name: dict(models={str(seed):dict(groups=ordinary_models[str(seed)]['paired_clean_groups'][name])
                                                   for seed in args.model_seeds}) for name in assignments}
        report['paired_clean']['stress_pooled']=pooled_models(list(report['paired_clean'].values()),args.model_seeds,metrics)
        for model in ordinary_models.values():
            del model['paired_clean_groups']
        report['model_seed_aggregates']=aggregate_model_seeds(report)
        report['ordinary_regression_passed'] = all(c['passed'] for c in report['ordinary_regressions']) if report['ordinary_regressions'] else None
        if report['ordinary_regression_passed'] is not None:
            print(('PASS' if report['ordinary_regression_passed'] else 'FAIL')+' ordinary regression summary (all evaluations completed)',flush=True)
        args.out_dir.mkdir(parents=True,exist_ok=True)
        text=json.dumps(report,indent=2,allow_nan=False)+'\n'
        account=os.environ.get('USER','')
        if re.search(r'/(?:project[a-z]+|usr\d+|home)/',text) or (account not in ('','root','nobody') and re.search(r'\b'+re.escape(account)+r'\b',text)):
            raise ValueError('Personal path leaked into portable result report')
        (args.out_dir/'metrics.json').write_text(text)
        (args.out_dir/'summary.md').write_text(summary(report))
        print('COMPLETE '+json.dumps(dict(datasets=len(report['datasets']),models_per_dataset=len(args.model_seeds))),flush=True)
    finally:
        if args.cleanup_stress:
            for path in generated_paths:
                shutil.rmtree(path)
            print('PILOT_STRESS_IMAGES_REMOVED',flush=True)


if __name__ == '__main__':
    main()
