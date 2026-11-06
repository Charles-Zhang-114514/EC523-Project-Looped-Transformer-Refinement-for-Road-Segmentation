#!/usr/bin/env python3
"""Offline confidence gating, fixed-answer inference and stored stress probes."""
import argparse
from collections import deque
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
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
from scripts import train_refiner as runner
from scripts import eval_stress as shared
from submission import predict as inference

DOMAINS = ('Urban', 'Rural', 'Overall')
TAUS = tuple(round(.5 + .05*i, 2) for i in range(10))
BANDS = ('[0.5,0.6)', '[0.6,0.7)', '[0.7,0.8)', '[0.8,0.9)', '[0.9,1.0]')
MODE = 'y0_answer_keep_residual'
_METRICS = None


def gated_mask(base, probability, tau):
    """Keep every native deletion; filter only additions by native confidence."""
    base = np.asarray(base, dtype=bool)
    native = probability >= .5
    added, deleted = ~base & native, base & ~native
    return (base & ~deleted) | (added & (probability >= tau))


def confidence_strata(base, probability, truth, valid):
    added = ~base.astype(bool) & (probability >= .5) & valid
    edges = np.array([.5, .6, .7, .8, .9, 1.], dtype=probability.dtype)
    result = []
    for index, name in enumerate(BANDS):
        upper = probability <= edges[index+1] if index == 4 else probability < edges[index+1]
        mask = added & (probability >= edges[index]) & upper
        result.append(dict(band=name, pixels=int(mask.sum()), correct=int((mask & truth).sum())))
    assert sum(row['pixels'] for row in result) == int(added.sum())
    return result


def initialize_worker(starter):
    global _METRICS
    shared.initialize_worker(starter)
    _METRICS = runner.probe.official_metrics(starter)


def score_task(row, base, probabilities, new_masks, taus):
    started = time.perf_counter()
    with Image.open(row['label']) as source:
        labels = np.array(source, copy=True)
    valid, truth = labels != 0, labels == 3
    native = {k: probability >= .5 for k, probability in probabilities.items()}
    # The existing direct evaluation path is an independent Counts reference.
    direct_predictions = dict(y0_draft=base.astype(np.uint8),
                              **{k: value.astype(np.uint8) for k, value in native.items()})
    direct = runner.score_image(row['domain'].title(), row['id'], direct_predictions,
                                labels, geometry=False)[2]

    def value(mask, counts):
        additions = ~base.astype(bool) & mask & valid
        return dict(counts=counts, added=int(additions.sum()), correct=int((additions & truth).sum()))

    outputs = {'y0_draft': value(base.astype(bool), direct['y0_draft']['counts'])}
    for k, probability in probabilities.items():
        outputs['direct:'+k] = value(native[k], direct[k]['counts'])
        for tau in taus:
            mask = gated_mask(base, probability, tau)
            assert not (mask & ~native[k]).any()
            assert np.array_equal(mask & base.astype(bool), native[k] & base.astype(bool))
            counts = asdict(_METRICS.image_counts(mask.astype(np.float32), labels))
            if tau == .5:
                assert np.array_equal(mask, native[k])
                assert counts == direct[k]['counts']
            outputs[f'{k}:{tau:.2f}'] = value(mask, counts)
    bands = confidence_strata(base, probabilities['4'], truth, valid)
    extra = shared.analyze_image(row, dict(y0_draft=base, **new_masks)) if new_masks else None
    if extra is not None:
        assert extra[2]['1']['counts'] == direct['1']['counts']
    return row['domain'].title(), outputs, bands, extra, time.perf_counter()-started


def choose_tau(selection, taus):
    """Only Val-sel can select; ascending tau breaks exact ties conservatively."""
    if selection['split'] != 'Val-sel':
        raise ValueError('Threshold selection must use Val-sel, never Val-rep')
    candidates = []
    for tau in sorted(taus):
        values = selection['groups']['Overall']
        score = .33*values[f'1:{tau:.2f}']['scores']['quality'] + .67*values[f'4:{tau:.2f}']['scores']['quality']
        candidates.append(dict(tau=f'{tau:.2f}', proxy=score))
    best = max(candidates, key=lambda row: row['proxy'])
    return dict(best, source='Val-sel', weights={'1': .33, '4': .67}, candidates=candidates,
                tie_break='lowest tau among exact ties')


def make_rows(split, part, limit):
    datasets = runner.protocol_datasets(split, part, limit)
    rows = [dict(id=f'{domain.lower()}/{image.stem}', domain=domain.lower(), condition='ordinary',
                 image=str(image), label=str(label))
            for domain, dataset in datasets.items() for image, label in dataset.pairs]
    return rows


def summarize_buckets(buckets, bands, part, metrics):
    groups = {}
    for domain in DOMAINS:
        domains = ('Urban', 'Rural') if domain == 'Overall' else (domain,)
        groups[domain] = {}
        for key in buckets['Urban']:
            counts = metrics.Counts()
            added = correct = 0
            for d in domains:
                item = buckets[d][key]
                counts = counts + metrics.Counts(**item['counts'])
                added += item['added']
                correct += item['correct']
            groups[domain][key] = dict(counts=asdict(counts), scores=metrics.summarize(counts),
                                       added_pixels=added, added_correct=correct,
                                       added_precision=correct/added if added else None)
        groups[domain]['strata_K4'] = []
        for index, name in enumerate(BANDS):
            pixels = sum(bands[d][index]['pixels'] for d in domains)
            correct = sum(bands[d][index]['correct'] for d in domains)
            groups[domain]['strata_K4'].append(dict(band=name, pixels=pixels, correct=correct,
                                                   precision=correct/pixels if pixels else None))
        assert sum(b['pixels'] for b in groups[domain]['strata_K4']) == groups[domain]['direct:4']['added_pixels']
        for k in ('1', '4'):
            assert groups[domain][k+':0.50'] == groups[domain]['direct:'+k]
    return dict(split=part, groups=groups, regression=dict(tau_050_equals_direct=True))


class CachedEncoder(torch.nn.Module):
    """CPU regression adapter: replay exact float32 features without encoding."""
    def __init__(self, cache):
        super().__init__()
        self.cache = cache

    def forward(self, images):
        index = int(images[0, 0, 0, 0].item())
        return torch.from_numpy(np.load(self.cache/f'{index:08d}.npy', allow_pickle=False)).to(images.device)


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, rows, domain):
        self.rows = [(index, row) for index, row in enumerate(rows) if row['domain']==domain.lower()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        index, row = self.rows[item]
        with Image.open(row['label']) as source:
            labels = np.array(source, copy=True)
        return np.full((3,1,1),index,np.float32), (labels==3).astype(np.float32), labels!=0, row['id'], labels


def cpu_direct_regression(rows, cache, head, name, observed, args):
    model,_ = shared.load_model(ROOT/'runs/exp'/name/'best.pt', torch.device('cpu'), inference.sha256(args.val_split))
    datasets = {d:CachedDataset(rows,d) for d in ('Urban','Rural')}
    direct = runner.evaluate(CachedEncoder(cache), head, model, datasets, [1,4], torch.device('cpu'),
                             loader_workers=0, metric_workers=args.metric_workers, geometry=False)
    for domain in DOMAINS:
        for k in ('1','4'):
            original = direct['groups'][domain][k]['scores']
            gated = observed['groups'][domain][k+':0.50']
            assert original['counts']==gated['counts']
            for metric in ('road_iou','cldice','boundary_iou','quality'):
                assert original[metric]==gated['scores'][metric]
    del model
    print('PASS CPU original runner.evaluate Counts/scores equal tau0.50 '+name,flush=True)
    return dict(passed=True, method='Original runner.evaluate with exact cached-encoder adapter; no re-encoding',
                groups={d:{k:direct['groups'][d][k]['scores'] for k in ('y0_draft','1','4')} for d in DOMAINS})


def evaluate_cached(rows, cache, head, name, part, args, device):
    path = ROOT/'runs/exp'/name/'best.pt'
    model, epoch = shared.load_model(path, device, inference.sha256(args.val_split))
    reference = json.loads((ROOT/'results/exp'/name/'metrics.json').read_text())
    if epoch != reference['best_epoch'] or reference['config'].get('untied_steps', 0):
        raise ValueError('Expected the selected shared checkpoint')
    expected_lambda = 10 if name.startswith('e5_noadd') else 0
    if reference['config'].get('noadd_lambda', 0) != expected_lambda:
        raise ValueError('Checkpoint family and reported objective disagree')
    keys = ['y0_draft', 'direct:1', 'direct:4'] + [f'{k}:{tau:.2f}' for k in (1, 4) for tau in args.taus]
    metrics = runner.probe.official_metrics(str(args.starter))
    buckets = {d: {k: dict(counts=asdict(metrics.Counts()), added=0, correct=0) for k in keys}
               for d in ('Urban', 'Rural')}
    bands = {d: [dict(pixels=0, correct=0) for _ in BANDS] for d in buckets}
    extra_buckets = shared.empty_buckets(['y0_draft', *map(str, args.intervention_ks)], metrics)
    intervene = part == 'Val-rep' and name.startswith('e0_v2')
    pending, worker_seconds = deque(), 0.
    refiner_times = dict.fromkeys(('1', '4'), 0.)
    intervention_times = dict.fromkeys(map(str, args.intervention_ks), 0.)
    peaks = dict.fromkeys(('1', '4'), None)
    started = time.perf_counter()

    def consume():
        nonlocal worker_seconds
        domain, outputs, strata, extra, seconds = pending.popleft().get()
        for key, item in outputs.items():
            for field, value in item['counts'].items():
                buckets[domain][key]['counts'][field] += value
            for field in ('added', 'correct'):
                buckets[domain][key][field] += item[field]
        for target, source in zip(bands[domain], strata):
            for field in ('pixels', 'correct'):
                target[field] += source[field]
        if extra is not None:
            shared.add_result(extra_buckets, extra)
        worker_seconds += seconds

    with mp.get_context('spawn').Pool(args.metric_workers, initializer=initialize_worker,
                                      initargs=(str(args.starter),)) as pool:
        for index, row in enumerate(rows):
            feature = np.load(cache/f'{index:08d}.npy', allow_pickle=False)
            if feature.dtype != np.float32:
                raise ValueError('Feature cache must remain float32')
            features = torch.from_numpy(feature).to(device)
            with torch.no_grad():
                draft = head(features).float()
            inference.validate_logits(draft, (1, 1, 1024, 1024))
            base = (draft.sigmoid()[0, 0] >= .5).cpu().numpy()
            probabilities = {}
            first_logits = None
            for k, answer, seconds, peak in runner.inference_steps(model, features, draft, [1, 4], device):
                inference.validate_logits(answer, draft.shape)
                probabilities[k] = answer.sigmoid()[0, 0].float().cpu().numpy()
                if k == '1':
                    first_logits = answer.clone()
                refiner_times[k] += seconds
                peaks[k] = max(peaks[k] or 0, peak) if peak is not None else None
            new_masks = {}
            if intervene:
                for k, answer, seconds, _ in runner.inference_steps(
                        model, features, draft, args.intervention_ks, device, MODE):
                    inference.validate_logits(answer, draft.shape)
                    if k == '1' and not torch.equal(answer, first_logits):
                        raise AssertionError('Fixed-answer intervention K1 differs from none logits')
                    new_masks[k] = (answer.sigmoid()[0, 0] >= .5).cpu().numpy().astype(np.uint8)
                    intervention_times[k] += seconds
            pending.append(pool.apply_async(score_task, (row, base, probabilities, new_masks, args.taus)))
            if len(pending) >= 2*args.metric_workers:
                consume()
            del features, draft, probabilities, new_masks, feature, answer, first_logits
        while pending:
            consume()
    result = summarize_buckets(buckets, bands, part, metrics)
    result.update(images=len(rows), epoch=epoch, checkpoint=f'runs/exp/{name}/best.pt',
                  checkpoint_sha256=inference.sha256(path), refiner_seconds=refiner_times,
                  peak_cuda_bytes=peaks, worker_seconds=worker_seconds,
                  wall_seconds=time.perf_counter()-started)
    if intervene:
        groups = shared.finalize(extra_buckets, metrics)
        result['intervention'] = dict(mode=MODE, groups={d:groups[d] for d in DOMAINS},
                                      refiner_seconds=intervention_times,
                                      regression=dict(K1_logits_equal_none=True, K1_counts_equal_none=True))
        for domain in DOMAINS:
            assert groups[domain]['1']['counts'] == result['groups'][domain]['direct:1']['counts']
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def stress_supplement(path):
    source = json.loads(Path(path).read_text())
    result = []
    for condition in ('ordinary', 'shadow', 'occlusion', 'blur'):
        for domain in DOMAINS:
            if condition == 'ordinary':
                dataset = source['datasets']['ordinary']
                clean = dataset
                key = domain
            else:
                dataset = source['combined_stress']
                clean = source['paired_clean']['stress_pooled']
                key = condition if domain == 'Overall' else domain+'/'+condition
            model_groups = [dataset['models'][str(s)]['groups'][key] for s in range(3)]
            assert all(g['y0_draft']['counts'] == model_groups[0]['y0_draft']['counts'] for g in model_groups)
            scores = model_groups[0]['y0_draft']['scores']
            clean_scores = clean['models']['0']['groups'][key]['y0_draft']['scores']
            fields = ('road_iou', 'cldice', 'boundary_iou', 'quality')
            precision = [g['4']['diagnostics']['added_accuracy'] for g in model_groups]
            line = scores['road_iou']/(1+scores['road_iou'])
            result.append(dict(condition=condition, domain=domain,
                               scores={f:scores[f] for f in fields},
                               paired_clean={f:clean_scores[f] for f in fields},
                               drop={f:clean_scores[f]-scores[f] for f in fields},
                               break_even=line, added_precision_mean=statistics.mean(precision),
                               added_precision_std=statistics.stdev(precision),
                               above_seeds=sum(p>line for p in precision)))
    return dict(rows=result, source='results/stress/v2/metrics.json',
                sha256=inference.sha256(path),
                clean_reference='Matched clean tile assignments for each category, Counts-pooled over stress seeds 523/524/525')


def intervention_comparison(report):
    rows = []
    limited = report['config']['limit_per_domain'] is not None
    for name, splits in report['gating'].items():
        if not name.startswith('e0_v2'):
            continue
        seed = int(name[-1])
        new = splits['Val-rep']['intervention']['groups']
        for domain in DOMAINS:
            item = new[domain]['4'] if '4' in new[domain] else None
            rows.append(dict(seed=seed, mode=MODE, domain=domain,
                             A={k:v['scores']['quality'] for k,v in new[domain].items() if k != 'y0_draft'},
                             diagnostics=item['diagnostics'] if item else None))
        if limited:
            continue
        original = json.loads((ROOT/f'results/exp_infer/e0_v2_s{seed}/metrics.json').read_text())
        if original['evaluated_epoch'] != splits['Val-rep']['epoch']:
            raise ValueError('Stored inference comparison uses another checkpoint epoch')
        for mode, evaluation in original['evaluation']['ablations'].items():
            for domain in DOMAINS:
                group = evaluation['groups'][domain]
                item = group['4']
                c, t = item['extra']['changes'], item['extra']['tracking']
                rows.append(dict(seed=seed, mode=mode, domain=domain,
                    A={str(k):group[str(k)]['scores']['A'] for k in (1,2,4,8)},
                    diagnostics=dict(added_valid=item['added_road_fraction'], deleted_valid=item['deleted_road_fraction'],
                                     added_accuracy=c['added_correct_fraction'], deleted_accuracy=c['deleted_correct_fraction'],
                                     gap_repair=t['gap_repair_fraction'], new_break=t['new_break_fraction'])))
    return dict(rows=rows, historical_full_set_comparisons=not limited,
                note='Historical full-Val-rep rows are omitted for a limited CPU subset')


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def number(value):
    return '--' if value is None else f'{value:.6f}'


def mean_std(values):
    if not values or any(v is None for v in values):
        return '--'
    return number(statistics.mean(values)) + (' +/- '+number(statistics.stdev(values)) if len(values)>1 else ' (n=1)')


def markdown(report):
    lines = ['# Inference probes', '', 'Observed report status: '+report['status']+'.', '',
             f"Observed scope: device={report['config']['device']}; limit per domain={report['config']['limit_per_domain']}; requested checkpoints={len(report['config']['checkpoints'])}.", '',
             '## Confidence gating', '',
             'Observed selection protocol: one tau per checkpoint, selected only on Val-sel by 0.33*A(K1)+0.67*A(K4). Exact ties keep the lowest tau. Val-rep curves are diagnostic only.']
    selected_rows, curve_rows, band_rows = [], [], []
    for name, splits in report['gating'].items():
        selected = report['selections'][name]
        for domain in DOMAINS:
            for k in ('1','4'):
                item = splits['Val-rep']['groups'][domain][k+':'+selected['tau']]
                score = item['scores']
                selected_rows.append([name, selected['tau'], selected['proxy'], domain, k,
                                      *[number(score[f]) for f in ('road_iou','cldice','boundary_iou','quality')],
                                      item['added_pixels'], number(item['added_precision'])])
            for part in ('Val-sel', 'Val-rep'):
                for k in ('1','4'):
                    for tau in report['config']['taus']:
                        item = splits[part]['groups'][domain][f'{k}:{tau:.2f}'];score=item['scores']
                        curve_rows.append([name,part,domain,k,f'{tau:.2f}',
                            *[number(score[f]) for f in ('road_iou','cldice','boundary_iou','quality')],
                            item['added_pixels'],number(item['added_precision'])])
            for b in splits['Val-rep']['groups'][domain]['strata_K4']:
                band_rows.append([name,domain,b['band'],b['pixels'],b['correct'],number(b['precision'])])
    if not selected_rows:
        lines += ['', 'Observed gating evaluation pending manual GPU execution.']
    else:
        lines += ['', table(['Checkpoint','Selected tau','Val-sel proxy','Domain','K','IoU','clDice','BIoU','A','Retained added pixels','Added precision'], selected_rows),
                  '', '### Observed full curves', '',
                  table(['Checkpoint','Set','Domain','K','Tau','IoU','clDice','BIoU','A','Retained added pixels','Added precision'],curve_rows),
                  '', '### Observed K4 confidence strata on Val-rep', '',
                  table(['Checkpoint','Domain','Band','Added pixels','Correct pixels','Precision'],band_rows),
                  '', 'Observed tau=0.50 equals direct evaluation for all masks, Counts and scores. Features remain float32; probabilities remain float32 in bounded worker tasks.']
    lines += ['', '## Fixed y0 answer with current residual', '',
              'Observed new-mode semantics: answer tokens use y0; hidden memory and the current-logit residual base accumulate normally. Historical no-mask-feedback instead uses y0 as both answer and residual base.']
    comparison = report.get('intervention_comparison', {'rows':[]})
    rows, diagnostic_rows = [], []
    for mode in (*runner.INFER_ABLATIONS, MODE):
        for domain in DOMAINS:
            items=[r for r in comparison['rows'] if r['mode']==mode and r['domain']==domain]
            if not items:
                continue
            rows.append([mode,domain,*[mean_std([r['A'].get(str(k)) for r in items]) for k in (1,2,4,8)]])
            diagnostic_rows.append([mode,domain,*[mean_std([r['diagnostics'][f] if r['diagnostics'] else None for r in items])
                for f in ('added_valid','added_accuracy','deleted_valid','deleted_accuracy','gap_repair','new_break')]])
    if rows:
        lines += ['',table(['Mode','Domain','A K1','A K2','A K4','A K8'],rows), '',
                  table(['Mode','Domain','Added/valid','Added precision','Deleted/valid','Deleted precision','Gap repair','New break'],diagnostic_rows),
                  '',comparison['note'], '', 'Observed new-mode K1 logits and official Counts exactly equal none on the same forward inputs.']
    else:
        lines += ['', 'Observed new-mode evaluation pending manual GPU execution.']
    stress = report['stress']['rows']
    lines += ['', '## Stored stress supplement', '',
              'Observed data only from existing stress results. Stress seeds 523/524/525 are Counts-pooled before official summarize in that source. Category drops compare the matched clean tile assignments, not an unmatched clean subset. Positive drop means clean minus stress. y0 Counts are identical across model seeds and are counted once.']
    lines += ['',table(['Condition','Domain','y0 IoU','y0 clDice','y0 BIoU','y0 A','IoU drop','clDice drop','BIoU drop','A drop'],[
        [r['condition'],r['domain'],*[number(r['scores'][f]) for f in ('road_iou','cldice','boundary_iou','quality')],
         *[number(r['drop'][f]) for f in ('road_iou','cldice','boundary_iou','quality')]] for r in stress]), '',
        table(['Condition','Domain','IoU break-even precision (%)','v2 K4 added precision mean +/- sample std (%)','Relation','Seeds above'],[
            [r['condition'],r['domain'],number(100*r['break_even']),
             number(100*r['added_precision_mean'])+' +/- '+number(100*r['added_precision_std']),
             'above' if r['added_precision_mean']>r['break_even'] else 'below',f"{r['above_seeds']}/3"] for r in stress]),
        '', 'Observed break-even formula: IoU_y0/(1+IoU_y0), an add-only IoU comparison, not an A-score guarantee. Undefined zero-denominator precisions are --.',
        '', 'Observed provenance: stress rows copy datasets.ordinary.models.0.groups.{domain}.y0_draft.scores or combined_stress.models.0.groups.{category/domain}.y0_draft.scores. Drops use paired_clean.stress_pooled for the same group. Added precision uses the corresponding three models at 4.diagnostics.added_accuracy.']
    return '\n'.join(lines)+'\n'


def write_report(args, report):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ('metrics.json', 'summary.md'):
        path = args.out_dir/name
        if path.exists():
            old = json.loads((args.out_dir/'metrics.json').read_text())
            if old.get('status') != 'stress_only' or old.get('stress',{}).get('sha256') != report['stress']['sha256']:
                raise ValueError('Refusing to overwrite a completed/different report')
    text = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if re.search(r'/(?:project[a-z]+|usr\d+|home)/', text):
        raise ValueError('Absolute personal path in portable report')
    (args.out_dir/'summary.md').write_text(markdown(report))
    (args.out_dir/'metrics.json').write_text(text)


def parse_taus(text):
    values = [float(value) for value in text.split(',')]
    if len(set(values)) != len(values) or .5 not in values or any(v not in TAUS for v in values):
        raise argparse.ArgumentTypeError('Use unique grid values 0.50..0.95 step 0.05, including 0.50')
    return sorted(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT/'results/infer_probe')
    parser.add_argument('--val-split', type=Path, default=ROOT/'configs/val_split_seed0.json')
    parser.add_argument('--stress-results', type=Path, default=ROOT/'results/stress/v2/metrics.json')
    parser.add_argument('--stress-only', action='store_true')
    parser.add_argument('--families', default='e0_v2,e5_noadd')
    parser.add_argument('--model-seeds', default='0,1,2')
    parser.add_argument('--taus', type=parse_taus, default=list(TAUS))
    parser.add_argument('--intervention-ks', type=runner.parse_ks, default=[1,2,4,8])
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--limit-per-domain', type=int)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--encoder-dir', type=Path, default=ROOT/'submission/encoder')
    parser.add_argument('--probe', type=Path, default=ROOT/'submission/baseline.safetensors')
    parser.add_argument('--loader-workers', type=int, default=1)
    parser.add_argument('--metric-workers', type=int, default=6)
    args = parser.parse_args()
    args.starter = Path(os.environ['EC523_STARTER_DIR'])
    if not args.out_dir.resolve().is_relative_to(ROOT):
        parser.error('Reports must stay inside the repository')
    families, seeds = args.families.split(','), [int(s) for s in args.model_seeds.split(',')]
    if not families or len(set(families)) != len(families) or any(f not in ('e0_v2','e5_noadd') for f in families):
        parser.error('Expected unique families e0_v2,e5_noadd')
    if not seeds or len(set(seeds)) != len(seeds) or any(s not in (0,1,2) for s in seeds):
        parser.error('Expected unique model seeds 0,1,2')
    if min(args.loader_workers, args.metric_workers-1) < 0:
        parser.error('Loader workers must be nonnegative and metric workers positive')
    if 1 not in args.intervention_ks or 4 not in args.intervention_ks or max(args.intervention_ks)>8:
        parser.error('Intervention K must include 1 and 4 and stay <=8')
    if args.limit_per_domain is not None and args.limit_per_domain<1:
        parser.error('Limit must be positive')
    slots = int(os.environ.get('NSLOTS','0'))
    if slots and max(args.loader_workers+1,args.metric_workers+1)>slots:
        parser.error('Active processes exceed NSLOTS')
    names = [f'{f}_s{s}' for f in families for s in seeds]
    report = dict(status='stress_only' if args.stress_only else 'complete',
        config=dict(checkpoints=names, taus=args.taus, limit_per_domain=args.limit_per_domain,
                    device=args.device, val_split_sha256=inference.sha256(args.val_split),
                    proxy_weights={'1':.33,'4':.67}, intervention_ks=args.intervention_ks,
                    loader_workers=args.loader_workers,metric_workers=args.metric_workers),
        stress=stress_supplement(args.stress_results), gating={}, selections={}, encoding={})
    if args.stress_only:
        write_report(args,report)
        print('COMPLETE stress_only: stored-result tables; GPU probes pending',flush=True)
        return
    for name in names:
        if not (ROOT/'runs/exp'/name/'best.pt').is_file():
            raise FileNotFoundError('Missing checkpoint '+name)
    runner.probe.initialize_data_worker(0)
    device = torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    split = json.loads(args.val_split.read_text())
    for part in ('Val-sel','Val-rep'):
        rows = make_rows(split,part,args.limit_per_domain)
        cache,_ = inference.create_cache(args.cache_dir,args.out_dir,len(rows))
        try:
            head,report['encoding'][part] = shared.encode_images(rows,cache,args,device)
            for name in names:
                result = evaluate_cached(rows,cache,head,name,part,args,device)
                if part=='Val-rep' and device.type=='cpu' and args.limit_per_domain is not None:
                    result['cpu_original_runner_regression'] = cpu_direct_regression(rows,cache,head,name,result,args)
                report['gating'].setdefault(name,{})[part] = result
                if part=='Val-sel':
                    report['selections'][name] = choose_tau(result,args.taus)
                print('EVALUATED '+part+' '+name+' tau050=PASS selected_tau='+report['selections'][name]['tau'],flush=True)
            del head
        finally:
            shutil.rmtree(cache)
            print('CACHE_REMOVED '+part,flush=True)
    report['intervention_comparison'] = intervention_comparison(report)
    write_report(args,report)
    print('COMPLETE all inference probes',flush=True)


if __name__ == '__main__':
    main()
