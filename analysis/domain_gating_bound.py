#!/usr/bin/env python3
"""Read-only domain-aware gating bounds; no inference or threshold fitting on report data."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.train_linear_probe import official_metrics


def combine(urban, rural, metrics):
    """Combine integer sufficient statistics, never average domain scores."""
    counts = metrics.Counts(**urban['counts']) + metrics.Counts(**rural['counts'])
    return dict(counts=asdict(counts), scores=metrics.summarize(counts))


def select(groups, taus):
    return {d: min(taus, key=lambda t: (-sum(w*groups[d][f'{k}:{t:.2f}']['scores']['quality']
                                           for k, w in (('1', .33), ('4', .67))), t))
            for d in ('Urban', 'Rural')}


def analyze(source, metrics):
    if source.get('status')!='complete':raise ValueError('Complete inference-probe results are required')
    rows = []
    for name, parts in source['gating'].items():
        rep = parts['Val-rep']['groups']
        for split in ('Val-sel', 'Val-rep'):
            thresholds = select(parts[split]['groups'], source['config']['taus'])
            for k in ('1', '4'):
                half = {d: rep[d][f'{k}:{thresholds[d]:.2f}'] for d in thresholds}
                groups = dict(half, Overall=combine(half['Urban'], half['Rural'], metrics))
                baseline = combine(rep['Urban'][k+':0.50'], rep['Rural'][k+':0.50'], metrics)
                assert baseline['counts'] == rep['Overall'][k+':0.50']['counts']
                assert baseline['scores'] == rep['Overall'][k+':0.50']['scores']
                # Check every equal-threshold combination against the stored overall path.
                for tau in source['config']['taus']:
                    same = combine(rep['Urban'][f'{k}:{tau:.2f}'], rep['Rural'][f'{k}:{tau:.2f}'], metrics)
                    assert same['counts'] == rep['Overall'][f'{k}:{tau:.2f}']['counts']
                    assert same['scores'] == rep['Overall'][f'{k}:{tau:.2f}']['scores']
                rows.append(dict(checkpoint=name, selected_on=split, optimistic=split=='Val-rep',
                                 thresholds=thresholds, K=int(k), groups={d: dict(
                                     counts=v['counts'], scores=v['scores'],
                                     delta_native=v['scores']['quality']-rep[d][k+':0.50']['scores']['quality'],
                                     delta_y0=v['scores']['quality']-rep[d]['y0_draft']['scores']['quality'])
                                     for d,v in groups.items()}))
    return dict(protocol='Domain-label-aware bound; domain is not known by the inference model',
                proxy_weights={'1':.33,'4':.67}, rows=rows, equal_threshold_regression=True)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def mean_sd(values):
    return f'{statistics.mean(values):.6f} +/- {statistics.stdev(values):.6f}'


def markdown(report):
    lines=['# Domain-aware gating bound', '',
           'Observed this is a bound with access to Urban/Rural labels, which the inference model does not know. '
           'Val-sel selection uses each domain\'s 0.33*A1 + 0.67*A4. Equal scores prefer lower tau. '
           'Overall uses Urban Counts + Rural Counts and the official summarize; domain A values are never averaged. '
           'This domain-wise proxy rule is not an exhaustive maximization of Overall over all threshold pairs.', '']
    for split in ('Val-sel','Val-rep'):
        lines += ['## '+('Observed Val-sel selection' if split=='Val-sel' else 'Observed optimistic diagnostic: thresholds selected on Val-rep'), '']
        rows=[]
        for r in report['rows']:
            if r['selected_on']!=split: continue
            rows.append([r['checkpoint'],r['K'],f"{r['thresholds']['Urban']:.2f}/{r['thresholds']['Rural']:.2f}",
                         *[f"{r['groups'][d]['scores']['quality']:.6f}; {r['groups'][d]['delta_native']:+.6f}; {r['groups'][d]['delta_y0']:+.6f}"
                           for d in ('Overall','Urban','Rural')]])
        lines += [table(['Checkpoint','K','Urban/Rural tau','Overall A; dA native; dA y0',
                         'Urban A; dA native; dA y0','Rural A; dA native; dA y0'],rows),'']
        rows=[]
        for family in ('e0_v2','e5_noadd'):
            for k in (1,4):
                records=[r for r in report['rows'] if r['selected_on']==split and r['K']==k and r['checkpoint'].startswith(family)]
                for d in ('Overall','Urban','Rural'):
                    vals=[r['groups'][d] for r in records]
                    rows.append([family,k,d,mean_sd([v['scores']['quality'] for v in vals]),
                                 mean_sd([v['delta_native'] for v in vals]),f"{sum(v['delta_native']>0 for v in vals)}/3",
                                 mean_sd([v['delta_y0'] for v in vals]),f"{sum(v['delta_y0']>0 for v in vals)}/3"])
        lines += [table(['Experiment','K','Domain','A mean +/- sample sd','dA native','Positive native','dA y0','Positive y0'],rows),'']
    lines += ['Observed all equal-threshold Count and score checks, including tau=0.50, passed.',
              'Observed source: results/infer_probe/metrics.json, gating.{checkpoint}.{split}.groups.{domain}.{K}:{tau}.counts.']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/infer_probe/metrics.json')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'results/domain_gating')
    args=parser.parse_args()
    if not args.out_dir.resolve().is_relative_to(ROOT):parser.error('Reports must stay inside the repository')
    report=analyze(json.loads(args.input.read_text()),official_metrics(os.environ['EC523_STARTER_DIR']))
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for name,text in [('metrics.json',json.dumps(report,indent=2,allow_nan=False)+'\n'),('summary.md',markdown(report))]:
        with (args.out_dir/name).open('x') as handle:handle.write(text)
    print('PASS equal-threshold Counts and official scores')


if __name__=='__main__':main()
