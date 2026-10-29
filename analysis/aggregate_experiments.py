#!/usr/bin/env python3
"""Report observed controlled runs, including unfinished experiment slots."""
import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ('e0_v2_s0', 'e0_v2_s1', 'e0_v2_s2', 'e1_reset_s0', 'e2_nofb_s0', 'e3_single_s0')
COLUMNS = ('experiment', 'seed', 'status', 'best_epoch', 'y0_A', 'A_K1', 'A_K2', 'A_K4', 'A_K6', 'A_K8', 'proxy',
           'Urban_A_K1', 'Rural_A_K1', 'Urban_A_K4', 'Rural_A_K4', 'gap_repair_K1', 'new_break_K1',
           'gap_repair_K4', 'new_break_K4', 'correct_added_K1', 'correct_deleted_K1', 'new_fragments_K1')


def read_run(name, path):
    try:
        experiment, seed_text = name.rsplit('_s', 1)
        seed = int(seed_text)
    except (ValueError, TypeError):
        return {'experiment': name, 'seed': None, 'status': 'missing', 'reason': 'invalid run directory name'}
    missing = {'experiment': experiment, 'seed': seed, 'status': 'missing'}
    try:
        data = json.loads(path.read_text())
        if data.get('protocol') != 'val_sel_rep':
            return {**missing, 'reason': 'not a controlled-protocol report'}
        groups = data['evaluation']['groups']
        overall = groups['Overall']
        if data['config']['seed'] != int(seed_text):
            return {**missing, 'reason': 'seed does not match run directory'}
        row = {'experiment': experiment, 'seed': data['config']['seed'], 'status': 'complete',
               'best_epoch': data['best_epoch'], 'y0_A': overall['y0_draft']['scores']['A'],
               'val_split_sha256': data['config']['val_split_sha256']}
        for k in (1, 2, 4, 6, 8):
            row['A_K'+str(k)] = overall[str(k)]['scores']['A']
        row['proxy'] = .25*row['A_K1'] + .25*row['A_K2'] + .5*row['A_K4']
        for k in (1, 4):
            for domain in ('Urban', 'Rural'):
                row[f'{domain}_A_K{k}'] = groups[domain][str(k)]['scores']['A']
            tracking = overall[str(k)]['extra']['tracking']
            row[f'gap_repair_K{k}'] = tracking['gap_repair_fraction']
            row[f'new_break_K{k}'] = tracking['new_break_fraction']
        changes = overall['1']['extra']['changes']
        row['correct_added_K1'] = changes['added_correct_fraction']
        row['correct_deleted_K1'] = changes['deleted_correct_fraction']
        row['new_fragments_K1'] = overall['1']['extra']['new_fragments']['count']
        return row
    except (OSError, ValueError, KeyError, TypeError):
        return {**missing, 'reason': 'metrics absent, incomplete or unreadable'}


def mean_row(rows):
    selected = [r for r in rows if r['experiment'] == 'e0_v2' and r['seed'] in (0, 1, 2) and r['status'] == 'complete']
    result = {'experiment': 'e0_v2 mean ± std (3 seeds)', 'seed': '0,1,2',
              'status': 'complete' if len(selected) == 3 else 'missing', 'completed_seeds': len(selected)}
    if len(selected) == 3:
        for name in COLUMNS[3:]:
            values = [r[name] for r in selected]
            result[name] = ({'mean': statistics.mean(values), 'std': statistics.stdev(values)}
                            if all(v is not None for v in values) else None)
    return result


def display(value):
    if value is None:
        return 'NA'
    if isinstance(value, dict):
        return f"{value['mean']:.6f} ± {value['std']:.6f}"
    return f'{value:.6f}' if isinstance(value, float) else str(value)


def aggregate_directory(directory):
    discovered = {p.parent.name for p in directory.glob('*/metrics.json') if '_s' in p.parent.name}
    names = list(EXPECTED) + sorted(discovered - set(EXPECTED))
    rows = [read_run(name, directory/name/'metrics.json') for name in names]
    rows.append(mean_row(rows))
    return {'rows': rows, 'standard_deviation': 'sample standard deviation (ddof=1)',
            'proxy_definition': '0.25*A(K1)+0.25*A(K2)+0.5*A(K4)',
            'observations': 'Observed Val-rep results. Unfinished runs are missing; no conclusions are inferred.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT/'results/exp')
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.results_dir
    if not out_dir.resolve().is_relative_to(ROOT):
        parser.error('Output must stay inside the repository')
    report = aggregate_directory(args.results_dir)
    lines = ['# Controlled experiments', '', report['observations'], '',
             'The e0_v2 aggregate requires all three seeds and uses sample standard deviation (ddof=1).', '',
             '| ' + ' | '.join(COLUMNS) + ' |', '|' + '---|'*len(COLUMNS)]
    lines += ['| ' + ' | '.join(display(row.get(key)) for key in COLUMNS) + ' |' for row in report['rows']]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (out_dir/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f"Observed {sum(r['status']=='complete' for r in report['rows'][:-1])} completed runs; wrote summary.md and summary.json", flush=True)


if __name__ == '__main__':
    main()
