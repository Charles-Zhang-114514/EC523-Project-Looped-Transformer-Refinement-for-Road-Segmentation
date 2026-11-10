#!/usr/bin/env python3
"""Confirm control, Lovasz and augmentation at seeds 1 and 2 on SCC."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import six_run_batch as base

REFERENCE = 'runs/six_run/20261101_192457Z_25415913'
METHODS = ('control', 'lovasz', 'augmentation')
TASKS = tuple({'name': f'{method}_seed{seed}', 'experiment': method, 'seed': seed}
              for seed in (1, 2) for method in METHODS)
ADDITIONS = ('scripts/repeat_refiner_batch.py', 'jobs/refiner_repeat.qsub',
             'jobs/README_refiner_repeat.txt', 'tests/test_repeat_refiner_batch.py')
REFERENCE_FILES = ('reports/metrics.json', 'runtime.json', 'command.json', 'SUCCESS')
IGNORED_CONFIG = {'seed', 'out_dir', 'ckpt_dir'}
METRICS = ('A1', 'A2', 'A4', 'A6', 'A8', 'Urban_A4', 'Rural_A4', 'report_proxy',
           'Urban_total_FP', 'Urban_new_FP', 'Urban_gap_repair', 'Urban_new_break',
           'Rural_gap_repair', 'Rural_new_break')


def read_json(path):
    return json.loads(path.read_text())


def validate_report(report, reference, seed, name):
    if report.get('protocol') != 'val_sel_rep' or len(report['epochs']) != 10:
        raise ValueError(f'{name}: expected ten epochs and Val-sel/Val-rep protocol')
    if report['config']['seed'] != seed:
        raise ValueError(f'{name}: wrong seed')
    settings = lambda d: {k:v for k,v in d['config'].items() if k not in IGNORED_CONFIG}
    if settings(report) != settings(reference):
        raise ValueError(f'{name}: training settings differ from the seed-0 method')
    for key in ('selection_files', 'evaluation_files'):
        if report[key] != reference[key]:
            raise ValueError(f'{name}: {key} differs from seed 0')
    for domain in ('Overall', 'Urban', 'Rural'):
        if report['evaluation']['groups'][domain]['y0_draft']['scores'] != reference['evaluation']['groups'][domain]['y0_draft']['scores']:
            raise ValueError(f'{name}: frozen-draft scores differ from seed 0 ({domain})')


def reference_state(reference):
    manifest = base.read_manifest(reference)
    base.validate_snapshot(reference, manifest)
    if manifest['seed'] != 0:
        raise ValueError('The reference batch must use seed 0')
    reports, runtimes = {}, {}
    for method in METHODS:
        run = reference/'source/runs'/method
        if not (run/'SUCCESS').is_file():
            raise ValueError(f'Reference {method} did not finish successfully')
        report = read_json(run/'reports/metrics.json')
        validate_report(report, report, 0, method)
        recipe = next(v for v in base.VARIANTS if v['name'] == method)
        if any(report['config'][k] != recipe[k] for k in ('loss', 'augmentation', 'pos_weight')):
            raise ValueError(f'Reference {method} does not match the planned recipe')
        if report['config'].get('resume') or report['config'].get('eval_only'):
            raise ValueError('Reference must be a fresh training run')
        reports[method], runtimes[method] = report, read_json(run/'runtime.json')
    control = reports['control']
    allowed = IGNORED_CONFIG | {'loss', 'augmentation'}
    for method, report in reports.items():
        if {k:v for k,v in report['config'].items() if k not in allowed} != {k:v for k,v in control['config'].items() if k not in allowed}:
            raise ValueError(f'Reference {method}: unexpected settings differ from control')
        for key in ('selection_files', 'evaluation_files'):
            if report[key] != control[key]:
                raise ValueError(f'Reference {method}: validation membership differs from control')
        if runtimes[method] != runtimes['control']:
            raise ValueError('Reference GPU/software runtimes do not match')
    return manifest, reports, runtimes['control']


def prepare_batch(repo, reference, max_running=2):
    if max_running not in range(1, 7):
        raise ValueError('max-running must be in 1..6')
    original, _, runtime = reference_state(reference)
    for key, value in original['expected_environment'].items():
        if os.environ.get(key) != value:
            raise ValueError(f'{key} must match seed 0; first source scripts/activate_scc.sh')
    if base.digest(repo/'scripts/activate_scc.sh') != original['files']['scripts/activate_scc.sh']:
        raise ValueError('Environment activation differs from the completed seed-0 batch')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    batch = repo/'runs/confirmation'/(stamp+'_'+uuid.uuid4().hex[:8])
    (batch/'source').mkdir(parents=True)
    (batch/'logs').mkdir()
    # Copy only recorded inputs, never seed-0 refiner weights or run outputs.
    for relative in original['files']:
        destination = batch/'source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference/'source'/relative, destination)
    base.validate_snapshot(batch, original)
    for relative in ADDITIONS:
        if relative in original['files']:
            if base.digest(repo / relative) == original['files'][relative]:
                continue
            raise ValueError(f'Confirmation helper would overwrite reference input: {relative}')
        destination = batch/'source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/relative, destination)
    for method in METHODS:
        for relative in REFERENCE_FILES:
            destination = batch/'reference'/method/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(reference/'source/runs'/method/relative, destination)
    files = {str(p.relative_to(batch/'source')): base.digest(p)
             for p in sorted((batch/'source').rglob('*')) if p.is_file()}
    reference_files = {str(p.relative_to(batch)): base.digest(p)
                       for p in sorted((batch/'reference').rglob('*')) if p.is_file()}
    manifest = {'schema': 1, 'kind': 'refiner_confirmation', 'created_utc': base.now(),
                'original_root': str(repo), 'reference_batch': str(reference),
                'reference_job_id': original['job_id'], 'expected_environment': original['expected_environment'],
                'gpu_type': original['gpu_type'], 'expected_runtime': runtime, 'max_running': max_running,
                'tasks': list(TASKS), 'files': files, 'reference_files': reference_files, 'job_id': None}
    base.write_json(batch/'manifest.json', manifest)
    return batch


def read_manifest(batch):
    manifest = read_json(batch/'manifest.json')
    if manifest.get('schema') != 1 or manifest.get('kind') != 'refiner_confirmation' or manifest['tasks'] != list(TASKS):
        raise ValueError('Unexpected confirmation manifest/tasks')
    return manifest


def validate_batch(batch, manifest):
    base.validate_snapshot(batch, manifest)
    for relative, expected in manifest['reference_files'].items():
        if base.digest(batch/relative) != expected:
            raise ValueError(f'Copied seed-0 evidence changed: {relative}')


def training_command(batch, task):
    original = read_json(batch/'reference'/task['experiment']/'command.json')
    if original[1:3] != ['-u', 'scripts/train_refiner.py']:
        raise ValueError('Unexpected reference training command')
    args = list(original[3:])
    if any(x in args for x in ('--resume', '--eval-only', '--ckpt')):
        raise ValueError('Confirmation runs must train from fresh initialization')
    for flag, value in (('--seed', str(task['seed'])),
                        ('--out-dir', f'runs/{task["name"]}/reports'),
                        ('--ckpt-dir', f'runs/{task["name"]}/checkpoints')):
        if args.count(flag) != 1:
            raise ValueError(f'Reference command must contain exactly one {flag}')
        args[args.index(flag)+1] = value
    return [sys.executable, '-u', 'scripts/train_refiner.py', *args]


def environment(source):
    return dict(os.environ, EC523_PROJECT_DIR=str(source),
                PYTHONPATH=str(source/'src')+os.pathsep+str(source), PYTHONDONTWRITEBYTECODE='1',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')


def preflight(source, env, gpu=False):
    base.run_tests(source, env)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                    '-p', 'test_repeat_refiner_batch.py', '-v'], cwd=source, env=env, check=True)
    base.check_inputs(source, env, gpu=gpu)


def task(batch, task_id):
    if task_id not in range(1, 7) or not os.environ.get('JOB_ID'):
        raise ValueError('Run as an SCC array task with ID 1..6')
    manifest = read_manifest(batch)
    item = manifest['tasks'][task_id-1]
    source = batch/'source'
    run = source/'runs'/item['name']
    run.mkdir()  # Existing attempts are never overwritten.
    base.write_json(run/'STARTED.json', {'utc': base.now(), 'job_id': os.environ['JOB_ID'], **item})
    try:
        validate_batch(batch, manifest)
        if base.digest(Path(manifest['original_root'])/'scripts/activate_scc.sh') != manifest['files']['scripts/activate_scc.sh']:
            raise ValueError('Original activation script changed since preparation')
        for key, value in manifest['expected_environment'].items():
            if os.environ.get(key) != value:
                raise ValueError(f'{key} changed since preparation')
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('An allocated CUDA GPU is required')
        runtime = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                   'cuda': torch.version.cuda, 'python': sys.version.split()[0]}
        base.write_json(run/'runtime.json', runtime)
        if runtime != manifest['expected_runtime']:
            raise RuntimeError(f'GPU/software must match seed 0: got {runtime}; expected {manifest["expected_runtime"]}')
        env = environment(source)
        preflight(source, env, gpu=True)
        command = training_command(batch, item)
        base.write_json(run/'command.json', command)
        print('TRAINING '+json.dumps(item), flush=True)
        subprocess.run(command, cwd=source, env=env, check=True)
        validate_batch(batch, manifest)
        report = read_json(run/'reports/metrics.json')
        reference = read_json(batch/'reference'/item['experiment']/'reports/metrics.json')
        validate_report(report, reference, item['seed'], item['name'])
        base.write_json(run/'SUCCESS', {'utc': base.now(), 'evaluated_epoch': report['evaluated_epoch']})
        print(f'TRAINING AND EVALUATION: SUCCESS ({item["name"]})', flush=True)
    except Exception as error:
        base.write_json(run/'FAILED.json', {'utc': base.now(), 'error': str(error)})
        raise


def resolve_batch(argument):
    if argument:
        return Path(argument).resolve()
    pointer = ROOT/'runs/confirmation/LATEST'
    if not pointer.is_file():
        raise ValueError('No submitted confirmation batch; supply --batch or submit first')
    return Path(pointer.read_text().strip())


def status(batch):
    manifest = read_manifest(batch)
    print(f'Batch: {batch}\nArray job: {manifest["job_id"]}\nSeed-0 reference job: {manifest["reference_job_id"]}')
    for number, item in enumerate(manifest['tasks'], 1):
        run = batch/'source/runs'/item['name']
        state = ('SUCCESS' if (run/'SUCCESS').is_file() else 'FAILED' if (run/'FAILED.json').is_file()
                 else 'STARTED (check qstat/log)' if (run/'STARTED.json').is_file() else 'NOT STARTED')
        print(f'{number}  {item["name"]:22} {state}')
    print(f'Logs: {batch}/logs/task_1.log through task_6.log')


def metric_row(report, method, seed):
    groups = report['evaluation']['groups']
    urban, rural = groups['Urban']['4'], groups['Rural']['4']
    row = {'experiment': method, 'seed': seed, 'status': 'complete', 'best_epoch': report['evaluated_epoch'],
           **{f'A{k}': groups['Overall'][str(k)]['scores']['A'] for k in (1, 2, 4, 6, 8)},
           'Urban_A4': urban['scores']['A'], 'Rural_A4': rural['scores']['A'],
           'report_proxy': sum(w*groups['Overall'][k]['scores']['A'] for k,w in report['config']['selection_ks'].items()),
           'Urban_total_FP': urban['structure']['fp_fragments'], 'Urban_new_FP': urban['extra']['new_fragments']['count']}
    for domain, values in (('Urban', urban), ('Rural', rural)):
        for metric, key in (('gap_repair', 'gap_repair_fraction'), ('new_break', 'new_break_fraction')):
            row[f'{domain}_{metric}'] = values['extra']['tracking'][key]
    return row


def summarize(rows):
    lookup = {(r['experiment'], r['seed']): r for r in rows if r['status'] == 'complete'}
    summaries, paired = {}, {}
    describe = lambda values: {'mean': statistics.mean(values), 'std': statistics.stdev(values)}
    for method in METHODS:
        present = [seed for seed in (0, 1, 2) if (method, seed) in lookup]
        summary = {'status': 'complete' if len(present) == 3 else 'incomplete', 'seeds': present}
        if len(present) == 3:
            summary['metrics'] = {key: describe([lookup[method,s][key] for s in present]) for key in METRICS}
        summaries[method] = summary
        if method != 'control' and all((m,s) in lookup for m in (method, 'control') for s in (0,1,2)):
            paired[method] = {}
            for key in METRICS:
                differences = [lookup[method,s][key]-lookup['control',s][key] for s in (0,1,2)]
                paired[method][key] = {**describe(differences), 'by_seed': dict(zip(('0','1','2'), differences))}
    return summaries, paired


def compare(batch):
    manifest = read_manifest(batch)
    validate_batch(batch, manifest)
    rows = []
    for method in METHODS:
        reference = read_json(batch/'reference'/method/'reports/metrics.json')
        rows.append(metric_row(reference, method, 0))
    for item in manifest['tasks']:
        run = batch/'source/runs'/item['name']
        if not (run/'SUCCESS').is_file():
            rows.append({'experiment': item['experiment'], 'seed': item['seed'],
                         'status': 'FAILED' if (run/'FAILED.json').is_file() else 'incomplete'})
            continue
        report = read_json(run/'reports/metrics.json')
        reference = read_json(batch/'reference'/item['experiment']/'reports/metrics.json')
        validate_report(report, reference, item['seed'], item['name'])
        if read_json(run/'runtime.json') != manifest['expected_runtime']:
            raise ValueError(f'{item["name"]}: GPU/software differs from seed 0')
        rows.append(metric_row(report, item['experiment'], item['seed']))
    summaries, paired = summarize(rows)
    lines = [f'Three-seed confirmation: {batch}', 'Seed 0 is copied from the completed screening batch; seeds 1 and 2 are new runs.',
             'Sample standard deviation (n-1); paired deltas compare matching seeds.',
             'Report proxy = 0.25*A1 + 0.25*A2 + 0.5*A4; stress evaluation is not included.', '',
             'Experiment      Seed  Epoch  Report proxy  Overall A4   Urban A4   Rural A4   Urban new FP']
    for row in sorted(rows, key=lambda r: (METHODS.index(r['experiment']), r['seed'])):
        prefix = f'{row["experiment"]:15} {row["seed"]:4}'
        if row['status'] != 'complete':
            lines.append(prefix+'  '+row['status'])
        else:
            lines.append(prefix+f' {row["best_epoch"]:6} {row["report_proxy"]:13.6f} {row["A4"]:11.6f} {row["Urban_A4"]:10.6f} {row["Rural_A4"]:10.6f} {row["Urban_new_FP"]:14}')
    for method, summary in summaries.items():
        lines += ['', f'{method}: {summary["status"]} ({len(summary["seeds"])}/3 seeds)']
        if summary['status'] == 'complete':
            for key, value in summary['metrics'].items():
                lines.append(f'  {key:20} {value["mean"]:.6f} +/- {value["std"]:.6f}')
    for method, metrics in paired.items():
        lines += ['', f'{method} minus control, paired seeds 0/1/2 (positive A delta is better; positive FP/break delta is worse):']
        for key, value in metrics.items():
            values = ', '.join(f'{v:+.6f}' for v in value['by_seed'].values())
            lines.append(f'  {key:20} {value["mean"]:+.6f} +/- {value["std"]:.6f}; seeds: [{values}]')
    lines += ['', 'Incomplete methods have no three-seed mean. Three seeds provide preliminary repeatability evidence, not a significance claim.',
              'Val-rep has informed development; it is not an untouched final test set. Inspect rural performance and connectivity as well as A.']
    (batch/'comparison.txt').write_text('\n'.join(lines)+'\n')
    base.write_json(batch/'comparison.json', {'rows': rows, 'summary': summaries, 'paired_deltas': paired})
    print('\n'.join(lines))
    return rows, summaries, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('prepare', 'submit'):
        p = sub.add_parser(name)
        p.add_argument('--reference', type=Path, default=ROOT/REFERENCE)
        p.add_argument('--max-running', type=int, default=2)
    for name in ('status', 'compare', 'task'):
        p = sub.add_parser(name)
        p.add_argument('--batch')
        if name == 'task':
            p.add_argument('--task', type=int, required=True)
    args = parser.parse_args()
    try:
        if args.action in ('prepare', 'submit'):
            reference = args.reference.resolve()
            gpu_type = base.read_manifest(reference)['gpu_type']
            if args.action == 'submit':
                gpu_type = base.gpu_name(gpu_type, subprocess.check_output(['qgpus', '-s'], text=True))
            batch = prepare_batch(ROOT, reference, args.max_running)
            print(f'Prepared batch: {batch}\nChecking frozen inputs and tests before submission...', flush=True)
            preflight(batch/'source', environment(batch/'source'))
            validate_batch(batch, read_manifest(batch))
            if args.action == 'submit':
                command = ['qsub', '-terse', '-l', f'gpu_type={gpu_type}', '-tc', str(args.max_running),
                           '-o', str(batch/'logs'), str(batch/'source/jobs/refiner_repeat.qsub'), str(batch)]
                output = subprocess.check_output(command, cwd=ROOT, text=True).strip()
                (batch/'qsub_output.txt').write_text(output+'\n')
                match = re.fullmatch(r'(\d+)(?:\.\d+-\d+:\d+)?', output)
                if not match:
                    raise RuntimeError(f'qsub returned {output!r}; inspect {batch}/qsub_output.txt before resubmitting')
                manifest = read_manifest(batch)
                manifest['job_id'] = match.group(1)
                base.write_json(batch/'manifest.json', manifest)
                (ROOT/'runs/confirmation/LATEST').write_text(str(batch)+'\n')
                print(f'Submitted six confirmation tasks as array {manifest["job_id"]}.')
                status(batch)
        elif args.action == 'task':
            task(resolve_batch(args.batch), args.task)
        elif args.action == 'status':
            status(resolve_batch(args.batch))
        else:
            compare(resolve_batch(args.batch))
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
