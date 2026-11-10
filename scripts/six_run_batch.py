#!/usr/bin/env python3
"""Prepare/submit/inspect a frozen six-task SCC experiment; stdlib orchestration."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
PROBE = Path('runs/linear_probe/checkpoints/dinov3_vitl16_web_bn_posw4.pt')
VARIANTS = (
    {'name': 'control', 'pos_weight': 4., 'loss': 'bce', 'augmentation': 'none'},
    {'name': 'posw3', 'pos_weight': 3., 'loss': 'bce', 'augmentation': 'none'},
    {'name': 'false_add', 'pos_weight': 4., 'loss': 'false_add', 'augmentation': 'none'},
    {'name': 'lovasz', 'pos_weight': 4., 'loss': 'lovasz', 'augmentation': 'none'},
    {'name': 'soft_cldice', 'pos_weight': 4., 'loss': 'soft_cldice', 'augmentation': 'none'},
    {'name': 'augmentation', 'pos_weight': 4., 'loss': 'bce', 'augmentation': 'd4'},
)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def gpu_name(requested, inventory):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', requested):
        raise ValueError('Invalid GPU model name')
    candidates = [line.split()[0] for line in inventory.splitlines() if line.split()]
    for name in candidates:
        if name.casefold() == requested.casefold():
            return name
    raise ValueError(f'{requested} is absent from qgpus -s. Inspect qgpus before choosing another model for the whole batch.')


def prepare_batch(repo, seed=0, gpu_type='L40S', max_running=2):
    if seed < 0 or max_running not in range(1, 7):
        raise ValueError('Seed must be nonnegative; max-running must be in 1..6')
    expected_env = {k: os.environ.get(k) for k in ('EC523_DATA_DIR', 'EC523_STARTER_DIR', 'HF_HOME')}
    if not all(expected_env.values()):
        raise ValueError('First run: source scripts/activate_scc.sh')
    if not (repo / PROBE).is_file():
        raise ValueError('The existing frozen probe checkpoint is missing')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    batch = repo / 'runs/six_run' / (stamp + '_' + uuid.uuid4().hex[:8])
    source = batch / 'source'
    source.mkdir(parents=True)
    (batch / 'logs').mkdir()
    # Snapshot actual working files, including uncommitted/new experiment code.
    # Encoder/cache/dataset files are not copied; the small frozen probe is.
    for folder in ('src', 'scripts', 'analysis', 'configs', 'tests', 'jobs'):
        for path in sorted((repo / folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.sh', '.qsub', '.json', '.txt') and '__pycache__' not in path.parts:
                target = source / path.relative_to(repo)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    target = source / PROBE
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / PROBE, target)
    files = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*')) if p.is_file()}
    (batch / 'source_commit.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True))
    (batch / 'source_changes.patch').write_text(subprocess.check_output(['git', 'diff', '--binary', 'HEAD', '--', 'src', 'scripts', 'analysis', 'configs', 'jobs', 'tests'], cwd=repo, text=True))
    manifest = {'schema': 1, 'created_utc': now(), 'original_root': str(repo), 'seed': seed,
                'gpu_type': gpu_type, 'max_running': max_running, 'expected_environment': expected_env,
                'variants': list(VARIANTS), 'files': files, 'job_id': None,
                'protocol': '10 epochs; fixed Val-sel selection, disjoint Val-rep report; seed matched across tasks',
                'false_add': '500 BCE updates, 8 BCE-only calibration batches, then fixed 0.1 logit-gradient ratio',
                'soft_cldice': 'weight 0.1, 20 thinning iterations, checkpointed predicted skeleton'}
    write_json(batch / 'manifest.json', manifest)
    return batch


def read_manifest(batch):
    manifest = json.loads((batch / 'manifest.json').read_text())
    if manifest.get('schema') != 1 or manifest['variants'] != list(VARIANTS):
        raise ValueError('Unexpected experiment manifest or variant definitions')
    return manifest


def validate_snapshot(batch, manifest):
    for relative, expected in manifest['files'].items():
        path = batch / 'source' / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Frozen input changed: {relative}')


def training_arguments(variant, seed):
    name = variant['name']
    return [
        '--head-type', 'conv_upsample', '--epochs', '10', '--batch-size', '4', '--lr', '3e-4',
        '--k-train', '4', '--seed', str(seed), '--device', 'cuda',
        '--loader-workers', '7', '--eval-loader-workers', '1', '--metric-workers', '6',
        '--monitor-every', '5', '--monitor-per-domain', '50', '--select-every', '1', '--select-per-domain', '100',
        '--selection-ks', '1:0.25,2:0.25,4:0.5', '--val-split', 'configs/val_split_seed0.json',
        '--eval-extra', '--eval-k', '1,2,4,6,8', '--pos-weight', str(variant['pos_weight']),
        '--loss', variant['loss'], '--augmentation', variant['augmentation'], '--aux-weight', '0.1',
        '--skeleton-iterations', '20', '--false-add-gradient-ratio', '0.1',
        '--false-add-warmup-steps', '500', '--false-add-calibration-batches', '8',
        '--out-dir', f'runs/{name}/reports', '--ckpt-dir', f'runs/{name}/checkpoints',
    ]


def run_tests(source, environment):
    for pattern in ('test_refiner*.py', 'test_controlled_exp.py', 'test_six_run_batch.py'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern, '-v'],
                       cwd=source, env=environment, check=True)


def check_inputs(source, environment, gpu=False):
    helper = Path('scripts/check_refiner_inputs.py')
    if not (source / helper).is_file():
        # Original screening snapshots stored the same helper under runs/.
        helper = Path('runs/bootstrap_tools/check_inputs.py')
    command = [sys.executable, '-u', str(helper), '--encoder', '--probe']
    if gpu:
        command.append('--gpu')
    subprocess.run(command, cwd=source, env=environment, check=True)


def task(batch, task_id):
    if task_id not in range(1, 7) or not os.environ.get('JOB_ID'):
        raise ValueError('Tasks must run as an SCC array job with task ID 1..6')
    manifest = read_manifest(batch)
    source = batch / 'source'
    variant = manifest['variants'][task_id - 1]
    run_dir = source / 'runs' / variant['name']
    run_dir.mkdir()  # Refuse to overwrite any prior attempt/checkpoint.
    write_json(run_dir / 'STARTED.json', {'utc': now(), 'job_id': os.environ['JOB_ID'], 'task': task_id})
    try:
        validate_snapshot(batch, manifest)
        original = Path(manifest['original_root'])
        if digest(original / 'scripts/activate_scc.sh') != manifest['files']['scripts/activate_scc.sh']:
            raise ValueError('Original environment activation script changed since batch preparation')
        for key, value in manifest['expected_environment'].items():
            if os.environ.get(key) != value:
                raise ValueError(f'{key} changed since batch preparation')
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('An allocated CUDA GPU is required')
        device_name = torch.cuda.get_device_name(0)
        if manifest['gpu_type'].casefold() not in device_name.casefold():
            raise RuntimeError(f'Expected {manifest["gpu_type"]}, got {device_name}')
        runtime = {'gpu': device_name, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'python': sys.version.split()[0]}
        write_json(run_dir / 'runtime.json', runtime)
        print('RUNTIME ' + json.dumps(runtime), flush=True)
        environment = dict(os.environ, EC523_PROJECT_DIR=str(source),
                           PYTHONPATH=str(source / 'src') + os.pathsep + str(source))
        run_tests(source, environment)
        check_inputs(source, environment, gpu=True)
        command = [sys.executable, '-u', 'scripts/train_refiner.py', *training_arguments(variant, manifest['seed'])]
        write_json(run_dir / 'command.json', command)
        subprocess.run(command, cwd=source, env=environment, check=True)
        validate_snapshot(batch, manifest)
        report = json.loads((run_dir / 'reports/metrics.json').read_text())
        if report.get('protocol') != 'val_sel_rep' or len(report['epochs']) != 10:
            raise RuntimeError('Expected ten completed epochs and a controlled-protocol report')
        if variant['loss'] == 'false_add' and report['config']['objective_state']['effective_aux_weight'] is None:
            raise RuntimeError('False-add calibration did not complete')
        write_json(run_dir / 'SUCCESS', {'utc': now(), 'evaluated_epoch': report['evaluated_epoch']})
        print(f'TRAINING AND EVALUATION: SUCCESS ({variant["name"]})', flush=True)
    except Exception as error:
        write_json(run_dir / 'FAILED.json', {'utc': now(), 'error': str(error)})
        raise


def resolve_batch(argument):
    if argument:
        return Path(argument).resolve()
    pointer = ROOT / 'runs/six_run/LATEST'
    if not pointer.is_file():
        raise ValueError('No submitted batch found; supply --batch or submit a new batch')
    return Path(pointer.read_text().strip())


def status(batch):
    manifest = read_manifest(batch)
    print(f'Batch: {batch}\nArray job: {manifest["job_id"]}\nGPU model: {manifest["gpu_type"]}')
    for index, variant in enumerate(manifest['variants'], 1):
        run = batch / 'source/runs' / variant['name']
        state = ('SUCCESS' if (run/'SUCCESS').is_file() else 'FAILED' if (run/'FAILED.json').is_file()
                 else 'STARTED (check qstat/log)' if (run/'STARTED.json').is_file() else 'NOT STARTED')
        print(f'{index}  {variant["name"]:14} {state}')
    print(f'Logs: {batch}/logs/task_1.log through task_6.log')


def compare(batch):
    manifest = read_manifest(batch)
    validate_snapshot(batch, manifest)
    reports, rows = {}, []
    for variant in manifest['variants']:
        name = variant['name']
        run = batch / 'source/runs' / name
        if not (run/'SUCCESS').is_file():
            rows.append({'name': name, 'status': 'FAILED' if (run/'FAILED.json').is_file() else 'incomplete'})
            continue
        data = json.loads((run/'reports/metrics.json').read_text())
        for key in ('loss', 'pos_weight', 'augmentation'):
            if data['config'][key] != variant[key]:
                raise ValueError(f'{name}: unexpected {key}')
        reports[name] = data
        groups = data['evaluation']['groups']
        row = {'name': name, 'status': 'complete', 'best_epoch': data['evaluated_epoch'],
               'selection_proxy': data['best_score'],
               'report_proxy': sum(w * groups['Overall'][k]['scores']['A'] for k, w in data['config']['selection_ks'].items()),
               'A4': groups['Overall']['4']['scores']['A'], 'Urban_A4': groups['Urban']['4']['scores']['A'],
               'Rural_A4': groups['Rural']['4']['scores']['A'],
               'Urban_new_FP': groups['Urban']['4']['extra']['new_fragments']['count'],
               'Urban_total_FP': groups['Urban']['4']['structure']['fp_fragments'],
               'Urban_gap_repair': groups['Urban']['4']['extra']['tracking']['gap_repair_fraction'],
               'Urban_new_break': groups['Urban']['4']['extra']['tracking']['new_break_fraction'],
               'runtime': json.loads((run/'runtime.json').read_text())}
        rows.append(row)
    if 'control' in reports:
        base = reports['control']
        allowed = {'pos_weight', 'loss', 'augmentation', 'objective_state', 'out_dir', 'ckpt_dir'}
        for name, data in reports.items():
            for key in ('selection_files', 'evaluation_files'):
                if data[key] != base[key]:
                    raise ValueError(f'{name}: {key} differs from control')
            if {k:v for k,v in data['config'].items() if k not in allowed} != {k:v for k,v in base['config'].items() if k not in allowed}:
                raise ValueError(f'{name}: other experiment settings differ from control')
        control_row = next(r for r in rows if r['name'] == 'control')
        for row in rows:
            if row['status'] == 'complete':
                if row['runtime'] != control_row['runtime']:
                    raise ValueError(f'{row["name"]}: GPU/software runtime differs from control')
                row['delta_A4'] = row['A4'] - control_row['A4']
                row['delta_Urban_A4'] = row['Urban_A4'] - control_row['Urban_A4']
                row['delta_Rural_A4'] = row['Rural_A4'] - control_row['Rural_A4']
    lines = [f'Six-run comparison: {batch}',
             'Val-sel chooses checkpoints; Val-rep reports results. One seed is exploratory.',
             'Threshold 0.5; K=4 below. Compare quality, repair, breaks and false additions together.',
             'Selection/report proxies are 0.25*A1+0.25*A2+0.5*A4, not the full stress-inclusive score.', '',
             'Experiment      Best   Sel-proxy  Rep-proxy  OverallA4  UrbanA4  RuralA4  DeltaA4']
    for r in rows:
        if r['status'] != 'complete':
            lines.append(f"{r['name']:15} {r['status']}")
        else:
            delta = f"{r['delta_A4']:+.6f}" if 'delta_A4' in r else 'NA'
            lines.append(f"{r['name']:15} {r['best_epoch']:4} {r['selection_proxy']:10.6f} {r['report_proxy']:10.6f} {r['A4']:10.6f} {r['Urban_A4']:8.6f} {r['Rural_A4']:8.6f} {delta}")
    lines += ['', 'Urban K=4 diagnostics:', 'Experiment       TotalFP   NewFP    Gap repair    New break']
    for r in rows:
        if r['status'] == 'complete':
            gap, broken = r['Urban_gap_repair'], r['Urban_new_break']
            gap_text = 'NA' if gap is None else f'{gap:.3%}'
            break_text = 'NA' if broken is None else f'{broken:.3%}'
            lines.append(f"{r['name']:15} {r['Urban_total_FP']:8} {r['Urban_new_FP']:7} {gap_text:>13} {break_text:>12}")
    lines += ['', 'A lower fragment count alone is not success: check rural/overall quality and road coverage.',
              'Full K curves and component metrics are in source/runs/<experiment>/reports/summary.md.']
    text = '\n'.join(lines) + '\n'
    (batch/'comparison.txt').write_text(text)
    write_json(batch/'comparison.json', {'rows': rows, 'batch': str(batch)})
    print(text, end='')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('prepare', 'submit'):
        p = sub.add_parser(name)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--gpu-type', default='L40S')
        p.add_argument('--max-running', type=int, default=2)
    for name in ('status', 'compare', 'task'):
        p = sub.add_parser(name)
        p.add_argument('--batch')
        if name == 'task':
            p.add_argument('--task', type=int, required=True)
    args = parser.parse_args()
    try:
        if args.action in ('prepare', 'submit'):
            selected_gpu = args.gpu_type
            if args.action == 'submit':
                inventory = subprocess.check_output(['qgpus', '-s'], text=True)
                selected_gpu = gpu_name(args.gpu_type, inventory)
            batch = prepare_batch(ROOT, args.seed, selected_gpu, args.max_running)
            source = batch/'source'
            env = dict(os.environ, PYTHONPATH=str(source/'src') + os.pathsep + str(source), PYTHONDONTWRITEBYTECODE='1',
                       EC523_PROJECT_DIR=str(source), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            run_tests(source, env)
            check_inputs(source, env)
            print(f'Prepared batch: {batch}', flush=True)
            if args.action == 'submit':
                command = ['qsub', '-terse', '-l', f'gpu_type={selected_gpu}', '-tc', str(args.max_running),
                           '-o', str(batch/'logs'), str(source/'jobs/refiner_six.qsub'), str(batch)]
                output = subprocess.check_output(command, cwd=ROOT, text=True).strip()
                (batch/'qsub_output.txt').write_text(output+'\n')
                match = re.fullmatch(r'(\d+)(?:\.\d+-\d+:\d+)?', output)
                if not match:
                    raise RuntimeError(f'qsub returned {output!r}; inspect {batch}/qsub_output.txt before resubmitting')
                manifest = read_manifest(batch)
                manifest['job_id'] = match.group(1)
                write_json(batch/'manifest.json', manifest)
                (ROOT/'runs/six_run/LATEST').write_text(str(batch)+'\n')
                print(f'Submitted array job {manifest["job_id"]}, tasks 1-6; at most {args.max_running} run together.')
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
