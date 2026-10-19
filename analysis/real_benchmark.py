#!/usr/bin/env python3
"""CPU-only structured errors on raw Train road labels; never reads Val.

Scores always pool official Counts before summarize. PNGs are optional derived
examples restricted to the repository's ignored tmp/ tree; no raw mask is saved.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from looproad.eval.structured_errors import (
    close, component_merges, connected_at, endpoint_connect, false_bridges,
    local_gaps, real_gap_sites, thicken, thin,
)

KEYS = ('road_iou', 'cldice', 'boundary_iou', 'quality')
LENGTHS = (8, 16, 32, 64)
CLOSING_RADII = (1, 2, 4, 8)
ENDPOINT_DISTANCES = (16, 32, 64)


def official_metrics():
    value = os.environ.get('EC523_STARTER_DIR')
    if not value:
        raise SystemExit('Set EC523_STARTER_DIR to the official course cli directory.')
    path = Path(value).resolve()
    if not (path / 'metrics.py').is_file() or not (path / 'common.py').is_file():
        raise SystemExit('EC523_STARTER_DIR must contain metrics.py and common.py.')
    sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location('ec523_official_metrics', path / 'metrics.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_labels(path):
    with Image.open(path) as image:
        labels = np.asarray(image).copy()
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f'Expected raw integer single-channel labels: {path.name}')
    if labels.min() < 0 or labels.max() > 7 or labels.shape != (1024, 1024):
        raise ValueError(f'Invalid label values or shape: {path.name}')
    return labels


def rng_for(seed, key):
    digest = hashlib.sha256(f'{seed}:{key}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'big'))


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def ratio_bin(value):
    if value < .5:
        return '[0,0.5)'
    if value < 1:
        return '[0.5,1)'
    if value < 2:
        return '[1,2)'
    if value < 4:
        return '[2,4)'
    return '[4,inf)'


def markdown_table(rows, fields):
    def fmt(value):
        return f'{value:.6f}' if isinstance(value, float) else str(value)
    return '\n'.join(['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---'] * len(fields)) + ' |'] +
                     ['| ' + ' | '.join(fmt(row[key]) for key in fields) + ' |' for row in rows])


def run(args):
    started = time.monotonic()
    metrics = official_metrics()
    if not os.environ.get('EC523_DATA_DIR'):
        raise SystemExit('Set EC523_DATA_DIR to the shared dataset root.')
    root = Path(os.environ['EC523_DATA_DIR']).resolve()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'results') and not out.is_relative_to(ROOT / 'tmp'):
        raise ValueError('Numeric output must be under repository results/ or tmp/.')
    if out.is_relative_to(root):
        raise ValueError('Never write outputs under EC523_DATA_DIR.')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh output directory; preserve prior results.')
    out.mkdir(parents=True, exist_ok=True)
    if args.examples:
        example_dir = args.examples.resolve()
        if not example_dir.is_relative_to(ROOT / 'tmp') or example_dir.is_relative_to(root):
            raise ValueError('Derived examples must stay under ignored repository tmp/.')
        example_dir.mkdir(parents=True, exist_ok=True)
    totals, changed, modified, image_numbers = {}, defaultdict(int), defaultdict(int), defaultdict(int)
    sample_rows, site_rows, event_rows, bridge_rows, repair_rows, false_rows, per_image = [], [], [], [], [], [], []
    def score(domain, image_id, condition, prediction, labels):
        prediction = prediction & (labels != 0)
        counts = metrics.image_counts(prediction.astype(np.float32), labels)
        nchanged = int(np.count_nonzero(prediction != (labels == 3)))
        for group in (domain, 'All'):
            key = (group, condition)
            totals[key] = totals.get(key, metrics.Counts()) + counts
            changed[key] += nchanged
            modified[key] += int(nchanged > 0)
            image_numbers[key] += 1
        value = metrics.summarize(counts)
        per_image.append(dict(domain=domain, image=image_id, condition=condition,
                              changed_valid_pixels=nchanged, **{key: value[key] for key in KEYS}))
    controls = [('close', radius) for radius in CLOSING_RADII] + [('endpoint', d) for d in ENDPOINT_DISTANCES]
    for domain in ('Urban', 'Rural'):
        directory = root / 'Train' / domain / 'masks_png'
        if not directory.is_dir():
            raise ValueError(f'Missing Train/{domain}/masks_png')
        files = sorted(directory.glob('*.png'))
        order = rng_for(args.seed, f'sample:{domain}').permutation(len(files))
        selected = []
        inspected = 0
        for index in order:
            labels = read_labels(files[index])
            inspected += 1
            if (labels == 3).any():
                own = metrics.summarize(metrics.image_counts((labels == 3).astype(np.float32), labels))
                if any(own[key] != 1.0 for key in KEYS):
                    raise RuntimeError(f'Clean self-score failed for Train/{domain}/{files[index].name}: {own}')
                selected.append(files[index])
            if len(selected) == args.n:
                break
        if len(selected) < args.n:
            raise ValueError(f'Train/{domain} contains fewer than {args.n} road-positive images.')
        print(f'{domain}: {len(selected)} road-positive Train masks selected after inspecting {inspected}. Clean self-scores all exactly 1.0.', flush=True)
        for image_index, path in enumerate(selected):
            image_id = path.stem
            labels = read_labels(path)
            mask, valid = labels == 3, labels != 0
            sample_rows.append(dict(split='Train', domain=domain, image=image_id,
                                    road_pixels=int(mask.sum()), valid_pixels=int(valid.sum()),
                                    label_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            score(domain, image_id, 'clean', mask, labels)
            sites, skeleton, distance, site_info = real_gap_sites(mask, valid, rng_for(args.seed, f'sites:{domain}:{image_id}'), limit=args.gaps_per_image)
            site_rows.append(dict(domain=domain, image=image_id, **site_info,
                                  reason='ok' if sites else 'no_site_with_clearance_and_straightness'))
            for length in LENGTHS:
                pred, events = local_gaps(mask, sites, length, skeleton=skeleton)
                score(domain, image_id, f'gap_{length}', pred, labels)
                for event_index, event in enumerate(events):
                    event_rows.append(dict(domain=domain, image=image_id, event=event_index, length=length,
                                           width=event['width'], length_width_ratio=event['length_width_ratio'],
                                           ratio_bin=ratio_bin(event['length_width_ratio']),
                                           center=json.dumps(event['center']), anchors=json.dumps(event['anchors']),
                                           window=json.dumps(event['window']), removed_pixels=event['removed_pixels'],
                                           status_4=event['status_4'], status_8=event['status_8']))
                for method, parameter in controls:
                    fixed = (close(pred, parameter) if method == 'close' else endpoint_connect(pred, valid, parameter)) & valid
                    condition = f'gap_{length}_{method}_{parameter}'
                    score(domain, image_id, condition, fixed, labels)
                    for event_index, event in enumerate(events):
                        for connectivity in (4, 8):
                            status = event[f'status_{connectivity}']
                            repaired = connected_at(fixed, event['anchors'], connectivity, event['window']) if status == 'valid' else False
                            repair_rows.append(dict(domain=domain, image=image_id, event=event_index,
                                                    length=length, width=event['width'], ratio_bin=ratio_bin(event['length_width_ratio']),
                                                    method=method, parameter=parameter, connectivity=connectivity,
                                                    status=status, repaired=int(repaired)))
                    if args.examples and image_index == 0 and length == 16 and (method, parameter) in [('close', 8), ('endpoint', 32)]:
                        save_example(example_dir / f'{domain.lower()}_gap16_{method}{parameter}.png', fixed, labels)
                if args.examples and image_index == 0 and length == 16:
                    save_example(example_dir / f'{domain.lower()}_gap16.png', pred, labels)
            pred, events, reason = false_bridges(mask, valid, rng_for(args.seed, f'bridge:{domain}:{image_id}'))
            score(domain, image_id, 'bridge_max3', pred, labels)
            bridge_rows.append(dict(domain=domain, image=image_id, bridges=len(events), reason=reason,
                                    events=json.dumps(events)))
            for radius in (1, 2, 3):
                score(domain, image_id, f'thicken_{radius}', thicken(mask, radius), labels)
            for radius in (1, 2):
                score(domain, image_id, f'thin_{radius}', thin(mask, radius), labels)
            for method, parameter in controls:
                fixed = (close(mask, parameter) if method == 'close' else endpoint_connect(mask, valid, parameter)) & valid
                score(domain, image_id, f'clean_{method}_{parameter}', fixed, labels)
                for connectivity in (4, 8):
                    merges = component_merges(mask, fixed, connectivity)
                    false_rows.append(dict(domain=domain, image=image_id, method=method, parameter=parameter,
                                           connectivity=connectivity, new_component_merges=merges,
                                           has_false_connection=int(merges > 0)))
            if args.examples and image_index == 0:
                save_example(example_dir / f'{domain.lower()}_clean.png', mask, labels)
                save_example(example_dir / f'{domain.lower()}_bridge.png', pred, labels)
            print(f'{domain} {image_index+1}/{args.n}: sites={len(sites)}, bridges={len(events)}, elapsed={time.monotonic()-started:.1f}s', flush=True)
    rows = []
    for (domain, condition), counts in sorted(totals.items()):
        value = metrics.summarize(counts)
        baseline = metrics.summarize(totals[(domain, 'clean')])
        quality_drop = baseline['quality'] - value['quality']
        pixels = changed[(domain, condition)]
        rows.append(dict(domain=domain, condition=condition, images=image_numbers[(domain, condition)],
                         modified_images=modified[(domain, condition)], changed_valid_pixels=pixels,
                         **{key: value[key] for key in KEYS},
                         **{f'drop_{key}': baseline[key] - value[key] for key in KEYS},
                         drop_quality_per_1000_changed_pixels=1000 * quality_drop / pixels if pixels else ''))
    repair_summary = []
    for domain in ('Urban', 'Rural', 'All'):
        subset = [r for r in repair_rows if domain == 'All' or r['domain'] == domain]
        for grouping in ('length', 'ratio_bin'):
            groups = defaultdict(list)
            for row in subset:
                groups[(row['method'], row['parameter'], row['connectivity'], str(row[grouping]))].append(row)
            for (method, parameter, connectivity, group), records in sorted(groups.items()):
                eligible = [r for r in records if r['status'] == 'valid']
                repaired = sum(r['repaired'] for r in eligible)
                repair_summary.append(dict(domain=domain, grouping=grouping, group=group,
                                           method=method, parameter=parameter, connectivity=connectivity,
                                           events=len(records), valid_events=len(eligible), repaired=repaired,
                                           bypassed=sum(r['status'] == 'bypassed' for r in records),
                                           invalid=sum(r['status'] not in ('valid', 'bypassed') for r in records),
                                           repair_rate=repaired / len(eligible) if eligible else ''))
    false_summary = []
    for domain in ('Urban', 'Rural', 'All'):
        for method, parameter in controls:
            for connectivity in (4, 8):
                subset = [r for r in false_rows if (domain == 'All' or r['domain'] == domain) and r['method'] == method and r['parameter'] == parameter and r['connectivity'] == connectivity]
                false_summary.append(dict(domain=domain, method=method, parameter=parameter, connectivity=connectivity,
                                          images=len(subset), images_with_false_connections=sum(r['has_false_connection'] for r in subset),
                                          new_component_merges=sum(r['new_component_merges'] for r in subset)))
    write_csv(out / 'metrics.csv', rows)
    write_csv(out / 'samples.csv', sample_rows)
    write_csv(out / 'gap_sites.csv', site_rows)
    write_csv(out / 'gap_events.csv', event_rows, ['domain', 'image', 'event', 'length', 'width', 'length_width_ratio', 'ratio_bin', 'center', 'anchors', 'window', 'removed_pixels', 'status_4', 'status_8'])
    write_csv(out / 'bridge_events.csv', bridge_rows)
    write_csv(out / 'repair_events.csv', repair_rows, ['domain', 'image', 'event', 'length', 'width', 'ratio_bin', 'method', 'parameter', 'connectivity', 'status', 'repaired'])
    write_csv(out / 'repair_rates.csv', repair_summary, ['domain', 'grouping', 'group', 'method', 'parameter', 'connectivity', 'events', 'valid_events', 'repaired', 'bypassed', 'invalid', 'repair_rate'])
    write_csv(out / 'clean_false_connections.csv', false_summary)
    write_csv(out / 'clean_false_connections_per_image.csv', false_rows)
    write_csv(out / 'per_image_metrics.csv', per_image)
    elapsed = time.monotonic() - started
    run_info = dict(seed=args.seed, n_per_domain=args.n, split='Train', gap_sites_per_image_limit=args.gaps_per_image,
                    elapsed_seconds=elapsed, counts={f'{d}/{c}': metrics.summarize(v)['counts'] for (d,c),v in totals.items()},
                    metrics_sha256=hashlib.sha256((Path(os.environ['EC523_STARTER_DIR']) / 'metrics.py').read_bytes()).hexdigest(),
                    versions=dict(numpy=np.__version__, scipy=__import__('scipy').__version__, skimage=__import__('skimage').__version__, pillow=__import__('PIL').__version__))
    (out / 'run_info.json').write_text(json.dumps(run_info, indent=2) + '\n')
    content = [f'# Real Train-label structured benchmark (N={args.n} per domain)', '',
               f'Seed {args.seed}; CPU elapsed {elapsed:.2f} seconds. Only road-positive Train masks were read. Val was never opened.', '',
               '## Methods and interpretation', '',
               '- Raw Pillow integer labels remain 0–7; road=3, ignore=0. Each sampled clean mask scored exactly 1.0 on all four official metrics before corruption. For Urban/Rural/All, official integer Counts are pooled before summarize; per-image CSVs are diagnostic, never averaged for headline scores.',
               '- Sampling is a seeded permutation with rejection of road-free masks: N road-positive masks per domain. Gap sites use a separately seeded stream, a maximum of three separated sites per mask, local PCA tangents, and clearance > 64/2 + width + 12 pixels from junctions, endpoints, image frame and ignore. Some positive masks have no eligible gap; report their unchanged scores and modified-image counts rather than resampling.',
               '- Gap cuts remove a straight tangent-aligned slab of length L and transverse half-width (local width + 2). Local width is twice the original foreground Euclidean distance at the centerline site. The recorded local ROI radius is ceil(L + 2*width + 16); anchors are real skeleton pixels beyond each side. An event is valid only if its anchors connect before cutting but disconnect after all cuts, inside this same ROI. Local alternate routes are bypassed; before-disconnected/missing-anchor cases are invalid. Separate four/eight-neighbor denominators exclude both. Global network connectivity is not used as a repair test.',
               '- Bridge search proposes nearest distinct-component boundary pairs at Euclidean separation 10–60 px. At most three independent connections per image; strokes cannot touch ignore or a third component. Width uses the smaller twice-EDT estimate at the nearest original centerline pixel on either side. Search is bounded at 4000 seeded boundary candidates; skip reasons are recorded. No guarantee of finding every eligible bridge.',
               '- Thicken/Thin use Euclidean outward/inward distance thresholds in pixels. Thicken 1 and Thin 1 describe opposite one-pixel boundary offsets, not spatial translation. Predictions are clipped to valid pixels for scoring and all connectivity checks.',
               '- Closing uses Euclidean disks r=1,2,4,8 and SciPy default zero padding. Endpoint connection uses D=16,32,64; both outward tangents must face the joining vector within 30 degrees. Tangents come from the first eight skeleton steps; width is the smaller maximum twice-EDT within 32 inward steps. Greedy shortest pairs use each endpoint once. These operations do not access original gap sites or labels beyond the valid mask.',
               '- New clean false connections are component merges: sum(m-1) for each resulting component overlapping m originally separate components, reported for four/eight-neighbor definitions. Intra-component shortcuts/cycle creation are not counted. This metric describes annotation connectivity, not road navigability or grade separation.',
               '- Per-1000 normalization is 1000 * pooled quality drop / total changed valid pixels, including skips as zero changes. It is descriptive and nonlinear, and depends on sample size/road density; not an additive per-pixel causal estimate. No confidence interval is estimated.', '',
               '## Error scores and clean self-score', '',
               markdown_table([r for r in rows if r['condition']=='clean' or r['condition'].startswith(('thicken_', 'thin_', 'bridge_')) or (r['condition'].startswith('gap_') and r['condition'].count('_')==1)], list(rows[0])), '',
               '## Repair-control scores', '', markdown_table([r for r in rows if r['condition'].count('_')>1], list(rows[0])), '',
               '## Repair rates by gap length', '', markdown_table([r for r in repair_summary if r['grouping']=='length'], ['domain','group','method','parameter','connectivity','events','valid_events','repaired','bypassed','invalid','repair_rate']), '',
               '## Repair rates by length / local width', '', markdown_table([r for r in repair_summary if r['grouping']=='ratio_bin'], ['domain','group','method','parameter','connectivity','events','valid_events','repaired','bypassed','invalid','repair_rate']), '',
               '## False connections on clean masks', '', markdown_table(false_summary, list(false_summary[0])), '',
               '## Skips and reproducibility', '', markdown_table(site_rows,list(site_rows[0])), '', markdown_table([{k:r[k] for k in ('domain','image','bridges','reason')} for r in bridge_rows], ['domain','image','bridges','reason']), '',
               'Configure course paths if needed, source scripts/activate_scc.sh, then run:', '',
               f'`python analysis/real_benchmark.py --n {args.n} --seed {args.seed} --output results/real_benchmark/<fresh-run>`', '',
               'Optional `--examples tmp/real_benchmark_examples` saves derived visualizations only in ignored tmp/. No data-derived PNG is written to results or staged in Git. See samples.csv for label hashes, gap_events.csv for exact local windows/anchors/statuses, repair_events.csv for event outcomes, and run_info.json for pooled counts/versions/timing.']
    (out / 'summary.md').write_text('\n'.join(content) + '\n')
    print(f'Finished {2*args.n} Train masks in {elapsed:.2f}s. Numeric results: {out.relative_to(ROOT)}', flush=True)


def save_example(path, prediction, labels):
    rgb = np.full((*labels.shape, 3), 25, dtype=np.uint8)
    rgb[prediction] = (235, 235, 235)
    rgb[labels == 0] = (180, 60, 170)
    Image.fromarray(rgb).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=100, help='Road-positive Train masks per domain.')
    parser.add_argument('--seed', type=int, default=523)
    parser.add_argument('--gaps-per-image', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/real_benchmark/full_n100')
    parser.add_argument('--examples', type=Path, default=None)
    args = parser.parse_args()
    if args.n <= 0 or args.gaps_per_image <= 0:
        parser.error('n and gaps-per-image must be positive.')
    run(args)


if __name__ == '__main__':
    main()
