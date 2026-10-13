#!/usr/bin/env python3
"""Read-only, CPU-only PNG audit; raw labels are never remapped or clipped."""
import argparse
import csv
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table(rows, fields):
    def fmt(value):
        return f'{value:.6f}' if isinstance(value, float) else str(value)
    return '\n'.join(['| ' + ' | '.join(fields) + ' |',
                      '| ' + ' | '.join(['---'] * len(fields)) + ' |'] +
                     ['| ' + ' | '.join(fmt(row[f]) for f in fields) + ' |' for row in rows])


def distribution(values):
    values = np.asarray(values)
    if not values.size:
        return dict(zip(['min', 'p10', 'p25', 'median', 'p75', 'p90', 'p95', 'max'], [float('nan')] * 8))
    return dict(zip(['min', 'p10', 'p25', 'median', 'p75', 'p90', 'p95', 'max'],
                    map(float, np.percentile(values, [0, 10, 25, 50, 75, 90, 95, 100]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'results/data_audit')
    parser.add_argument('--seed', type=int, default=523)
    parser.add_argument('--sample-size', type=int, default=50)
    parser.add_argument('--max-seconds', type=float, default=540)
    args = parser.parse_args()
    if not os.environ.get('EC523_DATA_DIR'):
        parser.error('Set EC523_DATA_DIR to the shared dataset root.')
    root = Path(os.environ['EC523_DATA_DIR']).resolve()
    out = args.output.resolve()
    if out == root or root in out.parents:
        parser.error('Output must be outside the read-only dataset.')
    if args.sample_size <= 0 or args.max_seconds <= 0:
        parser.error('Sample size and time limit must be positive.')
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def deadline():
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError('Audit time budget exceeded; no complete summary was written.')

    anomalies, integrity, labels_summary, samples, components, structure = [], [], [], [], [], []
    rng = np.random.default_rng(args.seed)
    square = np.ones((3, 3), dtype=bool)
    for split in ('Train', 'Val'):
        for domain in ('Urban', 'Rural'):
            deadline()
            group = dict(split=split, domain=domain)
            idir, mdir = root / split / domain / 'images_png', root / split / domain / 'masks_png'
            image_paths = sorted(idir.glob('*.png'))
            mask_paths = sorted(mdir.glob('*.png'))
            image_names, mask_names = {p.name for p in image_paths}, {p.name for p in mask_paths}
            def issue(name, kind, detail):
                anomalies.append(dict(**group, file=name, kind=kind, detail=detail))
            for directory in (idir, mdir):
                if not directory.is_dir():
                    issue('', 'missing_directory', str(directory.relative_to(root)))
            if not image_paths:
                issue('', 'empty_image_directory', 'No PNG images found')
            for name in sorted(image_names - mask_names):
                issue(name, 'missing_mask', 'No mask with the same filename')
            for name in sorted(mask_names - image_names):
                issue(name, 'orphan_mask', 'No image with the same filename')
            counts = np.zeros(8, dtype=np.int64)
            valid_names, zero_road, decoded_images, decoded_masks = [], 0, 0, 0
            for path in image_paths:
                deadline()
                try:
                    with Image.open(path) as im:
                        image = np.asarray(im)
                        mode = im.mode
                    decoded_images += 1
                    if mode != 'RGB' or image.shape != (1024, 1024, 3) or image.dtype != np.uint8:
                        issue(path.name, 'invalid_image', f'mode={mode}; shape={image.shape}; dtype={image.dtype}')
                except Exception as exc:
                    issue(path.name, 'image_decode_error', str(exc))
            # Visit every mask, including orphans, to enumerate all format errors.
            for path in mask_paths:
                deadline()
                try:
                    with Image.open(path) as im:
                        mask = np.asarray(im)
                    decoded_masks += 1
                    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
                        issue(path.name, 'invalid_mask_format', f'shape={mask.shape}; dtype={mask.dtype}')
                        continue
                    good = True
                    if mask.shape != (1024, 1024):
                        issue(path.name, 'invalid_mask_size', str(mask.shape))
                        good = False
                    hist = np.bincount(mask.ravel().astype(np.int64)) if mask.min() >= 0 else None
                    if hist is None or hist.size > 8:
                        issue(path.name, 'invalid_label_values', str(np.unique(mask).tolist()))
                        good = False
                    if good and path.name in image_names:
                        counts[:hist.size] += hist
                        zero_road += int(hist.size <= 3 or hist[3] == 0)
                        valid_names.append(path.name)
                except Exception as exc:
                    issue(path.name, 'mask_decode_error', str(exc))
            group_issues = sum(a['split'] == split and a['domain'] == domain for a in anomalies)
            integrity.append(dict(**group, images=len(image_paths), masks=len(mask_paths),
                                  decoded_images=decoded_images, decoded_masks=decoded_masks,
                                  valid_paired_masks=len(valid_names), anomalies=group_issues))
            total = int(counts.sum())
            effective = total - int(counts[0])
            labels_summary.append(dict(**group, images=len(valid_names), total_pixels=total,
                                       **{f'label_{i}_fraction': float(counts[i] / total) if total else float('nan') for i in range(8)},
                                       road_valid_fraction=float(counts[3] / effective) if effective else float('nan'),
                                       ignore_fraction=float(counts[0] / total) if total else float('nan'),
                                       no_road_images=zero_road))
            selected = sorted(rng.choice(valid_names, size=min(args.sample_size, len(valid_names)), replace=False).tolist())
            widths_group, sizes_group, sample_rows = [], [], []
            example_saved = False
            for index, name in enumerate(selected):
                deadline()
                with Image.open(mdir / name) as im:
                    road = np.asarray(im) == 3
                labeled, ncomponents = ndi.label(road, structure=square)
                sizes = np.bincount(labeled.ravel())[1:]
                skeleton = skeletonize(road, method='zhang')
                distance = ndi.distance_transform_edt(road)
                widths = (2 * distance[skeleton]).astype(np.float32)
                degree = ndi.convolve(skeleton.astype(np.uint8), square.astype(np.uint8), mode='constant') - skeleton
                endpoints = int(np.count_nonzero(skeleton & (degree == 1)))
                junction_pixels = skeleton & (degree >= 3)
                _, njunctions = ndi.label(junction_pixels, structure=square)
                row = dict(**group, file=name, road_pixels=int(road.sum()), components=int(ncomponents),
                           largest_component_pixels=int(sizes.max()) if sizes.size else 0,
                           skeleton_pixels=int(skeleton.sum()), endpoints=endpoints,
                           junction_clusters=int(njunctions), junction_pixels=int(junction_pixels.sum()),
                           width_median_px=float(np.median(widths)) if widths.size else float('nan'),
                           width_p90_px=float(np.percentile(widths, 90)) if widths.size else float('nan'))
                samples.append(row)
                sample_rows.append(row)
                components.extend(dict(**group, file=name, component=i + 1, pixels=int(size)) for i, size in enumerate(sizes))
                widths_group.append(widths)
                sizes_group.extend(sizes.tolist())
                if not example_saved and (road.any() or index == len(selected) - 1):
                    with Image.open(idir / name) as im:
                        rgb = np.asarray(im).copy()
                    if rgb.shape == (1024, 1024, 3) and rgb.dtype == np.uint8:
                        rgb[road] = np.rint(0.55 * rgb[road] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
                        Image.fromarray(rgb).save(out / f'example_{split.lower()}_{domain.lower()}.png')
                        example_saved = True
            all_widths = np.concatenate(widths_group) if widths_group else np.array([])
            n = len(sample_rows)
            structure.append(dict(**group, sampled_images=n, components_total=len(sizes_group),
                                  components_per_image_mean=float(np.mean([r['components'] for r in sample_rows])) if n else float('nan'),
                                  endpoints_per_image_mean=float(np.mean([r['endpoints'] for r in sample_rows])) if n else float('nan'),
                                  junction_clusters_per_image_mean=float(np.mean([r['junction_clusters'] for r in sample_rows])) if n else float('nan'),
                                  skeleton_pixels=int(all_widths.size),
                                  **{f'component_size_{k}_px': v for k, v in distribution(sizes_group).items()},
                                  **{f'width_{k}_px': v for k, v in distribution(all_widths).items()}))
            print(f'{split}/{domain}: {len(image_paths)} images, {group_issues} anomalies, {n} structure samples; elapsed={time.monotonic()-started:.1f}s', flush=True)
    deadline()
    write_csv(out / 'integrity.csv', integrity, list(integrity[0]))
    write_csv(out / 'label_statistics.csv', labels_summary, list(labels_summary[0]))
    write_csv(out / 'structure_summary.csv', structure, list(structure[0]))
    write_csv(out / 'structure_samples.csv', samples, list(samples[0]) if samples else ['split', 'domain', 'file'])
    write_csv(out / 'component_sizes.csv', components, ['split', 'domain', 'file', 'component', 'pixels'])
    write_csv(out / 'anomalies.csv', anomalies, ['split', 'domain', 'file', 'kind', 'detail'])
    elapsed = time.monotonic() - started
    text = ['# Shared dataset audit', '', f'Seed: {args.seed}. CPU-only elapsed time: {elapsed:.2f} seconds. Raw labels 0–7 were preserved; road=3, ignore=0.', '',
            'Reproduce from the repository root after activating the existing environment:', '',
            '```bash', 'python scripts/data_audit.py --seed 523 --sample-size 50', '```', '',
            'EC523_DATA_DIR must be set. Data were opened read-only with Pillow; no source data or masks were copied. Output PNGs are derived road overlays.', '',
            '## Integrity (all images and masks)', '', table(integrity, list(integrity[0])), '',
            '## Label proportions (all valid paired masks)', '',
            'Fractions use all mask pixels; road_valid_fraction uses only nonzero labels. Invalid masks are excluded explicitly through integrity counts.', '',
            table(labels_summary, list(labels_summary[0])), '',
            '## Connected components (fixed random subset)', '',
            '8-neighbor foreground connectivity. Sizes are in pixels; quantiles pool components rather than weighting images equally.', '',
            table(structure, ['split', 'domain', 'sampled_images', 'components_total', 'components_per_image_mean', 'component_size_min_px', 'component_size_median_px', 'component_size_p90_px', 'component_size_max_px']), '',
            '## Estimated road width distribution', '',
            'Width = 2 * Euclidean distance to nearest background at each Zhang centerline pixel. Quantiles pool centerline pixels, weighting longer roads more. Pixel discretization and intersection interiors can inflate estimates; these are calibration proxies, not physical measurements.', '',
            table(structure, ['split', 'domain', 'skeleton_pixels'] + [f'width_{k}_px' for k in distribution([])]), '',
            '## Centerline endpoints and junctions', '',
            'Endpoints have exactly one foreground neighbor in the 8-neighbor skeleton graph. Junctions are 8-connected clusters of degree >=3 pixels; junction_pixels is also retained per image. Diagonal staircases, adjacent branches, image boundaries and ignored areas can affect these counts; they are not validated road-network nodes.', '',
            table(structure, ['split', 'domain', 'sampled_images', 'endpoints_per_image_mean', 'junction_clusters_per_image_mean']), '',
            '## Anomalies', '', table(anomalies, ['split', 'domain', 'file', 'kind', 'detail']) if anomalies else 'No anomalies observed.', '',
            '## Examples', '',
            'The first road-containing filename in each sorted random subset supplies the example (last sample if all are road-free); red overlay marks raw road pixels. Exact sampled filenames and per-image statistics are in structure_samples.csv.', '',
            '| Group | Derived overlay |', '| --- | --- |']
    for row in integrity:
        split, domain = row['split'], row['domain']
        text.append(f'| {split}/{domain} | ![{split} {domain}](example_{split.lower()}_{domain.lower()}.png) |')
    text.extend(['', '## Observations and calibration limits', ''])
    for r, geom in zip(labels_summary, structure):
        text.append(f"- Observed {r['split']}/{r['domain']}: road/valid={100*r['road_valid_fraction']:.2f}%, ignore={100*r['ignore_fraction']:.2f}%, no-road images={r['no_road_images']}/{r['images']}; sampled width median={geom['width_median_px']:.2f}px, p90={geom['width_p90_px']:.2f}px.")
    text.append('Observed differences are descriptive. Width tails can include broad annotated regions and intersection geometry; inspect overlays before using them as synthetic road widths. Endpoints and components describe existing masks, not missing-road gap lengths. No artificial gap-length distribution was measured. The sample is fixed at 50 images per group and no uncertainty interval is estimated.')
    (out / 'summary.md').write_text('\n'.join(text) + '\n')
    print(f'Completed in {time.monotonic()-started:.2f}s; anomalies={len(anomalies)}', flush=True)


if __name__ == '__main__':
    main()
