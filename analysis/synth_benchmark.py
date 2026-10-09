"""CPU-only structured road benchmark using official pooled metrics."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
starter_dir = os.environ.get('EC523_STARTER_DIR')
if not starter_dir:
    raise SystemExit(
        'EC523_STARTER_DIR is not set. Obtain the course starter package, then '
        'export EC523_STARTER_DIR="<YOUR_STARTER_DIR>/cli" before running this script.'
    )
starter_path = Path(starter_dir).expanduser().resolve()
if not (starter_path / 'metrics.py').is_file() or not (starter_path / 'common.py').is_file():
    raise SystemExit('EC523_STARTER_DIR must point to the course cli directory containing metrics.py and common.py.')
sys.path.insert(0, str(starter_path))
from metrics import Counts, image_counts, summarize

KEYS = ['road_iou', 'cldice', 'boundary_iou', 'quality']
CONNECT = np.ones((3, 3), bool)


def draw_line(mask, points, width):
    image = Image.fromarray(mask.astype('uint8') * 255)
    ImageDraw.Draw(image).line(points, fill=255, width=int(width))
    return np.asarray(image) > 0


def generate(seed, size):
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), bool)
    scale = size / 1024
    xy = lambda v: int(round(v * scale))
    roads, targets, pairs = [], [], []
    ys = [xy(v + rng.integers(-15, 16)) for v in (150, 320, 490)]
    for j, y in enumerate(ys):
        width = int(rng.integers(4, 9) if j % 2 == 0 else rng.integers(12, 21))
        points = [(xy(70), y), (xy(930), y)]
        mask = draw_line(mask, points, width)
        roads.append(dict(points=points, width=width, role='intersecting_horizontal'))
        targets.append((y, xy(180)))
    for j, x in enumerate((350, 550, 750)):
        x = xy(x + rng.integers(-10, 11))
        width = int(rng.integers(12, 21) if j % 2 == 0 else rng.integers(4, 9))
        points = [(x, xy(70)), (x, xy(650))]
        mask = draw_line(mask, points, width)
        roads.append(dict(points=points, width=width, role='intersecting_vertical'))
    for ybase in (735, 825, 915):
        y = xy(ybase + rng.integers(-4, 5))
        width = int(rng.integers(4, 9))
        separation = width + int(rng.integers(6, 15))
        y2 = y + separation
        for yy in (y, y2):
            points = [(xy(120), yy), (xy(900), yy)]
            mask = draw_line(mask, points, width)
            roads.append(dict(points=points, width=width, role='disconnected_parallel'))
        pairs.append(dict(a=[y, xy(510)], b=[y2, xy(510)], width=width))
    labels = np.where(mask, 3, 1).astype('uint8')
    if seed % 3 == 0:
        labels[xy(300):xy(335), xy(525):xy(570)] = 0
    return labels, roads, targets, pairs


def analyze(mask):
    skel = skeletonize(mask, method='zhang')
    degree = ndi.convolve(skel.astype('uint8'), CONNECT.astype('uint8'), mode='constant') - skel
    endpoints = skel & (degree == 1)
    junctions = skel & (degree >= 3)
    distance = ndi.distance_transform_edt(mask)
    segments, n = ndi.label(skel & ~ndi.binary_dilation(junctions, iterations=2), CONNECT)
    records = []
    for ident, sl in enumerate(ndi.find_objects(segments), 1):
        if sl is None:
            continue
        local = segments[sl] == ident
        widths = 2 * distance[sl][local]
        records.append(dict(segment=ident, pixels=int(local.sum()), width_min=float(widths.min()),
                            width_median=float(np.median(widths)), width_max=float(widths.max())))
    return skel, distance, junctions, dict(endpoints=np.argwhere(endpoints).tolist(),
        junction_pixels=np.argwhere(junctions).tolist(), segments=records)


def gap_sites(skel, distance, junctions, targets, maximum):
    coords = np.argwhere(skel)
    jd = ndi.distance_transform_edt(~junctions)
    sites = []
    for target in targets:
        p = coords[np.argmin(np.sum((coords - target) ** 2, axis=1))]
        near = coords[np.sum((coords-p)**2, axis=1) <= 12**2]
        _, vectors = np.linalg.eigh(np.cov(near.T))
        tangent = vectors[:, -1]
        if tangent[1] < 0:
            tangent = -tangent
        width = float(2 * distance[tuple(p)])
        assert jd[tuple(p)] > maximum / 2 + width + 5
        sites.append(dict(center=p.tolist(), tangent=tangent.tolist(), width=width))
    return sites


def gaps(mask, sites, length):
    result = mask.copy()
    events = []
    for site in sites:
        p, t = np.array(site['center']), np.array(site['tangent'])
        radius = int(np.ceil(length / 2 + site['width'] + 5))
        lo = np.maximum(p-radius, 0).astype(int)
        hi = np.minimum(p+radius+1, mask.shape).astype(int)
        yy, xx = np.mgrid[lo[0]:hi[0], lo[1]:hi[1]]
        offset = np.stack([yy-p[0], xx-p[1]], axis=-1)
        along = offset @ t
        across = offset @ np.array([-t[1], t[0]])
        cut = (along >= -length/2) & (along < length/2) & (np.abs(across) <= site['width']+2)
        result[lo[0]:hi[0], lo[1]:hi[1]][cut] = False
        anchors = [np.rint(p + sign * (length/2+4)*t).astype(int).tolist() for sign in (-1, 1)]
        events.append(dict(**site, length=length, anchors=anchors, removed_pixels=int((mask[lo[0]:hi[0],lo[1]:hi[1]] & cut).sum())))
    components, _ = ndi.label(result, CONNECT)
    for event in events:
        a, b = event['anchors']
        assert components[tuple(a)] and components[tuple(b)]
        assert components[tuple(a)] != components[tuple(b)], 'Gap has an alternate route'
    return result, events


def close(mask, radius):
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    disk = xx*xx+yy*yy <= radius*radius
    return ndi.binary_closing(mask, structure=disk)


def save_png(path, mask, labels):
    rgb = np.full((*mask.shape, 3), 28, dtype='uint8')
    rgb[mask] = (235, 235, 235)
    rgb[labels == 0] = (180, 60, 170)
    Image.fromarray(rgb).save(path)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--seed', type=int, default=523)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--output', type=Path, default=ROOT/'tmp/synth_benchmark')
    parser.add_argument('--lengths', type=int, nargs='+', default=[4,8,16,32,64])
    parser.add_argument('--radii', type=int, nargs='+', default=[1,2,4,8])
    args = parser.parse_args()
    if args.n < 1 or args.size < 768:
        parser.error('Use n >= 1 and size >= 768; widths and gap lengths are pixel units.')
    out = args.output.resolve()
    if not out.is_relative_to(ROOT/'tmp'):
        parser.error('Output must be inside the repository tmp/ directory (ignored by Git).')
    out.mkdir(parents=True, exist_ok=True)
    (out/'examples').mkdir(exist_ok=True)
    start = time.monotonic()
    data = [generate(args.seed+i,args.size) for i in range(args.n)]
    totals = {}
    def score(name, prediction, labels):
        totals[name] = totals.get(name, Counts()) + image_counts(prediction.astype('float32'),labels)
    for labels, *_ in data:
        own = summarize(image_counts((labels==3).astype('float32'), labels))
        if any(own[k] != 1.0 for k in KEYS):
            raise RuntimeError(f'Clean self-score failed: {own}')
        score('clean',labels==3,labels)
    baseline = summarize(totals['clean'])
    print('Clean sanity check passed: all four metrics equal 1.0.', flush=True)
    metadata, repairs, bridges = [], {}, {}
    for i, (labels, roads, targets, pairs) in enumerate(data):
        mask = labels == 3
        skel, distance, junctions, structure = analyze(mask)
        sites = gap_sites(skel,distance,junctions,targets,max(args.lengths))
        original_cc, original_n = ndi.label(mask, CONNECT)
        for pair in pairs:
            assert original_cc[tuple(pair['a'])] != original_cc[tuple(pair['b'])]
        record = dict(image=i, seed=args.seed+i, roads=roads, structure=structure, errors=[])
        for length in args.lengths:
            pred, events = gaps(mask,sites,length)
            score(f'gap_{length}',pred,labels)
            record['errors'].append(dict(type='gap',length=length,events=events))
            for radius in args.radii:
                fixed = close(pred,radius) & (labels!=0)
                score(f'gap_{length}_close_{radius}',fixed,labels)
                cc, _ = ndi.label(fixed,CONNECT)
                key = (length,radius)
                repaired = sum(bool(cc[tuple(e['anchors'][0])]) and cc[tuple(e['anchors'][0])] == cc[tuple(e['anchors'][1])] for e in events)
                old = repairs.get(key,[0,0])
                repairs[key] = [old[0]+repaired,old[1]+len(events)]
                for event in events:
                    event.setdefault('repair_by_radius',{})[str(radius)] = bool(cc[tuple(event['anchors'][0])]) and bool(cc[tuple(event['anchors'][0])] == cc[tuple(event['anchors'][1])])
                if i < 2 and length in (8,32) and radius in (4,8):
                    save_png(out/'examples'/f'image_{i}_gap_{length}_close_{radius}.png',fixed,labels)
            if i < 2 and length in (8,32):
                save_png(out/'examples'/f'image_{i}_gap_{length}.png',pred,labels)
        for count in (1,3):
            pred = mask.copy()
            for pair in pairs[:count]:
                pred = draw_line(pred, [tuple(pair['a'][::-1]),tuple(pair['b'][::-1])],pair['width'])
            score(f'bridge_{count}',pred,labels)
            record['errors'].append(dict(type='bridge',count=count,events=pairs[:count]))
            if i < 2:
                save_png(out/'examples'/f'image_{i}_bridge_{count}.png',pred,labels)
        background_distance = ndi.distance_transform_edt(~mask)
        _, nearest_centerline = ndi.distance_transform_edt(~skel, return_indices=True)
        local_radius = distance[tuple(nearest_centerline)]
        for k in (1,2,3):
            pred = background_distance <= k
            score(f'thicken_k_{k}',pred,labels)
            record['errors'].append(dict(type='thicken',mode='uniform',k=k,added_pixels=int((pred&~mask).sum()),location='all road boundaries'))
        for fraction in (.25,.5):
            pred = background_distance <= fraction * (2*local_radius)
            score(f'thicken_fraction_{fraction}',pred,labels)
            record['errors'].append(dict(type='thicken',mode='proportional',fraction=fraction,added_pixels=int((pred&~mask).sum()),location='all road boundaries'))
            if i < 2:
                save_png(out/'examples'/f'image_{i}_thicken_fraction_{fraction}.png',pred,labels)
        for radius in args.radii:
            pred = close(mask,radius) & (labels!=0)
            score(f'clean_close_{radius}',pred,labels)
            cc, n = ndi.label(pred,CONNECT)
            merged = sum(cc[tuple(p['a'])] == cc[tuple(p['b'])] for p in pairs)
            # Count merges across every original component, not only designated pairs.
            groups = {}
            for component in range(1,original_n+1):
                memberships = np.unique(cc[original_cc==component])
                for member in memberships[memberships>0]:
                    groups.setdefault(int(member),set()).add(component)
            merges = sum(len(group)-1 for group in groups.values() if len(group)>1)
            old = bridges.get(radius,[0,0,0,0])
            bridges[radius] = [old[0]+merged,old[1]+len(pairs),old[2]+int(merges>0),old[3]+merges]
            if i < 2 and radius == 8:
                save_png(out/'examples'/f'image_{i}_clean_close_8.png',pred,labels)
        if i < 2:
            save_png(out/'examples'/f'image_{i}_clean.png',mask,labels)
        Image.fromarray(labels).save(out/'examples'/f'labels_{i:02d}.png')
        metadata.append(record)
        print(f'Completed image {i+1}/{args.n} ({time.monotonic()-start:.1f}s)',flush=True)
    rows = []
    for name, counts in totals.items():
        values = summarize(counts)
        rows.append(dict(condition=name,**{k:values[k] for k in KEYS},**{f'drop_{k}':baseline[k]-values[k] for k in KEYS}))
    repair_rows = [dict(length=l,radius=r,repaired=a,total=b,repair_rate=a/b) for (l,r),(a,b) in sorted(repairs.items())]
    bridge_rows = [dict(radius=r,connected_pairs=a,total_pairs=b,pair_rate=a/b,images_with_new_bridges=c,total_images=args.n,component_merges=d) for r,(a,b,c,d) in sorted(bridges.items())]
    write_csv(out/'metrics.csv',rows)
    write_csv(out/'gap_repair.csv',repair_rows)
    write_csv(out/'clean_false_bridges.csv',bridge_rows)
    (out/'metadata.json').write_text(json.dumps(dict(config=vars(args)|{'output':str(out.relative_to(ROOT))},elapsed_seconds=time.monotonic()-start,versions=dict(numpy=np.__version__,scipy=__import__('scipy').__version__,skimage=__import__('skimage').__version__,pillow=__import__('PIL').__version__),images=metadata,pooled_counts={k:summarize(v)['counts'] for k,v in totals.items()}),indent=2))
    def table(items):
        fields=list(items[0])
        lines=['| '+' | '.join(fields)+' |','| '+' | '.join(['---']*len(fields))+' |']
        for row in items:
            lines.append('| '+' | '.join(f'{row[k]:.6f}' if isinstance(row[k],float) else str(row[k]) for k in fields)+' |')
        return '\n'.join(lines)
    sections=[('Clean self-score',[r for r in rows if r['condition']=='clean']),('Gap sensitivity',[r for r in rows if r['condition'].startswith('gap_') and 'close' not in r['condition']]),('Bridge sensitivity',[r for r in rows if r['condition'].startswith('bridge_')]),('Uniform thickening',[r for r in rows if r['condition'].startswith('thicken_k')]),('Proportional thickening',[r for r in rows if r['condition'].startswith('thicken_fraction')]),('Closing damaged roads',[r for r in rows if r['condition'].startswith('gap_') and 'close' in r['condition']]),('Per-gap connectivity repair',repair_rows),('Closing clean roads',[r for r in rows if r['condition'].startswith('clean_close')]),('New bridges on clean roads',bridge_rows)]
    intro=f'''# Structured synthetic road benchmark\n\nDate: {__import__("datetime").date.today().isoformat()}. Seed: {args.seed}. Images: {args.n}. Size: {args.size} x {args.size}. CPU elapsed: {time.monotonic()-start:.1f} seconds.\n\nOfficial image_counts and summarize are imported through sys.path. Integer Counts are summed before summarization. All self-scores passed exactly 1.0 before experiments. Drops are clean score minus condition score. No image-score averaging is used.\n\n## Design and assumptions\n\n- Roads are axis-aligned intersecting networks with mixed nominal widths 4-8 and 12-20 pixels, plus three isolated parallel pairs. Seeded geometry and widths vary across images. Ignore rectangles occur every third seed and may intersect roads. Purple denotes ignore in visualizations; labels_*.png stores integer labels for every image.\n- Zhang centerlines, 8-neighbor endpoint/junction degrees, and twice the foreground Euclidean distance estimate local width. Junction neighborhoods are removed to summarize skeleton segments. Raster width estimates can differ from nominal drawing widths. Metadata records endpoints, junction pixels, segment width summaries, drawing coordinates, all error locations and repair outcomes.\n- Three nested gaps per image reuse centers and PCA tangents across lengths. Centers are away from junctions. Cuts span the full estimated width; anchors lie four pixels beyond each gap. Asserted disconnected anchors before closing prevent alternate-route false repair. Connectivity uses 8-neighbor components after masking ignore pixels.\n- Bridges connect designated pairs in separate original components, using their nominal drawing width. Closing uses Euclidean disk footprints. New clean bridges are reported both for designated pairs and as mergers among all original connected components.\n- Proportional thickening adds an outward radius equal to fraction times estimated full local width, taken from the nearest original centerline pixel. This increases full width by approximately twice that radius; it is not a fraction of half-width.\n- Clean ignore invariance is verified by official self-scoring; every prediction is masked by the official evaluator. Closing also masks ignore pixels before connectivity checks.\n\nThe synthetic masks contain straight, uniform-width, high-contrast geometry without image evidence, curved roads, occlusions, grade-separated crossings, complex junctions, or real segmentation noise. Observations may change with those structures, pixel scale, road density, or errors on cyclic networks. Gap component checks here intentionally use terminal branches.\n\n## Reproduction\n\nSet `EC523_STARTER_DIR` to the externally obtained course `cli/` directory, then run from the repository root: `python analysis/synth_benchmark.py`. Raw outputs default to ignored `tmp/synth_benchmark/`; publish only selected small result tables and visualizations. Configure `--n`, `--seed`, `--size`, `--lengths`, `--radii`, and `--output` as needed. Default parameters match the requested scan. Results are descriptive pooled scores, not statistical estimates of performance on real roads.\n\n'''
    (out/'results.md').write_text(intro+'\n\n'.join('## '+title+'\n\n'+table(items) for title,items in sections)+'\n')
    print(f'Finished in {time.monotonic()-start:.1f}s. Results: {out.relative_to(ROOT)}',flush=True)


if __name__ == '__main__':
    main()
