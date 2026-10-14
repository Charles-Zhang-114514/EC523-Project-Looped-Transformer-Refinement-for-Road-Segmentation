"""Shared CPU geometry, structured errors, and connectivity diagnostics.

Coordinates use (row, column), except Pillow line points which use (x, y).
Synthetic helper behavior is retained exactly for regression comparison.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from PIL import Image, ImageDraw

CONNECT = np.ones((3, 3), bool)

def draw_line(mask, points, width):
    image = Image.fromarray(mask.astype('uint8') * 255)
    ImageDraw.Draw(image).line(points, fill=255, width=int(width))
    return np.asarray(image) > 0

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


def connectivity_structure(connectivity):
    if connectivity not in (4, 8):
        raise ValueError('Connectivity must be 4 or 8.')
    return ndi.generate_binary_structure(2, 1 if connectivity == 4 else 2)


def connected_at(mask, anchors, connectivity=8, window=None, labels_map=None):
    """Require foreground anchors in the same component, optionally within an ROI.

    Window is (row_start, row_stop, col_start, col_stop); anchors are global.
    """
    anchors = np.asarray(anchors, dtype=int)
    if window is not None:
        y0, y1, x0, x1 = window
        mask = mask[y0:y1, x0:x1]
        anchors = anchors - np.array([y0, x0])
        labels_map = None
    if (anchors < 0).any() or (anchors >= np.asarray(mask.shape)).any():
        return False
    if labels_map is None:
        labels_map, _ = ndi.label(mask, connectivity_structure(connectivity))
    a, b = (int(labels_map[tuple(p)]) for p in anchors)
    return bool(a and a == b)


def component_merges(original, prediction, connectivity=8, original_labels=None,
                     prediction_labels=None):
    """Sum m-1 when a predicted component overlaps m original components.

    This retains the original synthetic benchmark's merge-count definition.
    Components removed entirely by an operation do not count as connections.
    """
    structure = connectivity_structure(connectivity)
    if original_labels is None:
        original_labels, _ = ndi.label(original, structure)
    if prediction_labels is None:
        prediction_labels, _ = ndi.label(prediction, structure)
    overlap = (original_labels > 0) & (prediction_labels > 0)
    base = int(original_labels.max()) + 1
    pairs = np.unique(prediction_labels[overlap].astype(np.int64) * base + original_labels[overlap])
    if not len(pairs):
        return 0
    memberships = np.bincount(pairs // base)
    return int(np.maximum(memberships - 1, 0).sum())


def distance_from(points):
    """Distance to a True pixel; infinity if no points exist."""
    return ndi.distance_transform_edt(~points) if points.any() else np.full(points.shape, np.inf)


def tangent_at(skeleton, point, radius=12):
    p = np.asarray(point)
    y0, x0 = np.maximum(p - radius, 0)
    y1, x1 = np.minimum(p + radius + 1, skeleton.shape)
    coords = np.argwhere(skeleton[y0:y1, x0:x1]) + np.array([y0, x0])
    coords = coords[np.sum((coords - p) ** 2, axis=1) <= radius ** 2]
    if len(coords) < 3:
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(coords.T))
    if eigenvalues[-1] <= 0 or eigenvalues[0] > .2 * eigenvalues[-1]:
        return None
    tangent = eigenvectors[:, -1]
    return tangent if tangent[1] >= 0 else -tangent


def real_gap_sites(mask, valid, rng, maximum=64, limit=3):
    """Seeded separated sites, away from endpoints/junctions/frame/ignore.

    Local straightness and clearance are deliberately conservative; report
    missing sites and do not claim representativeness of the whole network.
    """
    skeleton, distance, junctions, info = analyze(mask)
    degree = ndi.convolve(skeleton.astype('uint8'), CONNECT.astype('uint8'), mode='constant') - skeleton
    unsafe = junctions | (skeleton & (degree <= 1)) | ~valid
    clearance = distance_from(unsafe)
    coords = np.argwhere(skeleton)
    width = 2 * distance[skeleton]
    frame = np.minimum.reduce([coords[:, 0], coords[:, 1], mask.shape[0] - 1 - coords[:, 0], mask.shape[1] - 1 - coords[:, 1]])
    keep = (clearance[skeleton] > maximum / 2 + width + 12) & (frame > maximum / 2 + width + 12)
    candidates = coords[keep]
    sites = []
    for index in rng.permutation(len(candidates))[:2000]:
        point = candidates[index]
        local_width = float(2 * distance[tuple(point)])
        if any(np.linalg.norm(point - np.array(s['center'])) < 2 * maximum + local_width for s in sites):
            continue
        tangent = tangent_at(skeleton, point)
        if tangent is None:
            continue
        sites.append(dict(center=point.tolist(), tangent=tangent.tolist(), width=local_width))
        if len(sites) >= limit:
            break
    return sites, skeleton, distance, dict(candidate_pixels=int(len(candidates)), selected_sites=len(sites))


def local_gaps(mask, sites, length, skeleton=None):
    """Cut masks and classify each event within a recorded local window.

    Repair denominators are separate for 4/8 neighbors. Before-disconnected,
    missing-anchor and bypassed events are excluded from their denominator.
    """
    if skeleton is None:
        skeleton = skeletonize(mask, method='zhang')
    result = mask.copy()
    events = []
    for site in sites:
        p, tangent = np.array(site['center']), np.array(site['tangent'])
        radius = int(np.ceil(length / 2 + site['width'] + 5))
        lo = np.maximum(p - radius, 0).astype(int)
        hi = np.minimum(p + radius + 1, mask.shape).astype(int)
        yy, xx = np.mgrid[lo[0]:hi[0], lo[1]:hi[1]]
        offset = np.stack([yy - p[0], xx - p[1]], axis=-1)
        along = offset @ tangent
        across = offset @ np.array([-tangent[1], tangent[0]])
        cut = (along >= -length / 2) & (along < length / 2) & (np.abs(across) <= site['width'] + 2)
        before = mask[lo[0]:hi[0], lo[1]:hi[1]]
        result[lo[0]:hi[0], lo[1]:hi[1]][cut] = False
        coords = np.argwhere(skeleton[lo[0]:hi[0], lo[1]:hi[1]]) + lo
        projected = (coords - p) @ tangent
        perpendicular = np.abs((coords - p) @ np.array([-tangent[1], tangent[0]]))
        anchors = []
        for sign in (-1, 1):
            choose = (sign * projected >= length / 2 + 3) & (sign * projected <= length / 2 + 14) & (perpendicular <= max(4, site['width'] / 2))
            candidates = coords[choose]
            if len(candidates):
                target = p + sign * (length / 2 + 6) * tangent
                anchors.append(candidates[np.argmin(np.sum((candidates - target) ** 2, axis=1))].tolist())
        window_radius = int(np.ceil(length + 2 * site['width'] + 16))
        wl, wh = np.maximum(p - window_radius, 0), np.minimum(p + window_radius + 1, mask.shape)
        events.append(dict(**site, length=length, length_width_ratio=float(length / site['width']),
                           anchors=anchors, window=[int(wl[0]), int(wh[0]), int(wl[1]), int(wh[1])],
                           removed_pixels=int((before & cut).sum())))
    for event in events:
        for connectivity in (4, 8):
            if len(event['anchors']) != 2 or not all(result[tuple(p)] for p in event['anchors']):
                status = 'missing_anchor'
            elif not connected_at(mask, event['anchors'], connectivity, event['window']):
                status = 'before_disconnected'
            elif connected_at(result, event['anchors'], connectivity, event['window']):
                status = 'bypassed'
            else:
                status = 'valid'
            event[f'status_{connectivity}'] = status
    return result, events


def false_bridges(mask, valid, rng, maximum=3, minimum_distance=10, maximum_distance=60):
    """Connect initially separate 8-components using nearest boundary pairs.

    Distance is background separation between component pixels. Width is the
    smaller estimate at their nearest original centerline pixels, rounded.
    Candidate nearest-other pairs are searched within a maximum-distance ROI;
    accepted strokes may not cross ignored pixels or touch a third component.
    """
    original, count = ndi.label(mask, CONNECT)
    skeleton = skeletonize(mask, method='zhang')
    distance = ndi.distance_transform_edt(mask)
    _, nearest = ndi.distance_transform_edt(~skeleton, return_indices=True)
    widths = 2 * distance[tuple(nearest)]
    boundary = mask & ~ndi.binary_erosion(mask, structure=CONNECT)
    output, events = mask.copy(), []
    if count < 2:
        return output, events, 'fewer_than_two_components'
    candidates = np.argwhere(boundary & valid)
    used_pairs = set()
    for index in rng.permutation(len(candidates))[:4000]:
        p = candidates[index]
        ident = int(original[tuple(p)])
        lo = np.maximum(p - maximum_distance, 0)
        hi = np.minimum(p + maximum_distance + 1, mask.shape)
        local = original[lo[0]:hi[0], lo[1]:hi[1]]
        coords = np.argwhere((local > 0) & (local != ident)) + lo
        if not len(coords):
            continue
        squared = np.sum((coords - p) ** 2, axis=1)
        best = int(np.argmin(squared))
        gap_distance = float(np.sqrt(squared[best]))
        if not minimum_distance <= gap_distance <= maximum_distance:
            continue
        q = coords[best]
        other = int(original[tuple(q)])
        pair = tuple(sorted((ident, other)))
        if pair in used_pairs or connected_at(output, [p, q], 8):
            continue
        width = float(min(widths[tuple(p)], widths[tuple(q)]))
        stroke = draw_line(np.zeros_like(mask), [tuple(p[::-1]), tuple(q[::-1])], max(1, round(width)))
        if (stroke & ~valid).any() or not stroke[tuple(p)] or not stroke[tuple(q)]:
            continue
        touched = set(np.unique(original[ndi.binary_dilation(stroke, structure=CONNECT) & mask])) - {0}
        if touched != {ident, other}:
            continue
        output |= stroke
        events.append(dict(a=p.tolist(), b=q.tolist(), separation=gap_distance,
                           width=width, drawn_width=max(1, round(width)), components=list(pair),
                           added_pixels=int((stroke & ~mask).sum())))
        used_pairs.add(pair)
        if len(events) == maximum:
            break
    return output, events, 'ok' if events else 'no_eligible_pair'


def endpoint_connect(mask, valid, maximum_distance, angle_degrees=30):
    """Connect mutually facing endpoints with a local width line, without GT.

    Outward directions follow each endpoint's inward skeleton neighborhood.
    Require both directions to face the joining vector within angle_degrees.
    Each endpoint is used once; all pairs are chosen from the initial skeleton.
    Loops can legitimately connect endpoints in the same global component.
    """
    skeleton = skeletonize(mask, method='zhang')
    degree = ndi.convolve(skeleton.astype('uint8'), CONNECT.astype('uint8'), mode='constant') - skeleton
    endpoints = np.argwhere(skeleton & (degree == 1))
    distances = ndi.distance_transform_edt(mask)
    directions, widths = [], []
    for p in endpoints:
        lo, hi = np.maximum(p - 32, 0), np.minimum(p + 33, mask.shape)
        local_skeleton = skeleton[lo[0]:hi[0], lo[1]:hi[1]]
        reached = np.zeros_like(local_skeleton)
        reached[tuple(p - lo)] = True
        inward = np.zeros(2)
        for step in range(32):
            reached = ndi.binary_dilation(reached, structure=CONNECT) & local_skeleton
            if step == 7:
                inward = (np.argwhere(reached) + lo).mean(axis=0) - p
        norm = np.linalg.norm(inward)
        directions.append(-inward / norm if norm > 0 else np.zeros(2))
        widths.append(float(2 * distances[lo[0]:hi[0], lo[1]:hi[1]][reached].max()))
    candidates = []
    cosine = np.cos(np.deg2rad(angle_degrees))
    for i, p in enumerate(endpoints):
        for j in range(i + 1, len(endpoints)):
            q = endpoints[j]
            vector = q - p
            length = float(np.linalg.norm(vector))
            if not 0 < length <= maximum_distance:
                continue
            vector = vector / length
            if np.dot(directions[i], vector) >= cosine and np.dot(directions[j], -vector) >= cosine:
                candidates.append((length, i, j))
    output, used = mask.copy(), set()
    for _, i, j in sorted(candidates):
        if i in used or j in used:
            continue
        p, q = endpoints[i], endpoints[j]
        width = max(1, round(min(widths[i], widths[j])))
        stroke = draw_line(np.zeros_like(mask), [tuple(p[::-1]), tuple(q[::-1])], width)
        if (stroke & ~valid).any():
            continue
        output |= stroke
        used.update([i, j])
    return output & valid


def thicken(mask, radius, background_distance=None):
    if background_distance is None:
        background_distance = ndi.distance_transform_edt(~mask)
    return background_distance <= radius


def thin(mask, radius):
    return ndi.distance_transform_edt(mask) > radius
