"""Same-class weighted box fusion and geometry-aware tile deduplication."""
from dataclasses import replace
import numpy as np
from .boxes import Row, box_iou


def weighted_fusion(views, weights=None, iou=0.6):
    """Use each view once per group; preserve the largest member score."""
    if not 0 < iou <= 1:
        raise ValueError('IoU must be in (0, 1]')
    weights = [1.0] * len(views) if weights is None else weights
    if len(weights) != len(views) or any(not np.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('Each view needs one finite positive weight')
    candidates = sorted(((r, i) for i, rows in enumerate(views) for r in rows),
                        key=lambda item: -item[0].confidence)
    groups = []
    for row, view in candidates:
        eligible = [(float(box_iou(row.xyxy, np.array([g['row'].xyxy]))[0]), g)
                    for g in groups if row.class_id == g['row'].class_id]
        overlap, group = max(eligible, key=lambda item: item[0], default=(0, None))
        if group is None or overlap < iou:
            groups.append(dict(row=row, members=[(row, view)], views={view}))
        elif view not in group['views']:
            group['members'].append((row, view))
            group['views'].add(view)
            members = group['members']
            votes = [max(r.confidence, 1e-12) ** 3 * weights[v] for r, v in members]
            box = np.average([r.xyxy for r, _ in members], axis=0, weights=votes)
            x1, y1, x2, y2 = box
            group['row'] = replace(group['row'], x=float((x1+x2)/2), y=float((y1+y2)/2),
                                   w=float(x2-x1), h=float(y2-y1),
                                   confidence=max(r.confidence for r, _ in members))
    return sorted((g['row'] for g in groups), key=lambda row: -row.confidence)


def merge_tiles(base, tiles, iou=0.55, ios=0.85, limit=100):
    """Remove same-class duplicates using overlap, containment and geometry."""
    if not 0 < iou <= 1 or not 0 < ios <= 1 or limit < 1:
        raise ValueError('Invalid tile matching limits')
    final = list(base[:limit])
    for tile in sorted(tiles, key=lambda row: -row.confidence):
        duplicate = False
        for row in final:
            if row.class_id != tile.class_id:
                continue
            a, b = row.xyxy, tile.xyxy
            size_a, size_b = a[2:] - a[:2], b[2:] - b[:2]
            intersection = np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
            areas = np.array([np.prod(size_a), np.prod(size_b)])
            ratio = areas.min() / areas.max()
            axis = np.min(np.minimum(size_a, size_b) / np.maximum(size_a, size_b))
            distance = np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2) / min(np.linalg.norm(size_a), np.linalg.norm(size_b))
            if ((intersection / (areas.sum()-intersection) >= iou or intersection / areas.min() >= ios)
                    and ratio + 1e-12 >= 0.4 and axis + 1e-12 >= 0.4 and distance <= 0.35 + 1e-12):
                duplicate = True
                break
        if not duplicate and len(final) < limit:
            final.append(tile)
    return sorted(final, key=lambda row: -row.confidence)


def tile_windows(width, height, tile_size=(1100, 620), overlap=0.25):
    """Cover arbitrary image dimensions, including the final right/bottom edges."""
    if min(width, height, *tile_size) < 1 or not 0 <= overlap < 1:
        raise ValueError('Invalid image/tile dimensions or overlap')
    tw, th = min(width, tile_size[0]), min(height, tile_size[1])
    def starts(length, size):
        positions = list(range(0, length-size+1, max(1, round(size*(1-overlap)))))
        return sorted(set(positions + [length-size]))
    return [(x, y, x+tw, y+th) for y in starts(height, th) for x in starts(width, tw)]
