"""Render a reviewed C-position trial using explicit temporal anchors.

This is assisted annotation, not a trained choreographic-role classifier.
Anchors choose the C role; the detector only supplies intermediate body boxes.
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c_detect.overlays import narrow_c_boxes


def align_boxes(boxes, reference):
    if len(boxes) == 2:
        straight = np.abs(boxes - reference).mean()
        crossed = np.abs(boxes[::-1] - reference).mean()
        if crossed < straight:
            return boxes[::-1], True
    return boxes, False


def prepare_references(segment):
    anchors = segment['anchors']
    assert anchors and all(len(a['targets']) == segment['count'] for a in anchors)
    boxes, ids = [], []
    for anchor in anchors:
        current = np.array([t['box'] for t in anchor['targets']], dtype=float)
        current_ids = [t['track_id'] if t['track_id'] is not None else -1 for t in anchor['targets']]
        if boxes:
            current, flipped = align_boxes(current, boxes[-1])
            if flipped:
                current_ids.reverse()
        boxes.append(current)
        ids.append(current_ids)
    samples = np.array([a['frame'] for a in anchors])
    boxes = np.array(boxes)
    frames = np.arange(segment['start_frame'], segment['end_frame'])
    reference = np.empty((len(frames), segment['count'], 4))
    for person in range(segment['count']):
        for edge in range(4):
            reference[:, person, edge] = np.interp(frames, samples, boxes[:, person, edge])
    nearest = np.abs(frames[:, None] - samples[None, :]).argmin(axis=1)
    return reference, np.array(ids)[nearest], {a['frame']: (b, i) for a, b, i in zip(anchors, boxes, ids)}


def states_for_frame(row, reference, preferred_ids, hard_anchor=None):
    count = len(reference)
    if hard_anchor is not None:
        boxes, ids = hard_anchor
        return np.array([boxes]), np.array([ids]), np.zeros(1)
    candidates = []
    for p in row['people']:
        box = np.array(p['box'])
        if p['detection_confidence'] < .25 or box[3] - box[1] < .18:
            continue
        distance = np.abs(reference - box).mean(axis=1).min()
        if distance < .18:
            candidates.append((distance, box, p['track_id']))
    candidates.sort(key=lambda p: p[0])
    candidates = candidates[:10]
    for index, box in enumerate(reference):
        candidates.append((0, box, -index - 1))
    sets, ids, unary = [], [], []
    for subset in itertools.combinations(candidates, count):
        boxes = np.array([s[1] for s in subset])
        identities = np.array([s[2] for s in subset])
        if count == 2:
            inter = np.maximum(0, np.minimum(boxes[0, 2:], boxes[1, 2:]) - np.maximum(boxes[0, :2], boxes[1, :2])).prod()
            areas = ((boxes[:, 2:] - boxes[:, :2]).prod(axis=1))
            if inter / max(1e-6, areas.sum() - inter) > .82:
                continue
        boxes, flipped = align_boxes(boxes, reference)
        if flipped:
            identities = identities[::-1]
        cost = 2 * np.abs(boxes - reference).mean()
        cost += .16 * np.mean(identities < 0)
        cost -= .08 * np.mean((identities == preferred_ids) & (identities >= 0))
        sets.append(boxes)
        ids.append(identities)
        unary.append(cost)
    return np.array(sets), np.array(ids), np.array(unary)


def transition_cost(previous, current, previous_ids, current_ids):
    direct = 8 * np.abs(previous[:, None] - current[None, :]).mean(axis=(2, 3))
    comparable = (previous_ids[:, None] >= 0) & (current_ids[None, :] >= 0)
    direct += .045 * ((previous_ids[:, None] != current_ids[None, :]) & comparable).mean(axis=2)
    if previous.shape[1] == 2:
        reverse = current[:, ::-1]
        reverse_ids = current_ids[:, ::-1]
        swapped = 8 * np.abs(previous[:, None] - reverse[None, :]).mean(axis=(2, 3))
        comparable = (previous_ids[:, None] >= 0) & (reverse_ids[None, :] >= 0)
        swapped += .045 * ((previous_ids[:, None] != reverse_ids[None, :]) & comparable).mean(axis=2)
        direct = np.minimum(direct, swapped)
    return direct


def track_segment(segment, rows):
    reference, preferred, hard = prepare_references(segment)
    if segment.get('manual_interpolation'):
        return [(b, [-1] * segment['count'], 'reviewed_box_interpolation') for b in reference]
    history, pointers = [], []
    costs = None
    for offset, frame in enumerate(range(segment['start_frame'], segment['end_frame'])):
        boxes, ids, unary = states_for_frame(rows[frame], reference[offset], preferred[offset], hard.get(frame))
        if costs is None:
            costs = unary
            pointers.append(None)
        else:
            old_boxes, old_ids = history[-1]
            matrix = costs[:, None] + transition_cost(old_boxes, boxes, old_ids, ids)
            best = matrix.argmin(axis=0)
            costs = matrix[best, np.arange(len(boxes))] + unary
            pointers.append(best)
        history.append((boxes, ids))
    index = int(costs.argmin())
    selected = []
    for offset in range(len(history) - 1, -1, -1):
        boxes, ids = history[offset]
        method = 'detection_tracking' if np.all(ids[index] >= 0) else 'anchor_assisted_interpolation'
        selected.append((boxes[index], ids[index].tolist(), method))
        if pointers[offset] is not None:
            index = int(pointers[offset][index])
    selected.reverse()
    # Align paired rectangles before light smoothing. Never combine them.
    for index in range(1, len(selected)):
        boxes, ids, method = selected[index]
        boxes, flipped = align_boxes(boxes, selected[index - 1][0])
        selected[index] = (boxes, ids[::-1] if flipped else ids, method)
    raw = np.array([p[0] for p in selected])
    for index, (_, ids, method) in enumerate(selected):
        smooth = np.median(raw[max(0, index - 2):min(len(raw), index + 3)], axis=0)
        selected[index] = (smooth, ids, method)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=Path)
    parser.add_argument('--ffmpeg', default='/path/to/user/.local/bin/ffmpeg')
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    root = args.spec.parent
    rows = [json.loads(line) for line in Path(spec['cache']).read_text().splitlines()]
    assert len(rows) == spec['frames']
    annotation = []
    for index, segment in enumerate(spec['segments']):
        assert segment['start_frame'] == len(annotation)
        annotation.extend(track_segment(segment, rows))
        print(f'Reviewed segment {index + 1}/{len(spec["segments"])}: {segment["start_frame"] / spec["fps"]:.2f}–{segment["end_frame"] / spec["fps"]:.2f}s, C={segment["count"]}', flush=True)
    assert len(annotation) == len(rows)
    assert all(len(boxes) in (1, 2) for boxes, _, _ in annotation)
    with (root / 'reviewed_frames.jsonl').open('w') as stream:
        for index, (boxes, ids, method) in enumerate(annotation):
            stream.write(json.dumps({'frame': index, 'seconds': index / spec['fps'], 'c_count': len(boxes), 'boxes': boxes.tolist(), 'detection_ids': ids, 'method': method}) + '\n')
    cap = cv2.VideoCapture(spec['source'])
    if not cap.isOpened():
        raise RuntimeError('Cannot open source video')
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(annotation)
    output = root / 'c-main-red.mp4'
    cmd = [args.ffmpeg, '-hide_banner', '-loglevel', 'warning', '-n', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(spec['fps']), '-i', 'pipe:0', '-i', spec['source'], '-map', '0:v:0', '-map', '1:a?', '-c:v', 'libx264', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart', '-shortest', str(output)]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for index, (boxes, ids, method) in enumerate(annotation):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Source ended at {index}')
            torso_by_id = {p['track_id']: p['torso_x'] for p in rows[index]['people']}
            display_boxes = narrow_c_boxes(boxes, width, height,
                                           [torso_by_id.get(i) for i in ids],
                                           manual_pair=method == 'reviewed_box_interpolation')
            for x1, y1, x2, y2 in display_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            if index in [0, 601, 900, 1860, 3420, 4200, 4530, 4800, 6000, 6510]:
                cv2.imwrite(str(root / f'preview-{index:05}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            process.stdin.write(frame.tobytes())
            if index % 900 == 0:
                print(f'Rendered {index}/{len(annotation)} frames', flush=True)
    finally:
        cap.release()
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg export failed')
    print(output.resolve(), flush=True)


if __name__ == '__main__':
    main()
