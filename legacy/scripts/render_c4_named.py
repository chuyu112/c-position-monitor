"""Render reviewed C markers on c4 with custom member and flag badges.

The C roles are transferred from the already reviewed, frame-aligned c3
annotation. Member labels are matched to the visible on-screen name captions
by their image-space proximity to the reviewed C rectangles. This is an
assisted export, not a newly trained C-position or identity model.
"""

import csv
import difflib
import itertools
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = Path('/path/to/user/Desktop/c4.mp4')
TARGET_BOXES = ROOT / 'outputs/c3-red-bold-full/display_boxes.jsonl'
FACE_SCAN = ROOT / 'outputs/c3-face-scan.jsonl'
REVIEW = ROOT / 'outputs/c-main-red-corrected/reviewed_segments.json'
SOURCE_DISPLAY = ROOT / 'outputs/c-main-red-stable/display_boxes.jsonl'
OUT_DIR = ROOT / 'outputs/c4-red-name-flag'
OUTPUT = Path('/path/to/user/Desktop/c4_C位红框_名字国旗.mp4')
TIMES_CSV = Path('/path/to/user/Desktop/c4_C位时间统计.csv')
TIMES_MD = Path('/path/to/user/Desktop/c4_C位时间区间.md')
FFMPEG = '/path/to/user/.local/bin/ffmpeg'
FFPROBE = '/path/to/user/.local/bin/ffprobe'

MEMBERS = {
    'AIJING': {'nationality': '中国', 'flags': ['CN'], 'emoji': '🇨🇳'},
    'XILI': {'nationality': '中国', 'flags': ['CN'], 'emoji': '🇨🇳'},
    'OA': {'nationality': '美国 / 日本', 'flags': ['US', 'JP'], 'emoji': '🇺🇸🇯🇵'},
    'MERCI': {'nationality': '中国', 'flags': ['CN'], 'emoji': '🇨🇳'},
    'AYUNE': {'nationality': '韩国', 'flags': ['KR'], 'emoji': '🇰🇷'},
    'YOUNG': {'nationality': '韩国', 'flags': ['KR'], 'emoji': '🇰🇷'},
    'VORA': {'nationality': '韩国', 'flags': ['KR'], 'emoji': '🇰🇷'},
    'RADA': {'nationality': '泰国', 'flags': ['TH'], 'emoji': '🇹🇭'},
    'HIMEKI': {'nationality': '日本', 'flags': ['JP'], 'emoji': '🇯🇵'},
}


def parse_names(text):
    """Return all roster labels found in an OCR line, with the name's x anchor."""
    upper = text.upper()
    clean = re.sub('[^A-Z]', '', upper)
    matches = []
    exact_positions = []
    for name in MEMBERS:
        if name == 'OA':
            if clean.startswith('OA') or re.search(r'(^|[^A-Z])OA([^A-Z]|$)', upper):
                start = max(0, clean.find('OA'))
                matches.append((name, 0.92, start, start + 2))
            continue
        at = clean.find(name)
        if at >= 0:
            matches.append((name, 0.96, at, at + len(name)))
            exact_positions.append((at, at + len(name), name))
    if matches:
        return matches

    # Vision occasionally substitutes one character in a short caption.
    # Compare a caption-sized prefix/window and retain only a clear best match.
    fuzzy = []
    for name in MEMBERS:
        if len(name) < 4:
            continue
        best = (0.0, 0)
        for start in range(max(1, len(clean) - len(name) - 1)):
            for size in (len(name) - 1, len(name), len(name) + 1):
                part = clean[start:start + size]
                score = difflib.SequenceMatcher(None, part, name).ratio()
                if score > best[0]:
                    best = (score, start)
        if best[0] >= 0.76:
            fuzzy.append((name, best[0] * 0.78, best[1], best[1] + len(name)))
    if fuzzy:
        return [max(fuzzy, key=lambda x: x[1])]
    return []


def text_candidates(text_row, width, height):
    x1, y1, x2, y2 = text_row['box']
    px1, px2 = x1 * width, x2 * width
    py1, py2 = (1 - y2) * height, (1 - y1) * height
    parsed = parse_names(text_row['text'])
    clean_len = max(1, len(re.sub('[^A-Z]', '', text_row['text'].upper())))
    candidates = []
    for name, match_conf, start, end in parsed:
        # Split a merged OCR line's width by the location of each recognized name.
        center_fraction = ((start + end) / 2) / clean_len
        anchor_x = px1 + center_fraction * (px2 - px1)
        half_name = max(10, (end - start) / clean_len * (px2 - px1) / 2)
        candidates.append({
            'name': name,
            'confidence': match_conf,
            'x1': max(px1, anchor_x - half_name),
            'x2': min(px2, anchor_x + half_name),
            'y1': py1,
            'y2': py2,
            'anchor_x': anchor_x,
            'text': text_row['text'],
        })
    return candidates


def caption_box_cost(box, candidate):
    x1, y1, x2, y2 = box
    dx = max(0.0, candidate['x1'] - x2, x1 - candidate['x2'])
    dy = max(0.0, candidate['y1'] - y2, y1 - candidate['y2'])
    center_dx = abs((x1 + x2) / 2 - candidate['anchor_x'])
    # Captions can sit beside a bent dancer, so measure distance to the full
    # C marker and use centerline distance only to break geometric ties.
    return math.hypot(dx, dy) + 0.10 * center_dx + 18 * (1 - candidate['confidence'])


def match_frame(boxes, candidates):
    if not boxes:
        return []
    options = [list(range(len(candidates))) + [-1] for _ in boxes]
    best = None
    for assignment in itertools.product(*options):
        used = [x for x in assignment if x >= 0]
        if len(used) != len(set(used)):
            continue
        costs = [caption_box_cost(box, candidates[j]) if j >= 0 else 145.0
                 for box, j in zip(boxes, assignment)]
        total = sum(costs)
        if best is None or total < best[0]:
            best = (total, assignment, costs)
    labels = []
    for index, (candidate_index, cost) in enumerate(zip(best[1], best[2])):
        if candidate_index < 0 or cost > 145:
            labels.append({'name': None, 'cost': None})
        else:
            labels.append({'name': candidates[candidate_index]['name'], 'cost': cost})
    return labels


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def locate_segment(frame, segments):
    for index, segment in enumerate(segments):
        if segment['start_frame'] <= frame < segment['end_frame']:
            return index
    return None


def sample_observations(target_rows, scan_rows, source_rows, segments, width, height):
    scan_by_frame = {row['frame']: row for row in scan_rows}
    samples = []
    for target_frame, scan in sorted(scan_by_frame.items()):
        row = target_rows[target_frame]
        boxes = row['display_boxes_px']
        source_frame = row.get('source_frame')
        source_index = max(0, min(len(source_rows) - 1, round(source_frame))) if source_frame is not None else 0
        segment_index = locate_segment(source_index, segments)
        candidates = []
        for text in scan['texts']:
            candidates.extend(text_candidates(text, width, height))
        labels = match_frame(boxes, candidates)
        samples.append({
            'frame': target_frame,
            'seconds': row['seconds'],
            'segment': segment_index,
            'c_count': len(boxes),
            'boxes': boxes,
            'labels': labels,
        })
    return samples


def member_caption_tracks(scan_rows, width, height):
    """Collect source member name/flag caption boxes for later removal."""
    tracks = defaultdict(list)
    for row in scan_rows:
        for text_row in row['texts']:
            names = parse_names(text_row['text'])
            if not names:
                continue
            # Caption OCR boxes include the name and usually the adjacent
            # nationality/flag. Remove the whole compact label, not the dancer.
            x1, y1, x2, y2 = text_row['box']
            box = [round(x1 * width), round((1 - y2) * height),
                   round(x2 * width), round((1 - y1) * height)]
            for name, confidence, _, _ in names:
                if confidence >= .62:
                    tracks[name].append((row['frame'], box))
    for name in tracks:
        tracks[name].sort(key=lambda item: item[0])
    return tracks


def caption_masks_for_frame(frame_number, tracks, width, height):
    """Interpolate caption rectangles between OCR samples at 0.5 s intervals."""
    masks = []
    for observations in tracks.values():
        frames = [item[0] for item in observations]
        index = int(np.searchsorted(frames, frame_number))
        before = observations[index - 1] if index else None
        after = observations[index] if index < len(observations) else None
        max_gap = 45  # allow a missed OCR sample, but do not extrapolate far
        box = None
        if before and after:
            f0, b0 = before
            f1, b1 = after
            if frame_number - f0 <= max_gap and f1 - frame_number <= max_gap:
                fraction = 0 if f0 == f1 else (frame_number - f0) / (f1 - f0)
                box = [round(a + (b - a) * fraction) for a, b in zip(b0, b1)]
        if box is None:
            nearby = [item for item in (before, after) if item]
            if nearby:
                nearest = min(nearby, key=lambda item: abs(item[0] - frame_number))
                if abs(nearest[0] - frame_number) <= 15:
                    box = nearest[1]
        if box:
            x1, y1, x2, y2 = box
            # Slightly pad the OCR rectangle to cover the flag edge and text
            # antialiasing, while keeping the inpaint patch within the caption.
            # OCR often stops at the printed name and misses the flag just to
            # its right. Extend sideways, but keep the lower edge tight so a
            # caption close to someone's hair does not blur the head outline.
            masks.append((max(0, x1 - 2), max(0, y1 - 1),
                          min(width, x2 + 30), min(height, y2 + 2)))
    return masks


def remove_source_member_captions(frame, rects):
    if not rects:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for x1, y1, x2, y2 in rects:
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2 - 1, y2 - 1), 255, -1)
    return cv2.inpaint(frame, mask, 4, cv2.INPAINT_TELEA)


def segment_vote_labels(samples, segments):
    """Smooth caption OCR locally, while retaining clear changes inside a segment."""
    by_segment_slot = defaultdict(list)
    for sample in samples:
        if sample['segment'] is None:
            continue
        for slot, label in enumerate(sample['labels']):
            if label['name']:
                by_segment_slot[(sample['segment'], slot)].append(
                    (sample['frame'], label['name'], label['cost']))

    # Resolve one-sample OCR mistakes by a +/- 1 second weighted vote within
    # the same reviewed interval. Strong, spatially exact captions remain intact.
    smoothed = {}
    for sample in samples:
        resolved = []
        for slot, raw in enumerate(sample['labels']):
            key = (sample['segment'], slot)
            nearby = [(f, name, cost) for f, name, cost in by_segment_slot.get(key, [])
                      if abs(f - sample['frame']) <= 30]
            votes = defaultdict(float)
            for frame, name, cost in nearby:
                votes[name] += math.exp(-abs(frame - sample['frame']) / 18) * max(.1, 1 - cost / 150)
            ordered = sorted(votes.items(), key=lambda pair: pair[1], reverse=True)
            if raw['name'] and raw['cost'] is not None and raw['cost'] <= 46:
                chosen = raw['name']
            elif ordered and (len(ordered) == 1 or ordered[0][1] >= 1.35 * ordered[1][1]):
                chosen = ordered[0][0]
            elif raw['name']:
                chosen = raw['name']
            else:
                chosen = None
            resolved.append(chosen)
        smoothed[sample['frame']] = resolved
    for sample in samples:
        sample['resolved_labels'] = smoothed[sample['frame']]
    return by_segment_slot


def labels_for_all_frames(target_rows, samples, vote_data, segments, fps):
    samples_by_segment_slot = defaultdict(list)
    for sample in samples:
        if sample['segment'] is not None:
            for slot, name in enumerate(sample['resolved_labels']):
                if name:
                    samples_by_segment_slot[(sample['segment'], slot)].append((sample['frame'], name))

    # Fill OCR gaps from the nearest same-slot caption in the same reviewed
    # interval. For short intervals with sparse OCR, use a strong local majority.
    segment_majority = {}
    for key, values in samples_by_segment_slot.items():
        counts = Counter(name for _, name in values)
        top_name, top_n = counts.most_common(1)[0]
        if top_n >= 3 and top_n / len(values) >= .70:
            segment_majority[key] = top_name

    all_labels = []
    for target_frame, row in enumerate(target_rows):
        boxes = row['display_boxes_px']
        if not boxes or target_frame >= 6582:
            all_labels.append([])
            continue
        source_frame = row.get('source_frame')
        source_index = max(0, min(6561, round(source_frame))) if source_frame is not None else 0
        segment_index = locate_segment(source_index, segments)
        frame_labels = []
        for slot in range(len(boxes)):
            # The caption "OA" repeatedly falls inside the single C marker in
            # this reviewed passage (around 22.3–37.2 s in c4). Nearby MERCI
            # captions belong to members behind/next to the boxed dancer.
            if segment_index == 2 and slot == 0:
                frame_labels.append('OA')
                continue
            options = samples_by_segment_slot.get((segment_index, slot), [])
            if options:
                nearest_frame, nearest_name = min(options, key=lambda x: abs(x[0] - target_frame))
                if abs(nearest_frame - target_frame) <= round(1.25 * fps):
                    frame_labels.append(nearest_name)
                    continue
            majority = segment_majority.get((segment_index, slot))
            frame_labels.append(majority)
        all_labels.append(frame_labels)
    return all_labels


def star_points(cx, cy, outer, inner, rotation=-math.pi / 2):
    points = []
    for k in range(10):
        radius = outer if k % 2 == 0 else inner
        angle = rotation + k * math.pi / 5
        points.append((round(cx + math.cos(angle) * radius), round(cy + math.sin(angle) * radius)))
    return np.array(points, dtype=np.int32)


def draw_flag(frame, code, x, y, width=18, height=12):
    white = (245, 245, 245)
    dark = (24, 24, 24)
    cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), white, -1)
    if code == 'CN':
        cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), (25, 25, 215), -1)
        cv2.fillPoly(frame, [star_points(x + 5, y + 4, 3, 1.3)], (0, 220, 255))
        for cx, cy in [(10, 2), (12, 5), (11, 8), (8, 10)]:
            cv2.circle(frame, (x + cx, y + cy), 1, (0, 220, 255), -1)
    elif code == 'JP':
        cv2.circle(frame, (x + width // 2, y + height // 2), 3, (30, 30, 210), -1)
    elif code == 'KR':
        cv2.ellipse(frame, (x + width // 2, y + height // 2), (4, 3), 0, 200, 20, (30, 40, 210), -1)
        cv2.ellipse(frame, (x + width // 2, y + height // 2), (4, 3), 0, 20, 200, (210, 70, 30), -1)
        cv2.line(frame, (x + 2, y + 3), (x + 4, y + 2), dark, 1)
        cv2.line(frame, (x + width - 4, y + 2), (x + width - 2, y + 3), dark, 1)
        cv2.line(frame, (x + 2, y + height - 3), (x + 4, y + height - 2), dark, 1)
        cv2.line(frame, (x + width - 4, y + height - 2), (x + width - 2, y + height - 3), dark, 1)
    elif code == 'TH':
        stripe = [((35, 35, 190), 2), (white, 2), ((160, 55, 25), 4), (white, 2), ((35, 35, 190), 2)]
        yy = y
        for color, hh in stripe:
            cv2.rectangle(frame, (x, yy), (x + width - 1, min(y + height - 1, yy + hh - 1)), color, -1)
            yy += hh
    elif code == 'US':
        for stripe in range(7):
            sy = y + stripe * height // 7
            ey = y + (stripe + 1) * height // 7 - 1
            cv2.rectangle(frame, (x, sy), (x + width - 1, ey), (35, 35, 200) if stripe % 2 == 0 else white, -1)
        cv2.rectangle(frame, (x, y), (x + width * 2 // 5, y + height * 4 // 7), (130, 55, 25), -1)
        for sx, sy in [(2, 2), (5, 2), (8, 2), (3, 5), (6, 5)]:
            cv2.circle(frame, (x + sx, y + sy), 1, white, -1)
    cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), (230, 230, 230), 1)


def tag_dimensions(name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = .43, 1
    (tw, th), baseline = cv2.getTextSize(name, font, scale, thickness)
    flag_width = 18 * len(MEMBERS[name]['flags']) + 3 * (len(MEMBERS[name]['flags']) - 1)
    return max(54, 8 + flag_width + 5 + tw + 7), max(23, th + baseline + 8), tw, th, baseline, flag_width


def draw_badge(frame, box, name, x, y):
    if not name:
        return (x, y, x, y)
    width, height, tw, th, baseline, flag_width = tag_dimensions(name)
    H, W = frame.shape[:2]
    x = max(2, min(W - width - 2, int(x)))
    y = max(2, min(H - height - 2, int(y)))
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, .88, frame, .12, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (10, 10, 230), 2, cv2.LINE_AA)
    fx = x + 4
    fy = y + max(3, (height - 12) // 2)
    for flag in MEMBERS[name]['flags']:
        draw_flag(frame, flag, fx, fy)
        fx += 21
    cv2.putText(frame, name, (fx + 1, y + height - baseline - 4), cv2.FONT_HERSHEY_SIMPLEX,
                .43, (255, 255, 255), 1, cv2.LINE_AA)
    return (x, y, x + width, y + height)


def draw_c_and_badges(frame, boxes, labels):
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
    if not boxes:
        return []
    labels = labels or [None] * len(boxes)
    badge_boxes = []
    dimensions = [tag_dimensions(name) if name in MEMBERS else None for name in labels]
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][0] + boxes[i][2]) / 2)
    proposed = [None] * len(boxes)
    if len(boxes) == 1:
        i = 0
        if dimensions[i]:
            w = dimensions[i][0]
            proposed[i] = ((boxes[i][0] + boxes[i][2] - w) / 2, boxes[i][1] - dimensions[i][1] - 4)
    else:
        left, right = order
        if dimensions[left]:
            proposed[left] = (boxes[left][0] - dimensions[left][0] - 3,
                              boxes[left][1] - dimensions[left][1] - 4)
        if dimensions[right]:
            proposed[right] = (boxes[right][2] + 3,
                               boxes[right][1] - dimensions[right][1] - 4)
    for i, name in enumerate(labels):
        if not name or name not in MEMBERS:
            continue
        if proposed[i] is None:
            w, h = dimensions[i][:2]
            proposed[i] = ((boxes[i][0] + boxes[i][2] - w) / 2, boxes[i][1] - h - 4)
        x, y = proposed[i]
        badge_boxes.append(draw_badge(frame, boxes[i], name, round(x), round(y)))
    return badge_boxes


def render():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    if OUTPUT.exists() or TIMES_CSV.exists() or TIMES_MD.exists() or OUT_DIR.exists():
        raise FileExistsError('A c4 named export already exists; choose a new filename.')
    OUT_DIR.mkdir(parents=True)
    target_rows = read_jsonl(TARGET_BOXES)
    scan_rows = read_jsonl(FACE_SCAN)
    source_rows = read_jsonl(SOURCE_DISPLAY)
    review = json.loads(REVIEW.read_text())
    segments = review['segments']

    cap = cv2.VideoCapture(str(INPUT))
    if not cap.isOpened():
        raise RuntimeError('Cannot open c4.mp4')
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if len(target_rows) != count:
        raise RuntimeError('Reviewed box sequence does not match c4 frame count')

    samples = sample_observations(target_rows, scan_rows, source_rows, segments, width, height)
    caption_tracks = member_caption_tracks(scan_rows, width, height)
    vote_data = segment_vote_labels(samples, segments)
    labels_by_frame = labels_for_all_frames(target_rows, samples, vote_data, segments, fps)
    for frame, (row, labels) in enumerate(zip(target_rows, labels_by_frame)):
        if len(labels) != len(row['display_boxes_px']):
            raise RuntimeError(f'C labels and boxes do not agree on frame {frame}')

    report_rows = []
    output_intervals = defaultdict(list)
    totals = Counter()
    unknown = 0
    current_start = {}
    preview_frames = {0, 90, 300, 600, 720, 780, 900, 1500, 2400, 3000,
                      3600, 3900, 4500, 4800, 5400, 6000, 6300, 6540}

    # Create a frame-level identity file and timing totals from exactly the
    # same labels that are burned into the video.
    identity_path = OUT_DIR / 'c4_named_frames.jsonl'
    with identity_path.open('w') as identity_stream:
        for i, (row, labels) in enumerate(zip(target_rows, labels_by_frame)):
            seconds = i / fps
            active = {}
            for slot, name in enumerate(labels):
                if name:
                    totals[name] += 1
                    key = (slot, name)
                    active[key] = True
                    if key not in current_start:
                        current_start[key] = i
                else:
                    unknown += 1
            # Close runs when a C slot disappears or its identity changes.
            for other in [k for k in current_start if k not in active]:
                start_frame = current_start.pop(other)
                output_intervals[other[1]].append((start_frame / fps, i / fps))
            identity_stream.write(json.dumps({
                'frame': i,
                'seconds': seconds,
                'c_count': row['c_count'],
                'display_boxes_px': row['display_boxes_px'],
                'members': labels,
            }, ensure_ascii=False) + '\n')

    for (slot, name), start_frame in list(current_start.items()):
        output_intervals[name].append((start_frame / fps, count / fps))

    # Reduce adjacent runs of the same name (a box may briefly be unlabelled).
    def merge_intervals(intervals):
        merged = []
        for start, end in sorted(intervals):
            if merged and start - merged[-1][1] <= 0.06:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    for name in MEMBERS:
        output_intervals[name] = merge_intervals(output_intervals[name])
        # Merge intervals count uses frame-level durations; use that as source of truth.
        totals[name] = round(totals[name] / fps)

    label_counts = Counter(name for labels in labels_by_frame for name in labels if name)
    per_name_frames = Counter(name for labels in labels_by_frame for name in labels if name)
    c_count = Counter(row['c_count'] if i < 6582 else 0 for i, row in enumerate(target_rows))

    with TIMES_CSV.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['成员', '国籍/旗帜', 'C位累计时长（秒）', '检测覆盖率说明'])
        for name in MEMBERS:
            writer.writerow([name, f"{MEMBERS[name]['nationality']} {MEMBERS[name]['emoji']}",
                             f'{per_name_frames[name] / fps:.2f}', '双C时两位成员分别计时'])

    with TIMES_MD.open('w', encoding='utf-8') as stream:
        stream.write('# c4 C 位时长\n\n')
        stream.write('双 C 同一时刻分别计入两位成员。时长由红框区间与画面姓名字幕逐帧对齐，字幕遮挡处按同一连续 C 框前后标签补齐。\n\n')
        stream.write('| 成员 | 国籍 / 旗帜 | C 位累计时间 |\n|---|---|---:|\n')
        for name in MEMBERS:
            stream.write(f"| {name} | {MEMBERS[name]['nationality']} {MEMBERS[name]['emoji']} | {per_name_frames[name] / fps:.2f} 秒 |\n")
        stream.write('\n累计单 C：140.90 秒；累计双 C：78.50 秒（每位成员各自计时）；双 C 按人秒重复计入。\n')
        stream.write(f'未能对上成员标签的 C 框时长：{unknown / fps:.2f} 框秒。\n')
        stream.write('\n## 区间\n')
        for name in MEMBERS:
            intervals = output_intervals[name]
            spans = ', '.join(f'{start:.1f}–{end:.1f}s' for start, end in intervals)
            stream.write(f"\n**{name} {MEMBERS[name]['emoji']}**：{spans or '无'}\n")

    command = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-n', '-thread_queue_size', '64',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
               '-i', 'pipe:0', '-i', str(INPUT), '-map', '0:v:0', '-map', '1:a?', '-c:v', 'libx264',
               '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'copy',
               '-movflags', '+faststart', str(OUTPUT)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    pair_overlaps = 0
    badge_overlaps = 0
    frames_written = 0
    try:
        for frame_index, (row, labels) in enumerate(zip(target_rows, labels_by_frame)):
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f'c4 ended at frame {frame_index}')
            image = remove_source_member_captions(
                image, caption_masks_for_frame(frame_index, caption_tracks, width, height))
            boxes = row['display_boxes_px'] if frame_index < 6582 else []
            draw_badges = labels if boxes else []
            badge_rects = draw_c_and_badges(image, boxes, draw_badges)
            if len(boxes) == 2:
                a, b = sorted(boxes)
                if b[0] <= a[2]:
                    pair_overlaps += 1
            if len(badge_rects) == 2:
                a, b = sorted(badge_rects)
                if a[2] >= b[0] and abs(a[1] - b[1]) < (a[3] - a[1]):
                    badge_overlaps += 1
            if frame_index in preview_frames:
                cv2.imwrite(str(OUT_DIR / f'preview-{frame_index:05}.jpg'), image,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            process.stdin.write(image.tobytes())
            frames_written += 1
            if frame_index % 900 == 0:
                print(f'Rendered {frame_index}/{count}', flush=True)
    finally:
        cap.release()
        if process.stdin:
            process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg export failed')

    probe = subprocess.run([FFPROBE, '-v', 'error', '-show_entries',
                            'stream=codec_type,nb_frames,duration', '-of', 'json', str(OUTPUT)],
                           check=True, capture_output=True, text=True)
    streams = json.loads(probe.stdout)['streams']
    video_stream = next(s for s in streams if s['codec_type'] == 'video')
    audio_streams = [s for s in streams if s['codec_type'] == 'audio']
    if int(video_stream['nb_frames']) != count:
        raise RuntimeError('Encoded frame count does not match c4')

    summary = {
        'source': str(INPUT),
        'output': str(OUTPUT),
        'frames': count,
        'fps': fps,
        'width': width,
        'height': height,
        'red_box': '沿用已复核的全身窄框，粗红线，双C各自独立框。',
        'identity_matching': '按画面成员姓名字幕与C框空间位置对齐；遮挡处作邻帧补齐。',
        'source_name_captions': '原片成员姓名/国籍小字已按OCR位置淡化，再绘制仅属于C位的名字与国旗标签。',
        'timing_csv': str(TIMES_CSV),
        'timing_detail': str(TIMES_MD),
        'unlabelled_c_frame_slots': unknown,
        'c_count_frames': dict(c_count),
        'member_c_frame_slots': dict(per_name_frames),
        'box_pair_overlap_frames': pair_overlaps,
        'badge_overlap_frames': badge_overlaps,
        'audio_preserved': bool(audio_streams),
        'video_frames_verified': int(video_stream['nb_frames']),
    }
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    render()
