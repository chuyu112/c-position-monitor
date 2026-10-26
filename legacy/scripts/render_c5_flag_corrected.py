"""Re-render c5 with flags beside C names and posture-aware C boxes.

The box refinement is intentionally limited to the OA/AIJING two-C passage
shown by the user. It uses visible body keypoints to widen boxes that cut
through a dancer during turns or bends, then smooths the result over time.
"""

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
INPUT = Path('/path/to/user/Desktop/c5.mp4')
ANNOTATIONS = ROOT / 'outputs/c4-red-name-flag/c4_named_frames.jsonl'
OUTPUT = Path('/path/to/user/Desktop/c5_C位红框_名字国旗_修正版.mp4')
TIMES_CSV = Path('/path/to/user/Desktop/c5_C位时间统计_修正版.csv')
TIMES_MD = Path('/path/to/user/Desktop/c5_C位时间区间_修正版.md')
OUT_DIR = ROOT / 'outputs/c5-red-name-flag-corrected'
POSE_MODEL = ROOT / 'yolo11s-pose.pt'
FFMPEG = '/path/to/user/.local/bin/ffmpeg'
FFPROBE = '/path/to/user/.local/bin/ffprobe'

OLD_WIDTH, OLD_HEIGHT = 544, 720
FRAME_OFFSET = 9
RESIZE_SCALE = 720 / 544
VERTICAL_CROP = 128
TOP_MARGIN = 34
BOTTOM_MARGIN = 28
SIDE_MARGIN = 4
FLAG_CODES = {
    'AIJING': ['CN'], 'XILI': ['CN'], 'OA': ['US', 'JP'], 'MERCI': ['CN'],
    'AYUNE': ['KR'], 'YOUNG': ['KR'], 'VORA': ['KR'], 'RADA': ['TH'],
    'HIMEKI': ['JP'],
}
MEMBERS = list(FLAG_CODES)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def map_frame(frame_index, old_rows):
    old_index = max(0, min(len(old_rows) - 1, frame_index - FRAME_OFFSET))
    row = old_rows[old_index]
    if old_index >= 6582:
        return [], []
    boxes = []
    for x1, y1, x2, y2 in row['display_boxes_px']:
        x1 = round(x1 * RESIZE_SCALE)
        x2 = round(x2 * RESIZE_SCALE)
        y1 = round((y1 + VERTICAL_CROP) * RESIZE_SCALE)
        y2 = round((y2 + VERTICAL_CROP) * RESIZE_SCALE)
        x1 = max(SIDE_MARGIN, min(720 - SIDE_MARGIN - 1, x1))
        x2 = max(x1 + 1, min(720 - SIDE_MARGIN, x2))
        y1 = max(TOP_MARGIN, min(1280 - BOTTOM_MARGIN - 2, y1))
        y2 = max(y1 + 2, min(1280 - BOTTOM_MARGIN, y2))
        boxes.append([x1, y1, x2, y2])
    return boxes, list(row['members'])


def star_polygon(cx, cy, radius, rotation=-np.pi / 2):
    points = []
    for i in range(10):
        r = radius if i % 2 == 0 else radius * .43
        angle = rotation + i * np.pi / 5
        points.append((round(cx + np.cos(angle) * r), round(cy + np.sin(angle) * r)))
    return np.array(points, dtype=np.int32)


def draw_flag(frame, code, x, y, width=22, height=16):
    white = (245, 245, 245)
    dark = (24, 24, 24)
    if code == 'CN':
        cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), (18, 18, 220), -1)
        yellow = (0, 220, 255)
        cv2.fillPoly(frame, [star_polygon(x + 6, y + 5, 3.5)], yellow)
        for cx, cy, angle in [(13, 3, -0.5), (16, 5, 0.0), (16, 9, 0.45), (12, 12, 0.8)]:
            cv2.fillPoly(frame, [star_polygon(x + cx, y + cy, 1.6, angle)], yellow)
    elif code == 'JP':
        cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), white, -1)
        cv2.circle(frame, (x + width // 2, y + height // 2), 4, (35, 35, 210), -1)
    elif code == 'KR':
        cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), white, -1)
        center = (x + width // 2, y + height // 2)
        cv2.ellipse(frame, center, (5, 4), 0, 200, 20, (35, 40, 210), -1)
        cv2.ellipse(frame, center, (5, 4), 0, 20, 200, (210, 70, 35), -1)
        for dx, dy, flip in [(4, 3, 0), (width - 7, 3, 1), (4, height - 5, 1), (width - 7, height - 5, 0)]:
            for j in range(2 if flip else 3):
                yy = y + dy + j * 2
                cv2.line(frame, (x + dx, yy), (x + dx + 3, yy), dark, 1)
    elif code == 'TH':
        stripes = [((35, 35, 190), 2), (white, 2), ((160, 55, 25), 6), (white, 2), ((35, 35, 190), 2)]
        yy = y
        for color, hh in stripes:
            cv2.rectangle(frame, (x, yy), (x + width - 1, min(y + height - 1, yy + hh - 1)), color, -1)
            yy += hh
    elif code == 'US':
        for stripe in range(7):
            sy = y + stripe * height // 7
            ey = y + (stripe + 1) * height // 7 - 1
            color = (35, 35, 200) if stripe % 2 == 0 else white
            cv2.rectangle(frame, (x, sy), (x + width - 1, ey), color, -1)
        cv2.rectangle(frame, (x, y), (x + width * 2 // 5, y + height * 4 // 7), (130, 55, 25), -1)
        for sx, sy in [(2, 2), (6, 2), (9, 4), (3, 7), (7, 7)]:
            cv2.circle(frame, (x + sx, y + sy), 1, white, -1)
    cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), (230, 230, 230), 1)


def label_size(name):
    flags = FLAG_CODES[name]
    scale = .46
    (tw, th), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    flag_width = 22 * len(flags) + 3 * (len(flags) - 1)
    width = 8 + flag_width + 5 + tw + 8
    return width, max(26, 20 + 6), tw, th, baseline


def draw_frame(frame, boxes, names):
    h, w = frame.shape[:2]
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5, cv2.LINE_AA)

    labels = []
    for box, name in zip(boxes, names):
        if name not in FLAG_CODES:
            labels.append(None)
            continue
        x1, y1, x2, _ = box
        bw, bh, tw, th, baseline = label_size(name)
        left = round((x1 + x2 - bw) / 2)
        left = max(SIDE_MARGIN, min(w - SIDE_MARGIN - bw, left))
        top = max(2, y1 - bh)
        labels.append([left, top, left + bw, top + bh])

    # If two badges collide, place the left badge against its box's left edge
    # and the right badge against its box's right edge before shrinking either.
    if len(labels) == 2 and labels[0] and labels[1]:
        a, b = sorted(range(2), key=lambda i: boxes[i][0])
        if labels[a][2] > labels[b][0]:
            wa = labels[a][2] - labels[a][0]
            wb = labels[b][2] - labels[b][0]
            labels[a][0] = max(SIDE_MARGIN, boxes[a][0])
            labels[a][2] = labels[a][0] + wa
            labels[b][2] = min(w - SIDE_MARGIN, boxes[b][2])
            labels[b][0] = labels[b][2] - wb

    for box, name, rect in zip(boxes, names, labels):
        if not rect:
            continue
        left, top, right, bottom = rect
        bw, bh, tw, th, baseline = label_size(name)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 235), -1)
        fx = left + 4
        fy = top + (bh - 16) // 2
        for flag in FLAG_CODES[name]:
            draw_flag(frame, flag, fx, fy)
            fx += 25
        text_y = top + (bh + th - baseline) // 2
        cv2.putText(frame, name, (fx + 1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    .46, (255, 255, 255), 1, cv2.LINE_AA)
    return labels


def pose_bounds_for_window(old_rows, count, fps):
    """Find a narrow head/torso/leg envelope for OA and AIJING in the shown passage."""
    # This is the passage with the visible OA + AIJING pair, around 18–21 s.
    start, end, step = 500, 660, 3
    model = YOLO(str(POSE_MODEL))
    cap = cv2.VideoCapture(str(INPUT))
    sampled = {0: {}, 1: {}}
    sampled_frames = []
    for frame_index in range(start, min(end, count), step):
        boxes, names = map_frame(frame_index, old_rows)
        if len(boxes) != 2 or set(names) != {'OA', 'AIJING'}:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        result = model.predict(frame, conf=.20, imgsz=640, verbose=False)[0]
        detections = []
        if result.keypoints is not None:
            xy = result.keypoints.xy.cpu().numpy()
            conf = result.keypoints.conf.cpu().numpy()
            # Nose, shoulders, hips, knees and ankles describe the body axis;
            # hands and elbows are excluded so sweeping arms do not bloat boxes.
            joints = [0, 5, 6, 11, 12, 13, 14, 15, 16]
            for person in range(len(xy)):
                xs = [float(xy[person, j, 0]) for j in joints if conf[person, j] >= .30]
                if len(xs) >= 4:
                    detections.append((float(np.median(xs)), min(xs), max(xs)))
        if not detections:
            continue
        order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
        used = set()
        this = {}
        for slot in order:
            x1, _, x2, _ = boxes[slot]
            center = (x1 + x2) / 2
            candidates = [(abs(det[0] - center), j, det) for j, det in enumerate(detections) if j not in used]
            if not candidates:
                continue
            distance, detection_index, det = min(candidates, key=lambda z: z[0])
            if distance > 180:
                continue
            used.add(detection_index)
            _, px1, px2 = det
            core_width = px2 - px1
            old_width = x2 - x1
            target_width = max(old_width * 1.18, core_width * .82)
            target_center = (px1 + px2) / 2
            lx = max(SIDE_MARGIN, target_center - target_width / 2)
            rx = min(720 - SIDE_MARGIN, target_center + target_width / 2)
            this[slot] = (lx, rx)
        if len(this) == 2:
            sampled_frames.append(frame_index)
            for slot, bounds in this.items():
                sampled[slot][frame_index] = bounds
    cap.release()
    if len(sampled_frames) < 3:
        return {}

    # Median-filter sampled bounds, then interpolate per frame for stable motion.
    refined = {}
    for frame_index in range(start, min(end, count)):
        boxes, names = map_frame(frame_index, old_rows)
        if len(boxes) != 2 or set(names) != {'OA', 'AIJING'}:
            continue
        frame_bounds = []
        for slot in range(2):
            observations = sorted(sampled[slot].items())
            if not observations:
                frame_bounds = []
                break
            before = [item for item in observations if item[0] <= frame_index]
            after = [item for item in observations if item[0] >= frame_index]
            a = before[-1] if before else observations[0]
            b = after[0] if after else observations[-1]
            if b[0] == a[0]:
                vals = a[1]
            else:
                t = (frame_index - a[0]) / (b[0] - a[0])
                vals = tuple(v0 + (v1 - v0) * t for v0, v1 in zip(a[1], b[1]))
            neighborhood = [v for f, v in observations if abs(f - frame_index) <= 9]
            if neighborhood:
                vals = tuple(float(np.median([v[k] for v in neighborhood])) for k in range(2))
            frame_bounds.append(vals)
        if len(frame_bounds) == 2:
            frame_bounds.sort(key=lambda x: (x[0] + x[1]) / 2)
            # Preserve a clean gap between the two red boxes.
            if frame_bounds[0][1] + 4 > frame_bounds[1][0]:
                mid = (frame_bounds[0][1] + frame_bounds[1][0]) / 2
                frame_bounds[0] = (frame_bounds[0][0], mid - 2)
                frame_bounds[1] = (mid + 2, frame_bounds[1][1])
            refined[frame_index] = frame_bounds
    return refined


def main():
    for path in [INPUT, ANNOTATIONS, POSE_MODEL]:
        if not path.exists():
            raise FileNotFoundError(path)
    if OUTPUT.exists() or TIMES_CSV.exists() or TIMES_MD.exists() or OUT_DIR.exists():
        raise FileExistsError('Corrected c5 export already exists; choose a new output name.')
    OUT_DIR.mkdir(parents=True)
    old_rows = read_rows(ANNOTATIONS)
    cap = cv2.VideoCapture(str(INPUT))
    if not cap.isOpened():
        raise RuntimeError('Cannot open c5.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if (width, height) != (720, 1280):
        raise RuntimeError(f'Unexpected source size: {width}x{height}')

    refined_x = pose_bounds_for_window(old_rows, count, fps)
    per_member = Counter()
    c_count = Counter()
    overlap_frames = 0
    preview_frames = {0, 573, 585, 600, 615, 630, 1500, 3000, 4500, 6000}
    command = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-n', '-thread_queue_size', '64',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
               '-i', 'pipe:0', '-i', str(INPUT), '-map', '0:v:0', '-map', '1:a?', '-c:v', 'libx264',
               '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'copy',
               '-movflags', '+faststart', str(OUTPUT)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    boxes_jsonl = (OUT_DIR / 'corrected_boxes.jsonl').open('w', encoding='utf-8')
    try:
        for frame_index in range(count):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'c5 ended at frame {frame_index}')
            boxes, names = map_frame(frame_index, old_rows)
            if frame_index in refined_x and len(boxes) == 2:
                for i, (x1, x2) in enumerate(refined_x[frame_index]):
                    boxes[i][0] = max(SIDE_MARGIN, min(720 - SIDE_MARGIN - 1, round(x1)))
                    boxes[i][2] = max(boxes[i][0] + 1, min(720 - SIDE_MARGIN, round(x2)))
            c_count[len(boxes)] += 1
            for name in names:
                if name in FLAG_CODES:
                    per_member[name] += 1
            if len(boxes) == 2:
                left, right = sorted(boxes, key=lambda b: b[0])
                if left[2] >= right[0]:
                    overlap_frames += 1
            draw_frame(frame, boxes, names)
            boxes_jsonl.write(json.dumps({'frame': frame_index, 'members': names,
                                          'boxes': boxes}, ensure_ascii=False) + '\n')
            if frame_index in preview_frames:
                cv2.imwrite(str(OUT_DIR / f'preview-{frame_index:05}.jpg'), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            process.stdin.write(frame.tobytes())
            if frame_index % 900 == 0:
                print(f'Rendered {frame_index}/{count}', flush=True)
    finally:
        cap.release()
        boxes_jsonl.close()
        if process.stdin:
            process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg export failed')

    with TIMES_CSV.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['成员', 'C位累计时间（秒）', '国籍'])
        nationalities = {'AIJING': '中国', 'XILI': '中国', 'OA': '美国 / 日本',
                         'MERCI': '中国', 'AYUNE': '韩国', 'YOUNG': '韩国',
                         'VORA': '韩国', 'RADA': '泰国', 'HIMEKI': '日本'}
        for name in MEMBERS:
            writer.writerow([name, f'{per_member[name] / fps:.2f}', nationalities[name]])
    with TIMES_MD.open('w', encoding='utf-8') as stream:
        stream.write('# c5 C 位时间（修正版）\n\n双 C 同一时刻分别计入两位成员。\n\n')
        stream.write('| 成员 | C 位累计时间 | 国籍 |\n|---|---:|---|\n')
        for name in MEMBERS:
            stream.write(f'| {name} | {per_member[name] / fps:.2f} 秒 | {nationalities[name]} |\n')

    probe = subprocess.run([FFPROBE, '-v', 'error', '-show_entries',
                            'stream=codec_type,nb_frames,duration', '-of', 'json', str(OUTPUT)],
                           check=True, capture_output=True, text=True)
    streams = json.loads(probe.stdout)['streams']
    video_stream = next(s for s in streams if s['codec_type'] == 'video')
    audio_stream = next((s for s in streams if s['codec_type'] == 'audio'), None)
    if int(video_stream['nb_frames']) != count:
        raise RuntimeError('Encoded frame count does not match c5')
    summary = {
        'source': str(INPUT), 'output': str(OUTPUT), 'frames': count, 'fps': fps,
        'duration_seconds': count / fps, 'source_annotation': str(ANNOTATIONS),
        'identity_rule': 'retained reviewed c4 identities and C intervals',
        'flag_badges': 'simple visible flag icons beside the member names',
        'pose_refinement_frames': len(refined_x),
        'pose_refinement_scope': 'OA / AIJING two-C passage around 16.7–22.0 s',
        'c_count_frames': dict(c_count), 'member_c_frames': dict(per_member),
        'box_pair_overlap_frames': overlap_frames, 'audio_preserved': bool(audio_stream),
        'video_frames_verified': int(video_stream['nb_frames']),
    }
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
