"""Transfer reviewed C-role annotations to c5 and render simple name tags."""

import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
INPUT = Path('/path/to/user/Desktop/c5.mp4')
ANNOTATIONS = ROOT / 'outputs/c4-red-name-flag/c4_named_frames.jsonl'
OUTPUT = Path('/path/to/user/Desktop/c5_C位红框_名字跟框.mp4')
TIMES_CSV = Path('/path/to/user/Desktop/c5_C位时间统计.csv')
TIMES_MD = Path('/path/to/user/Desktop/c5_C位时间区间.md')
OUT_DIR = ROOT / 'outputs/c5-red-name-only'
FFMPEG = '/path/to/user/.local/bin/ffmpeg'
FFPROBE = '/path/to/user/.local/bin/ffprobe'

OLD_WIDTH, OLD_HEIGHT = 544, 720
FRAME_OFFSET = 9  # c5 starts about 0.30 s before the matched c4 sequence
RESIZE_SCALE = 720 / 544
VERTICAL_CROP = 128  # c4 was a centered vertical crop of the 9:16 c5 frame
TOP_MARGIN = 34
BOTTOM_MARGIN = 28
SIDE_MARGIN = 4
MEMBERS = ['AIJING', 'XILI', 'OA', 'MERCI', 'AYUNE', 'YOUNG', 'VORA', 'RADA', 'HIMEKI']


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def map_frame(frame, old_rows):
    old_frame = max(0, min(len(old_rows) - 1, frame - FRAME_OFFSET))
    row = old_rows[old_frame]
    if old_frame >= 6582:
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


def tag_size(name):
    (tw, th), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, .55, 1)
    return tw + 16, max(27, th + baseline + 8), tw, th, baseline


def draw_frame(frame, boxes, names):
    H, W = frame.shape[:2]
    badge_rects = []
    # Draw each slim, full-body marker with comfortable space from the frame edge.
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5, cv2.LINE_AA)

    for box, name in zip(boxes, names):
        if not name:
            badge_rects.append(None)
            continue
        x1, y1, x2, _ = box
        width, height, tw, th, baseline = tag_size(name)
        width = min(width, max(48, x2 - x1 - 2))
        left = round((x1 + x2 - width) / 2)
        top = y1 - height
        left = max(SIDE_MARGIN, min(W - SIDE_MARGIN - width, left))
        top = max(2, top)
        # A simple red name tab touches the matching red frame; no flag artwork.
        cv2.rectangle(frame, (left, top), (left + width, y1), (0, 0, 235), -1)
        text_x = left + (width - tw) // 2
        baseline_y = y1 - (height - th - baseline) // 2 - baseline
        cv2.putText(frame, name, (text_x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (255, 255, 255), 1, cv2.LINE_AA)
        badge_rects.append([left, top, left + width, y1])
    return badge_rects


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start - merged[-1][1] <= .06:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def render():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    if not ANNOTATIONS.exists():
        raise FileNotFoundError(ANNOTATIONS)
    if OUTPUT.exists() or OUT_DIR.exists():
        raise FileExistsError('A c5 name-only export already exists.')
    OUT_DIR.mkdir(parents=True)
    old_rows = read_jsonl(ANNOTATIONS)
    cap = cv2.VideoCapture(str(INPUT))
    if not cap.isOpened():
        raise RuntimeError('Cannot open c5.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if (width, height) != (720, 1280):
        raise RuntimeError(f'Unexpected c5 dimensions: {width}x{height}')

    per_member = Counter()
    member_intervals = defaultdict(list)
    current_start = {}
    c_count = Counter()
    pair_overlaps = 0
    badge_overlaps = 0
    previews = {0, 600, 1500, 2400, 3000, 3900, 4500, 5400, 6000, 6540}

    command = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-n', '-thread_queue_size', '64',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
               '-i', 'pipe:0', '-i', str(INPUT), '-map', '0:v:0', '-map', '1:a?', '-c:v', 'libx264',
               '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'copy',
               '-movflags', '+faststart', str(OUTPUT)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame_index in range(count):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'c5 ended at frame {frame_index}')
            boxes, names = map_frame(frame_index, old_rows)
            if len(boxes) != len(names):
                raise RuntimeError(f'Names and C boxes differ at frame {frame_index}')
            c_count[len(boxes)] += 1
            active = set()
            for name in names:
                if name:
                    per_member[name] += 1
                    active.add(name)
                    current_start.setdefault(name, frame_index)
            for name in list(current_start):
                if name not in active:
                    member_intervals[name].append((current_start.pop(name) / fps,
                                                   frame_index / fps))

            badge_rects = draw_frame(frame, boxes, names)
            if len(boxes) == 2:
                a, b = sorted(boxes, key=lambda r: r[0])
                if b[0] <= a[2]:
                    pair_overlaps += 1
            rects = [r for r in badge_rects if r]
            if len(rects) == 2:
                a, b = sorted(rects, key=lambda r: r[0])
                if a[2] > b[0] and max(a[1], b[1]) < min(a[3], b[3]):
                    badge_overlaps += 1
            if frame_index in previews:
                cv2.imwrite(str(OUT_DIR / f'preview-{frame_index:05}.jpg'), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 94])
            process.stdin.write(frame.tobytes())
            if frame_index % 900 == 0:
                print(f'Rendered {frame_index}/{count}', flush=True)
    finally:
        cap.release()
        if process.stdin:
            process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg export failed')

    for name, start in current_start.items():
        member_intervals[name].append((start / fps, count / fps))
    member_intervals = {name: merge_intervals(member_intervals[name]) for name in MEMBERS}
    with TIMES_CSV.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['成员', 'C位累计时间（秒）', '说明'])
        for name in MEMBERS:
            writer.writerow([name, f'{per_member[name] / fps:.2f}', '双C时两位成员分别计时'])
    with TIMES_MD.open('w', encoding='utf-8') as stream:
        stream.write('# c5 C 位时间\n\n双 C 同一时刻分别计入两位成员。\n\n')
        stream.write('| 成员 | C 位累计时间 |\n|---|---:|\n')
        for name in MEMBERS:
            stream.write(f'| {name} | {per_member[name] / fps:.2f} 秒 |\n')
        stream.write('\n## 区间\n')
        for name in MEMBERS:
            spans = ', '.join(f'{a:.1f}–{b:.1f}s' for a, b in member_intervals[name])
            stream.write(f'\n**{name}**：{spans or "无"}\n')

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
        'frame_mapping': f'c5 frame - {FRAME_OFFSET} -> reviewed c4 annotation frame',
        'box_transform': {'x_scale': RESIZE_SCALE, 'y_scale': RESIZE_SCALE,
                          'vertical_crop': VERTICAL_CROP},
        'margins_px': {'top_min': TOP_MARGIN, 'bottom_min': BOTTOM_MARGIN},
        'c_count_frames': dict(c_count), 'member_c_frames': dict(per_member),
        'box_pair_overlap_frames': pair_overlaps, 'name_tag_overlap_frames': badge_overlaps,
        'audio_preserved': bool(audio_stream), 'video_frames_verified': int(video_stream['nb_frames']),
    }
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    render()
