"""Render C annotations with explicit user corrections and stable gray name badges."""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from render_c5_flag_corrected import FLAG_CODES, draw_flag as draw_other_flag
from c5_live_ranking import LiveRanking
from c5_text import CHINESE_NAMES, draw_sprite, text_sprite


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/path/to/user/Desktop/c5.mp4')
BASE = ROOT / 'outputs/c5-red-name-flag-corrected/corrected_boxes.jsonl'
OUT_DIR = ROOT / 'outputs/c5-red-name-cn-emoji'
OUTPUT = Path('/path/to/user/Desktop/c5_C位红框_五星红旗版.mp4')
FFMPEG = '/path/to/user/.local/bin/ffmpeg'
FFPROBE = '/path/to/user/.local/bin/ffprobe'
CORRECTIONS = ROOT / 'annotations/c5_member_overrides.json'
ROLE_CORRECTIONS = ROOT / 'annotations/c5_role_overrides.json'


def load_flag(code='CN'):
    rgba = cv2.imread(str(ROOT / f'assets/flags/{code.lower()}-emoji.png'), cv2.IMREAD_UNCHANGED)
    ys, xs = np.where(rgba[:, :, 3] > 0)
    rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # Give the five stars enough pixels while keeping the label compact.
    return cv2.resize(rgba, (30, 22), interpolation=cv2.INTER_AREA)


CN_FLAG = load_flag()
EMOJI_FLAGS = {'CN': CN_FLAG, 'KR': load_flag('KR'), 'TH': load_flag('TH')}


def badge_size(name):
    widths = [30 if flag == 'CN' else 22 for flag in FLAG_CODES[name]]
    (_, th), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, .46, 1)
    tw = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, .46, 1)[0][0]
    if name in CHINESE_NAMES:
        tw = max(tw, text_sprite(CHINESE_NAMES[name]).shape[1])
    width = 8 + sum(widths) + 3 * (len(widths) - 1) + 5 + tw + 8
    height = 42 if name in CHINESE_NAMES else 26
    return width, height, th, baseline


def draw_frame(frame, boxes, names, reserved_rects=()):
    h, w = frame.shape[:2]
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5, cv2.LINE_AA)
    labels = []
    for box, name in zip(boxes, names):
        if name not in FLAG_CODES:
            labels.append(None)
            continue
        bw, bh, _, _ = badge_size(name)
        left = max(4, min(w - 4 - bw, round((box[0] + box[2] - bw) / 2)))
        top = max(2, box[1] - bh)
        labels.append([left, top, left + bw, top + bh])
    if len(labels) == 2 and all(labels):
        a, b = sorted(range(2), key=lambda i: boxes[i][0])
        if labels[a][2] > labels[b][0]:
            wa, wb = labels[a][2] - labels[a][0], labels[b][2] - labels[b][0]
            labels[a][0] = max(4, boxes[a][0])
            labels[a][2] = labels[a][0] + wa
            labels[b][2] = min(w - 4, boxes[b][2])
            labels[b][0] = labels[b][2] - wb
    # Keep the badge attached horizontally to its C box, below a reserved leaderboard
    # when its usual position above the box would be hidden by that panel.
    for rect in labels:
        if rect is None:
            continue
        for rx1, ry1, rx2, ry2 in reserved_rects:
            if rect[0] < rx2 and rect[2] > rx1 and rect[1] < ry2 and rect[3] > ry1:
                bh = rect[3] - rect[1]
                rect[1], rect[3] = ry2 + 6, ry2 + 6 + bh
                assert rect[3] < h - 4
    for name, rect in zip(names, labels):
        if rect is None:
            continue
        left, top, right, bottom = rect
        _, bh, th, baseline = badge_size(name)
        cv2.rectangle(frame, (left, top), (right, bottom), (198, 194, 190), -1)
        fx = left + 4
        for flag in FLAG_CODES[name]:
            if flag in EMOJI_FLAGS:
                # KR and TH use the same clear 30x22 system emoji as CN.
                # Reclaim eight pixels of the existing right padding, keeping
                # the badge rectangle unchanged for existing video exports.
                rgba = EMOJI_FLAGS[flag]
                fy = top + (bh - 22) // 2
                region = frame[fy:fy + 22, fx:fx + 30]
                alpha = rgba[:, :, 3:4].astype(np.float32) / 255
                region[:] = np.rint(rgba[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)
                fx += 33
            else:
                draw_other_flag(frame, flag, fx, top + (bh - 16) // 2)
                fx += 25
        text_y = top + 16 if name in CHINESE_NAMES else top + (bh + th - baseline) // 2
        cv2.putText(frame, name, (fx + 1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, .46, (45, 42, 40), 1, cv2.LINE_AA)
        if name in CHINESE_NAMES:
            draw_sprite(frame, text_sprite(CHINESE_NAMES[name]), fx + 1, top + 22)
    return labels


def apply_member_corrections(rows):
    """Apply user-supplied labels to C slots, never move boxes to follow a face."""
    changes = []
    if not CORRECTIONS.exists():
        return changes
    config = json.loads(CORRECTIONS.read_text())
    assert config['video'] == str(SOURCE) and config['fps'] == 30
    for correction in config['corrections']:
        assert correction['c_slot'] in {'left', 'right', 'only'}
        for i in range(correction['start_frame'], correction['end_frame_exclusive']):
            row = rows[i]
            count = 1 if correction['c_slot'] == 'only' else 2
            assert len(row['boxes']) == len(row['members']) == count
            choose = max if correction['c_slot'] == 'right' else min
            slot = choose(range(count), key=lambda j: row['boxes'][j][0])
            previous = row['members'][slot]
            assert previous in correction['allowed_previous_names']
            if count == 2:
                other = correction['other_member']
                assert row['members'][1 - slot] in (other if isinstance(other, list) else [other])
            if previous != correction['member']:
                changes.append({'frame': i, 'slot': slot, 'before': previous,
                                'after': correction['member']})
                row['members'][slot] = correction['member']
    return changes


def apply_role_corrections(rows):
    """Apply reviewed C-role keyframes without waiting for identity recognition."""
    changes = []
    if not ROLE_CORRECTIONS.exists():
        return changes
    config = json.loads(ROLE_CORRECTIONS.read_text())
    assert config['video'] == str(SOURCE) and config['fps'] == 30
    for correction in config['corrections']:
        operation = correction.get('operation', 'replace')
        assert operation in {'replace', 'add_left', 'add_right', 'adjust_box', 'clear', 'remove_member'}
        start, end = correction['start_frame'], correction['end_frame_exclusive']
        if operation in {'clear', 'remove_member'}:
            # During an unresolved handoff, neither the departing body nor its timer stays active.
            for i in range(start, end):
                row = rows[i]
                assert row['members'] == correction['expected_previous_members']
                assert len(row['boxes']) == len(row['members'])
                if operation == 'clear':
                    members, new_boxes = [], []
                else:
                    assert correction['member'] in row['members']
                    keep = [j for j, name in enumerate(row['members'])
                            if name != correction['member']]
                    members = [row['members'][j] for j in keep]
                    new_boxes = [row['boxes'][j] for j in keep]
                if 'turned_away_member' in correction:
                    assert correction['turned_away_member'] not in members
                changes.append({'frame': i, 'previous_members': row['members'],
                                'previous_boxes': row['boxes'], 'members': members,
                                'boxes': new_boxes, 'operation': operation})
                row['members'], row['boxes'] = members, new_boxes
            continue
        keyframes = correction['box_keyframes']
        times = np.array([item['frame'] for item in keyframes])
        coordinates = np.array([item['box'] for item in keyframes], dtype=float)
        assert times[0] == start and times[-1] == end - 1
        assert np.all(np.diff(times) > 0)
        boxes = np.column_stack([np.interp(np.arange(start, end), times, coordinates[:, axis])
                                 for axis in range(4)])
        # Offline symmetric smoothing preserves body movement without frame-to-frame jitter.
        kernel = np.array([1, 2, 3, 2, 1], dtype=float) / 9
        boxes = np.column_stack([np.convolve(np.pad(boxes[:, axis], (2, 2), mode='edge'),
                                             kernel, mode='valid') for axis in range(4)])
        integer_boxes = np.rint(boxes).astype(int)
        widths = coordinates[:, 2] - coordinates[:, 0]
        if np.all(widths == widths[0]):
            # Round the translated left edge once so a fixed-width box never gains/loses a pixel.
            integer_boxes[:, 2] = integer_boxes[:, 0] + int(round(widths[0]))
        for i, box in zip(range(start, end), integer_boxes.tolist()):
            row = rows[i]
            assert row['members'] == correction['expected_previous_members']
            expected_count = (len(correction['expected_previous_members'])
                              if operation in {'adjust_box', 'replace'} else 1)
            assert len(row['boxes']) == len(row['members']) == expected_count
            assert 4 <= box[0] < box[2] <= 716 and 34 <= box[1] < box[3] <= 1252
            if operation == 'add_left':
                assert box[2] + 6 <= row['boxes'][0][0], (i, box, row['boxes'])
                members, new_boxes = [correction['member']] + row['members'], [box] + row['boxes']
            elif operation == 'add_right':
                assert row['boxes'][0][2] + 6 <= box[0], (i, box, row['boxes'])
                members, new_boxes = row['members'] + [correction['member']], row['boxes'] + [box]
            elif operation == 'adjust_box':
                members, new_boxes = row['members'][:], [b[:] for b in row['boxes']]
                new_boxes[correction['slot']] = box
                if len(new_boxes) == 2:
                    left, right = sorted(new_boxes)
                    assert left[2] + 6 <= right[0], (i, new_boxes)
            else:
                members, new_boxes = [correction['member']], [box]
            if 'turned_away_member' in correction:
                assert correction['turned_away_member'] not in members
            changes.append({'frame': i, 'previous_members': row['members'],
                            'previous_boxes': row['boxes'], 'members': members,
                            'boxes': new_boxes, 'operation': operation})
            row['members'] = members
            row['boxes'] = new_boxes
    return changes


def main():
    global OUTPUT, OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--live-ranking', action='store_true')
    args = parser.parse_args()
    OUTPUT, OUT_DIR = args.output, args.out_dir
    if OUTPUT.exists() or OUT_DIR.exists():
        raise FileExistsError('Output already exists')
    rows = [json.loads(line) for line in BASE.read_text().splitlines()]
    changes = apply_member_corrections(rows)
    role_changes = apply_role_corrections(rows)
    cap = cv2.VideoCapture(str(SOURCE))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert (width, height) == (720, 1280) and count == len(rows)
    OUT_DIR.mkdir(parents=True)
    (OUT_DIR / 'member-corrections.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2))
    (OUT_DIR / 'role-corrections.json').write_text(json.dumps(role_changes, ensure_ascii=False, indent=2))
    cmd = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-n', '-thread_queue_size', '64',
           '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
           '-i', 'pipe:0', '-i', str(SOURCE), '-map', '0:v:0', '-map', '1:a?',
           '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
           '-c:a', 'copy', '-movflags', '+faststart', str(OUTPUT)]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    member_frames = Counter()
    leaderboard = LiveRanking(fps, FLAG_CODES) if args.live_ranking else None
    reserved_rects = ([(leaderboard.x, leaderboard.y,
                        leaderboard.x + leaderboard.width,
                        leaderboard.y + leaderboard.height)] if leaderboard else [])
    ranking_log = []
    try:
        for i, row in enumerate(rows):
            assert row['frame'] == i
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Source ended at frame {i}')
            labels = draw_frame(frame, row['boxes'], row['members'], reserved_rects)
            member_frames.update(row['members'])
            if leaderboard:
                ranking_log.append(leaderboard.step(row['boxes'], row['members']))
                leaderboard.draw(frame)
            if i in {0, 573, 585, 600, 615, 630, 1500, 3000, 3072, 3090, 3108, 3132, 3156, 3157,
                     3830, 3915, 3947, 4044, 4045, 4500, 6000, 6378, 6392, 6407, 6408,
                     6435, 6475, 6500, 6542, 6590,
                     1455, 1470, 1487, 1488, 2760, 2899, 2911, 2915, 4560, 4565,
                     4659, 4675, 4698, 5330, 5342, 5364, 5388, 5389, 5555, 6591, 6823}:
                cv2.imwrite(str(OUT_DIR / f'preview-{i:05}.jpg'), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            if i == 573:
                rect = labels[row['members'].index('AIJING')]
                x1, y1, x2, y2 = rect
                crop = frame[max(0, y1 - 5):y2 + 8, max(0, x1 - 5):x2 + 5]
                cv2.imwrite(str(OUT_DIR / 'flag-label-preview.png'),
                            cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
            process.stdin.write(frame.tobytes())
            if i % 900 == 0:
                print(f'Rendered {i}/{count}', flush=True)
    finally:
        cap.release()
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')
    (OUT_DIR / 'corrected_boxes.jsonl').write_text(''.join(
        json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    if leaderboard:
        assert leaderboard.counts == member_frames
        (OUT_DIR / 'ranking_frames.jsonl').write_text(''.join(
            json.dumps(row, ensure_ascii=False) + '\n' for row in ranking_log))
    result = subprocess.run([FFPROBE, '-v', 'error', '-show_entries',
                             'stream=codec_type,nb_frames,duration', '-of', 'json', str(OUTPUT)],
                            check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout)['streams']
    assert int(next(s for s in streams if s['codec_type'] == 'video')['nb_frames']) == count
    assert any(s['codec_type'] == 'audio' for s in streams)
    summary = {'output': str(OUTPUT), 'frames': count, 'fps': fps,
               'duration_seconds': count / fps, 'audio_preserved': True,
               'cn_icon': 'Actual Apple Color Emoji 🇨🇳, 30 × 22 pixels',
               'label_background': '#BEC2C6 gray', 'label_text': '#282A2D dark gray',
               'chinese_names': CHINESE_NAMES,
               'annotations': str(BASE), 'boxes_unchanged': not bool(role_changes),
               'role_correction_rules': str(ROLE_CORRECTIONS),
               'corrected_role_frames': len({change['frame'] for change in role_changes}),
               'c_count_frames': dict(Counter(len(row['boxes']) for row in rows)),
               'member_correction_rules': str(CORRECTIONS),
               'live_ranking': bool(leaderboard),
               'timing_rule': 'Each visible red box adds 1/fps seconds to its named member; unboxed members pause. Double C counts both members.',
               'corrected_member_frames': len(changes),
               'member_c_frames': dict(member_frames),
               'member_c_seconds': {name: frames / fps for name, frames in member_frames.items()}}
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
