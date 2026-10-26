"""Rebuild badges and exact live counts when only an annotated export remains.

This path permits supplied name corrections and flag artwork refreshes. It requires identical box
geometry and badge dimensions, so no reconstruction of dancer pixels is needed.
The leaderboard uses an opaque gray panel to fully replace its previous text.
"""

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import render_c5_cn_flag as renderer
from c5_live_ranking import LiveRanking


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--previous-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--refresh-flags', nargs='+', choices=['KR', 'TH'], default=[])
    args = parser.parse_args()
    assert args.input.is_file()
    assert not args.output.exists() and not args.out_dir.exists()
    previous = read_rows(args.previous_data / 'corrected_boxes.jsonl')
    rows = read_rows(renderer.BASE)
    renderer.apply_member_corrections(rows)
    renderer.apply_role_corrections(rows)
    assert len(rows) == len(previous)
    changed = []
    for i, (old, row) in enumerate(zip(previous, rows)):
        assert old['frame'] == row['frame'] == i
        assert old['boxes'] == row['boxes'], (i, 'Geometry change requires raw source')
        assert len(old['members']) == len(row['members'])
        for before, after in zip(old['members'], row['members']):
            assert renderer.badge_size(before)[:2] == renderer.badge_size(after)[:2]
        if old['members'] != row['members']:
            changed.append(i)
    refresh_flags = set(args.refresh_flags)
    assert changed or refresh_flags
    cap = cv2.VideoCapture(str(args.input))
    width, height, count = [int(cap.get(k)) for k in
                            (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                             cv2.CAP_PROP_FRAME_COUNT)]
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert (width, height, fps, count) == (720, 1280, 30, len(rows))
    ranking = LiveRanking(fps, renderer.FLAG_CODES)
    reserved = [(ranking.x, ranking.y, ranking.x + ranking.width,
                 ranking.y + ranking.height)]
    args.out_dir.mkdir(parents=True)
    for path in [renderer.CORRECTIONS, renderer.ROLE_CORRECTIONS,
                 renderer.ROOT / 'annotations/c5_member_names.json']:
        shutil.copy2(path, args.out_dir / path.name)
    if refresh_flags:
        (args.out_dir / 'flags').mkdir()
        for code in refresh_flags:
            path = renderer.ROOT / f'assets/flags/{code.lower()}-emoji.png'
            shutil.copy2(path, args.out_dir / 'flags' / path.name)
    process = subprocess.Popen([
        renderer.FFMPEG, '-hide_banner', '-loglevel', 'error', '-n',
        '-thread_queue_size', '64', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0',
        '-i', str(args.input), '-map', '0:v:0', '-map', '1:a:0',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '16',
        '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart',
        str(args.output)], stdin=subprocess.PIPE)
    logs = []
    changed_set = set(changed)
    refreshed_badges = Counter()
    preview_frames = {0, 2750, 5342, 5400, 5420, 5500, len(rows) - 1}
    if changed:
        preview_frames.update([changed[0] - 1, changed[0], changed[-1], changed[-1] + 1])
    try:
        for i, row in enumerate(rows):
            ok, frame = cap.read()
            assert ok, i
            refresh_slots = {slot for slot, name in enumerate(row['members'])
                             if refresh_flags.intersection(renderer.FLAG_CODES[name])}
            if i in changed_set or refresh_slots:
                # Render in scratch space and copy only the same-sized badge.
                # All red boxes and all pixels of the underlying bodies survive.
                scratch = np.zeros_like(frame)
                rects = renderer.draw_frame(scratch, row['boxes'], row['members'], reserved)
                for slot, rect in enumerate(rects):
                    if previous[i]['members'][slot] == row['members'][slot] and slot not in refresh_slots:
                        continue
                    x1, y1, x2, y2 = rect
                    frame[y1:y2 + 1, x1:x2 + 1] = scratch[y1:y2 + 1, x1:x2 + 1]
                    if slot in refresh_slots:
                        refreshed_badges[row['members'][slot]] += 1
            # Clear the previous leaderboard completely; use consistent gray
            # across the whole export, including before the correction interval.
            x, y, w, h = ranking.x, ranking.y, ranking.width, ranking.height
            logs.append(ranking.step(row['boxes'], row['members']))
            if changed:
                frame[y:y + h + 1, x:x + w + 1] = (230, 229, 226)
                ranking.draw(frame)
            if i in preview_frames:
                cv2.imwrite(str(args.out_dir / f'preview-{i}.jpg'), frame)
            process.stdin.write(frame.tobytes())
            if i % 900 == 0:
                print(f'Rendered {i}/{count}', flush=True)
    finally:
        cap.release()
        process.stdin.close()
    assert process.wait() == 0
    probe = json.loads(subprocess.check_output([
        renderer.FFPROBE, '-v', 'error', '-show_entries',
        'stream=codec_type,nb_frames,duration', '-of', 'json', str(args.output)]))
    assert int(next(s for s in probe['streams'] if s['codec_type'] == 'video')['nb_frames']) == count
    assert any(s['codec_type'] == 'audio' for s in probe['streams'])
    for name, values in [('corrected_boxes.jsonl', rows), ('ranking_frames.jsonl', logs)]:
        (args.out_dir / name).write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n'
                                                for v in values))
    old_counts = Counter(name for row in previous for name in row['members'])
    new_counts = Counter(name for row in rows for name in row['members'])
    assert new_counts == ranking.counts
    summary = {
        'output': str(args.output), 'input_annotated_export': str(args.input),
        'frames': count, 'fps': fps, 'duration_seconds': count / fps,
        'audio_preserved': True, 'boxes_unchanged': True,
        'leaderboard_background': ('opaque gray, replacing old rendered text'
                                   if changed else 'preserved from input'),
        'refreshed_flag_codes': sorted(refresh_flags),
        'flag_style': 'Apple Color Emoji, 30x22 pixels for CN, KR and TH',
        'refreshed_badge_frames': dict(refreshed_badges),
        'changed_member_frames': changed,
        'member_c_frames': dict(new_counts),
        'member_c_seconds': {name: n / fps for name, n in new_counts.items()},
        'member_frame_delta': {name: new_counts[name] - old_counts[name]
                               for name in new_counts if new_counts[name] != old_counts[name]},
        'c_count_frames': dict(Counter(len(row['boxes']) for row in rows)),
    }
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'changed_member_frames'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
