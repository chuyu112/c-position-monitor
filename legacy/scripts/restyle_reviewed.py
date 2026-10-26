"""Re-export reviewed C roles with narrow, separate full-height markers."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c_detect.overlays import narrow_c_boxes
from c_detect.stabilization import stabilize_display_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('review_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--stabilize', action='store_true', help='Smooth marker positions and sizes within reviewed role intervals')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit('Choose an empty output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads((args.review_dir / 'reviewed_segments.json').read_text())
    reviewed = [json.loads(line) for line in (args.review_dir / 'reviewed_frames.jsonl').read_text().splitlines()]
    detections = [json.loads(line) for line in Path(spec['cache']).read_text().splitlines()]
    cap = cv2.VideoCapture(spec['source'])
    if not cap.isOpened():
        raise RuntimeError('Cannot read source video')
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert len(reviewed) == len(detections) == int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    displays, ratios = [], []
    for row, detected in zip(reviewed, detections):
        torso_by_id = {p['track_id']: p['torso_x'] for p in detected['people']}
        torso_xs = [torso_by_id.get(i) for i in row['detection_ids']]
        boxes = narrow_c_boxes(row['boxes'], width, height, torso_xs,
                              manual_pair=row['method'] == 'reviewed_box_interpolation')
        assert len(boxes) == row['c_count']
        for box, original in zip(boxes, row['boxes']):
            x1, y1, x2, y2 = box
            assert 0 <= x1 < x2 < width and 0 <= y1 < y2 < height
            assert abs(y1 - original[1] * height) <= 3 and abs(y2 - original[3] * height) <= 3
            ratios.append((x2 - x1) / ((original[2] - original[0]) * width))
        displays.append(boxes)
    stabilization = None
    if args.stabilize:
        displays, stabilization = stabilize_display_boxes(displays, spec['segments'], width, height, spec['fps'])
    for row, boxes in zip(reviewed, displays):
        assert len(boxes) == row['c_count']
        for x1, y1, x2, y2 in boxes:
            assert 0 <= x1 < x2 < width and 0 <= y1 < y2 < height
        if len(boxes) == 2:
            left, right = sorted(boxes)
            assert right[0] - left[2] >= 8
    with (args.output_dir / 'display_boxes.jsonl').open('w') as stream:
        for row, boxes in zip(reviewed, displays):
            stream.write(json.dumps({'frame': row['frame'], 'seconds': row['seconds'], 'c_count': row['c_count'], 'display_boxes_px': boxes, 'detection_ids': row['detection_ids']}) + '\n')
    output = args.output_dir / ('c-main-red-stable.mp4' if args.stabilize else 'c-main-red-narrow.mp4')
    ffmpeg = '/path/to/user/.local/bin/ffmpeg'
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-thread_queue_size', '64',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(spec['fps']),
               '-i', 'pipe:0', '-i', spec['source'], '-map', '0:v:0', '-map', '1:a?',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p',
               '-c:a', 'copy', '-movflags', '+faststart', '-shortest', str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index, boxes in enumerate(displays):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Missing frame {index}')
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            if index in [0, 601, 1260, 1358, 1380, 1485, 1500, 2340, 4530, 4590, 4800, 6510]:
                cv2.imwrite(str(args.output_dir / f'preview-{index:05}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            process.stdin.write(frame.tobytes())
            if index % 1800 == 0:
                print(f'Rendered {index}/{len(displays)}', flush=True)
    finally:
        cap.release()
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video export failed')
    summary = {
        'source': spec['source'], 'role_annotations': str((args.review_dir / 'reviewed_frames.jsonl').resolve()),
        'output': str(output.resolve()), 'frames': len(reviewed), 'fps': spec['fps'],
        'style': '全身高度；横向窄框；1像素细红线；双C框之间至少8像素间隔。',
        'role_selection_changed': bool(spec.get('corrections')), 'original_audio_preserved': True,
        'reviewed_corrections': spec.get('corrections', []),
        'stabilization': stabilization,
        'mean_width_ratio_vs_original_boxes': sum(ratios) / len(ratios),
        'c_count_frames': dict(Counter(row['c_count'] for row in reviewed)),
        'geometry_validation': {'all_frames_checked': True, 'overlapping_pairs': 0,
                                'full_height_style': True, 'all_rectangles_in_frame': True},
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
