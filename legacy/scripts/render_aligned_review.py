"""Transfer reviewed markers to a visually aligned copy of the same footage.

The alignment file contains a verified pixel transform and temporal anchors.
This does not infer C roles in unrelated footage.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c_detect.overlays import constrain_display_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    parser.add_argument('markers', type=Path)
    parser.add_argument('review_spec', type=Path)
    parser.add_argument('alignment', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--line-width', type=int, default=3)
    parser.add_argument('--content-end-frame', type=int, required=True, help='Exclusive end of verified dance footage; no boxes on outro')
    args = parser.parse_args()
    if args.output.exists() or args.report_dir.exists():
        raise SystemExit('Output file and report directory must be new')
    if not 1 <= args.line_width <= 12:
        raise SystemExit('Line width must be between 1 and 12 pixels')
    rows = [json.loads(line) for line in args.markers.read_text().splitlines()]
    spec = json.loads(args.review_spec.read_text())
    alignment = json.loads(args.alignment.read_text())
    matrix = np.array(alignment['matrix'], dtype=float)
    # A robust local offset avoids matching ambiguity during held poses.
    anchors = alignment['anchors']
    anchor_frames = np.array([a['target_frame'] for a in anchors])
    offsets = np.array([a['offset_frames'] for a in anchors], dtype=float)
    offsets = np.array([np.median(offsets[max(0, i - 2):i + 3]) for i in range(len(offsets))])
    role_segments = np.empty(len(rows), dtype=int)
    for index, segment in enumerate(spec['segments']):
        role_segments[segment['start_frame']:segment['end_frame']] = index
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError('Cannot open target video')
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if abs(fps - round(fps)) < .0001:
        fps = float(round(fps))
    if not 0 < args.content_end_frame <= frame_count:
        raise ValueError('Content endpoint is outside the video')
    args.report_dir.mkdir(parents=True)
    # More room between centerlines keeps the thicker strokes separated too.
    gap = 8 + args.line_width + 1
    displays = []
    for frame in range(frame_count):
        if frame >= args.content_end_frame:
            displays.append({'frame': frame, 'seconds': frame / fps, 'c_count': 0, 'display_boxes_px': [], 'source_frame': None})
            continue
        source_frame = float(np.clip(frame - np.interp(frame, anchor_frames, offsets), 0, len(rows) - 1))
        lo, hi = int(np.floor(source_frame)), int(np.ceil(source_frame))
        nearest = int(round(source_frame))
        if role_segments[lo] == role_segments[hi]:
            boxes = np.asarray(rows[lo]['display_boxes_px'], dtype=float)
            boxes += (source_frame - lo) * (np.asarray(rows[hi]['display_boxes_px']) - boxes)
        else:
            boxes = np.asarray(rows[nearest]['display_boxes_px'], dtype=float)
        warped = []
        for x1, y1, x2, y2 in boxes:
            corners = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]]) @ matrix.T
            warped.append([*corners.min(axis=0), *corners.max(axis=0)])
        rectangles = constrain_display_boxes(warped, width, height, gap=gap)
        assert len(rectangles) in (1, 2)
        for x1, y1, x2, y2 in rectangles:
            assert 0 <= x1 < x2 < width and 0 <= y1 < y2 < height
        if len(rectangles) == 2:
            left, right = sorted(rectangles)
            assert right[0] - left[2] >= gap
        displays.append({'frame': frame, 'seconds': frame / fps, 'c_count': len(rectangles),
                         'display_boxes_px': rectangles, 'source_frame': source_frame})
    (args.report_dir / 'display_boxes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in displays))
    command = ['/path/to/user/.local/bin/ffmpeg', '-hide_banner', '-loglevel', 'error', '-n',
               '-thread_queue_size', '64', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0', '-i', str(args.video),
               '-map', '0:v:0', '-map', '1:a?', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
               '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart', str(args.output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    previews = {0, 90, 600, 1378, 1505, 1520, 3000, 4560, 6000, 6540, args.content_end_frame}
    try:
        for index, row in enumerate(displays):
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f'Missing target frame {index}')
            for x1, y1, x2, y2 in row['display_boxes_px']:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), args.line_width, cv2.LINE_AA)
            if index in previews:
                cv2.imwrite(str(args.report_dir / f'preview-{index:05}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            process.stdin.write(image.tobytes())
            if index % 1500 == 0:
                print(f'Rendered {index}/{frame_count}', flush=True)
    finally:
        cap.release()
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video export failed')
    probe = subprocess.run(['/path/to/user/.local/bin/ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=nb_frames,duration', '-of', 'json', str(args.output)],
                           check=True, capture_output=True, text=True)
    video_stream = json.loads(probe.stdout)['streams'][0]
    if int(video_stream['nb_frames']) != frame_count:
        raise RuntimeError('Encoded frame count does not match the input')
    summary = {'source': str(args.video.resolve()), 'output': str(args.output.resolve()),
               'frames': frame_count, 'fps': fps, 'width': width, 'height': height,
               'line_width_pixels': args.line_width, 'minimum_pair_centerline_gap_pixels': gap,
               'style': '全身窄红框；沿用已平滑的显示轨迹；双C各自独立框。',
               'reviewed_markers': str(args.markers.resolve()), 'alignment': str(args.alignment.resolve()),
               'median_source_offset_frames': float(np.median(offsets)), 'spatial_transform': matrix.tolist(),
               'content_end_frame': args.content_end_frame, 'outro_unannotated': True,
               'c_count_frames': dict(Counter(row['c_count'] for row in displays)),
               'geometry_validation': {'all_frames_checked': True, 'overlapping_pairs': 0, 'all_rectangles_in_frame': True},
               'original_audio_preserved': True, 'encoded_video_frames_verified': int(video_stream['nb_frames'])}
    (args.report_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
