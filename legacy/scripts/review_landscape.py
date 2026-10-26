"""Create time-labelled contact sheets from a detection or reviewed-frame cache."""

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    parser.add_argument('cache', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--step', type=float, default=2)
    parser.add_argument('--count', type=int, default=15)
    parser.add_argument('--crop', type=int, nargs=4, default=[0, 190, 852, 450])
    args = parser.parse_args()
    rows = []
    with args.cache.open() as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    tile_width = 680
    crop = args.crop
    tile_height = round((crop[3] - crop[1]) * tile_width / (crop[2] - crop[0])) + 30
    canvas = Image.new('RGB', (tile_width * 3, tile_height * ((args.count + 2) // 3)), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    for n in range(args.count):
        sec = args.start + n * args.step
        index = round(sec * fps)
        if index >= len(rows):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        row = rows[index]
        h, w = frame.shape[:2]
        if 'people' in row:
            for p in row['people']:
                box = p['box']
                if max(box) <= 1.01:
                    box = [v * (w if k % 2 == 0 else h) for k, v in enumerate(box)]
                x1, y1, x2, y2 = map(round, box)
                name = p.get('person_id', str(p['track_id']))
                color = (0, 255, 255) if p['track_id'] in row.get('selected_ids', []) else (100, 255, 100)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.rectangle(frame, (x1, y1 - 15), (x1 + len(name) * 9, y1), (15, 15, 15), -1)
                cv2.putText(frame, name, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
        else:
            for box, name in zip(row['boxes'], row['members']):
                x1, y1, x2, y2 = map(round, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, name, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv2.LINE_AA)
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).crop(crop)
        im = im.resize((tile_width, tile_height - 30))
        x, y = n % 3 * tile_width, n // 3 * tile_height
        canvas.paste(im, (x, y + 30))
        draw.text((x + 8, y + 8), f'{sec:.2f}s / frame {index}', fill='white')
    cap.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=94)


if __name__ == '__main__':
    main()
