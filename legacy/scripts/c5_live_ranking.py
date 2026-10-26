"""Frame-exact on-screen C time, driven only by visible red-box annotations."""

from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from c5_text import CHINESE_NAMES, draw_sprite, text_sprite


class LiveRanking:
    def __init__(self, fps, members):
        self.fps = fps
        self.order = list(members)
        self.counts = Counter({name: 0 for name in members})
        self.active = set()
        self.frame = -1
        # Keep the source's corner watermark visible, above this compact panel.
        self.x, self.y, self.width = 12, 76, 224
        self.row_height = 20
        self.height = 32 + len(members) * self.row_height + 6
        title = Image.new('RGBA', (self.width - 16, 24))
        font = ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc', 16)
        ImageDraw.Draw(title).text((0, 0), 'C 位实时排名', font=font, fill=(40, 42, 45, 255))
        self.title = np.asarray(title)
        self.name_sprites = {
            name: text_sprite(f'{name} {CHINESE_NAMES[name]}' if name in CHINESE_NAMES else name, 14)
            for name in members
        }

    def step(self, boxes, members):
        assert len(boxes) == len(members) <= 2
        assert len(set(members)) == len(members)
        assert all(name in self.counts for name in members)
        self.frame += 1
        self.active = set(members)
        # Exactly one frame per visible box. No increment for unboxed members.
        self.counts.update(members)
        # Keep the preceding order on exact ties; change rank only on an overtake.
        self.order.sort(key=lambda name: -self.counts[name])
        return {
            'frame': self.frame,
            'active': list(members),
            'order': self.order[:],
            'cumulative_frames': dict(self.counts),
        }

    def draw(self, frame):
        x, y, w, h = self.x, self.y, self.width, self.height
        panel = frame[y:y + h, x:x + w]
        # An opaque panel keeps moving red boxes and dancers from showing
        # through the stationary names and times, especially the bottom rows.
        panel[:] = (230, 229, 226)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (181, 181, 179), 1)
        title_region = frame[y + 5:y + 29, x + 8:x + w - 8]
        alpha = self.title[:, :, 3:4].astype(np.float32) / 255
        title_region[:] = np.rint(self.title[:, :, :3][:, :, ::-1] * alpha
                                 + title_region * (1 - alpha)).astype(np.uint8)
        cv2.line(frame, (x + 8, y + 28), (x + w - 8, y + 28), (174, 176, 178), 1)
        for index, name in enumerate(self.order):
            top = y + 32 + index * self.row_height
            baseline = top + 14
            active = name in self.active
            if active:
                cv2.rectangle(frame, (x + 3, top), (x + w - 3, top + 19),
                              (209, 210, 211), -1)
                cv2.circle(frame, (x + 9, top + 9), 3, (0, 0, 235), -1, cv2.LINE_AA)
            text_color = (45, 42, 40)
            cv2.putText(frame, str(index + 1), (x + 18, baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, .43, (100, 98, 96), 1, cv2.LINE_AA)
            sprite = self.name_sprites[name]
            draw_sprite(frame, sprite, x + 39, top + (self.row_height - sprite.shape[0]) // 2)
            seconds = f'{self.counts[name] / self.fps:05.2f}s'
            tw = cv2.getTextSize(seconds, cv2.FONT_HERSHEY_SIMPLEX, .43, 1)[0][0]
            assert x + 39 + sprite.shape[1] + 5 < x + w - 9 - tw
            cv2.putText(frame, seconds, (x + w - 9 - tw, baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, .43,
                        (0, 0, 210) if active else text_color, 1, cv2.LINE_AA)
