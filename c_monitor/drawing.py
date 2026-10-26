"""Shared red boxes, supplied member badges and stable live leaderboard."""

from functools import lru_cache
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .dataset import DATA

COUNTRIES = {"CN": "中国", "US": "美国", "JP": "日本", "KR": "韩国", "TH": "泰国"}


def find_font(explicit=None):
    requested = explicit or os.environ.get("C_MONITOR_FONT")
    if requested:
        if not Path(requested).is_file():
            raise ValueError("Font does not exist: " + str(requested))
        return str(requested)
    candidates = ["/System/Library/Fonts/STHeiti Medium.ttc",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                  "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
                  "C:/Windows/Fonts/msyh.ttc"]
    for path in candidates:
        if Path(path).is_file():
            return path
    raise ValueError("Chinese font required: pass --font or set C_MONITOR_FONT")


def intersects(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


class Overlay:
    def __init__(self, profile, width, height, font):
        self.profile, self.width, self.height = profile, width, height
        self.portrait = profile["name"] == "portrait"
        self.scale = width / 720 if self.portrait else height / 480
        self.font_path = font
        # Adapt the original portrait and landscape panel placements.
        self.panel = (12, 76, 224, 218) if self.portrait else (6, 0, 290, 178)
        self.background, self.ink = (230, 229, 226), (40, 42, 45)
        self.flags = {}
        for codes in profile.get("flags", {}).values():
            for code in codes:
                if code not in self.flags:
                    im = Image.open(DATA / "flags" / (code + ".png")).convert("RGBA")
                    self.flags[code] = im.crop(im.getchannel("A").getbbox())
        self.panel_rect = [round(v * self.scale) for v in
                           (self.panel[0], self.panel[1], self.panel[0] + self.panel[2], self.panel[1] + self.panel[3])]
        if self.panel_rect[2] > width or self.panel_rect[3] > height or self.scale < .25:
            raise ValueError("Output too small for readable ranking; increase --height")

    @lru_cache(maxsize=16)
    def font(self, size):
        return ImageFont.truetype(self.font_path, max(1, round(size * self.scale)))

    @lru_cache(maxsize=32)
    def flag(self, code, width, height):
        return self.flags[code].resize((width, height), Image.Resampling.LANCZOS)

    @lru_cache(maxsize=16)
    def badge(self, name):
        s = self.scale
        font, small = self.font(13), self.font(12)
        chinese = self.profile.get("chinese_names", {}).get(name, "")
        fw, fh, gap, pad = [max(1, round(v * s)) for v in (27, 20, 4, 5)]
        codes = self.profile.get("flags", {}).get(name, [])
        flag_width = len(codes) * (fw + gap)
        text_width = max(font.getlength(name), small.getlength(chinese))
        im = Image.new("RGB", (round(3 * pad + flag_width + text_width), round((36 if chinese else 25) * s)), (190, 194, 198))
        x = pad
        for code in codes:
            flag = self.flag(code, fw, fh)
            im.paste(flag, (x, (im.height - fh) // 2), flag)
            x += fw + gap
        draw = ImageDraw.Draw(im)
        draw.text((x + pad, round(3 * s)), name, font=font, fill=self.ink)
        if chinese:
            draw.text((x + pad, round(20 * s)), chinese, font=small, fill=self.ink)
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    def draw_panel(self, image, counter):
        s = self.scale
        _, _, width, height = self.panel
        im = Image.new("RGB", (round(width * s), round(height * s)), self.background)
        draw = ImageDraw.Draw(im)

        def xy(*values):
            return tuple(round(v * s) for v in values)

        draw.rectangle((0, 0, im.width - 1, im.height - 1), outline=(179, 181, 181))
        draw.text(xy(9, 5), "C 位实时排名", font=self.font(16), fill=self.ink)
        draw.line(xy(8, 28, width - 9, 28), fill=(174, 176, 178))
        row_height = 20 if self.portrait else 16
        font = self.font(14 if self.portrait else 13)
        for rank, name in enumerate(counter.order):
            y = 32 + rank * row_height
            active = name in counter.active
            if active:
                draw.rectangle(xy(3, y - 1, width - 4, y + row_height - 2), fill=(211, 210, 209))
                draw.ellipse(xy(5, y + 4, 11, y + 10), fill=(235, 0, 0))
            draw.text(xy(16, y), str(rank + 1), font=font, fill=(98, 100, 102))
            display = name
            chinese = self.profile.get("chinese_names", {}).get(name)
            if chinese:
                display += " " + chinese
            draw.text(xy(34 if not self.portrait else 39, y), display, font=font, fill=self.ink)
            if not self.portrait:
                codes = self.profile.get("flags", {}).get(name, [])
                x = 141 if len(codes) == 2 else 158
                for code in codes:
                    flag = self.flag(code, max(1, round(16 * s)), max(1, round(12 * s)))
                    im.paste(flag, xy(x, y + 1), flag)
                    x += 17
                draw.text(xy(178, y + 2), "/".join(COUNTRIES[c] for c in codes), font=self.font(10), fill=(65, 68, 70))
            value = f"{counter.counts[name] / counter.fps:05.2f}s"
            draw.text((round((width - 8) * s) - draw.textlength(value, font=font), round(y * s)), value,
                      font=font, fill=(210, 0, 0) if active else self.ink)
        x, y = self.panel_rect[:2]
        image[y:y + im.height, x:x + im.width] = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    def draw(self, image, row, counter):
        aw, ah = self.profile["annotation_size"]
        boxes = [[round(x1 * self.width / aw), round(y1 * self.height / ah),
                  round(x2 * self.width / aw), round(y2 * self.height / ah)] for x1, y1, x2, y2 in row["boxes"]]
        occupied = [self.panel_rect]
        for name, box in zip(row["members"], boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), max(1, round(2 * self.scale)), cv2.LINE_AA)
            sprite = self.badge(name)
            bh, bw = sprite.shape[:2]
            if bw + 4 > self.width or bh + 4 > self.height:
                raise ValueError("Member badge does not fit output canvas")
            left = max(2, min(self.width - bw - 2, round((x1 + x2 - bw) / 2)))
            top = max(2, y1 - bh - max(2, round(4 * self.scale)))
            # Keep labels out of the ranking and each other, moving upward first.
            for _ in range(20):
                rect = [left, top, left + bw, top + bh]
                collisions = [r for r in occupied if intersects(rect, r)]
                if not collisions:
                    break
                if self.panel_rect in collisions and self.panel_rect[2] + 3 + bw < self.width:
                    left = self.panel_rect[2] + 3
                    continue
                new_top = min(r[1] for r in collisions) - bh - 3
                if new_top >= 2:
                    top = new_top
                else:
                    left = max(r[2] for r in collisions) + 3
                    if left + bw >= self.width:
                        left = 2
                        top = max(r[3] for r in collisions) + 3
                if top + bh >= self.height:
                    raise ValueError("No room for member badge at frame " + str(row["frame"]))
            else:
                raise ValueError("Cannot resolve member badge collision")
            if abs((left + bw / 2) - (x1 + x2) / 2) > 5 * self.scale or abs(top + bh - y1) > 8 * self.scale:
                cv2.line(image, (left + bw // 2, top + bh), ((x1 + x2) // 2, y1), (0, 0, 240), max(1, round(self.scale)), cv2.LINE_AA)
            image[top:top + bh, left:left + bw] = sprite
            occupied.append([left, top, left + bw, top + bh])
        # The opaque ranking stays readable even when a body or leader crosses it.
        self.draw_panel(image, counter)
        return image
