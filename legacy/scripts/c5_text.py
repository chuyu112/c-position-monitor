"""Cached Chinese display text using the user's supplied spellings."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHINESE_NAMES = json.loads((Path(__file__).resolve().parents[1]
                           / 'annotations/c5_member_names.json').read_text())


@lru_cache(maxsize=64)
def text_sprite(text, size=15):
    font = ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc', size)
    left, top, right, bottom = font.getbbox(text)
    im = Image.new('RGBA', (right - left + 2, bottom - top + 2))
    ImageDraw.Draw(im).text((1 - left, 1 - top), text, font=font, fill=(40, 42, 45, 255))
    return np.asarray(im)


def draw_sprite(frame, sprite, x, y):
    h, w = sprite.shape[:2]
    region = frame[y:y + h, x:x + w]
    assert region.shape[:2] == (h, w), (x, y, w, h)
    alpha = sprite[:, :, 3:4].astype(np.float32) / 255
    region[:] = np.rint(sprite[:, :, :3][:, :, ::-1] * alpha + region * (1 - alpha)).astype(np.uint8)
