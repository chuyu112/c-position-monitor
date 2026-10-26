"""Member leaderboard for the landscape video, using supplied name labels.

No names are inferred from faces or raw tracker IDs. A preview can show the
roster with pending times; an actual counter requires a name for every C box.
"""

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


MEMBERS = ['AIJING', 'XILI', 'OA', 'MERCI', 'AYUNE', 'YOUNG', 'VORA', 'RADA', 'HIMEKI']
DISPLAY = {'AIJING': 'AIJING 盛艾敬', 'XILI': 'XILI 梁淅沥', 'MERCI': 'MERCI 马西桐'}
FONT = '/System/Library/Fonts/STHeiti Medium.ttc'


class MemberLeaderboard:
    def __init__(self, fps=30, scale=1):
        self.scale = scale
        self.fps = fps
        self.order = MEMBERS[:]
        self.counts = Counter({name: 0 for name in MEMBERS})
        self.active = set()
        self.next_frame = 0
        self.ready = False
        self.x, self.y, self.width = 10, 64, 238
        self.row_height = 16
        self.height = 30 + 9*self.row_height + 16
        self.font = ImageFont.truetype(FONT, round(13*scale))
        self.title_font = ImageFont.truetype(FONT, round(16*scale))
        self.small_font = ImageFont.truetype(FONT, round(11*scale))

    def step(self, frame, boxes, names):
        if frame != self.next_frame:
            raise ValueError('Leaderboard frames must be consecutive from zero')
        if len(boxes) != len(names) or len(names) > 2 or len(set(names)) != len(names):
            raise ValueError('Each visible C box needs one distinct supplied member name')
        if any(name not in MEMBERS for name in names):
            raise ValueError('Unknown or missing member name; do not infer it from track IDs')
        self.active = set(names)
        self.counts.update(names)
        # Stable sorting leaves exact ties in their previous displayed order.
        self.order.sort(key=lambda name: -self.counts[name])
        self.ready = True
        self.next_frame += 1
        return {'frame': frame, 'active': names[:], 'order': self.order[:],
                'cumulative_frames': dict(self.counts)}

    def draw(self, frame):
        s = self.scale
        im = Image.new('RGB', (round(self.width*s), round(self.height*s)), (230, 229, 226))
        native = ImageDraw.Draw(im)
        class ScaledDraw:
            def rectangle(_, box, **kwargs): native.rectangle(tuple(round(v*s) for v in box), **kwargs)
            def line(_, box, **kwargs): native.line(tuple(round(v*s) for v in box), **kwargs)
            def ellipse(_, box, **kwargs): native.ellipse(tuple(round(v*s) for v in box), **kwargs)
            def text(_, point, text, **kwargs): native.text(tuple(round(v*s) for v in point), text, **kwargs)
            def textlength(_, text, **kwargs): return native.textlength(text, **kwargs)/s
        draw = ScaledDraw()
        draw.rectangle((0,0,self.width-1,self.height-1), outline=(179,181,181))
        draw.text((9,5), 'C 位实时排名', font=self.title_font, fill=(40,42,45))
        draw.line((8,28,self.width-9,28), fill=(174,176,178))
        for index,name in enumerate(self.order):
            y = 31 + index*self.row_height
            active = name in self.active
            if active:
                draw.rectangle((3,y-1,self.width-4,y+14), fill=(211,210,209))
                draw.ellipse((6,y+4,12,y+10), fill=(235,0,0))
            draw.text((18,y),str(index+1),font=self.font,fill=(98,100,102))
            draw.text((39,y),DISPLAY.get(name,name),font=self.font,fill=(40,42,45))
            value = f'{self.counts[name]/self.fps:05.2f}s' if self.ready else '—'
            length = draw.textlength(value,font=self.font)
            draw.text((self.width-9-length,y),value,font=self.font,
                      fill=(210,0,0) if active else (40,42,45))
        footer = '有红框才计时 · 双 C 分别累计' if self.ready else '样式预览 · 成员对应待补充'
        draw.text((8,self.height-15),footer,font=self.small_font,fill=(95,98,101))
        panel=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
        h,w=frame.shape[:2]
        x,y=round(self.x*s),round(self.y*s)
        ph,pw=panel.shape[:2]
        if x+pw>w or y+ph>h:
            raise ValueError('Leaderboard lies outside the video frame')
        frame[y:y+ph,x:x+pw]=panel
        return frame
