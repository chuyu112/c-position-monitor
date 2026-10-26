"""Top-aligned nationalities for the reviewed recording; reuse frame counters."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from landscape_live_ranking import MemberLeaderboard, DISPLAY, FONT

COUNTRIES={'CN':'中国','US':'美国','JP':'日本','KR':'韩国','TH':'泰国'}

class CountryLeaderboard(MemberLeaderboard):
    def __init__(self,fps,scale,flags,flag_dir,show_footer=True):
        super().__init__(fps,scale)
        self.x,self.y,self.width=6,0,290
        self.show_footer=show_footer
        if not show_footer:self.height-=12
        self.flags=flags
        self.country_font=ImageFont.truetype(FONT,round(10*scale))
        self.flag_images={}
        for code in COUNTRIES:
            im=Image.open(flag_dir/f'{code}.png').convert('RGBA')
            self.flag_images[code]=im.crop(im.getchannel('A').getbbox()).resize(
                (round(16*scale),round(12*scale)),Image.Resampling.LANCZOS)

    def draw(self,frame):
        s=self.scale
        im=Image.new('RGB',(round(self.width*s),round(self.height*s)),(230,229,226))
        d=ImageDraw.Draw(im)
        def xy(*args):return tuple(round(v*s) for v in args)
        d.rectangle((0,0,im.width-1,im.height-1),outline=(179,181,181))
        d.text(xy(9,5),'C 位实时排名',font=self.title_font,fill=(40,42,45))
        d.line(xy(8,28,self.width-9,28),fill=(174,176,178))
        for i,n in enumerate(self.order):
            y=31+i*self.row_height
            if n in self.active:
                d.rectangle(xy(3,y-1,self.width-4,y+14),fill=(211,210,209))
                d.ellipse(xy(5,y+4,11,y+10),fill=(235,0,0))
            d.text(xy(16,y),str(i+1),font=self.font,fill=(98,100,102))
            d.text(xy(34,y),DISPLAY.get(n,n),font=self.font,fill=(40,42,45))
            codes=self.flags[n]
            x=141 if len(codes)==2 else 158
            for code in codes:
                flag=self.flag_images[code];im.paste(flag,xy(x,y+1),flag);x+=17
            d.text(xy(178,y+2),'/'.join(COUNTRIES[c] for c in codes),
                   font=self.country_font,fill=(65,68,70))
            text=f'{self.counts[n]/self.fps:05.2f}s'
            d.text((round((self.width-8)*s)-d.textlength(text,font=self.font),round(y*s)),
                   text,font=self.font,fill=(210,0,0) if n in self.active else (40,42,45))
        if self.show_footer:
            d.text(xy(8,self.height-15),'有红框才计时 · 双 C 分别累计',font=self.small_font,fill=(95,98,101))
        panel=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
        x,y=xy(self.x,self.y);frame[y:y+im.height,x:x+im.width]=panel
        return frame
