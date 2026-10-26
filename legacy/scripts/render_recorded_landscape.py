"""Render reviewed choreography labels on the user's high resolution recording.

The audio-matched copy shares the horizontal performance, so positions and
user-supplied labels come from its reviewed annotations, never from faces.
Decode by presentation timestamps: OpenCV random seeking is inaccurate on
this ScreenCaptureKit variable-frame-rate recording.
"""
import argparse
from collections import Counter
from functools import lru_cache
import itertools
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from landscape_live_ranking import MemberLeaderboard, MEMBERS, FONT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/neon-screen-recording'
SOURCE = Path('/path/to/user/Documents/全屏录屏/录屏_2026-10-19_12-10-10_4D8AA1.mp4')
BASE = ROOT / 'outputs/neon-landscape/named-ranking-v2-720p'
FFMPEG = '/path/to/user/.local/bin/ffmpeg'
START = 1.917375
FPS, WIDTH, HEIGHT, COUNT = 30, 3456, 1944, 6657
CHINESE = json.loads((ROOT / 'annotations/c5_member_names.json').read_text())
FLAGS = {'AIJING':['CN'], 'XILI':['CN'], 'OA':['US','JP'], 'MERCI':['CN'],
         'AYUNE':['KR'], 'YOUNG':['KR'], 'VORA':['KR'], 'RADA':['TH'], 'HIMEKI':['JP']}
FLAG_DIR = OUT / 'flags'
PANEL = (10,64,248,254)
BADGE_GAP = 2
PREVIEWS = set(range(0, COUNT, 120)) | {0, 87, 135, 294, 300, 450, 1390, 1414,
    1450, 1471, 1473, 1500, 1800, 2400, 3537, 3637, 4800, 6559, 6561, COUNT-4}

# Additional reviewed crossings and role handovers in the third version.
PREVIEWS |= {690, 1440, 1455, 1610, 1650, 2145, 2165, 2195, 2340, 2370, 2380, 2678, 2730, 2770, 2775, 2785, 2790, 2865, 2885, 3165, 3190, 3210, 3540, 3570, 3690, 3750, 3810, 4820, 4946, 5400, 5470, 6375, 6565}


@lru_cache(maxsize=32)
def badge(name, scale):
    font = ImageFont.truetype(FONT, round(13*scale))
    cn_font = ImageFont.truetype(FONT, round(12*scale))
    chinese = CHINESE.get(name)
    tw = max(font.getlength(name), cn_font.getlength(chinese) if chinese else 0)
    fw, fh, gap, pad = round(27*scale), round(20*scale), round(4*scale), round(5*scale)
    flagwidth = len(FLAGS[name])*fw + (len(FLAGS[name])-1)*gap
    bh = round((36 if chinese else 25)*scale)
    bw = round(3*pad + flagwidth + tw)
    im = Image.new('RGB', (bw,bh), (190,194,198))
    x = pad
    for code in FLAGS[name]:
        flag = Image.open(FLAG_DIR/f'{code}.png').convert('RGBA')
        flag = flag.crop(flag.getchannel('A').getbbox()).resize((fw,fh),Image.Resampling.LANCZOS)
        im.paste(flag,(x,(bh-fh)//2),flag);x += fw+gap
    x += pad-gap
    d = ImageDraw.Draw(im)
    d.text((x,round(3*scale)),name,font=font,fill=(40,42,45))
    if chinese:d.text((x,round(20*scale)),chinese,font=cn_font,fill=(40,42,45))
    return cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)


def intersects(a,b):
    return a[0]<b[2] and a[2]>b[0] and a[1]<b[3] and a[3]>b[1]


def draw_members(image, row):
    h,w = image.shape[:2];s = h/480
    boxes = [[round(b[0]*w/852),round(b[1]*h/480),round(b[2]*w/852),round(b[3]*h/480)] for b in row['boxes']]
    panel = [round(v*s) for v in PANEL]
    sprites = [badge(name,s) for name in row['members']]
    rects=[]
    for b,im in zip(boxes,sprites):
        x1,y1,x2,y2=b
        assert 0<=x1<x2<w and 0<=y1<y2<h,b
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),max(2,round(2*s)),cv2.LINE_AA)
        bh,bw=im.shape[:2]
        left=max(2,min(w-bw-2,round((x1+x2-bw)/2)))
        # A kneeling lead's badge must not cover the standing duet partner.
        # Keep both badges above the higher head, with leaders to their boxes.
        badge_anchor=min(b[1] for b in boxes) if len(boxes)==2 else y1
        top=max(2,badge_anchor-bh-round(BADGE_GAP*s));r=[left,top,left+bw,top+bh]
        if intersects(r,panel):
            left=panel[2]+round(5*s);r=[left,top,left+bw,top+bh]
        rects.append(r)
    if len(rects)==2 and intersects(rects[0],rects[1]):
        a,b=sorted(range(2),key=lambda j:boxes[j][0])
        gap=round(5*s);overlap=rects[a][2]+gap-rects[b][0]
        move=(overlap+1)//2
        rects[a][0]-=move;rects[a][2]-=move
        rects[b][0]+=move;rects[b][2]+=move
        # If the left badge would enter the leaderboard, stack it just above
        # the second badge; keep both badges outside the C bodies.
        if intersects(rects[a],panel):
            bh=rects[a][3]-rects[a][1]
            rects[a][3]=min(rects[a][3],rects[b][1]-gap)
            rects[a][1]=rects[a][3]-bh
    for im,r,b in zip(sprites,rects,boxes):
        x,y,x2,y2=r
        assert 0<=x<x2<w and 0<=y<y2<h,r
        assert not intersects(r,panel),('badge-panel overlap',r,panel)
        if abs((x+x2-b[0]-b[2])/2)>5*s or abs(y2-b[1])>8*s:
            cv2.line(image,((x+x2)//2,y2),((b[0]+b[2])//2,b[1]),(0,0,240),max(1,round(s)),cv2.LINE_AA)
        image[y:y2,x:x2]=im
    assert len(rects)<2 or not intersects(*rects),rects
    return boxes,rects


def prepare_rows():
    source=[json.loads(l) for l in (BASE/'corrected_boxes.jsonl').read_text().splitlines()]
    rows=[]
    for f in range(COUNT):
        # Across the PTS-decoded comparison anchors, the median video offset
        # is one reference frame beyond the audio-aligned output frame.
        ref=f+1
        row=dict(source[ref]);row['frame']=f;row['seconds']=f/FPS
        row['reference_frame']=ref;row['recording_seconds']=START+f/FPS
        rows.append(row)
    return rows


def write_reports(rows,ranking,rankrows,video,music_start_frame=0):
    counts=ranking.counts
    spec=json.loads((BASE/'annotation_spec.json').read_text())
    summary={'source':str(SOURCE),'output':str(video),'width':WIDTH,'height':HEIGHT,
      'fps':FPS,'frames':COUNT,'duration_seconds':COUNT/FPS,'recording_trim_start_seconds':START,
      'screen_crop_xywh':[0,178,WIDTH,HEIGHT],'music_start_frame':music_start_frame,
      'music_start_seconds':music_start_frame/FPS,
      'visual_reference_offset_frames':1,
      'reviewed_horizontal_annotations':str(BASE/'corrected_boxes.jsonl'),
      'portrait_reference':spec['portrait_video'],'selection_rules':spec['selection_rules'],
      'name_provenance':spec['name_provenance'],'chinese_names':CHINESE,'flag_codes':FLAGS,
      'flag_style':'System color emoji, same artwork style as portrait reference',
      'timing_rule':'Only visible C red boxes accrue time; each dual C accrues independently; stable sorting on ties.',
      'ranking_footer_visible':getattr(ranking,'show_footer',True),
      'member_c_frames':dict(counts),'member_c_seconds':{n:counts[n]/FPS for n in MEMBERS},
      'final_order':ranking.order,'person_seconds':sum(counts.values())/FPS,
      'c_count_frames':dict(Counter(len(r['members']) for r in rows)),
      'video_encoding':'HEVC VideoToolbox 50 Mbps, native cropped pixel dimensions, 30 fps',
      'audio_encoding':'Original recording audio, start-aligned and encoded AAC 256 kbps',
      'status':'First reviewable high-resolution version using previously reviewed role order and body positions.'}
    corrections=OUT/'corrections.json'
    if corrections.exists():
        summary['review_corrections']=json.loads(corrections.read_text())
        summary['status']='High-resolution review incorporating user screenshot corrections; all counters recomputed from visible boxes.'
        summary['reviewed_horizontal_annotations']=str(OUT/'reviewed_rows.jsonl')
        summary['selection_rules'].append(summary['review_corrections']['selection_rule'])
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    intervals=[]
    for names,g in itertools.groupby(rows,key=lambda r:tuple(r['members'])):
        segment=list(g);a=segment[0]['frame'];b=segment[-1]['frame']+1
        intervals.append({'start_frame':a,'end_frame_exclusive':b,'start_seconds':a/FPS,'end_seconds':b/FPS,'members':list(names)})
    (OUT/'timing_intervals.json').write_text(json.dumps(intervals,ensure_ascii=False,indent=2))
    (OUT/'corrected_boxes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (OUT/'ranking_frames.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rankrows))
    from recording_country_ranking import COUNTRIES
    lines=['NEON 横屏录屏高清版 — C 位时间统计','', f'从视频 {music_start_frame/FPS:.2f} 秒音乐起计时；此前不画框、不累计。只给 C 位画红框、标姓名和国旗。',
           '两人主表演为双 C，各自计时；三人主表演只选中间；舞蹈转身继续计时，退出主舞、静止陪衬和辅助不计时。',
           '结合双脚落点、身体前后移动和主舞角色判断前排。双 C 分别累计，因此总人秒可超过视频时长。','', '名次\t姓名\t国籍\tC位秒数']
    for i,n in enumerate(ranking.order,1):
        country='/'.join(COUNTRIES[c] for c in FLAGS[n])
        lines.append(f'{i}\t{n} {CHINESE.get(n,"")}\t{country}\t{counts[n]/FPS:.2f}')
    (OUT/'C位时间统计.txt').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    global OUT, PANEL, BADGE_GAP, PREVIEWS
    p=argparse.ArgumentParser();p.add_argument('--preview-only',action='store_true')
    p.add_argument('--output-dir',type=Path);p.add_argument('--rows',type=Path)
    p.add_argument('--country-ranking',action='store_true')
    p.add_argument('--hide-ranking-footer',action='store_true')
    p.add_argument('--music-start-seconds',type=float,default=0);args=p.parse_args()
    proxy=OUT/'review-proxy.mp4'
    if args.output_dir:OUT=args.output_dir
    (OUT/'review').mkdir(parents=True,exist_ok=True)
    rows=[json.loads(l) for l in args.rows.read_text().splitlines()] if args.rows else prepare_rows()
    assert len(rows)==COUNT
    music_start_frame=round(args.music_start_seconds*FPS)
    assert 0<=music_start_frame<COUNT
    assert all(not r['members'] and not r['boxes'] for r in rows[:music_start_frame]), 'Music pre-roll must not accrue C time'
    scale=(720 if args.preview_only else HEIGHT)/480
    if args.country_ranking:
        from recording_country_ranking import CountryLeaderboard
        ranking=CountryLeaderboard(FPS,scale,FLAGS,FLAG_DIR,show_footer=not args.hide_ranking_footer)
        PANEL=(ranking.x,ranking.y,ranking.x+ranking.width,ranking.y+ranking.height)
        BADGE_GAP=13
        PREVIEWS |= {4218,4225,4235,4245,4534,4540,4585,4600,4615,4660,4705,4720,
            4842,4860,4944,5030,5045,5060,5070,5090,5120,5140,5155,5910,5921,5924,5930,
            5970,6015,6030,6060,6090,6223,6240,6270,6310,6332}
    else:ranking=MemberLeaderboard(FPS,scale)
    rankrows=[]
    video=OUT/'NEON_横屏录屏高清_C位红框_国旗姓名排行榜.mp4'
    decoder=encoder=None;cap=None;tiles=[]
    if args.preview_only:cap=cv2.VideoCapture(str(proxy))
    else:
        if video.exists():raise FileExistsError(video)
        log=(OUT/'render.log').open('w')
        decoder=subprocess.Popen([FFMPEG,'-v','error','-nostdin','-hwaccel','videotoolbox','-ss',str(START),'-i',str(SOURCE),
            '-vf',f'crop={WIDTH}:{HEIGHT}:0:178,fps={FPS}','-frames:v',str(COUNT),'-an','-f','rawvideo','-pix_fmt','bgr24','pipe:1'],stdout=subprocess.PIPE,stderr=log)
        encoder=subprocess.Popen([FFMPEG,'-v','error','-nostdin','-n','-thread_queue_size','8',
            '-f','rawvideo','-pix_fmt','bgr24','-s',f'{WIDTH}x{HEIGHT}','-r',str(FPS),'-i','pipe:0',
            '-ss',str(START),'-i',str(SOURCE),'-map','0:v:0','-map','1:a:0',
            '-c:v','hevc_videotoolbox','-b:v','50M','-bf','0','-tag:v','hvc1','-pix_fmt','yuv420p',
            '-c:a','aac','-b:a','256k','-t',str(COUNT/FPS),'-movflags','+faststart',str(video)],stdin=subprocess.PIPE,stderr=log)
    begun=time.monotonic()
    try:
        for row in rows:
            f=row['frame'];rankrows.append(ranking.step(f,row['boxes'],row['members']))
            if args.preview_only and f not in PREVIEWS:continue
            if cap:
                cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,image=cap.read();assert ok,f
            else:
                raw=decoder.stdout.read(WIDTH*HEIGHT*3)
                if len(raw)!=WIDTH*HEIGHT*3:raise RuntimeError(f'Decoder stopped at {f}: {len(raw)} bytes')
                image=np.frombuffer(raw,dtype=np.uint8).reshape(HEIGHT,WIDTH,3).copy()
            ranking.draw(image);boxes,badges=draw_members(image,row)
            if f in PREVIEWS:
                cv2.imwrite(str(OUT/'review'/f'annotated-{f:05}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,95])
                small=cv2.resize(image,(852,480))
                im=Image.fromarray(cv2.cvtColor(small[220:465],cv2.COLOR_BGR2RGB)).resize((680,196))
                tiles.append((f,row['members'],im))
            if encoder:encoder.stdin.write(image.tobytes())
            if encoder and f%300==0:
                progress={'frame':f,'total':COUNT,'percent':round(100*f/COUNT,1),'elapsed_seconds':round(time.monotonic()-begun,1)}
                (OUT/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
    finally:
        if cap:cap.release()
        if decoder:decoder.stdout.close()
        if encoder:encoder.stdin.close()
    if decoder and decoder.wait():raise RuntimeError('Native decode failed; see render.log')
    if encoder and encoder.wait():raise RuntimeError('Encode failed; see render.log')
    for batch in range((len(tiles)+11)//12):
        canvas=Image.new('RGB',(2040,4*222),(20,20,20));d=ImageDraw.Draw(canvas)
        for j,(f,names,im) in enumerate(tiles[batch*12:batch*12+12]):
            x=j%3*680;y=j//3*222;canvas.paste(im,(x,y+26));d.text((x+8,y+6),f'{f/FPS:.2f}s  '+ ' + '.join(names),fill='white')
        canvas.save(OUT/'review'/f'annotated-contact-{batch:02}.jpg',quality=94)
    summary=write_reports(rows,ranking,rankrows,video,music_start_frame)
    print(json.dumps({'frames':COUNT,'order':summary['final_order'],'preview_only':args.preview_only},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
