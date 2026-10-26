"""Render C-only red boxes, member labels and a frame-exact live leaderboard."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from landscape_live_ranking import MemberLeaderboard,MEMBERS,FONT

ROOT=Path(__file__).resolve().parents[1]
FFMPEG='/path/to/user/.local/bin/ffmpeg'
FONT_NAME=ImageFont.truetype(FONT,13)


def draw_c(frame,boxes,names):
    h,w=frame.shape[:2];rects=[]
    scale=h/480
    font=ImageFont.truetype(FONT,round(13*scale))
    boxes=[[round(b[0]*w/852),round(b[1]*h/480),round(b[2]*w/852),round(b[3]*h/480)] for b in boxes]
    for b,n in zip(boxes,names):
        x1,y1,x2,y2=b;cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),round(2*scale),cv2.LINE_AA)
        bw=round(font.getlength(n)+12*scale);bh=round(21*scale)
        x=max(2,min(w-2-bw,round((x1+x2-bw)/2)));y=max(2,y1-bh-round(2*scale))
        rects.append([x,y,x+bw,y+bh])
    if len(rects)==2:
        a,b=rects
        if a[2]+4>b[0] and a[1]<b[3] and b[1]<a[3]:
            move=(a[2]+4-b[0]+1)//2
            a[0]-=move;a[2]-=move;b[0]+=move;b[2]+=move
    for n,box,r in zip(names,boxes,rects):
        x,y,x2,y2=r
        if x<248*scale and x2>10*scale and y<254*scale and y2>64*scale:
            dy=round(258*scale)-y;y+=dy;y2+=dy;r[1]=y;r[3]=y2
        assert 0<=x<x2<w and 0<=y<y2<h
        im=Image.new('RGB',(x2-x,y2-y),(198,194,190));d=ImageDraw.Draw(im)
        d.text((round(6*scale),round(3*scale)),n,font=font,fill=(40,42,45))
        frame[y:y2,x:x2]=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
        if abs((x+x2-box[0]-box[2])/2)>5:
            cv2.line(frame,((x+x2)//2,y2),((box[0]+box[2])//2,box[1]),(0,0,240),1,cv2.LINE_AA)
    return rects


def metadata(rows,out,spec,counts,ranking):
    hist=Counter(len(r['members']) for r in rows[:spec['performance_end_frame']]);expected=Counter(n for r in rows for n in r['expected_members'])
    intervals=[]
    for names,g in itertools.groupby(rows,key=lambda r:tuple(r['members'])):
        a=list(g);s=a[0]['frame'];e=a[-1]['frame']+1
        intervals.append({'start_frame':s,'end_frame_exclusive':e,'start_seconds':s/30,'end_seconds':e/30,'members':list(names)})
    summary={'output':str(out/'NEON_横屏_C位红框_姓名与实时排行.mp4'),'source':spec['source'],
      'frames':len(rows),'fps':30,'width':spec.get('width',852),'height':spec.get('height',480),'audio_preserved':True,
      'selection_rules':spec['selection_rules'],'music_start_frame':spec.get('music_start_frame'),
      'chinese_names':{'AIJING':'盛艾敬','XILI':'梁淅沥','MERCI':'马西桐'},
      'portrait_reference':spec['portrait_video'],'alignment_offset_seconds':spec['offset_frames']/30,
      'name_provenance':spec['name_provenance'],'status':spec['status'],
      'timing_rule':'每个可见 C 位红框每帧给对应成员累计 1/30 秒；无框停表；双 C 分别累计；同分保持上一帧次序。',
      'member_c_frames':dict(counts),'member_c_seconds':{n:counts[n]/30 for n in MEMBERS},
      'final_order':ranking.order,'c_count_frames':dict(hist),
      'person_seconds':sum(counts.values())/30,'unboxed_performance_seconds':hist[0]/30,
      'unresolved_role_person_seconds':(sum(expected.values())-sum(counts.values()))/30,
      'performance_end_seconds':spec['performance_end_frame']/30}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (out/'timing_intervals.json').write_text(json.dumps(intervals,ensure_ascii=False,indent=2)+'\n')
    lines=['NEON 横屏 — C 位姓名、红框与实时排行榜','',
      '依据竖屏已复核的 C 位顺序，按音乐对齐后迁移到横屏；只为 C 位标名字、画红框。',
      '成员名称来自竖屏已确认标注及用户确认的表演顺序；未使用人脸识别。',
      '红框位置结合横屏人体检测、服装/站位和动作连续性校对；遮挡及交接仍可能存在误差。',
      '每个可见红框每帧累计 1/30 秒；双 C 分别累计，无框停表。','', '名次\t成员\tC 位时间(s)']
    for i,n in enumerate(ranking.order,1):lines.append(f'{i}\t{n}\t{counts[n]/30:.2f}')
    lines+=['',f'累计人秒：{sum(counts.values())/30:.2f}',f'表演内无红框画面：{hist[0]/30:.2f} 秒',
            f'参考角色因遮挡或定位缺失未计入：{summary["unresolved_role_person_seconds"]:.2f} 人秒',
            '计时逐帧一致不代表识别已达到逐帧人工标注的准确率。']
    (out/'C位成员时间统计.txt').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/neon-landscape/named-ranking-v1');p.add_argument('--preview-only',action='store_true');args=p.parse_args();out=args.output_dir.resolve()
    rows=[json.loads(l) for l in (out/'corrected_boxes.jsonl').read_text().splitlines()]
    config=out/'annotation_spec.json'
    if not config.exists():config=ROOT/'annotations/neon_landscape_member_transfer.json'
    spec=json.loads(config.read_text());ranking=MemberLeaderboard(scale=spec.get('height',480)/480);counts=Counter({n:0 for n in MEMBERS});rankrows=[]
    cap=cv2.VideoCapture(str(ROOT/spec['source']));assert int(cap.get(7))==len(rows)
    video=out/'NEON_横屏_C位红框_姓名与实时排行.mp4';process=None
    if not args.preview_only:
        if video.exists():raise FileExistsError(video)
        size=f"{spec.get('width',852)}x{spec.get('height',480)}"
        process=subprocess.Popen([FFMPEG,'-v','error','-nostdin','-n','-thread_queue_size','64','-f','rawvideo','-pix_fmt','bgr24','-s',size,'-r','30','-i','pipe:0','-i',str(ROOT/spec['source']),'-map','0:v','-map','1:a:0','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    previewdir=out/'review';previewdir.mkdir(exist_ok=True)
    previews={f for f in range(0,6695,60)}|{87,135,294,300,450,1414,1473,1800,2400,3537,3637,4800,6694,6800,7125};tiles=[]
    try:
        for row in rows:
            f=row['frame'];names=row['members'];boxes=row['boxes'];counts.update(names)
            rankingrow=ranking.step(f,boxes,names);rankrows.append(rankingrow)
            if process or f in previews:
                if not process:cap.set(cv2.CAP_PROP_POS_FRAMES,f)
                ok,frame=cap.read();assert ok
                ranking.draw(frame);rects=draw_c(frame,boxes,names)
                if f in previews:
                    cv2.imwrite(str(previewdir/f'preview-{f:05}.jpg'),frame,[cv2.IMWRITE_JPEG_QUALITY,95])
                    # Keep full-stage C boxes visible on body-crop contact sheets.
                    im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((852,480)).crop((0,245,852,450)).resize((680,164))
                    tiles.append((f,names,im))
                if process:process.stdin.write(frame.tobytes())
            if process and f%1500==0:print('Rendered',f,flush=True)
    finally:
        cap.release()
        if process:process.stdin.close()
    if process and process.wait():raise RuntimeError('Video encode failed')
    for batch in range((len(tiles)+14)//15):
        canvas=Image.new('RGB',(2040,5*190),'#171717');d=ImageDraw.Draw(canvas)
        for j,(f,names,im) in enumerate(tiles[batch*15:batch*15+15]):
            x=j%3*680;y=j//3*190;canvas.paste(im,(x,y+26));d.text((x+8,y+7),f'{f/30:.2f}s '+ ' + '.join(names),fill='white')
        canvas.save(previewdir/f'contact-{batch:02}.jpg',quality=94)
    (out/'ranking_frames.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rankrows))
    summary=metadata(rows,out,spec,counts,ranking);print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
