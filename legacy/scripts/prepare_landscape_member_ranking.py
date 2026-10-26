"""Prepare the requested ranking design and an explicit position-label sheet."""

import json
from pathlib import Path

import cv2
from PIL import Image,ImageDraw,ImageFont

from landscape_live_ranking import MemberLeaderboard,MEMBERS,FONT


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/neon-landscape/member-ranking'
SOURCE=ROOT/'outputs/neon-landscape/source_30fps.mp4'


def main():
    OUT.mkdir(exist_ok=True)
    annotations=[json.loads(l) for l in (ROOT/'outputs/neon-landscape/final/frames.jsonl').read_text().splitlines()]
    detection=[json.loads(l) for l in (ROOT/'outputs/neon-landscape/detection/frames.jsonl').read_text().splitlines()]
    cap=cv2.VideoCapture(str(SOURCE))
    cap.set(cv2.CAP_PROP_POS_FRAMES,120);ok,frame=cap.read()
    assert ok
    original=frame.copy()
    for j,(x1,y1,x2,y2) in enumerate(annotations[120]['boxes']):
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2,cv2.LINE_AA)
        cv2.putText(frame,f'C{j+1}',(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1,cv2.LINE_AA)
    MemberLeaderboard().draw(frame)
    cv2.imwrite(str(OUT/'左侧成员排行榜_样式预览.png'),frame)
    # The labels below describe positions in this one image only. The user is
    # asked to supply the names directly, rather than comparing face photos.
    people=sorted(detection[120]['people'],key=lambda p:p['box'][0])
    assert len(people)==9
    image=Image.fromarray(cv2.cvtColor(original,cv2.COLOR_BGR2RGB)).resize((1704,960))
    draw=ImageDraw.Draw(image);font=ImageFont.truetype(FONT,25)
    numbered=[]
    for j,p in enumerate(people,1):
        x1,y1,x2,y2=[round(v*(1704 if k%2==0 else 960)) for k,v in enumerate(p['box'])]
        cx=round((x1+x2)/2)
        if j==6:
            # The detector's sixth box spans both centre performers here.
            # Narrow its numbered marker to the right-hand performer.
            cx=round(p['torso_x']*1704)
            x1,x2=cx-30,cx+30
        label=f'{j:02}'
        y=460 if j in (5,6) else 420
        draw.line((cx,y+36,cx,y1-4),fill=(255,207,50),width=3)
        draw.rectangle((cx-23,y,cx+23,y+35),fill=(255,207,50),outline=(30,30,30),width=2)
        draw.text((cx-17,y+2),label,font=font,fill=(20,20,20))
        draw.rectangle((x1,y1,x2,y2),outline=(255,207,50),width=2)
        numbered.append({'position':label,'frame':120,'seconds':4.0,'raw_track_id_at_reference':p['track_id'],
                         'name':None,'box_normalized':p['box']})
    footer=Image.new('RGB',(1704,1040),(247,247,247));footer.paste(image,(0,0));d=ImageDraw.Draw(footer)
    d.text((22,975),'横屏 00:04：请直接提供 01–09 的姓名。05 / 06 是中间左 / 右两位。',font=font,fill=(25,25,25))
    footer.save(OUT/'横屏人物位置_待填姓名.png')
    (OUT/'member_positions.template.json').write_text(json.dumps({'source':str(SOURCE),'members':MEMBERS,
        'reference_frame':120,'positions':numbered,'note':'User-supplied labels required. These are position labels, not inferred identities.'},ensure_ascii=False,indent=2)+'\n')
    cap.release()
    print(OUT)


if __name__=='__main__':main()
