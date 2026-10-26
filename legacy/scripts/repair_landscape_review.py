"""Apply supplied corrections without replacing unrelated reviewed body tracks.

The portrait final supplies role order, while the horizontal user screenshots
override local handoffs. Body keyframes only repair explicitly reviewed gaps.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from c_detect.overlays import narrow_c_boxes,constrain_display_boxes
from c_detect.stabilization import _smooth

def read(path):return [json.loads(s) for s in path.read_text().splitlines()]
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

# Pixel body observations on the original 852 x 480 coordinate system.
# Interpolation is restricted to these reviewed, visible-body detection gaps.
KEYFRAMES={
 'AIJING':{128:[430,281,520,435],135:[434,281,526,435],142:[442,280,537,427],
           289:[460,274,540,451],294:[472,275,548,449],298:[472,279,552,451],302:[466,275,543,449]},
 'XILI':{87:[408,263,446,428],90:[405,269,448,432],94:[403,278,456,441],
         135:[403,278,465,432],
         285:[368,282,446,443],290:[386,274,451,445],294:[404,268,460,442],
         298:[408,268,468,441],302:[412,267,469,442],306:[413,268,467,443],311:[408,272,469,441]},
}

def body(name,f,detections,track):
    people=[p for p in detections[f]['people'] if p['track_id']==track]
    if people:
        p=people[0]
        return {'box':p['box'],'torso_x':p.get('torso_x'),'track_id':track,
                'method':'user_reviewed_local_body_continuity'}
    bounds=[]
    for direction in [-1,1]:
        for distance in range(1,10):
            q=f+direction*distance
            if not 0<=q<len(detections):break
            found=[p for p in detections[q]['people'] if p['track_id']==track]
            if found:bounds.append((q,found[0]));break
    if len(bounds)==2 and bounds[1][0]-bounds[0][0]<=10:
        (a,p),(b,q)=bounds;t=(f-a)/(b-a)
        box=np.array(p['box'])*(1-t)+np.array(q['box'])*t
        return {'box':box.tolist(),'torso_x':None,'track_id':track,
                'method':'reviewed_local_short_gap'}
    points=KEYFRAMES.get(name,{})
    before=[k for k in points if k<=f];after=[k for k in points if k>=f]
    if not before or not after:raise ValueError((name,f,'No reviewed gap bounds'))
    a,b=max(before),min(after)
    assert b-a<=12,(name,f,a,b)
    t=(f-a)/(b-a) if b>a else 0
    box=np.array(points[a])*(1-t)+np.array(points[b])*t
    return {'box':(box/np.array([852,480,852,480])).tolist(),
            'torso_x':float((box[0]+box[2])/1704),'track_id':None,
            'method':'user_reviewed_body_keyframes'}

def fill(rows,name,start,end,track,detections):
    missing=[f for f in range(start,end) if name not in rows[f]['members']]
    for _,g in itertools.groupby(enumerate(missing),key=lambda q:q[1]-q[0]):
        fs=[f for _,f in g];observations=[body(name,f,detections,track) for f in fs]
        boxes=np.array([narrow_c_boxes([p['box']],852,480,[p['torso_x']])[0] for p in observations],float)
        # Include the existing neighboring rectangles to join the repaired run.
        left=rows[fs[0]-1] if fs[0] else None
        right=rows[fs[-1]+1] if fs[-1]+1<len(rows) else None
        prefix=[left['boxes'][left['members'].index(name)]] if left and name in left['members'] else []
        suffix=[right['boxes'][right['members'].index(name)]] if right and name in right['members'] else []
        padded=np.array(prefix+boxes.tolist()+suffix,float)
        for k in range(4):padded[:,k]=_smooth(padded[:,k],1.1)
        boxes=padded[len(prefix):len(prefix)+len(fs)]
        for f,p,b in zip(fs,observations,boxes):
            row=rows[f];row['members'].append(name);row['observations'].append(p);row['boxes'].append(b.tolist())
            if name not in row['expected_members']:row['expected_members'].append(name)
            row.pop('collision',None)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/neon-landscape/named-ranking-v2-720p');args=parser.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    base=ROOT/'outputs/neon-landscape/named-ranking-v1';old=read(base/'corrected_boxes.jsonl');rows=deepcopy(old)
    detections=read(ROOT/'outputs/neon-landscape/detection/frames.jsonl')
    spec=json.loads((ROOT/'annotations/neon_landscape_member_transfer.json').read_text())
    if not (base/'annotation_spec.json').exists():save(base/'annotation_spec.json',spec)
    spec.update({'source':'outputs/neon-landscape/source_720p_30fps.mp4','width':1280,'height':720,
                 'music_start_frame':0,'movement_start_frame':0,
                 'portrait_annotations':'outputs/c5-red-name-aijing-himeki/corrected_boxes.jsonl',
                 'portrait_video':'/path/to/user/Documents/c5_C位标注_最终版.mp4'})
    spec['selection_rules']=[
        '开场从音乐起给 AIJING 盛艾敬画框和计时，不等待舞蹈动作开始。',
        '三人主表演组合只有中间一人为 C；两人主表演组合为双 C，分别计时。',
        '仍在主舞中的舞蹈转身保留 C；转身退场、持续背对观众退出主舞、静止陪衬或辅助动作不当 C。',
        '交接按已确认竖屏最终版的表演顺序，结合横屏实际动作；原主舞未结束时不提前撤框。',
        '用户在横屏截图直接确认的成员姓名及交接优先于旧标注；只给 C 位画红框、标名字。',
        '结尾定格及完全遮挡保持本次横屏已复核的停止计时边界。']
    fill(rows,'AIJING',0,681,1,detections)
    fill(rows,'XILI',87,681,13,detections)
    # The user confirms that the outgoing HIMEKI is still co-leading here.
    fill(rows,'HIMEKI',1390,1472,104,detections)
    # The front body's tracker ID was reused at this handoff. The user names
    # the incoming performer AYUNE; this is a local label correction only.
    for row in rows[1472:1515]:
        for key in ['members','expected_members']:
            row[key]=['AYUNE' if n=='HIMEKI' else n for n in row[key]]
        row['review_note']='User identifies the incoming C at 49.10 s as AYUNE.'
    # Incorporate the final portrait's later name corrections. Its short 81 s
    # geometric handoff is already covered by the horizontal local override.
    for start,end,before,after in [(3524,3554,'XILI','AIJING'),(3631,3642,'MERCI','HIMEKI')]:
        for row in rows[start:end]:
            for key in ['members','expected_members']:
                row[key]=[after if n==before else n for n in row[key]]
            row['review_note']='Latest user-confirmed portrait final; audio offset +27 frames.'
    changed=[]
    for f,row in enumerate(rows):
        if row==old[f]:continue
        if row['boxes']:
            row['boxes']=[list(b) for b in constrain_display_boxes(row['boxes'],852,480,gap=8)]
            order=sorted(range(len(row['members'])),key=lambda j:row['boxes'][j][0])
            for k in ['members','boxes','observations']:row[k]=[row[k][j] for j in order]
        changed.append(f)
    before=Counter(n for r in old for n in r['members']);after=Counter(n for r in rows for n in r['members'])
    intervals=[]
    for _,g in itertools.groupby(enumerate(changed),key=lambda q:q[1]-q[0]):
        fs=[f for _,f in g];intervals.append([fs[0],fs[-1]+1])
    manifest={'baseline':str(base/'corrected_boxes.jsonl'),
      'baseline_sha256':hashlib.sha256((base/'corrected_boxes.jsonl').read_bytes()).hexdigest(),
      'changed_frame_intervals':intervals,'member_frame_deltas':{n:after[n]-before[n] for n in after},
      'music_start_frame':0,'opening_continuous_duet':[87,681],
      'himeki_aijing_duet':[1390,1472],'incoming_ayune':[1472,1515],
      'portrait_final_name_corrections':[[3524,3554,'AIJING'],[3631,3642,'HIMEKI']],
      'source_quality':'Native 1280 x 720 Bilibili HTML5 rendition; no upscaling.',
      'keyframes':KEYFRAMES}
    spec['review_corrections']=manifest
    (out/'corrected_boxes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    save(out/'annotation_spec.json',spec);save(out/'corrections.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
