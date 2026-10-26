"""Keep duet markers readable during crossings and review two local motions.

All identities and role decisions are supplied by the user. Display layout
may move a marker a little outward; the reviewed body anchor is preserved.
"""
from collections import Counter
from copy import deepcopy
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/neon-screen-recording-v5'
OUT=ROOT/'outputs/neon-screen-recording-v6'
old=[json.loads(l) for l in (BASE/'corrected_boxes.jsonl').read_text().splitlines()]
rows=deepcopy(old);notes=[]

def head_motion(name,start,end,points,reason):
    keys=sorted(points)
    for f in range(start,end):
        x,top,bottom,width=[np.interp(f,keys,[points[k][j] for k in keys]) for j in range(4)]
        b=(np.array([x-width/2,top,x+width/2,bottom])*[852/1280,480/720,852/1280,480/720]).tolist()
        r=rows[f];j=r['members'].index(name)
        r['boxes'][j]=b;r['observations'][j]={'box':(np.array(b)/[852,480,852,480]).tolist(),
          'method':'user_reviewed_head_and_body_motion','evidence':reason}
        r['review_note']=reason
        r.pop('body_boxes_before_spacing',None)
    notes.append({'frames':[start,end],'member':name,'head_body_keyframes_720p':points,'reason':reason})

# The song lead changes as OA turns out of the main performance, before
# the old handover. AYUNE's new body anchor continues into her existing solo.
for r in rows[2468:2477]:
    r['members']=['AYUNE'];r['expected_members']=['AYUNE'];r['role_review']='用户确认：OA 转身退出主表演后由 AYUNE 接 C。'
notes.append({'frames':[2468,2477],'members':['AYUNE'],'reason':'用户截图确认 OA 已转身，主 C 应为 AYUNE；交接提前到 82.267 秒。'})
head_motion('AYUNE',2468,2491,{
2468:[652,395,633,44],2472:[650,395,633,44],2476:[650,394,633,44],
2480:[650,391,633,44],2485:[650,388,634,44],2490:[651,387,631,44]},
'01:22 OA 转身退入辅助，红框移交给中间正面接唱的 AYUNE，随她走到前面。')

head_motion('AIJING',4115,4180,{
4114:[651.3,420,655.5,44],4115:[652,415,634,44],4120:[653,414,635,44],
4125:[648,414,635,44],4130:[643,411,635,44],4135:[632,390,598,44],
4140:[647,381,600,44],4145:[648,383,635,44],4146:[650,384,635,44],
4150:[649,389,632,44],4155:[640,391,631,44],4160:[631,379,631,44],
4165:[632,389,634,44],4170:[632,400,634,44],4175:[631,392,600,44],
4179:[634,415,635,44],4180:[634.8,420,639,40.5]},
'02:18 AIJING 起跳、转身时连续跟随头部和身体，修正框偏左及框顶落在头部以下。')

# During the kneeling crossing, the old torso centers crossed later than
# the visible heads. Keep the two user-identified performers correctly
# assigned to the left/right markers when restoring normal width.
head_motion('AYUNE',2721,2745,{
2720:[661,407,632,42],2725:[670,402,633,42],2729:[669,400,634,42],
2733:[670,399,634,42],2737:[672,400,633,42],2741:[670,400,633,42],
2745:[667.5,400,633,42]},
'01:31 AYUNE 站立、AIJING 下蹲交叉时，按两人的可见头部校准左右关系，避免宽框偏到邻人。')
head_motion('AIJING',2721,2745,{
2720:[706,496,670,42],2725:[701,483,672,42],2729:[682,481,674,42],
2733:[655,474,669,42],2737:[636,472,662,42],2741:[626,474,655,42],
2745:[622,478.5,653,42]},
'01:31 AIJING 下蹲后向左移动，框随可见头部及时换侧，不继续停在 AYUNE 的同一条竖线上。')

# Restore pre-compression anchors, then widen the outer edges instead of
# crushing both frames into strips. A short median reduces width jitter
# when the body is in profile, without smoothing the person's position.
raw={}
for r in rows:
    if len(r['boxes'])==2:
        raw[r['frame']]=deepcopy(r.get('body_boxes_before_spacing',r['boxes']))
layout=[];minimum=28.;gap=4.;widths=[];shifts=[]
for f,bodies in raw.items():
    r=rows[f];target=[]
    for j,name in enumerate(r['members']):
        local=[]
        for k in range(max(0,f-5),min(len(rows),f+6)):
            if k in raw and set(rows[k]['members'])==set(r['members']):
                q=raw[k][rows[k]['members'].index(name)];local.append(q[2]-q[0])
        target.append(max(minimum,float(np.median(local))))
    boxes=deepcopy(bodies)
    centers=[(b[0]+b[2])/2 for b in bodies]
    i,j=sorted(range(2),key=lambda k:centers[k])
    c1,c2=centers[i],centers[j]
    need=(target[i]+target[j])/2+gap
    delta=max(0.,(need-(c2-c1))/2)
    displayed=centers[:];displayed[i]-=delta;displayed[j]+=delta
    for k in range(2):
        boxes[k][0]=displayed[k]-target[k]/2;boxes[k][2]=displayed[k]+target[k]/2
        widths.append(target[k]);shifts.append(abs(displayed[k]-centers[k]))
    r['body_boxes_before_spacing']=bodies
    r['boxes']=boxes
    r['display_spacing']='双 C 至少保持正常可读框宽；靠近时两框适度向外展开并留间隔，不压成细条。'
    r['display_width_floor']=minimum
    if boxes!=old[f]['boxes']:layout.append(f)

for f,r in enumerate(rows):
    assert len(r['members'])==len(r['boxes'])==len(r['observations'])
    assert (not r['members']) if f<60 else 1<=len(r['members'])<=2
    for b in r['boxes']:assert 0<=b[0]<b[2]<852 and 0<=b[1]<b[3]<480,(f,b)
    if len(r['boxes'])==2:
        a,b=sorted(r['boxes']);assert a[2]+3.999<=b[0],f
        assert min(q[2]-q[0] for q in r['boxes'])>=minimum-1e-8,f
    if not 2468<=f<2477:assert r['members']==old[f]['members'],f
before=Counter(n for r in old for n in r['members']);after=Counter(n for r in rows for n in r['members'])
deltas={n:after[n]-before[n] for n in before}
assert {n:d for n,d in deltas.items() if d}=={'OA':-9,'AYUNE':9},deltas
turn_checks=[(270,360,['AIJING','XILI']),(4115,4281,['AIJING']),
             (2468,2621,['AYUNE']),(4796,4944,['OA','YOUNG'])]
for a,b,names in turn_checks:
    assert all(set(r['members'])==set(names) for r in rows[a:b]),(a,b)
# User's explicit brief full occlusion remains the only missing AYUNE box
# inside the long AIJING / AYUNE duet, without removing the visible AIJING.
assert all(r['members']==['AIJING'] for r in rows[2773:2788])
assert all(set(r['members'])=={'AIJING','AYUNE'} for r in rows[2621:2773]+rows[2788:2895])
changed=[f for f in range(len(rows)) if rows[f]!=old[f]]
manifest={'baseline':str(BASE/'corrected_boxes.jsonl'),
 'baseline_sha256':hashlib.sha256((BASE/'corrected_boxes.jsonl').read_bytes()).hexdigest(),
 'corrections':notes,'changed_frames':len(changed),'duet_layout_frames':len(layout),
 'minimum_duet_width_annotation_pixels':minimum,'duet_gap_annotation_pixels':gap,
 'maximum_outward_display_shift_annotation_pixels':max(shifts),
 'member_frame_deltas':deltas,'music_start_frame':60,
 'selection_rule':'舞蹈转身连续保留 C；OA 在约 1:22 转身退出主表演时交给 AYUNE。双框保持可读宽度并留间隔，不压成细线。',
 'prior_corrections_preserved':True}
validation={'frames':len(rows),'music_start_frame':60,'pre_music_zero_frames':60,
 'empty_stage_frames_after_music':0,'nonoverlapping_duet_boxes':True,
 'minimum_duet_width_native_pixels':min(round(b[2]*3456/852)-round(b[0]*3456/852) for r in rows if len(r['boxes'])==2 for b in r['boxes']),
 'counter_totals':dict(after),'member_frame_deltas':deltas,
 'turn_continuity_intervals':[{'frames':[a,b],'members':n} for a,b,n in turn_checks],
 'full_occlusion_exception_frames':[2773,2788]}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'reviewed_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
for name,val in [('corrections.json',manifest),('annotation-validation.json',validation)]:
 (OUT/name).write_text(json.dumps(val,ensure_ascii=False,indent=2)+'\n')
(OUT/'第六版修订记录.txt').write_text('''第六版：双人红框宽度、转身交接与定位复核。

双 C 红框设定最低可读宽度，靠近时适度向外展开，保持并排留间隔；不再压成细条。重点复核 00:23、01:19、01:29–01:33、02:41。
约 01:22.267：OA 转身退出主表演后交给 AYUNE，提前 0.30 秒切换，重新计算两人的累计。
约 02:18：重新校准 AIJING 起跳、转身时的头部、身体及框顶位置。
舞蹈动作中的转身保持 C 位及红框连续；退出主表演才交接。保留 AYUNE 完全被 AIJING 遮挡时的短暂隐藏例外。
第 2 秒音乐起计时；保留前五版其他已确认角色与交接。
原生 3456×1944、30 fps、221.9 秒，原声和视频时间线不变。
''')
print(json.dumps({'changed_frames':len(changed),'duet_layout_frames':len(layout),'min_duet_width':minimum,'counter_deltas':deltas},ensure_ascii=False))
