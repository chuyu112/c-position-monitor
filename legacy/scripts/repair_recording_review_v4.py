"""Apply screenshot corrections to exact CFR output frames; preserve v1 and v2.

Labels are supplied by the user. Local body motion, head position and feet
resolve placement, without face identity matching. All display keyframes are
in the clean 1280 x 720 CFR review copy and scale to native output pixels.
"""
from collections import Counter
from copy import deepcopy
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from c_detect.overlays import narrow_c_boxes,constrain_display_boxes
from c_detect.stabilization import _smooth
BASE=ROOT/'outputs/neon-screen-recording-v3'
OUT=ROOT/'outputs/neon-screen-recording-v4'

def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
old=read(BASE/'corrected_boxes.jsonl');rows=deepcopy(old)
dets=read(ROOT/'outputs/neon-landscape/detection/frames.jsonl')
notes=[]
def rename(a,b,mapping,reason):
    for r in rows[a:b]:
        for k in ['members','expected_members']:r[k]=[mapping.get(n,n) for n in r[k]]
        r['review_note']=reason
    notes.append({'frames':[a,b],'rename':mapping,'reason':reason})

def set_member(f,n,box,evidence,replace=False):
    r=rows[f]
    if replace:
        r['members']=[];r['boxes']=[];r['observations']=[];r['expected_members']=[]
    ob={'box':(np.array(box)/[852,480,852,480]).tolist(),
        'method':'user_reviewed_spatial_motion','evidence':evidence}
    if n in r['members']:
        j=r['members'].index(n);r['boxes'][j]=list(box);r['observations'][j]=ob
    else:
        r['members'].append(n);r['boxes'].append(list(box));r['observations'].append(ob)
    if n not in r['expected_members']:r['expected_members'].append(n)
    r['review_note']=evidence;r.pop('collision',None)

def lerp(points,f):
    keys=sorted(points)
    return np.array([np.interp(f,keys,[points[k][j] for k in keys]) for j in range(len(points[keys[0]]))])

def head_keyframes(name,points,a,b,reason,only_missing=False,replace=False):
    # center x, top of head, lower body bound, marker width in 720p pixels.
    for f in range(a,b):
        if only_missing and name in rows[f]['members']:continue
        x,top,bottom,width=lerp(points,f)
        box=np.array([x-width/2,top,x+width/2,bottom])*[852/1280,480/720,852/1280,480/720]
        set_member(f,name,box.tolist(),reason,replace)
    notes.append({'frames':[a,b],'member':name,'head_body_keyframes_720p':points,'reason':reason})

def track_body(name,a,b,ids,xkeys,reason,replace=False):
    observations=[]
    for f in range(a,b):
        x=float(lerp(xkeys,f)[0]);people=[p for p in dets[f+1]['people'] if p['track_id'] in ids and abs(p['torso_x']*1280-x)<55]
        p=min(people,key=lambda p:abs(p['torso_x']*1280-x)) if people else None
        observations.append(p)
    valid=[i for i,p in enumerate(observations) if p]
    assert valid
    boxes=[]
    for i,p in enumerate(observations):
        if p:
            box=narrow_c_boxes([p['box']],852,480,[p['torso_x']])[0]
        else:
            before=max([k for k in valid if k<i],default=valid[0]);after=min([k for k in valid if k>i],default=valid[-1])
            assert abs(after-before)<=20,(name,a+i,before,after)
            pa,pb=observations[before],observations[after]
            t=(i-before)/(after-before) if before!=after else 0
            bounds=np.array(pa['box'])*(1-t)+np.array(pb['box'])*t
            torso=pa['torso_x']*(1-t)+pb['torso_x']*t
            box=narrow_c_boxes([bounds],852,480,[torso])[0]
        boxes.append(box)
    boxes=np.array(boxes,float)
    for j in range(4):boxes[:,j]=_smooth(boxes[:,j],1.1)
    for f,box in zip(range(a,b),boxes):set_member(f,name,box.tolist(),reason,replace)
    notes.append({'frames':[a,b],'member':name,'local_tracks':ids,'spatial_anchors_720p':xkeys,'reason':reason})

# Role and drawing visibility are independent: occlusion does not transfer C.
def role(a,b,names,reason):
    for r in rows[a:b]:
        r['expected_members']=list(names)
        keep=[j for j,n in enumerate(r['members']) if n in names]
        for k in ['members','boxes','observations']:r[k]=[r[k][j] for j in keep]
        r['role_review']=reason
    notes.append({'frames':[a,b],'role_members':names,'reason':reason})

def hide(a,b,n,reason):
    for r in rows[a:b]:
        if n in r['members']:
            j=r['members'].index(n)
            for k in ['members','boxes','observations']:r[k].pop(j)
        r.setdefault('temporarily_hidden_members',[]).append(n)
        r['visibility_review']=reason
    notes.append({'frames':[a,b],'hidden_member':n,'reason':reason})

# 00:47: retain the body on the left through the crossing.
track_body('HIMEKI',1388,1426,[104],{1388:[465],1395:[440],1400:[429],1407:[423],1415:[432],1420:[444],1425:[484]},
 '用户指出 00:47 HIMEKI 框偏了一位；沿左前排原身体走位重新跟踪，再衔接已修正的 00:48 交叉。')
role(1514,1620,['AYUNE','XILI'],'用户确认 00:54 仍是金字塔中间 AYUNE 与最右侧 XILI 双 C；队形解开前不提前改为 YOUNG。')
head_keyframes('AYUNE',{1545:[621,398,625,40],1550:[620,397,628,40],1556:[633,394,629,40],
 1562:[637,397,630,40],1568:[634,400,631,40],1574:[631,403,631,40],1580:[625,402,631,40],
 1586:[632,406,631,40],1592:[632,406,632,40],1598:[633,404,631,40],1604:[629,400,631,40],
 1610:[633,398,631,40],1616:[629,396,631,40],1619:[618,395,632,40]},1545,1620,
 '00:52–00:54 以用户指定金字塔中间 AYUNE 的头部为锚点，红框左移回本人。')
track_body('XILI',1590,1620,[314],{1590:[827],1600:[826],1610:[824],1619:[817]},
 '补回金字塔队形最右侧 XILI，直到双人主表演实际结束。')
# AIJING shifts left of OA, returns, then travels left at the handover.
head_keyframes('AIJING',{4450:[614,379,627,40],4455:[619,384,635,40],4460:[613,398,631,40],
 4465:[608,405,630,40],4470:[607,411,631,40],4475:[614,416,633,40],4480:[616,386,632,40],
 4485:[635,400,649,40],4490:[624,402,650,40],4495:[651,387,650,40],4500:[649,399,650,40],
 4505:[655,389,650,40],4510:[649,389,650,40],4515:[644,405,642,40],4520:[605,390,636,40],
 4525:[555,388,639,40],4530:[512,394,643,40],4533:[493,391,642,40]},4450,4534,
 '用户指出 02:29 AIJING 框落在 OA 身上；按用户指定黄发长袖成员的连续身体运动修正，包含退场交接。')
track_body('VORA',5501,5614,[1576],{5501:[758],5511:[737],5521:[705],5531:[694],5541:[689],
 5551:[687],5561:[678],5571:[672],5581:[645],5591:[654],5601:[671],5611:[746],5613:[749]},
 '用户指出 03:04 VORA 右框未对准；连续跟随短发短袖长裤身体，不跳到后面的长发辅助者。')
head_keyframes('OA',{4810:[625,394,616,40],4815:[619,392,616,40],4820:[619,390,616,40],
 4825:[621,394,616,40],4830:[636,402,616,40],4835:[648,415,616,40],4840:[641,410,616,40],
 4845:[611,397,616,40]},4810,4846,
 '02:41 双人贴近时按 OA 头部与身体位置定位，避免躯干交叉导致两个框挤到同一人。')
head_keyframes('YOUNG',{4810:[663,392,620,40],4815:[662,384,620,40],4820:[659,388,620,40],
 4825:[655,390,620,40],4830:[664,387,620,40],4835:[679,395,620,40],4840:[672,391,620,40],
 4845:[651,390,620,40]},4810,4846,
 '02:41 YOUNG 侧倾时保持框对准头部，不以偏向 OA 的躯干中心作锚点。')
notes.append({'frames':[5924,6104],'role_members':['RADA'],'reason':'用户撤回 03:22 双 C 的说法，最新确认只有 RADA；保留第三版此段单 C。','unchanged':True})

# Display-only spacing, with original body coordinates preserved for audit.
layout=[]
for r in rows:
 if len(r['boxes'])!=2:continue
 order=sorted(range(2),key=lambda j:sum(r['boxes'][j][::2]))
 i,j=order;left,right=np.array(r['boxes'][i],float),np.array(r['boxes'][j],float)
 gap=4.0
 if left[2]+gap<=right[0]:continue
 r['body_boxes_before_spacing']=deepcopy(r['boxes'])
 c1=(left[0]+left[2])/2;c2=(right[0]+right[2])/2
 half1=(left[2]-left[0])/2;half2=(right[2]-right[0])/2;mid=(c1+c2)/2
 # At native resolution: minimum width 40 px, gap 16 px.
 if c2-c1<14.0:c1=mid-7;c2=mid+7
 room=(c2-c1-gap)/2;h1=min(half1,room);h2=min(half2,room)
 left[[0,2]]=[c1-h1,c1+h1];right[[0,2]]=[c2-h2,c2+h2]
 r['boxes'][i]=left.tolist();r['boxes'][j]=right.tolist()
 r['display_spacing']='用户要求双框并排不重合；靠近时收窄，身体轨迹和计时不变。'
 layout.append(r['frame'])
notes.append({'display_spacing_frames':layout,'reason':'双 C 红框并排留间隙；只调整显示宽度和必要的少量横向错开，不改变成员或计时。'})
changed=[]
for f,r in enumerate(rows):
 if r!=old[f]:changed.append(f)
 assert 1<=len(r['members'])==len(set(r['members']))==len(r['boxes'])<=2
 assert set(r['expected_members'])-set(r['members'])==set(r.get('temporarily_hidden_members',[]))
 for b in r['boxes']:assert 0<=b[0]<b[2]<852 and 0<=b[1]<b[3]<480,(f,b)
 if len(r['boxes'])==2:
  a,b=sorted(r['boxes']);assert a[2]+3.99<=b[0],(f,a,b)
before=Counter(n for r in old for n in r['members']);after=Counter(n for r in rows for n in r['members'])
# Placement/layout fixes do not change clocks; only the pyramid handover does.
assert {n:after[n]-before[n] for n in before if after[n]!=before[n]}=={'XILI':30,'AYUNE':13,'YOUNG':-13}
assert rows[1614]['members']==['AYUNE','XILI']
assert rows[6072]['members']==['RADA']
manifest={'baseline':str(BASE/'corrected_boxes.jsonl'),'baseline_sha256':hashlib.sha256((BASE/'corrected_boxes.jsonl').read_bytes()).hexdigest(),
 'frame_coordinate_system':'30 fps music-trimmed output; start-inclusive/end-exclusive','corrections':notes,
 'changed_frames':len(changed),'member_frame_deltas':{n:after[n]-before[n] for n in before},
 'selection_rule':'沿用第三版表演角色；本次仅将 00:54 提前切换纠正为 AYUNE/XILI 双 C，03:22 保持用户最新确认的 RADA 单 C。',
 'timing_rule':'可见红框逐帧计时，双 C 分别累计；位置和宽度调整不改变计时。',
 'layout':'双框靠近时收窄并排，避免重合；保留左上贴顶国籍排行榜、姓名国旗及原始 3456×1944 清晰度。'}
OUT.mkdir(exist_ok=True,parents=True)
(OUT/'reviewed_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
save(OUT/'corrections.json',manifest)
save(OUT/'annotation-validation.json',{'frames':len(rows),'empty_stage_frames':0,'max_members':2,
 'nonoverlapping_duet_boxes':True,'display_spacing_adjusted_frames':len(layout),
 'hidden_intervals':[{'frames':[2773,2788],'member':'AYUNE','role_retained':True}],
 'counter_totals':dict(after),'changed_frames':len(changed)})
(OUT/'第四版修订记录.txt').write_text('''第四版：根据用户在第三版上的截图反馈修正。

00:47 HIMEKI：重接左前排身体轨迹，修正框偏到相邻者的问题。
00:52 AYUNE：以金字塔中间主舞头部为基准左移框，持续跟随。
00:54：保持 AYUNE 与最右侧 XILI 双 C 至队形解开，撤销提前出现的 YOUNG 单 C。
02:29 AIJING：从 OA 身上移回用户指定的黄发成员，连续修正移动与交接。
02:41 及其他双框靠近处：适度收窄并排，避免两框重合。此项不改变计时。
03:04 VORA：对准前排短发长裤身体，避免跟到后面的人。
03:22：按用户最后确认，保持 RADA 单 C。

保持 3456×1944、30 fps、221.9 秒及原录屏音频；各成员累计时间重新计算。
''')
print(json.dumps({'changed_frames':len(changed),'spacing_frames':len(layout),'frame_deltas':manifest['member_frame_deltas']},ensure_ascii=False))
