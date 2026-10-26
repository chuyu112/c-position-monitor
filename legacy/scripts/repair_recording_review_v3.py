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
BASE=ROOT/'outputs/neon-screen-recording-v2'
OUT=ROOT/'outputs/neon-screen-recording-v3'

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

# 00:48: follow the adjacent short-sleeved dancer through the crossing.
track_body('HIMEKI',1426,1464,[286,104],
 {1426:[493],1430:[509],1435:[527],1440:[540],1445:[544],1450:[548],1455:[548],1460:[512],1463:[501]},
 '用户指出 00:48 HIMEKI 是旧红框右边一位；在交叉走位中换用相邻身体轨迹，回到左前排后衔接原轨迹。')
# VORA remains the lead until AIJING reaches the front, not until a detector misses.
role(2135,2165,['VORA'],'VORA 保留 C 至 AIJING 上前接位；摄像机经过不提前撤框。')
head_keyframes('VORA',{2130:[653,391,630,42],2135:[658,390,632,42],2145:[672,394,633,42],
 2150:[673,397,637,42],2155:[666,400,641,42],2160:[639,405,639,42],2165:[607,408,638,42]},
 2135,2165,'延续短发长裤主舞的身体移动，直到前排交接。')
role(2165,2219,['AIJING'],'黄发 AIJING 已上前承接 C，原 VORA 标记在检测 ID 复用后应切为 AIJING。')
head_keyframes('AIJING',{2165:[641,411,646,44],2170:[651,411,646,44],2175:[657,409,649,44],
 2185:[666,408,646,44],2195:[667,386,637,44],2205:[660,387,641,44],2218:[628,398,641,44]},
 2165,2219,'以用户指出的黄发成员走到前排后的连续动作确认交接。')
# 01:19: both body markers were delayed during lateral movement.
track_body('OA',2311,2477,[619],{2311:[755],2320:[757],2340:[748],2350:[717],2360:[664],
 2370:[625],2380:[619],2400:[619],2440:[622],2476:[625]},
 'OA 是前排长上衣成员；重算连续身体位置，避免横移时落到右边相邻者。')
role(2311,2458,['OA','MERCI'],'保留前排 OA、MERCI 双 C；两框分别跟随用户指定的左侧相邻身体。')
# MERCI is the long-sleeved shorts dancer, initially left of OA, then right.
head_keyframes('MERCI',{2311:[663,385,635,43],2320:[666,385,634,43],2330:[693,431,645,43],
 2340:[705,428,642,43],2345:[689,421,640,43],2350:[675,407,634,43],2355:[668,394,633,43],
 2360:[678,386,635,43],2365:[685,398,638,43],2370:[687,411,639,43],2380:[689,394,637,43]},
 2311,2380,'01:19 MERCI 跟随长袖短裤主舞的头部；不再框右侧长裤辅助者。')
track_body('MERCI',2380,2458,[699],{2380:[689],2400:[691],2430:[693],2457:[695]},
 '横移交叉结束后，MERCI 接续右前排同一身体轨迹。')
# Visible duet continues to the next lead taking over at 01:36.50.
role(2621,2895,['AIJING','AYUNE'],'用户确认 01:29–01:36 仍是 AIJING、AYUNE 双 C；动作及短暂遮挡不改变角色。')
track_body('AYUNE',2858,2895,[619],{2858:[686],2870:[682],2880:[679],2885:[702],2894:[739]},
 '补回 01:36 已经从 AIJING 身后露出的 AYUNE，持续到实际交接。')
# AYUNE passes directly behind AIJING; the old tracker latched onto a neighbour.
hide(2773,2788,'AYUNE','用户指定例外：完全躲到 AIJING 身后时暂时隐藏 AYUNE 框，角色保持；重新露出立即恢复。')
head_keyframes('AYUNE',{2788:[657,388,623,34],2790:[662,385,625,34],2795:[664,387,629,38],2800:[668,391,630,40]},
 2788,2800,'从 AIJING 身后重新露出后恢复原双 C，不跟随旁边抬手辅助者。')
# RADA takes the vocal lead as OA turns away, even while entering from the side.
role(3164,3241,['RADA'],'OA 转身退入辅助，RADA 从左侧开始接唱；不等到画面中心才计 C。')
track_body('RADA',3164,3241,[893],{3164:[289],3180:[320],3190:[374],3200:[442],3210:[521],3220:[581],3230:[619],3240:[629]},
 '沿用户指定从左侧走来的长裤成员连续跟踪，排除原中心蹲下的 OA。')
# 01:58: the same physical performer remains AIJING across reused tracker IDs.
role(3525,3630,['AIJING'],'用户确认 01:58 黄发 AIJING 主 C，其他人辅助；保留至下一个主舞交接。')
track_body('AIJING',3525,3565,[895],{3525:[508],3540:[489],3555:[484],3564:[503]},
 '重定位用户明确指出的黄发 AIJING，不把同名框贴到黑发长裤辅助者。')
head_keyframes('AIJING',{3565:[508,416,643,42],3570:[522,418,643,42],3575:[539,405,641,42],
 3580:[554,404,640,42],3585:[574,398,638,42],3590:[595,398,637,42],3595:[607,400,640,42],
 3600:[620,407,645,42],3610:[646,402,634,42],3620:[668,400,637,42],3629:[680,395,638,42]},
 3565,3630,'AIJING 在摄像机后向右回到前排中间，保持同一身体轨迹。')
# 02:03: the fifth dancer in the nine-person formation, not the second from left.
role(3630,3827,['HIMEKI'],'用户确认九人横排第五位 HIMEKI 是 C。')
track_body('HIMEKI',3630,3827,[1138,1174,1195],{3630:[628],3650:[607],3690:[625],3730:[634],
 3750:[635],3770:[630],3790:[638],3810:[641],3826:[665]},
 '以用户指定九人中的第五位作为身体锚点，保持收拢队形前后的同一主舞。')
# The OA/YOUNG duet survives crane occlusion and runs until MERCI takes over.
role(4796,4944,['OA','YOUNG'],'用户确认 02:40–02:45 OA、YOUNG 双 C，摄像机经过不撤 C。')
track_body('OA',4812,4842,[1488],{4812:[627],4820:[625],4830:[625],4841:[620]},'保持左侧前排 OA；排除摄像机遮挡造成的检测空段。')
track_body('YOUNG',4812,4842,[1582],{4812:[686],4820:[687],4830:[685],4841:[677]},'保持右侧前排 YOUNG；与 OA 分别累计时间。')
# The last five frames use the same-role interpolation below.
# 03:00: vocalist in the middle, flanking dancers are support.
role(5368,5501,['RADA'],'用户确认中间 RADA 唱高音，两侧为辅助，此段只计 RADA。')
track_body('RADA',5368,5501,[1808],{5368:[648],5380:[631],5400:[644],5420:[647],
 5440:[649],5460:[647],5480:[638],5500:[627]},'以中间主唱 RADA 身体为 C，移除两侧辅助者的 C 框。')
# 03:32 through final pose: centre lead is OA.
role(6333,6657,['OA'],'用户确认 03:32 中间 OA 单 C，沿中央前排保持至结束画面。')
track_body('OA',6333,6657,[2089],{6333:[654],6360:[671],6400:[689],6430:[690],6460:[695],
 6500:[698],6560:[695],6600:[678],6656:[667]},'跟随用户指定中间 OA 的前排身体与脚落点，直到片尾。')
role(683,703,['XILI','AIJING'],'OA 接位前保留开场双 C，移动退场交接期间不出现空白帧。')
# Other full-stage gaps: preserve the outgoing lead until the next takes over.
role(4524,4534,['AIJING'],'交接空段保留前一主舞 AIJING，OA 与 AYUNE 接位时再撤下。')
track_body('YOUNG',1607,1662,[377],{1607:[571],1621:[548],1637:[490],1650:[451],1661:[430]},
 '短暂漏检不结束 C：沿下一帧已确认的 YOUNG 身体回接原主舞。')

# Dense head anchors resolve fast crossings and the floor movement explicitly.
head_keyframes('AIJING',{683:[678,411,665,42],686:[660,419,665,42],690:[634,430,675,42],
 695:[588,425,667,42],700:[548,421,653,42],702:[530,416,650,42]},683,703,
 '开场交接双人分向两侧：按黄发头部逐段跟随，不冻结在旧中心。')
head_keyframes('XILI',{683:[650,400,630,40],686:[653,408,632,40],690:[673,421,640,40],
 695:[710,410,640,40],700:[755,408,640,40],702:[767,413,644,40]},683,703,
 '开场交接中 XILI 向右移动，保持与 AIJING 分开的实际身体位置。')
head_keyframes('YOUNG',{1607:[583,395,624,40],1610:[583,397,625,40],1616:[576,393,625,40],
 1620:[537,397,625,40],1621:[523,398,628,40]},1607,1622,
 '从金字塔队形后左侧起身的主舞连续向左上前；不把未来检测位置提前画在空处。')
head_keyframes('AIJING',{2670:[584,405,650,42],2675:[615,403,650,42],2680:[630,401,650,42],
 2685:[649,413,654,42],2690:[659,435,657,42],2695:[684,480,661,42],2700:[708,503,660,42],
 2710:[711,502,666,42],2720:[706,496,670,42],2730:[677,488,675,42],2740:[627,479,656,42],
 2750:[617,478,650,42],2760:[624,448,642,42],2770:[638,410,645,42],2773:[643,403,645,42]},
 2670,2773,'AIJING 下蹲、转向再起身仍是双 C；按头部和落脚位置连续跟随，不用检测中断作为退 C。')
head_keyframes('AYUNE',{2670:[650,399,632,40],2680:[655,395,635,40],2690:[650,398,632,40],
 2700:[651,410,631,40],2710:[660,406,632,40],2720:[661,407,632,40],2730:[660,405,634,40],
 2740:[665,403,633,40],2750:[670,397,633,40],2760:[665,397,632,40],2770:[659,389,632,38],
 2773:[655,390,633,38]},2670,2773,'AYUNE 在 AIJING 地板动作期间继续主舞；重叠时保持各自头部锚点，不把框挤成细线。')
head_keyframes('AYUNE',{2858:[668,392,646,40],2865:[666,394,646,40],2870:[666,395,648,40],
 2875:[664,399,650,40],2880:[658,399,649,40],2885:[677,411,650,40],2890:[714,405,650,40],
 2894:[739,394,650,40]},2858,2895,'01:36 AYUNE 后仰时以头部为红框基准，避免框仅落在抬起的手臂上。')
head_keyframes('AIJING',{2877:[626,393,647,42],2880:[625,395,649,42],2885:[610,395,650,42],
 2890:[596,403,655,42],2894:[575,403,657,42]},2877,2895,'双 C 后仰动作保留 AIJING 完整身体纵向范围。')

# Fill brief detector holes from the same reviewed role, never change identities.
# Hidden-member intervals are an explicit exception and are never backfilled.
def missing_spans(name):
 out=[];start=None
 for f,r in enumerate(rows+[{'expected_members':[],'members':[]} ]):
  missing=name in r['expected_members'] and name not in r['members'] and name not in r.get('temporarily_hidden_members',[])
  if missing and start is None:start=f
  if not missing and start is not None:out.append((start,f));start=None
 return out
for name in ['AIJING','XILI','OA','MERCI','AYUNE','YOUNG','VORA','RADA','HIMEKI']:
 for a,b in missing_spans(name):
  assert b-a<=50,(name,a,b)
  valid=[f for f in range(max(0,a-45),min(len(rows),b+45)) if name in rows[f]['members']]
  assert valid,(name,a,b)
  lo=max([f for f in valid if f<a],default=valid[0]);hi=min([f for f in valid if f>=b],default=valid[-1])
  ba=np.array(rows[lo]['boxes'][rows[lo]['members'].index(name)],float)
  bb=np.array(rows[hi]['boxes'][rows[hi]['members'].index(name)],float)
  for f in range(a,b):
   t=np.clip((f-lo)/(hi-lo),0,1) if hi!=lo else 0
   set_member(f,name,(ba*(1-t)+bb*t).tolist(),'同一主舞角色的短漏检，按前后身体位置连续插值；不更换 C。')
  notes.append({'frames':[a,b],'member':name,'interpolation_frames':[lo,hi],'reason':'同一角色漏检补全'})

changed=[]
for f,r in enumerate(rows):
 if r!=old[f]:
  # Keep each marker centred on its performer; natural overlap is allowed.
  order=sorted(range(len(r['members'])),key=lambda j:r['boxes'][j][0])
  for k in ['members','boxes','observations']:r[k]=[r[k][j] for j in order]
  changed.append(f)
 assert 1<=len(r['members'])==len(set(r['members']))==len(r['boxes'])<=2,(f,r['members'])
 assert set(r['members'])<=set(r['expected_members'])
 assert set(r['expected_members'])-set(r['members'])==set(r.get('temporarily_hidden_members',[])),f
 for b in r['boxes']:assert 0<=b[0]<b[2]<852 and 0<=b[1]<b[3]<480,(f,b)
before=Counter(n for r in old for n in r['members']);after=Counter(n for r in rows for n in r['members'])
manifest={'baseline':str(BASE/'corrected_boxes.jsonl'),'baseline_sha256':hashlib.sha256((BASE/'corrected_boxes.jsonl').read_bytes()).hexdigest(),
 'frame_coordinate_system':'30 fps music-trimmed output; start-inclusive/end-exclusive','corrections':notes,
 'changed_frames':len(changed),'member_frame_deltas':{n:after[n]-before[n] for n in before},
 'selection_rule':'先判定主唱、领舞及交接，站位和脚落点辅助；角色与框的可见性分离。双 C 持续到交接。舞蹈转身和短漏检不撤框；AYUNE 完全躲在 AIJING 身后时仅暂停其框，露出恢复。',
 'timing_rule':'保持原有口径：可见红框逐帧计时，双 C 分别累计；明确的完全遮挡期间不累计被隐藏者。',
 'layout':'保留 v2 左上贴顶国籍排行榜、上移姓名国旗及原始 3456×1944 清晰度。'}
OUT.mkdir(exist_ok=True,parents=True)
(OUT/'reviewed_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
save(OUT/'corrections.json',manifest)
save(OUT/'annotation-validation.json',{'frames':len(rows),'empty_stage_frames':0,'max_members':max(len(r['members']) for r in rows),
 'hidden_intervals':[{'frames':[2773,2788],'member':'AYUNE','role_retained':True}],
 'counter_totals':dict(after),'changed_frames':len(changed)})
print(json.dumps({'changed_frames':len(changed),'frame_deltas':manifest['member_frame_deltas']},ensure_ascii=False))
