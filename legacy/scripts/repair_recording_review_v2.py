"""Apply screenshot corrections to exact CFR output frames; preserve v1.

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
BASE=ROOT/'outputs/neon-screen-recording'
OUT=ROOT/'outputs/neon-screen-recording-v2'

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

rename(1471,1514,{'AYUNE':'HIMEKI'},'用户最新截图确认：此处是 HIMEKI；到 1514 帧才是后续 AYUNE 交接。')
# Preserve a dance turn in the middle of the three-person lead group.
gap={4215:[653,387,630,44],4220:[675,391,639,44],4225:[687,407,650,44],
     4230:[682,397,640,43],4235:[665,398,653,43],4240:[660,414,655,42],
     4245:[655,408,632,42],4250:[652,394,623,41],4260:[646,386,629,41],4270:[641,393,640,41]}
head_keyframes('AIJING',gap,4216,4269,'用户确认：三人中间主舞持续，舞蹈转身不撤框。',only_missing=True)
# Foreground pair: heads lean outwards while torsos overlap the rear row.
lean_x={4534:(596,693),4540:(587,706),4555:(574,718),4570:(563,722),4585:(560,724),
 4600:(560,722),4615:(556,718),4630:(560,723),4645:(552,717),4660:(548,718),
 4675:(546,714),4690:(557,715),4705:(570,686),4720:(605,654),4724:(609,648)}
lean_top={4534:(394,397),4540:(414,416),4555:(424,427),4570:(426,430),4585:(435,437),
 4600:(434,439),4615:(438,439),4630:(443,438),4645:(442,441),4660:(442,441),
 4675:(447,442),4690:(441,441),4705:(444,425),4720:(408,401),4724:(403,400)}
for j,n in enumerate(['OA','AYUNE']):
    points={f:[xs[j],lean_top[f][j],622 if j==0 else 619,40 if f<4710 else 34] for f,xs in lean_x.items()}
    head_keyframes(n,points,4534,4725,'用户确认 OA、AYUNE：以各自前排头部位置为红框基准，保持双 C。')
rename(4842,4949,{'OA':'YOUNG','YOUNG':'OA'},'用户确认 2:42 的姓名对应相反：左 OA、右 YOUNG；沿用至各自交接。')
# MERCI remains upright behind the kneeling foreground dancer, then moves
# behind the front line and comes out to the left. Do not follow the reused ID.
merci={5030:[628,376,616,45],5035:[630,380,620,44],5040:[630,383,614,44],
 5045:[640,383,614,44],5050:[648,383,610,44],5055:[653,386,610,44],
 5060:[656,387,610,44],5065:[658,386,611,44],5070:[646,385,611,44],
 5075:[615.1,392,614,44],5080:[572.3,384,615,43],5085:[542.8,382,610,43],
 5090:[501.6,388,610,43],5095:[460.5,389,610,43],5100:[420.1,384,610,43],
 5105:[380.6,389,620,43],5110:[357.8,388,625,43],5115:[343.0,387,625,43],
 5120:[340.0,395,625,43],5125:[339.1,393,627,43],5130:[349.5,392,628,43],
 5135:[372.3,389,628,43],5140:[398.5,385,630,43],5145:[421.0,385,630,43],
 5150:[441.7,387,630,43],5155:[464.9,382,630,44],5160:[484.5,383,632,46]}
head_keyframes('MERCI',merci,5035,5160,'用户确认 2:49 MERCI 框错位：沿直立主舞头部及身体连续移动，排除前方辅助者；5070 帧后用局部光流复核向左走位。',replace=True)
rename(5903,5924,{'RADA':'YOUNG'},'用户确认 3:17 仍为 YOUNG；RADA 走到前排中间后再交接。')
rx={5924:[660],5930:[637],5940:[620],5960:[625],5980:[634],6000:[633],6020:[641],6030:[642],6060:[655],6090:[636],6103:[549]}
track_body('RADA',5924,6104,[1905,2027],rx,'用户确认后续主 C 是 RADA：从前排脚落点与身体轨迹定位，移除误跟 XILI 的框。',replace=True)
track_body('MERCI',6223,6333,[2075],{6223:[597],6240:[598],6270:[595],6300:[575],6332:[563]},
 '用户确认 3:29 双 C 为 MERCI 与 XILI：依据前排脚落点，继续前一双 C 轨迹。',replace=True)
track_body('XILI',6223,6333,[2086],{6223:[730],6240:[730],6270:[728],6300:[727],6332:[714]},
 '用户确认 3:29 双 C 为 MERCI 与 XILI：依据前排脚落点，继续前一双 C 轨迹。')

changed=[]
for f,r in enumerate(rows):
    if r==old[f]:continue
    if r['boxes']:
        r['boxes']=[list(b) for b in constrain_display_boxes(r['boxes'],852,480,gap=5)]
        order=sorted(range(len(r['members'])),key=lambda j:r['boxes'][j][0])
        for k in ['members','boxes','observations']:r[k]=[r[k][j] for j in order]
    assert len(r['members'])==len(set(r['members']))==len(r['boxes'])<=2
    changed.append(f)
before=Counter(n for r in old for n in r['members']);after=Counter(n for r in rows for n in r['members'])
manifest={'baseline':str(BASE/'corrected_boxes.jsonl'),'baseline_sha256':hashlib.sha256((BASE/'corrected_boxes.jsonl').read_bytes()).hexdigest(),
 'frame_coordinate_system':'30 fps music-trimmed output; start-inclusive/end-exclusive',
 'corrections':notes,'changed_frames':len(changed),'member_frame_deltas':{n:after[n]-before[n] for n in before},
 'layout':'排行榜左上贴顶；姓名、国旗、国籍；人名国旗距框顶 13/480 画面高度；保持原始 3456×1944。',
 'selection_rule':'前排主舞以双脚落点、身体连续移动及主舞角色判定；不能仅按画面水平居中。'}
OUT.mkdir(exist_ok=True,parents=True)
(OUT/'reviewed_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
save(OUT/'corrections.json',manifest)
assert all('AIJING' in r['members'] for r in rows[4085:4524])
assert all(set(r['members'])=={'MERCI','XILI'} for r in rows[6104:6333])
assert all(r['members']==['RADA'] for r in rows[5924:6104])
assert all('HIMEKI' in r['members'] and 'AYUNE' not in r['members'] for r in rows[1471:1514])
print(json.dumps({'changed_frames':len(changed),'frame_deltas':manifest['member_frame_deltas']},ensure_ascii=False))
