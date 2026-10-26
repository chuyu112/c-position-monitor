"""User-confirmed music start and local body-placement / role corrections.

Keep the existing video time axis. Coordinates are reviewed on the clean CFR
1280x720 proxy; names come from user instructions, not face identification.
"""
from collections import Counter
from copy import deepcopy
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/neon-screen-recording-v4'
OUT=ROOT/'outputs/neon-screen-recording-v5'
FPS=30
def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
old=read(BASE/'corrected_boxes.jsonl');rows=deepcopy(old);notes=[]

def role(a,b,names,reason):
    for r in rows[a:b]:
        r['expected_members']=list(names)
        keep=[j for j,n in enumerate(r['members']) if n in names]
        for k in ['members','boxes','observations']:r[k]=[r[k][j] for j in keep]
        r['role_review']=reason
    notes.append({'frames':[a,b],'members':names,'reason':reason})

def heads(n,a,b,points,reason):
    keys=sorted(points)
    for f in range(a,b):
        x,top,bottom,width=[np.interp(f,keys,[points[k][j] for k in keys]) for j in range(4)]
        box=(np.array([x-width/2,top,x+width/2,bottom])*[852/1280,480/720,852/1280,480/720]).tolist()
        ob={'box':(np.array(box)/[852,480,852,480]).tolist(),'method':'user_reviewed_head_and_body_motion','evidence':reason}
        r=rows[f]
        if n in r['members']:
            j=r['members'].index(n);r['boxes'][j]=box;r['observations'][j]=ob
        else:
            r['members'].append(n);r['boxes'].append(box);r['observations'].append(ob)
        assert n in r['expected_members'],(f,n)
        r['review_note']=reason
    notes.append({'frames':[a,b],'member':n,'head_body_keyframes_720p':points,'reason':reason})

# User confirms the song starts at 00:02 in the delivered video. Do not
# confuse the source alignment offset (1.917375 s) with the music start.
role(0,60,[],'用户明确音乐从第四版视频第 2 秒开始；前 60 帧不画框、不计时，保留原音画时间线。')
for r in rows[:60]:r['timing_status']='before_music'

# AIJING remains the lead; the old tracker jumped to the bent neighbour.
heads('AIJING',3367,3401,{
3367:[645,402,636,38],3370:[632,399,640,38],3373:[624,400,642,38],
3376:[627,411,647,38],3379:[632,409,651,38],3382:[635,404,650,38],
3385:[639,409,649,38],3388:[636,419,647,38],3391:[625,421,643,38],
3394:[606,412,639,38],3397:[585,410,637,38],3400:[574,409,638,38]},
'01:53 按用户指定的浅色头发长袖 AIJING 连续动作定位，不跟到旁边俯身的成员。')
# Exactly two old frames jumped onto the member the user calls XILI.
heads('AIJING',3523,3525,{3522:[506,403,619,36],3525:[508,405,617,36]},
'01:57–01:58 始终保持 AIJING；修正 117.433–117.500 秒瞬间跳到 XILI 身上的框。')

# User: OA solo -> OA lowers her head, AYUNE solo -> OA and AYUNE duet.
role(3827,3930,['OA'],'用户确认此段 OA 单 C；AYUNE 尚未接过主表演时不提前加框。')
role(3930,3958,['AYUNE'],'OA 低头俯身辅助，AYUNE 直立主表演：按用户确认切为 AYUNE 单 C。')
role(3958,4071,['AYUNE','OA'],'OA 抬头恢复与 AYUNE 并排主表演，按用户确认恢复双 C。')

heads('OA',3827,3930,{
3827:[624,405,630,36],3832:[632,407,628,36],3837:[638,404,625,36],
3842:[650,404,624,36],3847:[652,413,621,36],3852:[659,394,616,36],
3857:[657,395,618,36],3862:[665,409,620,36],3867:[650,426,620,36],
3872:[649,438,620,36],3877:[660,440,624,36],3882:[674,433,626,36],
3887:[674,430,627,36],3892:[675,433,626,36],3897:[658,444,625,36],
3902:[647,442,626,36],3907:[645,411,626,36],3912:[646,394,625,36],
3917:[633,390,627,36],3922:[634,394,625,36],3927:[653,395,625,36],
3929:[665,405,625,36]},
'02:08–02:11 OA 的框以用户指定的中间长上衣成员头部为基准，持续随动作移动，避免偏到邻人或空处。')

heads('AYUNE',3930,3958,{
3930:[621,391,615,36],3934:[622,394,615,36],3938:[622,395,616,36],
3942:[624,392,615,36],3946:[622,393,614,36],3950:[620,393,614,36],
3954:[625,392,615,36],3957:[625,394,615,36]},
'02:11 其他成员俯身时只框中间直立的 AYUNE，框顶跟住头部，不沿错误检测框下滑。')

heads('OA',3958,4071,{
3958:[686,394,630,36],3964:[674,393,624,36],3970:[653,405,621,36],
3976:[648,427,623,36],3982:[653,407,626,36],3988:[671,402,625,36],
3994:[684,434,624,36],4000:[682,427,626,36],4006:[672,399,623,36],
4012:[651,421,621,36],4018:[651,429,625,36],4024:[655,402,636,36],
4030:[659,389,637,36],4036:[666,393,632,36],4042:[668,394,631,36],
4048:[674,396,634,36],4054:[680,394,633,36],4060:[682,395,631,36],
4066:[685,398,630,36],4070:[683,401,629,36]},
'恢复双 C 后逐段对齐 OA 头部和长上衣身体，红框随侧倾移动，不滞留在两人之间。')
heads('AYUNE',3958,4071,{
3958:[625,397,616,36],3964:[624,389,616,36],3970:[600,398,612,36],
3976:[580,420,615,36],3982:[579,410,615,36],3988:[598,396,612,36],
3994:[621,423,615,36],4000:[615,415,616,36],4006:[600,398,617,36],
4012:[572,410,615,36],4018:[568,416,614,36],4024:[578,391,620,36],
4030:[590,382,620,36],4036:[599,392,616,36],4042:[608,391,617,36],
4048:[611,394,618,36],4054:[617,393,617,36],4060:[618,392,617,36],
4066:[622,395,616,36],4070:[626,400,615,36]},
'恢复双 C 后按 AYUNE 的头部、短上衣与裙装身体连续移动校准，保持与 OA 的两个框并排。')

# Preserve the fourth version's display separation. Reapply only if these
# local head corrections bring two display markers too close together.
spacing=[]
for r in rows:
    if len(r['boxes'])!=2:continue
    i,j=sorted(range(2),key=lambda k:sum(r['boxes'][k][::2]))
    left,right=np.array(r['boxes'][i],float),np.array(r['boxes'][j],float)
    if left[2]+4<=right[0]:continue
    r['body_boxes_before_spacing']=deepcopy(r['boxes'])
    c1=(left[0]+left[2])/2;c2=(right[0]+right[2])/2;mid=(c1+c2)/2
    h1=(left[2]-left[0])/2;h2=(right[2]-right[0])/2
    if c2-c1<14:c1=mid-7;c2=mid+7
    room=(c2-c1-4)/2
    left[[0,2]]=[c1-min(h1,room),c1+min(h1,room)]
    right[[0,2]]=[c2-min(h2,room),c2+min(h2,room)]
    r['boxes'][i]=left.tolist();r['boxes'][j]=right.tolist();spacing.append(r['frame'])

changed=[f for f,r in enumerate(rows) if r!=old[f]]
for f,r in enumerate(rows):
    assert len(r['members'])==len(set(r['members']))==len(r['boxes'])==len(r['observations'])
    assert (not r['members']) if f<60 else 1<=len(r['members'])<=2
    assert set(r['expected_members'])-set(r['members'])==set(r.get('temporarily_hidden_members',[]))
    for b in r['boxes']:assert 0<=b[0]<b[2]<852 and 0<=b[1]<b[3]<480,(f,b)
    if len(r['boxes'])==2:
        a,b=sorted(r['boxes']);assert a[2]+3.99<=b[0],f
before=Counter(n for r in old for n in r['members']);after=Counter(n for r in rows for n in r['members'])
deltas={n:after[n]-before[n] for n in before}
assert {n:v for n,v in deltas.items() if v}=={'AIJING':-60,'OA':-28,'AYUNE':-15},deltas
assert rows[6072]['members']==['RADA']
for f in range(3347,3630):assert rows[f]['members']==['AIJING']
manifest={'baseline':str(BASE/'corrected_boxes.jsonl'),
 'baseline_sha256':hashlib.sha256((BASE/'corrected_boxes.jsonl').read_bytes()).hexdigest(),
 'frame_coordinate_system':'Original V4 time axis retained, 30 fps; ranges are start-inclusive/end-exclusive.',
 'music_start_frame':60,'music_start_seconds':2.0,'music_start_provenance':'用户明确回复第四版视频 2s。',
 'corrections':notes,'changed_frames':len(changed),'member_frame_deltas':deltas,
 'selection_rule':'第 2 秒音乐起才画框计时；01:53 及 01:57–01:58 连续跟随 AIJING；02:10–02:12 按 OA 单 C、AYUNE 单 C、两人双 C 的实际动作切换。',
 'timing_rule':'2 秒以前零累计；其后可见 C 框逐帧计时，双 C 分别累计；坐标修正不转移身份或计时。',
 'new_display_spacing_frames':spacing,'prior_corrections_preserved':True}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'reviewed_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
save(OUT/'corrections.json',manifest)
save(OUT/'annotation-validation.json',{'frames':len(rows),'music_start_frame':60,'music_start_seconds':2.0,
 'pre_music_zero_frames':60,'empty_stage_frames_after_music':0,'max_members':2,
 'nonoverlapping_duet_boxes':True,'counter_totals':dict(after),'changed_frames':len(changed),
 'verified_role_intervals':[{'frames':[3827,3930],'members':['OA']},{'frames':[3930,3958],'members':['AYUNE']},{'frames':[3958,4071],'members':['AYUNE','OA']}]})
(OUT/'第五版修订记录.txt').write_text('''第五版：从用户确认的第 2 秒音乐起计时。

00:00–00:02：不画框、不累计，排行榜所有成员保持 0 秒；视频本身时间线不变。
01:53：红框跟随浅色头发的 AIJING，修正误框旁边俯身成员。
01:57–01:58：始终跟随 AIJING，修正 117.433–117.500 秒瞬间跳到 XILI 的红框。
02:07.567–02:11.000：OA 单 C，重新校准中间 OA 的头部和身体位置。
02:11.000–02:11.933：OA 俯身，AYUNE 单 C。
02:11.933–02:15.700：OA 抬头恢复，OA 与 AYUNE 双 C；两框继续对准各自头部与身体。

保留第四版其他修正，包括 03:22 RADA 单 C、双框并排不重合。
保持 3456×1944、30 fps、221.9 秒及原音频；排行榜及统计重新计算。
''')
print(json.dumps({'changed_frames':len(changed),'frame_deltas':deltas,'music_start_frame':60},ensure_ascii=False))
