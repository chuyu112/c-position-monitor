"""Correct the user-confirmed MERCI-left/OA-right duet at 1:17."""
from collections import Counter
from copy import deepcopy
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/neon-screen-recording-v6'
OUT=ROOT/'outputs/neon-screen-recording-v7'
old=[json.loads(l) for l in (BASE/'corrected_boxes.jsonl').read_text().splitlines()]
rows=deepcopy(old)
POINTS={
 'MERCI':{2290:[683,412,652,44],2293:[683,411,651,44],2294:[682,409,650,44],
          2296:[680,401,647,44],2300:[674,390,640,44],2304:[665,383,635,44],
          2307:[664,381,635,44],2311:[663,385,635,43]},
 'OA':{2294:[725,421,637,44],2296:[728,417,635,44],2300:[737,397,640,44],
       2304:[745,387,641,44],2307:[749,386,642,44],2311:[751.807,388.5,641.878,42.067]}}
reason='用户确认约 01:17 左侧长袖短裤为 MERCI、右侧长上衣为 OA，属于双 C；修复旧跟踪在此处换人造成的姓名误挂及 OA 漏框。'
for f in range(2290,2311):
 r=rows[f];names=['MERCI','OA'] if f>=2294 else ['MERCI']
 bodies=[];obs=[]
 for name in names:
  keys=sorted(POINTS[name]);x,top,bottom,width=[np.interp(f,keys,[POINTS[name][k][j] for k in keys]) for j in range(4)]
  b=(np.array([x-width/2,top,x+width/2,bottom])*[852/1280,480/720,852/1280,480/720]).tolist()
  bodies.append(b);obs.append({'box':(np.array(b)/[852,480,852,480]).tolist(),
    'method':'user_reviewed_continuous_body_position','evidence':reason})
 boxes=deepcopy(bodies)
 if len(boxes)==2:
  overlap=max(0,boxes[0][2]+4-boxes[1][0])/2
  boxes[0][0]-=overlap;boxes[0][2]-=overlap
  boxes[1][0]+=overlap;boxes[1][2]+=overlap
  r['body_boxes_before_spacing']=bodies
  r['display_width_floor']=28
  r['display_spacing']='保持正常可读宽度及双框间隔；姓名随各自身体移动。'
 r.update({'members':names,'expected_members':names[:],'boxes':boxes,'observations':obs,'review_note':reason})

before=Counter(n for r in old for n in r['members']);after=Counter(n for r in rows for n in r['members'])
delta={n:after[n]-before[n] for n in before}
assert {n:d for n,d in delta.items() if d}=={'OA':17}
for f,r in enumerate(rows):
 if not 2290<=f<2311:assert r==old[f],f
 assert len(r['members'])==len(r['boxes'])==len(r['observations'])
 assert not r['members'] if f<60 else 1<=len(r['members'])<=2
 if len(r['boxes'])==2:
  a,b=sorted(r['boxes'],key=lambda b:b[0])
  assert a[2]+3.999<=b[0],f
  assert min(x[2]-x[0] for x in r['boxes'])>=28-1e-8,f
assert all(set(r['members'])=={'MERCI','OA'} for r in rows[2294:2458])
assert all(r['boxes'][0][2]<r['boxes'][1][0] for r in rows[2294:2311])
manifest={'baseline':str(BASE/'corrected_boxes.jsonl'),
 'baseline_sha256':hashlib.sha256((BASE/'corrected_boxes.jsonl').read_bytes()).hexdigest(),
 'corrections':[{'frames':[2290,2311],'head_body_keyframes_720p':POINTS,'reason':reason},
                {'frames':[2294,2311],'added_member':'OA','reason':'补齐并排双 C，MERCI 左、OA 右。'},
                {'display_change':'删除排行榜底部说明并收紧面板高度，不改变计时规则。'}],
 'member_frame_deltas':delta,'ranking_footer_visible':False,'music_start_frame':60,
 'selection_rule':'约 01:16.467 起延续 MERCI / OA 双 C，按用户给定左右姓名校准；其余第六版角色、框宽和交接保持不变。',
 'prior_corrections_preserved':True}
validation={'frames':len(rows),'pre_music_zero_frames':60,'empty_stage_frames_after_music':0,
 'all_other_rows_unchanged':True,'corrected_duet_frames':[2294,2311],
 'counter_totals':dict(after),'member_frame_deltas':delta,'nonoverlapping_duet_boxes':True,
 'ranking_footer_visible':False}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'reviewed_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
for name,value in [('corrections.json',manifest),('annotation-validation.json',validation)]:
 (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
(OUT/'第七版修订记录.txt').write_text('''第七版：1:17 双 C 姓名、红框及排行榜文字修正。

约 01:16.467–01:17.033：左边 MERCI、右边 OA，补齐双 C；修复 MERCI 姓名误挂到 OA 身上。两人的标记分别跟随身体移动，接续后面的双 C 段落。
补回 OA 的 17 帧（约 0.57 秒），所有排行榜累计重新计算，其他成员时长不变。
删除排行榜底部“有红框才计时 · 双 C 分别累计”，同时收紧面板高度；计时规则不变。
保留第六版双框宽度、间隔及其他已确认修正；音乐从视频第 2 秒开始计时。
原生 3456×1944、30 fps、221.9 秒，原声与视频时间线不变。
''')
print(json.dumps({'changed_frames':21,'added_duet_frames':17,'member_frame_deltas':delta},ensure_ascii=False))
