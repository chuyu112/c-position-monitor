"""Correct the user-confirmed AIJING solo at 2:16 and its body placement."""
from collections import Counter
from copy import deepcopy
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/neon-screen-recording-v7'
OUT=ROOT/'outputs/neon-screen-recording-v8'
old=[json.loads(l) for l in (BASE/'corrected_boxes.jsonl').read_text().splitlines()]
rows=deepcopy(old)
# Head center, head top, body bottom, width in the 1280 x 720 CFR proxy.
points={4071:[566,400,629,44],4075:[589,402,634,44],4080:[615,400,653,44],
        4084:[627,424,643,44],4088:[644,425,641,44],4092:[649,428,638,44],
        4096:[652,432,637,44],4100:[654,432,637,44],4104:[655,430,637,44],
        4108:[654,421,635,44],4111:[654,418,635,44],4114:[652,416,635,44],
        4115:[652,415,634,44]}
reason='用户确认 02:16 应为中间浅色头发的 AIJING 单 C；取消短暂误标 XILI，沿 AIJING 向中间移动及俯身动作连续跟框。'
keys=sorted(points)
for f in range(4071,4115):
 x,top,bottom,width=[np.interp(f,keys,[points[k][j] for k in keys]) for j in range(4)]
 b=(np.array([x-width/2,top,x+width/2,bottom])*[852/1280,480/720,852/1280,480/720]).tolist()
 rows[f].update({'members':['AIJING'],'expected_members':['AIJING'],'boxes':[b],
    'observations':[{'box':(np.array(b)/[852,480,852,480]).tolist(),
       'method':'user_reviewed_continuous_body_position','evidence':reason}],
    'review_note':reason})
 rows[f].pop('body_boxes_before_spacing',None)
before=Counter(n for r in old for n in r['members']);after=Counter(n for r in rows for n in r['members'])
delta={n:after[n]-before[n] for n in before}
assert {n:d for n,d in delta.items() if d}=={'AIJING':14,'XILI':-14}
assert all(r['members']==['AIJING'] for r in rows[4071:4281])
for f,r in enumerate(rows):
 if not 4071<=f<4115:assert r==old[f],f
 assert len(r['members'])==len(r['boxes'])==len(r['observations'])
 assert not r['members'] if f<60 else 1<=len(r['members'])<=2
 for b in r['boxes']:assert 0<=b[0]<b[2]<852 and 0<=b[1]<b[3]<480,(f,b)
manifest={'baseline':str(BASE/'corrected_boxes.jsonl'),
 'baseline_sha256':hashlib.sha256((BASE/'corrected_boxes.jsonl').read_bytes()).hexdigest(),
 'corrections':[{'frames':[4071,4085],'members':['AIJING'],'reason':reason},
                {'frames':[4071,4115],'head_body_keyframes_720p':points,
                 'reason':'按头部与身体连续移动校准红框，修正交接后框底超出脚下的位置。'}],
 'member_frame_deltas':delta,'ranking_footer_visible':False,'music_start_frame':60,
 'selection_rule':'02:15.700 起由 AIJING 单 C 接续，修正短暂误标为 XILI 的 14 帧；第七版其他标注不变。',
 'prior_corrections_preserved':True}
validation={'frames':len(rows),'pre_music_zero_frames':60,'empty_stage_frames_after_music':0,
 'all_other_rows_unchanged':True,'corrected_role_frames':[4071,4085],
 'corrected_position_frames':[4071,4115],'counter_totals':dict(after),
 'member_frame_deltas':delta,'ranking_footer_visible':False}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'reviewed_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
for name,value in [('corrections.json',manifest),('annotation-validation.json',validation)]:
 (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
(OUT/'本次修订记录.txt').write_text('''c7 更新：2:16 的 C 位修正为 AIJING。

02:15.700–02:16.167：将短暂误标 XILI 的 14 帧修正为 AIJING 单 C，红框移到中间浅色头发的 AIJING 身上。
连续校准至 02:17.167 的身体位置和框底，接续此前已经修正的 AIJING 片段。
AIJING 累计增加约 0.47 秒，XILI 减少约 0.47 秒；其他成员累计不变。
保留此前左右双 C、框宽与间隔、音乐第 2 秒起计时、排行榜无底部说明等修正。
原生 3456×1944、30 fps、221.9 秒，原声和视频时间线不变。
''')
print(json.dumps({'changed_position_frames':44,'corrected_role_frames':14,'member_frame_deltas':delta},ensure_ascii=False))
