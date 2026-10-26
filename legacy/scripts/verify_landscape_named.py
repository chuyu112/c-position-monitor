"""Validate encoded media and the shared box/name/time/ranking contract."""
from collections import Counter
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from landscape_live_ranking import MemberLeaderboard,MEMBERS
from render_landscape_named import draw_c

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/neon-landscape/named-ranking-v1'
FFMPEG='/path/to/user/.local/bin/ffmpeg'
FFPROBE='/path/to/user/.local/bin/ffprobe'

def rows(path):return [json.loads(l) for l in path.read_text().splitlines()]
def audio_hash(path):
    return subprocess.check_output([FFMPEG,'-v','error','-i',str(path),'-map','0:a:0','-c','copy','-f','hash','-hash','sha256','pipe:1'],text=True).strip()

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=OUT);args=parser.parse_args();OUT=args.output_dir
    data=rows(OUT/'corrected_boxes.jsonl');ranks=rows(OUT/'ranking_frames.jsonl')
    config=OUT/'annotation_spec.json'
    if not config.exists():config=ROOT/'annotations/neon_landscape_member_transfer.json'
    spec=json.loads(config.read_text());summary=json.loads((OUT/'summary.json').read_text())
    counter=Counter({n:0 for n in MEMBERS});ranker=MemberLeaderboard();canvas=np.zeros((spec.get('height',480),spec.get('width',852),3),np.uint8)
    for f,(row,rank) in enumerate(zip(data,ranks)):
        assert row['frame']==rank['frame']==f
        names,boxes=row['members'],row['boxes'];assert len(names)==len(boxes)<=2 and len(names)==len(set(names))
        assert set(names)<=set(row['expected_members'])
        for x1,y1,x2,y2 in boxes:assert 0<=x1<x2<852 and 0<=y1<y2<480
        if len(boxes)==2:
            a,b=sorted(boxes);assert b[0]-a[2]>=8
        if f<spec['movement_start_frame'] or f>=spec['movement_end_frame']:assert not names
        for n,intervals in spec['hidden_intervals'].items():
            if any(a<=f/30<b for a,b in intervals):assert n not in names
        if 4086<=f<4525:assert set(names)<= {'AIJING'}
        rects=draw_c(canvas,boxes,names)
        for a,b in zip(rects,rects[1:]):assert not (a[0]<b[2] and a[2]>b[0] and a[1]<b[3] and a[3]>b[1])
        expected=ranker.step(f,boxes,names);assert expected==rank
        counter.update(names)
    assert len(data)==len(ranks)==7126
    assert dict(counter)==summary['member_c_frames']
    for n in MEMBERS:assert abs(counter[n]/30-summary['member_c_seconds'][n])<1e-10
    assert sum(counter.values())==sum(len(row['boxes']) for row in data)
    assert data[120]['members']==['XILI','AIJING']
    if 'review_corrections' in spec:
        fix=spec['review_corrections'];baseline=rows(Path(fix['baseline']))
        for f,(a,b) in enumerate(zip(data,baseline)):
            if not any(s<=f<e for s,e in fix['changed_frame_intervals']):assert a==b
        assert all(r['members']==['AIJING'] for r in data[:87])
        assert all(set(r['members'])=={'XILI','AIJING'} for r in data[87:681])
        assert all(set(r['members'])=={'HIMEKI','AIJING'} for r in data[1390:1472])
        assert all('AYUNE' in r['members'] and 'HIMEKI' not in r['members'] for r in data[1472:1515])
        previous=Counter(n for r in baseline for n in r['members'])
        assert {n:counter[n]-previous[n] for n in MEMBERS}==fix['member_frame_deltas']
    video=Path(summary['output']);probe=json.loads(subprocess.check_output([FFPROBE,'-v','error','-show_streams','-show_format','-of','json',str(video)],text=True))
    v=next(s for s in probe['streams'] if s['codec_type']=='video');a=next(s for s in probe['streams'] if s['codec_type']=='audio')
    assert int(v['nb_frames'])==len(data) and v['avg_frame_rate']=='30/1'
    assert (v['width'],v['height'])==(spec.get('width',852),spec.get('height',480))
    source=ROOT/spec['source'];h1=audio_hash(source);h2=audio_hash(video);assert h1==h2
    subprocess.run([FFMPEG,'-v','error','-i',str(video),'-map','0:v','-map','0:a','-f','null','-'],check=True)
    report={'frames_verified':len(data),'all_name_box_timer_deltas_match':True,
       'all_ranks_and_ties_verified':True,'all_boxes_and_labels_in_bounds':True,
       'pair_boxes_and_labels_do_not_overlap':True,'recorded_stationary_and_hidden_exclusions_verified':True,
       'user_opening_duet_handoff_name_corrections_verified':'review_corrections' in spec,
       'native_source_dimensions':[v['width'],v['height']],
       'three_person_phrase_keeps_only_center_role':True,'audio_bitstream_preserved':True,
       'audio_sha256':h2.split('=')[-1],'full_video_audio_decode_passed':True,
       'video_bytes':video.stat().st_size,'member_c_seconds':summary['member_c_seconds'],
       'verification_scope':'数据、排版、编码及已记录排除区间校验；不代表每帧 C 位判定或姓名绑定经过独立人工真值验证。'}
    (OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
