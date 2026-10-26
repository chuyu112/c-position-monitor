"""Check native export dimensions, CFR frames, audio and encoded review frames."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/neon-screen-recording-v8'
VIDEO=OUT/'NEON_横屏录屏高清_C位红框_国旗姓名排行榜.mp4'
SOURCE=Path('/path/to/user/Documents/全屏录屏/录屏_2026-10-19_12-10-10_4D8AA1.mp4')
FFMPEG='/path/to/user/.local/bin/ffmpeg';FFPROBE='/path/to/user/.local/bin/ffprobe'
START=1.917375
p=json.loads(subprocess.check_output([FFPROBE,'-v','error','-show_entries',
 'stream=index,codec_name,codec_type,width,height,r_frame_rate,start_time,duration,nb_frames,sample_rate,channels:format=duration,size','-of','json',str(VIDEO)]))
v,a=p['streams'];assert (v['width'],v['height'],v['nb_frames'],v['r_frame_rate'])==(3456,1944,'6657','30/1')
assert a['sample_rate']=='48000' and a['channels']==2
assert abs(float(a['duration'])-float(v['duration']))<.04
rows=[json.loads(l) for l in (OUT/'corrected_boxes.jsonl').read_text().splitlines()]
ranks=[json.loads(l) for l in (OUT/'ranking_frames.jsonl').read_text().splitlines()]
summary=json.loads((OUT/'summary.json').read_text())
assert len(rows)==len(ranks)==6657
assert summary['music_start_frame']==60 and summary['music_start_seconds']==2
counts={n:0 for n in summary['member_c_frames']}
min_gap=3456
min_width=3456
for f,(row,rank) in enumerate(zip(rows,ranks)):
 assert row['frame']==rank['frame']==f
 assert len(row['members'])==len(row['boxes'])
 if f<60:assert not row['members'] and not any(rank['cumulative_frames'].values())
 else:assert 1<=len(row['members'])<=2
 for n in row['members']:counts[n]+=1
 assert counts==rank['cumulative_frames'],f
 assert set(rank['active'])==set(row['members']),f
 if len(row['boxes'])==2:
  for b in row['boxes']:
   width=round(b[2]*3456/852)-round(b[0]*3456/852)
   assert width>=113,(f,width)
   min_width=min(min_width,width)
  left,right=sorted(row['boxes'],key=lambda b:b[0])
  gap=round(right[0]*3456/852)-round(left[2]*3456/852)
  assert gap>=15,(f,gap)
  min_gap=min(min_gap,gap)
assert counts==summary['member_c_frames']
assert summary['ranking_footer_visible'] is False
assert all(set(r['members'])=={'MERCI','OA'} for r in rows[2294:2458])
for r in rows[2294:2311]:
 assert r['members']==['MERCI','OA']
 assert r['boxes'][0][2]<r['boxes'][1][0]
old=[json.loads(l) for l in (ROOT/'outputs/neon-screen-recording-v7/corrected_boxes.jsonl').read_text().splitlines()]
assert all(r==old[f] for f,r in enumerate(rows) if not 4071<=f<4115)
for start,end,names in [(3827,3930,['OA']),(3930,3958,['AYUNE']),(3958,4071,['AYUNE','OA'])]:
 assert all(set(r['members'])==set(names) for r in rows[start:end])
assert all(r['members']==['AIJING'] for r in rows[3347:3630])
assert rows[6072]['members']==['RADA']
assert all(r['members']==['AIJING'] for r in rows[4071:4281])
for start,end,names in [(270,360,['AIJING','XILI']),(4115,4281,['AIJING']),
 (2468,2621,['AYUNE']),(4796,4944,['OA','YOUNG']),
 (2621,2773,['AIJING','AYUNE']),(2773,2788,['AIJING']),(2788,2895,['AIJING','AYUNE'])]:
 assert all(set(r['members'])==set(names) for r in rows[start:end]),(start,end)
baseline=json.loads((ROOT/'outputs/neon-screen-recording-v7/summary.json').read_text())
delta={n:counts[n]-baseline['member_c_frames'][n] for n in counts}
assert {n:d for n,d in delta.items() if d}=={'AIJING':14,'XILI':-14},delta
validation=json.loads((OUT/'annotation-validation.json').read_text())
validation.update({'native_duet_gap_min_pixels':min_gap,'native_duet_width_min_pixels':min_width,'verified_counter_delta_from_v7':delta,'all_frame_counter_validation':True,
 'all_frame_badge_geometry_checked_during_render':True})
(OUT/'annotation-validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
checks=[]
def audio(path,t):
 data=subprocess.check_output([FFMPEG,'-v','error','-ss',str(t),'-i',str(path),'-t','3','-vn','-ac','1','-ar','48000','-f','f32le','pipe:1'])
 return np.frombuffer(data,np.float32)
for t in [20,100,200]:
 x=audio(SOURCE,t+START);y=audio(VIDEO,t);n=min(len(x),len(y));x=x[:n].astype(float);y=y[:n].astype(float)
 x-=x.mean();y-=y.mean();size=1<<(2*n-1).bit_length()
 c=np.fft.irfft(np.fft.rfft(x,size)*np.conj(np.fft.rfft(y,size)),size)
 lags=np.arange(-2400,2401);lag=int(lags[np.argmax(c[lags%size])]);corr=float(np.corrcoef(x,y)[0,1])
 assert abs(lag)<=1 and corr>.98,(t,lag,corr)
 checks.append({'seconds':t,'waveform_correlation':corr,'alignment_error_samples':lag})
exports=[]
frames=[0,59,60,2307,3524,3941,3967,4065,4070,4071,4075,4080,4084,4085,4088,4092,4100,4108,4114,4115,4146,4837,6072,6653]
# select emits frames in source order, irrespective of expression order.
frames=sorted(set(frames))
# Decode from the beginning and select by frame index. Random -ss near an
# IDR can omit a reordered frame even though the full video is valid CFR.
select='+'.join(f'eq(n,{f})' for f in frames)
subprocess.run([FFMPEG,'-v','error','-xerror','-y','-hwaccel','videotoolbox','-i',str(VIDEO),
 '-vf',f"select='{select}'",'-vsync','0','-start_number','0','-q:v','2',
 str(OUT/'review'/'verified-sample-%02d.jpg')],check=True)
for i,f in enumerate(frames):
 decoded=cv2.imread(str(OUT/'review'/f'verified-sample-{i:02}.jpg'))
 ref=cv2.imread(str(OUT/'review'/f'annotated-{f:05}.jpg'))
 assert decoded.shape==(1944,3456,3)
 if ref.shape!=decoded.shape:decoded=cv2.resize(decoded,(ref.shape[1],ref.shape[0]),interpolation=cv2.INTER_AREA)
 psnr=float(cv2.PSNR(decoded,ref));assert psnr>30,(f,psnr)
 exports.append({'frame':f,'psnr_vs_preencode_jpeg95':psnr,'selection':'sequential decode frame index'})
result={'probe':p,'audio_checks':checks,'encoded_frames':exports,'annotation_validation':json.loads((OUT/'annotation-validation.json').read_text())}
(OUT/'media-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({'native_dimensions':[3456,1944],'frames':6657,'audio_checks':checks,'encoded_samples':len(exports)}))
