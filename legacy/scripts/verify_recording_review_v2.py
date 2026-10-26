"""Check native export dimensions, CFR frames, audio and encoded review frames."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/neon-screen-recording-v2'
VIDEO=OUT/'NEON_横屏录屏高清_C位红框_国旗姓名排行榜.mp4'
SOURCE=Path('/path/to/user/Documents/全屏录屏/录屏_2026-10-19_12-10-10_4D8AA1.mp4')
FFMPEG='/path/to/user/.local/bin/ffmpeg';FFPROBE='/path/to/user/.local/bin/ffprobe'
START=1.917375
p=json.loads(subprocess.check_output([FFPROBE,'-v','error','-show_entries',
 'stream=index,codec_name,codec_type,width,height,r_frame_rate,start_time,duration,nb_frames,sample_rate,channels:format=duration,size','-of','json',str(VIDEO)]))
v,a=p['streams'];assert (v['width'],v['height'],v['nb_frames'],v['r_frame_rate'])==(3456,1944,'6657','30/1')
assert a['sample_rate']=='48000' and a['channels']==2
assert abs(float(a['duration'])-float(v['duration']))<.04
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
frames=[0,1473,4225,4585,4860,5070,5120,5921,6030,6270,6653]
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
