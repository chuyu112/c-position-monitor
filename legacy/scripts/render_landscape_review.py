"""Render an auditable landscape C-role candidate pass and frame-exact timing.

Role anchors are clip-specific review judgements. Track IDs are retained as
evidence, never aggregated into presumed member identities.
"""

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from c_detect.overlays import narrow_c_boxes, constrain_display_boxes
from c_detect.stabilization import _smooth


def build_annotations(spec, rows, width, height):
    output = []
    for segment in spec['segments']:
        start, end, count = (segment[k] for k in ('start_frame', 'end_frame', 'count'))
        assert start == len(output) and start < end <= len(rows)
        anchors = segment['anchors']
        times = np.array([a['frame'] for a in anchors])
        boxes = np.array([[t['box'] for t in a['targets']] for a in anchors])
        previous = []
        for index in range(start, end):
            row = {'frame': index, 'seconds': index/spec['fps'], 'segment': segment['id'],
                   'reason': segment['reason'], 'boxes': [], 'selected_ids': [],
                   'method': 'unresolved' if index < spec['performance_end_frame'] else 'outro'}
            if count:
                ref = np.array([[np.interp(index, times, boxes[:, slot, edge])
                                 for edge in range(4)] for slot in range(count)])
                near = int(np.argmin(np.abs(times-index)))
                preferred = [t['track_id'] for t in anchors[near]['targets']]
                people = [p for p in rows[index]['people']
                          if p['box'][3]-p['box'][1] > .12 and p['detection_confidence'] >= .25]
                # Slot order is spatial, not a member identity. Enumerate pairs
                # so a detection cannot be assigned to both C slots.
                best, chosen = float('inf'), []
                for candidates in itertools.permutations(people, count):
                    cost = 0
                    for slot,p in enumerate(candidates):
                        b=np.asarray(p['box'])
                        dx=abs((b[0]+b[2]-ref[slot,0]-ref[slot,2])/2)
                        dy=abs((b[1]+b[3]-ref[slot,1]-ref[slot,3])/2)
                        shape=abs((b[3]-b[1])-(ref[slot,3]-ref[slot,1]))
                        cost += 2*dx+.35*dy+.12*shape
                        cost -= .025*(p['track_id']==preferred[slot])
                        cost -= .012*(slot<len(previous) and p['track_id']==previous[slot])
                    if cost < best:
                        best,chosen=cost,candidates
                if chosen and best/count < .085:
                    # Reject a slot that is much farther from its review path
                    # than the other slot; an occluded C must not jump to a peer.
                    valid=[]
                    for slot,p in enumerate(chosen):
                        b=np.asarray(p['box'])
                        dx=abs((b[0]+b[2]-ref[slot,0]-ref[slot,2])/2)
                        if dx<.07:
                            valid.append(p)
                    if valid:
                        row['selected_ids']=[p['track_id'] for p in valid]
                        row['boxes']=[list(b) for b in narrow_c_boxes(
                            [p['box'] for p in valid],width,height,[p.get('torso_x') for p in valid],gap=8)]
                        row['method']='review_anchor_and_body_detection'
                previous=row['selected_ids']
            row['c_count']=len(row['boxes'])
            output.append(row)
        print(f'Prepared {segment["id"]}: {start/spec["fps"]:.2f}-{end/spec["fps"]:.2f}s',flush=True)
    assert len(output)==len(rows)
    # Bridge only a short detector dropout between the same reviewed role and
    # nearly identical body locations. Never bridge a role handoff or outro.
    for index in range(1,len(output)-1):
        if output[index]['c_count'] or output[index]['method']=='outro':
            continue
        if not output[index-1]['c_count']:
            continue
        end=index
        while end<len(output) and not output[end]['c_count'] and end-index<=6:
            end+=1
        if end>=len(output) or end-index>6:
            continue
        before,after=output[index-1],output[end]
        if before['segment']!=after['segment'] or before['c_count']!=after['c_count']:
            continue
        a,b=np.asarray(before['boxes']),np.asarray(after['boxes'])
        if np.max(np.abs(a-b))>30:
            continue
        for f in range(index,end):
            t=(f-index+1)/(end-index+1)
            output[f].update(boxes=np.rint(a+(b-a)*t).astype(int).tolist(),
                selected_ids=before['selected_ids'][:],c_count=before['c_count'],
                method='interpolated_short_detection_gap')
    # Smooth only a contiguous, spatially continuous role; reset on handoffs,
    # visibility changes and large position jumps.
    start=0
    for end in range(1,len(output)+1):
        boundary=(end==len(output) or output[end]['segment']!=output[start]['segment']
                  or output[end]['c_count']!=output[start]['c_count'])
        if not boundary and output[end]['c_count']:
            boundary=np.max(np.abs(np.asarray(output[end]['boxes'])-np.asarray(output[end-1]['boxes'])))>65
        if boundary:
            if output[start]['c_count']:
                raw=np.asarray([r['boxes'] for r in output[start:end]],dtype=float)
                center=(raw[:,:,:2]+raw[:,:,2:])/2
                size=raw[:,:,2:]-raw[:,:,:2]
                for slot in range(raw.shape[1]):
                    for axis in range(2):
                        center[:,slot,axis]=_smooth(center[:,slot,axis],.07*spec['fps'])
                        size[:,slot,axis]=_smooth(size[:,slot,axis],.12*spec['fps'])
                stable=np.concatenate([center-size/2,center+size/2],axis=2)
                for row,b in zip(output[start:end],stable):
                    row['boxes']=[list(v) for v in constrain_display_boxes(b,width,height,gap=8)]
            start=end
    return output


FONT = '/System/Library/Fonts/STHeiti Medium.ttc'


def panel_image(single, double, uncertain, active, segment, elapsed, outro):
    im=Image.new('RGB',(236,141),(234,232,229));d=ImageDraw.Draw(im)
    font=ImageFont.truetype(FONT,15);small=ImageFont.truetype(FONT,12)
    d.text((10,7),'C 位计时 · 横屏复核初版',font=font,fill=(38,42,46))
    d.line((10,30,225,30),fill=(184,184,182))
    for y,label,value in [(36,'单 C 画面',single),(57,'双 C 画面',double),(78,'C 位累计人秒',single+2*double)]:
        d.text((10,y),label,font=small,fill=(55,58,61))
        d.text((145,y),f'{value:6.2f}s',font=small,fill=(193,38,48))
    status='片尾 · 停表' if outro else ('交接/遮挡 · 暂停' if not active else f'{segment} · {"双 C" if active==2 else "单 C"} · {elapsed:.2f}s')
    d.text((10,104),status,font=small,fill=(55,58,61))
    d.text((10,122),'按可见红框计时；未绑定成员姓名',font=small,fill=(95,98,101))
    return cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('spec',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    spec=json.loads(args.spec.read_text())
    source=ROOT/spec['source'];cache=ROOT/spec['cache'];fps=spec['fps']
    rows=[json.loads(l) for l in cache.read_text().splitlines()]
    cap=cv2.VideoCapture(str(source))
    width,height=int(cap.get(3)),int(cap.get(4))
    assert len(rows)==spec['frames']==int(cap.get(7))
    annotations=build_annotations(spec,rows,width,height)
    output=args.output_dir/'NEON_横屏_C位识别与计时.mp4'
    command=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n',
             '-thread_queue_size','64','-f','rawvideo','-pix_fmt','bgr24',
             '-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-i',str(source),
             '-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','fast',
             '-crf','18','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(output)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    counts=Counter();segments=Counter();tracks=Counter();intervals=[]
    previews={120,420,840,1200,1560,1860,2340,3000,4200,4680,4980,5700,6420,6660,6900}
    try:
        for row in annotations:
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Source ended early')
            n=row['c_count'];outro=row['method']=='outro'
            if not outro:counts[n]+=1
            if n:segments[row['segment']]+=1
            tracks.update(row['selected_ids'])
            for j,(x1,y1,x2,y2) in enumerate(row['boxes']):
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2,cv2.LINE_AA)
                text='C' if n==1 else f'C{j+1}'
                tw=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.43,1)[0][0]
                x=max(2,min(width-tw-9,(x1+x2-tw)//2-3));y=max(17,y1)
                cv2.rectangle(frame,(x,y-18),(x+tw+7,y),(198,194,190),-1)
                cv2.putText(frame,text,(x+3,y-5),cv2.FONT_HERSHEY_SIMPLEX,.43,(45,42,40),1,cv2.LINE_AA)
            frame[64:205,10:246]=panel_image(counts[1]/fps,counts[2]/fps,counts[0]/fps,
                                            n,row['segment'],segments[row['segment']]/fps,outro)
            if row['frame'] in previews:
                cv2.imwrite(str(args.output_dir/f'preview-{row["frame"]:05}.jpg'),frame,[cv2.IMWRITE_JPEG_QUALITY,95])
            process.stdin.write(frame.tobytes())
            key=(row['segment'],n,row['method'])
            if intervals and intervals[-1]['key']==key:
                intervals[-1]['end_frame']=row['frame']+1
            else:
                intervals.append({'key':key,'start_frame':row['frame'],'end_frame':row['frame']+1,
                                  'segment':row['segment'],'c_count':n,'method':row['method']})
            if row['frame']%1500==0:print('Rendered',row['frame'],flush=True)
    finally:
        cap.release();process.stdin.close()
    if process.wait():raise RuntimeError('Encoder failed')
    for interval in intervals:
        del interval['key']
        interval.update(start_seconds=interval['start_frame']/fps,end_seconds=interval['end_frame']/fps,
                        duration_seconds=(interval['end_frame']-interval['start_frame'])/fps)
    segment_totals=[]
    for segment in spec['segments']:
        subset=annotations[segment['start_frame']:segment['end_frame']]
        histogram=Counter(r['c_count'] for r in subset)
        segment_totals.append({'segment':segment['id'],'start_seconds':segment['start_frame']/fps,
            'end_seconds':segment['end_frame']/fps,'single_seconds':histogram[1]/fps,
            'double_seconds':histogram[2]/fps,'unboxed_seconds':histogram[0]/fps,
            'person_seconds':(histogram[1]+2*histogram[2])/fps,'reason':segment['reason']})
    summary={'source':str(source),'original_source':str(ROOT/spec['original_source']),
        'output':str(output.resolve()),'frames':len(annotations),'fps':fps,'width':width,'height':height,
        'single_c_seconds':counts[1]/fps,'double_c_seconds':counts[2]/fps,
        'visible_c_seconds':(counts[1]+counts[2])/fps,'person_seconds':(counts[1]+2*counts[2])/fps,
        'unresolved_performance_seconds':counts[0]/fps,
        'performance_end_seconds':spec['performance_end_frame']/fps,
        'outro_seconds':(len(annotations)-spec['performance_end_frame'])/fps,
        'method':'YOLO11s Pose + ByteTrack + clip-specific spatial role review + symmetric box smoothing',
        'status':'辅助复核候选初版，尚未获用户确认；不是自动识别准确率评测结果',
        'identity_scope':spec['identity_scope'],'timing_rule':'显示一个红框累计 1/30 人秒，双框同一帧累计 2/30 人秒，无框暂停。',
        'source_review':str(args.spec.resolve()),'segment_totals':segment_totals,
        'track_c_seconds':{str(k):v/fps for k,v in tracks.items()},
        'limitations':['480P远景、遮挡和密集换位会使人体检测及C位判断出错。',
                       '区间边界是画面复核的初始判断，尚需用户核对。',
                       '跟踪编号会断开或交换，不能作为成员身份或成员总时长。']}
    (args.output_dir/'frames.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in annotations))
    (args.output_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (args.output_dir/'timing_intervals.json').write_text(json.dumps(intervals,ensure_ascii=False,indent=2)+'\n')
    lines=['NEON 横屏 C 位识别与计时 — 复核初版','',
           f'单 C 画面：{counts[1]/fps:.2f} 秒',f'双 C 画面：{counts[2]/fps:.2f} 秒',
           f'至少一个可见 C 框：{(counts[1]+counts[2])/fps:.2f} 秒',
           f'C 位累计人秒（双 C 分别计时）：{(counts[1]+2*counts[2])/fps:.2f}',
           f'表演内无框/暂不确定：{counts[0]/fps:.2f} 秒',
           '','计时规则：每个可见红框每帧累加 1/30 秒，无框暂停。',
           '区间编号 S01 等不代表成员。尚未把断开的轨迹合并为真实成员，不能据此生成成员姓名排行榜。',
           '本版使用现有 YOLO11s-Pose / ByteTrack，加本视频的角色复核锚点和防抖。',
           '原片 480P，保留原声；角色和交接边界仍属初版判断，计时精度不等于识别准确率。',
           '', '区间\t开始(s)\t结束(s)\t单C(s)\t双C(s)\t人秒\t说明']
    for s in segment_totals:
        lines.append(f'{s["segment"]}\t{s["start_seconds"]:.2f}\t{s["end_seconds"]:.2f}\t{s["single_seconds"]:.2f}\t{s["double_seconds"]:.2f}\t{s["person_seconds"]:.2f}\t{s["reason"]}')
    (args.output_dir/'C位时间统计.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['segment_totals','track_c_seconds']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
