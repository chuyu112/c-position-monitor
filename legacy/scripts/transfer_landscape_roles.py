"""Transfer the user-confirmed portrait role sequence to the same choreography.

Names come from the reviewed portrait annotations and the user's statement that
performance / C-role order matches. Spatial anchors use stage positions, costume
and body motion, not face comparison. Raw IDs are only local observations.
"""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from c_detect.overlays import narrow_c_boxes, constrain_display_boxes
from c_detect.stabilization import _smooth

# Local observations at reviewed choreography / costume anchors, in seconds.
OBS = {
 'AIJING': {0:1,2:1,4:1,6:1,8:1,12:1,14:1,16:1,18:1,20:1,22:1,46:265,48:265,
             88:656,92:724,94:724,96:724,112:974,114:993,116:895,120:1004,
             140:1322,142:1385,144:1385,148:1385,150:1385,208:2096,210:2096},
 'XILI': {4:13,8:13,12:37,16:37,20:37,22:37,50:314,52:314,54:314,
           59:435,60:435,62:435,64:435,66:435,98:905,100:905,102:905,
           118:1004,136:1285,204:2086,206:2086},
 'OA': {24:2,26:2,28:2,30:2,32:2,34:2,36:2,44:189,
         78:619,80:619,82:619,104:930,127:1122,134:1174,
         152:1455,154:1455,156:1455,158:1564,160:1488,162:1582,164:1582,
         214:2089,216:2089,218:2089,220:2089,222:2089,223:2089},
 'MERCI': {38:2,40:2,42:2,44:2,74:576,76:576,80:699,82:699,
            121:1051,166:1640,168:1640,170:1640,172:1640,204:2075,206:2075},
 'HIMEKI': {46:104,48:104,50:104,122:1141,124:1141,126:1141,
             178:1771,180:1771,184:1771,212:1771},
 'AYUNE': {50:267,51:267,84:619,86:619,88:619,90:619,92:619,94:619,
            132:1228,134:1228,152:1488,154:1488,156:1488,212:1825},
 'YOUNG': {54:377,56:377,58:377,59:377,98:895,100:895,102:895,
            160:1582,162:1488,164:1488,174:1760,176:1760,178:1760,180:1760,
            188:1905,190:1905,192:1905,194:1905,196:1905,208:2094,210:2094},
 'VORA': {68:514,70:514,73:514,182:1798,184:1798,186:1576,213.5:2108},
 'RADA': {108:893,110:893,198:1943,200:1943,202:1943},
}
# Direct spatial observations where a body is visible but YOLO missed / merged it.
MANUAL = {
 'AYUNE': {52:[410,273,454,399], 53:[413,276,454,401]},
 'MERCI': {78:[511,295,563,414]},
 'OA': {128:[384,261,421,404],129:[408,269,450,411],130:[422,278,464,425],
        131:[424,270,467,420],132:[440,266,477,420]},
 'AIJING': {90:[449,332,496,430],138:[397,280,444,418]},
}
# Full foreground camera occlusion: do not time an invisible body.
HIDDEN = {'VORA':[(71.2,72.75)],'AYUNE':[(95.3,96.54)],
          'OA':[(105.5,106.14)],'HIMEKI':[(185.4,186.17)]}


def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def make_spec(detections):
    anchors=[]
    for name,points in OBS.items():
        for t,tid in points.items():
            f=round(t*30);found=[]
            for q in sorted(range(max(0,f-12),min(len(detections),f+13)),key=lambda q:abs(q-f)):
                found=[p for p in detections[q]['people'] if p['track_id']==tid]
                if found:break
            if not found:
                print('No observation:',name,t,tid);continue
            p=found[0];box=p['box']
            anchors.append({'name':name,'frame':q,'track_id':tid,'box':box,
                            'torso_x':p.get('torso_x'), 'evidence':'reviewed choreography position and costume'})
    for name,points in MANUAL.items():
        for t,box in points.items():
            anchors.append({'name':name,'frame':round(t*30),'track_id':None,
                            'box':[v/([852,480,852,480][k]) for k,v in enumerate(box)],
                            'torso_x':(box[0]+box[2])/1704,'evidence':'visible body spatial keyframe; detector merged or missed'})
    return {'fps':30,'offset_frames':27,'audio_offset_seconds':.89,
            'source':'outputs/neon-landscape/source_30fps.mp4',
            'portrait_annotations':'outputs/c5-red-name-oa-double-turns/corrected_boxes.jsonl',
            'portrait_video':'/path/to/user/Desktop/c5_C位红框_OA双C与转身修正版.mp4',
            'frames':7126,'performance_end_frame':6695,'anchors':anchors,
            'movement_start_frame':82,'movement_end_frame':6561,
            'selection_rules':[
              '用户最新确认：三人主表演组合只有中间一人为 C。',
              '用户最新确认：两人主表演组合为双 C，两人分别计时。',
              '转身、背对观众、站着不跳舞及辅助表演者不是 C，撤框并停止计时。',
              '用户最新的站立不跳舞排除规则优先于旧竖屏 ending 静止例外。',
              '仅 C 位显示姓名和红框，其他人物不标注。'],
            'hidden_intervals':HIDDEN,
            'name_provenance':'User confirms identical performance and C-role order. Labels transferred from reviewed portrait, aligned by audio, choreography, costume and stage position; no face recognition.',
            'status':'按竖屏已确认顺序迁移并抽帧校对；横屏交接边界和遮挡仍可能有误差。'}


def track_run(name,start,end,anchors,detections,spec):
    # A fresh path for every active run prevents stale IDs crossing a handoff.
    nearby=[a for a in anchors if a['name']==name and start-75<=a['frame']<end+75]
    if not nearby:
        raise ValueError(f'No spatial observations for {name} {start/30:.2f}-{end/30:.2f}')
    inside=[a for a in nearby if start<=a['frame']<end]
    if inside:
        # Include the adjacent observation only to support a continuous boundary.
        before=[a for a in nearby if a['frame']<start]
        after=[a for a in nearby if a['frame']>=end]
        nearby=inside+([max(before,key=lambda a:a['frame'])] if before else [])+([min(after,key=lambda a:a['frame'])] if after else [])
    nearby.sort(key=lambda a:a['frame'])
    times=np.array([a['frame'] for a in nearby]);bs=np.array([a['box'] for a in nearby]);
    centers=np.array([a['torso_x'] if a['torso_x'] is not None else (a['box'][0]+a['box'][2])/2 for a in nearby])
    layers=[];costs=[];backs=[]
    for f in range(start,end):
        ref=np.array([np.interp(f,times,bs[:,k]) for k in range(4)])
        cx=float(np.interp(f,times,centers));cy=(ref[1]+ref[3])/2;bh=ref[3]-ref[1]
        nearest=min(nearby,key=lambda a:abs(a['frame']-f));candidates=[];unaries=[]
        hidden=any(a<=f/30<b for a,b in spec['hidden_intervals'].get(name,[]))
        for p in ([] if hidden else detections[f]['people']):
            b=np.array(p['box']);pc=(b[0]+b[2])/2
            torso=p.get('torso_x');torso=pc if torso is None else torso
            pc=.55*torso+.45*pc
            dx=abs(pc-cx);dy=abs((b[1]+b[3])/2-cy);dh=abs(b[3]-b[1]-bh)
            if dx>.065 or dy>.17 or p['detection_confidence']<.25:continue
            near_id=p['track_id']==nearest['track_id'] and abs(nearest['frame']-f)<=75
            u=2.4*dx+.32*dy+.22*dh-.026*near_id
            candidates.append({'box':p['box'],'torso_x':p.get('torso_x'),
                               'track_id':p['track_id'],'method':'anchored_body_detection'})
            unaries.append(u)
        # Spatial interpolation is only available around a directly reviewed
        # missing/merged body, not as a general license to invent a person.
        manual=[a for a in nearby if a['track_id'] is None and abs(a['frame']-f)<=18]
        if manual and not hidden:
            candidates.append({'box':ref.tolist(),'torso_x':cx,'track_id':None,
                               'method':'reviewed_spatial_keyframe'})
            unaries.append(.018)
        candidates.append(None);unaries.append(.075 if not hidden else 0.)
        exact=[a for a in nearby if a['frame']==f]
        if exact and not hidden:
            for i,c in enumerate(candidates):
                correct=(c is not None and c['track_id']==exact[0]['track_id'])
                if not correct:unaries[i]+=2
        if not layers:
            costs=np.array(unaries);back=[-1]*len(candidates)
        else:
            prev=layers[-1];total=[];back=[]
            for i,c in enumerate(candidates):
                v=[]
                for j,p in enumerate(prev):
                    if c is None or p is None:
                        trans=.028 if (c is None)!=(p is None) else .006
                    else:
                        a=np.array(c['box']);b=np.array(p['box'])
                        trans=3*abs((a[0]+a[2]-b[0]-b[2])/2)+.4*abs((a[1]+a[3]-b[1]-b[3])/2)
                        trans+=.025*(c['track_id']!=p['track_id'])
                    v.append(costs[j]+trans)
                j=int(np.argmin(v));back.append(j);total.append(v[j]+unaries[i])
            costs=np.array(total)
        layers.append(candidates);backs.append(back)
    j=int(np.argmin(costs));path=[]
    for k in range(len(layers)-1,-1,-1):
        path.append(layers[k][j]);j=backs[k][j]
    path=path[::-1]
    # Close <= 0.3 s dropouts only between continuous observations of this role.
    i=1
    while i<len(path)-1:
        if path[i] is not None or path[i-1] is None:i+=1;continue
        e=i
        while e<len(path) and path[e] is None:e+=1
        if e<len(path) and e-i<=9 and not any(a<=(start+i)/30<b for a,b in spec['hidden_intervals'].get(name,[])):
            a=np.array(path[i-1]['box']);b=np.array(path[e]['box'])
            if np.max(abs(a-b))<.07:
                for q in range(i,e):
                    t=(q-i+1)/(e-i+1);path[q]={'box':(a+(b-a)*t).tolist(),'torso_x':None,
                        'track_id':None,'method':'short_body_gap_interpolation'}
        i=e
    return path


def build(spec,detections,portrait):
    expected=[]
    for f in range(spec['frames']):
        pf=f-spec['offset_frames']
        if f>=spec['movement_end_frame'] or f<spec['movement_start_frame']:names=[]
        elif pf>=6591:names=['OA']
        elif pf<0:names=['AIJING']
        else:names=portrait[pf]['members'][:]
        # Stage handoffs can differ slightly from the music-aligned relay take.
        if 87<=f<100:names=['XILI','AIJING']
        if 1122<=f<1150:names=['MERCI']
        if 687<=f<704:names=[]
        if 2459<=f<2478:names=['OA']
        if 4086<=f<4109:names=['AIJING']
        names=[n for n in names if not any(a<=f/30<b for a,b in spec['hidden_intervals'].get(n,[]))]
        expected.append(names)
    result=[{'frame':f,'seconds':f/30,'portrait_frame':max(0,f-27),
             'expected_members':names,'members':[],'boxes':[],'observations':[]} for f,names in enumerate(expected)]
    for name in OBS:
        for active,g in itertools.groupby(range(len(expected)),key=lambda f:name in expected[f]):
            fs=list(g)
            if not active:continue
            s,e=fs[0],fs[-1]+1
            path=track_run(name,s,e,spec['anchors'],detections,spec)
            print(name,f'{s/30:.2f}-{e/30:.2f}',sum(p is not None for p in path),'/',len(path),flush=True)
            for f,p in zip(fs,path):
                if p is not None:
                    result[f]['members'].append(name);result[f]['observations'].append(p)
    for row in result:
        # One detector box must never credit two different names.
        if len(row['members'])==2:
            a,b=row['observations'];duplicate=a['track_id'] is not None and a['track_id']==b['track_id']
            if duplicate:
                row['members']=[];row['observations']=[];row['collision']=True
        if row['members']:
            row['boxes']=[list(b) for b in narrow_c_boxes([p['box'] for p in row['observations']],852,480,
                                                [p['torso_x'] for p in row['observations']],gap=8)]
    # Each continuous named role is independently smoothed; role handoffs reset.
    for name in OBS:
        for active,g in itertools.groupby(range(len(result)),key=lambda f:name in result[f]['members']):
            fs=list(g)
            if not active:continue
            slots=[result[f]['members'].index(name) for f in fs]
            raw=np.array([result[f]['boxes'][j] for f,j in zip(fs,slots)],float)
            # Split at an implausibly large step to avoid sweeping a red box.
            borders=[0]+(np.where(np.max(abs(np.diff(raw,axis=0)),axis=1)>55)[0]+1).tolist()+[len(fs)]
            for s,e in zip(borders[:-1],borders[1:]):
                a=raw[s:e];center=(a[:,:2]+a[:,2:])/2;size=a[:,2:]-a[:,:2]
                for k in range(2):center[:,k]=_smooth(center[:,k],2.0);size[:,k]=_smooth(size[:,k],3.5)
                smooth=np.concatenate([center-size/2,center+size/2],axis=1)
                for f,j,b in zip(fs[s:e],slots[s:e],smooth):result[f]['boxes'][j]=b.tolist()
    for row in result:
        if row['boxes']:
            row['boxes']=[list(b) for b in constrain_display_boxes(row['boxes'],852,480,gap=8)]
            order=sorted(range(len(row['boxes'])),key=lambda j:row['boxes'][j][0])
            for key in ['boxes','members','observations']:row[key]=[row[key][j] for j in order]
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/neon-landscape/named-ranking-v1');p.add_argument('--refresh-spec',action='store_true');args=p.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    detections=read_rows(ROOT/'outputs/neon-landscape/detection/frames.jsonl')
    config=ROOT/'annotations/neon_landscape_member_transfer.json'
    if args.refresh_spec or not config.exists():
        spec=make_spec(detections);config.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
    else:spec=json.loads(config.read_text())
    rows=build(spec,detections,read_rows(ROOT/spec['portrait_annotations']))
    (out/'corrected_boxes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    counts=Counter(n for r in rows for n in r['members']);print('TOTALS',dict(counts));print('MISSING',sum(len(r['expected_members'])-len(r['members']) for r in rows))

if __name__=='__main__':main()
