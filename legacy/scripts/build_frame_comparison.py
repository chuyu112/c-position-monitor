#!/usr/bin/env python3
"""Build a paired screenshot gallery and synchronized source-video viewer."""

from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "neon-vocal-comparison"
MANIFEST = OUT / "member_comparison.json"
OFFICIAL = ROOT / "downloads" / "BV1VVpo6gEYg" / "BV1VVpo6gEYg_720p.mp4"
LIVE = ROOT / "downloads" / "BV1sVnP6kEdn" / "BV1sVnP6kEdn_720p.mp4"
FRAMES = OUT / "frames"


def take_frame(video: Path, seconds: float, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{seconds:.3f}", "-i", str(video), "-frames:v", "1",
            "-vf", "scale=640:360:force_original_aspect_ratio=decrease,"
            "pad=640:360:(ow-iw)/2:(oh-ih)/2:black",
            "-q:v", "3", str(destination),
        ],
        check=True,
    )


def main() -> None:
    if not OFFICIAL.exists() or not LIVE.exists():
        raise SystemExit("Expected downloaded 720p official and live videos in downloads/.")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = []
    for segment in data["segments"]:
        seg_id = segment["id"]
        duration = float(segment["snippet_duration_seconds"])
        # Use each clip's locally refined alignment rather than the global offset.
        official_t = float(segment["official_audio_start_seconds"]) + duration / 2
        live_t = float(segment["live_audio_start_seconds"]) + duration / 2
        official_img = FRAMES / "official" / f"{seg_id}.jpg"
        live_img = FRAMES / "live" / f"{seg_id}.jpg"
        take_frame(OFFICIAL, official_t, official_img)
        take_frame(LIVE, live_t, live_img)
        records.append(
            {
                "id": seg_id,
                "members": segment["dance_role_candidates"],
                "duration": round(duration, 3),
                "official_time": round(official_t, 3),
                "live_time": round(live_t, 3),
                "alignment": segment["alignment_status"],
                "fit": segment["local_chroma_fit"],
                "official_image": official_img.relative_to(OUT).as_posix(),
                "live_image": live_img.relative_to(OUT).as_posix(),
            }
        )

    (OUT / "frame_comparison.json").write_text(
        json.dumps(
            {
                "sources": {"official": str(OFFICIAL), "live": str(LIVE)},
                "mapping": "Each still is sampled at the midpoint of its locally aligned audio excerpt.",
                "assignment_warning": data["member_assignment_warning"],
                "segments": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    build_gallery(records, data["member_assignment_warning"])
    build_video_viewer(records, data["member_assignment_warning"])
    print(f"Created {len(records)} aligned screenshot pairs in {FRAMES}")
    print(f"Gallery: {OUT / 'frame_comparison.html'}")
    print(f"Video viewer: {OUT / 'video_comparison.html'}")


def build_gallery(records: list[dict], warning: str) -> None:
    cards = []
    for r in records:
        names = " + ".join(html.escape(n) for n in r["members"])
        review = " · 需复核对齐" if r["alignment"] == "review_alignment" else ""
        href = f"video_comparison.html?segment={r['id']}"
        cards.append(f"""<article class="card" id="{r['id']}">
  <header><div><b>{r['id']}</b><span class="names">{names}</span></div><span class="duration">片段 {r['duration']:.1f}s{review}</span></header>
  <div class="pair">
    <figure><figcaption>官方录音版 · {r['official_time']:.2f}s</figcaption><img loading="lazy" src="{r['official_image']}" alt="{r['id']} 官方录音版画面"></figure>
    <figure><figcaption>现场消音版 · {r['live_time']:.2f}s</figcaption><img loading="lazy" src="{r['live_image']}" alt="{r['id']} 现场版画面"></figure>
  </div>
  <footer><span>局部对齐参考 {r['fit']:.2f}</span><a href="{href}">打开这一段的同步视频 →</a></footer>
</article>""")
    navbuttons = "".join(
        f'<button data-target="{r["id"]}">{html.escape(" + ".join(r["members"]))} · {r["id"]}</button>'
        for r in records
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEON 开嗓对照片段截图</title>
<style>
:root{{color-scheme:dark;--bg:#10131a;--panel:#191e29;--muted:#aab4c5;--accent:#97f3cd;--line:#2a3242}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:#edf2fa;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1440px;margin:auto;padding:28px 24px 60px}}h1{{margin:0 0 8px;font-size:27px}}.intro{{color:var(--muted);max-width:950px;margin:0 0 18px}}
.notice{{border:1px solid #66562b;background:#292416;color:#f1dfaa;border-radius:10px;padding:12px 15px;margin:16px 0 24px}}
.toolbar{{position:sticky;top:0;z-index:2;background:#10131aeb;backdrop-filter:blur(12px);padding:12px 0;display:flex;gap:8px;flex-wrap:wrap}}
.toolbar button{{background:var(--panel);border:1px solid var(--line);color:#edf2fa;padding:8px 12px;border-radius:8px;cursor:pointer}}.toolbar button:hover{{border-color:var(--accent)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;margin:14px 0;padding:14px;scroll-margin-top:80px}}
header,footer{{display:flex;align-items:center;justify-content:space-between;gap:12px}}header{{margin-bottom:10px}}.names{{margin-left:12px;color:var(--accent);font-weight:650}}.duration,footer{{color:var(--muted);font-size:13px}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}figure{{margin:0;min-width:0}}figcaption{{font-size:13px;color:#c7d1df;margin:0 0 5px}}img{{display:block;width:100%;height:auto;aspect-ratio:16/9;object-fit:contain;background:#050608;border-radius:8px}}
footer{{padding-top:10px}}a{{color:var(--accent);text-decoration:none}}a:hover{{text-decoration:underline}}
@media(max-width:700px){{main{{padding:18px 12px 40px}}.pair{{grid-template-columns:1fr}}header{{align-items:flex-start;flex-direction:column}}}}
</style></head><body><main>
<h1>官方录音版 × 现场消音版｜逐段截图</h1>
<p class="intro">每张截图取自对应音频片段的中点。可以按舞蹈版的候选成员顺序浏览；点卡片底部可进入两路视频同步播放，并切换试听音轨。</p>
<div class="notice"><b>成员归属说明：</b>{html.escape(warning)} 这些截图用于人工核对人和段落；对齐分只表示音频特征的时间匹配，不是演唱得分。</div>
<nav class="toolbar"><button onclick="window.scrollTo({{top:0,behavior:'smooth'}})">回到顶部</button>{navbuttons}</nav>
{''.join(cards)}
</main><script>document.querySelectorAll('.toolbar [data-target]').forEach(b=>b.addEventListener('click',()=>document.getElementById(b.dataset.target).scrollIntoView({{behavior:'smooth',block:'start'}})))</script></body></html>"""
    (OUT / "frame_comparison.html").write_text(page, encoding="utf-8")


def build_video_viewer(records: list[dict], warning: str) -> None:
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    entries = []
    for r in records:
        name = html.escape(" + ".join(r["members"]))
        entries.append(
            f'<button class="chapter" data-id="{r["id"]}"><b>{r["id"]}</b><span>{name}</span><small>{r["duration"]:.1f}s</small></button>'
        )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NEON 同步视频对比</title>
<style>
:root{{color-scheme:dark;--bg:#10131a;--panel:#191e29;--muted:#aab4c5;--accent:#97f3cd;--line:#2a3242}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:#edf2fa;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1440px;margin:auto;padding:24px}}h1{{margin:0 0 6px}}.muted{{color:var(--muted)}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:18px 0}}figure{{margin:0;min-width:0;background:#050608;border-radius:12px;overflow:hidden}}figcaption{{padding:10px 12px;background:var(--panel)}}video{{display:block;width:100%;height:auto;aspect-ratio:16/9;background:#050608;object-fit:contain}}
.controls,.chapterbox{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin:12px 0}}.row{{display:flex;align-items:center;gap:10px;flex-wrap:wrap}}button{{background:#222a38;color:#edf2fa;border:1px solid #384255;padding:8px 12px;border-radius:8px;cursor:pointer}}button:hover,button.active{{border-color:var(--accent);color:var(--accent)}}input[type=range]{{flex:1;min-width:180px}}.sound{{margin-top:12px}}.chapters{{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:7px;margin-top:12px;max-height:290px;overflow:auto}}.chapter{{display:flex;align-items:center;justify-content:space-between;gap:8px;text-align:left}}.chapter span{{color:var(--accent);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.chapter small{{color:var(--muted);white-space:nowrap}}.warn{{border:1px solid #66562b;background:#292416;color:#f1dfaa;padding:12px;border-radius:10px;margin:12px 0}}a{{color:var(--accent)}}
@media(max-width:760px){{main{{padding:14px}}.pair{{grid-template-columns:1fr}}}}
</style></head><body><main>
<a href="frame_comparison.html">← 返回逐段截图</a><h1>官方版与现场版同步视频</h1><p class="muted">画面按段落局部音频对齐；播放/暂停和拖动进度会同步两边。一次只播放一个音轨。</p>
<div class="warn">{html.escape(warning)} 这是舞蹈角色候选，不是已确认的主唱归属。画面和音轨对齐也不构成演唱质量分数。</div>
<section class="pair"><figure><figcaption>左侧｜官方录音版（评分基准）</figcaption><video id="official" preload="metadata" playsinline src="../../downloads/BV1VVpo6gEYg/BV1VVpo6gEYg_720p.mp4"></video></figure><figure><figcaption>右侧｜现场消音版</figcaption><video id="live" preload="metadata" playsinline src="../../downloads/BV1sVnP6kEdn/BV1sVnP6kEdn_720p.mp4"></video></figure></section>
<section class="controls"><div class="row"><button id="play">▶ 播放</button><button id="pause">暂停</button><button id="back">−5秒</button><button id="forward">+5秒</button><span id="clock">00:00 / 00:00</span><input id="seek" type="range" min="0" max="1000" value="0"></div>
<div class="row sound"><b>试听：</b><button data-sound="official" class="active">官方音轨</button><button data-sound="live">现场音轨</button><button data-sound="mute">静音</button><span class="muted">（避免两条人声叠在一起）</span></div></section>
<section class="chapterbox"><b>按舞蹈版顺序跳转段落</b><div class="chapters">{''.join(entries)}</div></section>
</main><script>
const segments={payload}; const byId=Object.fromEntries(segments.map(x=>[x.id,x]));
const a=document.getElementById('official'), b=document.getElementById('live'), seek=document.getElementById('seek');
let localOffset=0.302, correcting=false;
function fmt(x){{if(!Number.isFinite(x))return '00:00';let m=Math.floor(x/60),s=Math.floor(x%60);return String(m).padStart(2,'0')+':'+String(s).padStart(2,'0')}}
function alignLive(force=false){{if(!Number.isFinite(a.currentTime))return;const target=Math.max(0,a.currentTime+localOffset);if(force||Math.abs(b.currentTime-target)>.16){{correcting=true;b.currentTime=target;setTimeout(()=>correcting=false,80)}}}}
function start(){{alignLive(true);b.play().catch(()=>{{}});a.play().catch(()=>{{}});document.getElementById('play').textContent='▶ 播放中'}}
function stop(){{a.pause();b.pause();document.getElementById('play').textContent='▶ 播放'}}
document.getElementById('play').onclick=start;document.getElementById('pause').onclick=stop;
document.getElementById('back').onclick=()=>jump(Math.max(0,a.currentTime-5));document.getElementById('forward').onclick=()=>jump(Math.min(a.duration||0,a.currentTime+5));
function jump(t,liveT=null){{stop();a.currentTime=t;b.currentTime=liveT===null?Math.max(0,t+localOffset):liveT;update()}}
a.addEventListener('timeupdate',()=>{{if(!correcting)alignLive();update()}});a.addEventListener('seeking',()=>{{if(!correcting)alignLive(true)}});
b.addEventListener('play',()=>{{if(a.paused&&!correcting){{a.play().catch(()=>{{}})}}}});a.addEventListener('pause',()=>{{if(!b.paused)b.pause()}});
function update(){{if(Number.isFinite(a.duration)&&a.duration>0)seek.value=String(Math.round(a.currentTime/a.duration*1000));document.getElementById('clock').textContent=fmt(a.currentTime)+' / '+fmt(a.duration)}}
seek.addEventListener('input',()=>{{if(a.duration)jump(a.duration*(Number(seek.value)/1000))}});
function chooseSegment(id,scroll=false){{const x=byId[id];if(!x)return;localOffset=x.live_time-x.official_time;jump(x.official_time,x.live_time);document.querySelectorAll('.chapter').forEach(btn=>btn.classList.toggle('active',btn.dataset.id===id));if(scroll)window.scrollTo({{top:0,behavior:'smooth'}});history.replaceState(null,'','?segment='+encodeURIComponent(id))}}
document.querySelectorAll('.chapter').forEach(btn=>btn.addEventListener('click',()=>chooseSegment(btn.dataset.id)));
document.querySelectorAll('[data-sound]').forEach(btn=>btn.addEventListener('click',()=>{{const v=btn.dataset.sound;a.muted=v!=='official';b.muted=v!=='live';document.querySelectorAll('[data-sound]').forEach(x=>x.classList.toggle('active',x===btn))}}));
a.addEventListener('loadedmetadata',update);const initial=new URLSearchParams(location.search).get('segment');if(initial&&byId[initial])chooseSegment(initial);
</script></body></html>"""
    (OUT / "video_comparison.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
