#!/usr/bin/env python3
"""Prepare the user-selected first OA segment for sequential A/B review."""

from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "neon-vocal-comparison"
SEGMENT_ID = "dance_003"
HOP = 512
SAMPLE_RATE = 22050
OFFICIAL_VIDEO = ROOT / "downloads" / "BV1VVpo6gEYg" / "BV1VVpo6gEYg_720p.mp4"
LIVE_VIDEO = ROOT / "downloads" / "BV1sVnP6kEdn" / "BV1sVnP6kEdn_720p.mp4"
RUBRIC_WEIGHTS = {"intonation_accuracy": 0.40, "pitch_control": 0.30, "timbre_quality": 0.30}


def local_pitch_diagnostic() -> dict:
    comparison = json.loads((OUT / "member_comparison.json").read_text(encoding="utf-8"))
    clip = next(s for s in comparison["segments"] if s["id"] == SEGMENT_ID)
    alignment = json.loads((OUT / "alignment.json").read_text(encoding="utf-8"))
    row = next(s for s in alignment["segments"] if s["id"] == SEGMENT_ID)
    official = np.load(OUT / "pitch" / "official.npz")
    live = np.load(OUT / "pitch" / "live.npz")
    frame_seconds = HOP / SAMPLE_RATE
    phrase = next(s for s in alignment["segments"] if s["id"] == SEGMENT_ID)
    start = float(phrase["dance_start_seconds"]) + 0.2
    end = float(phrase["dance_end_seconds"]) - 0.2
    official_shift = float(row["official_local_alignment"]["best_offset_seconds"])
    live_shift = float(row["live_local_alignment"]["best_offset_seconds"])
    official_times = np.arange(start + official_shift, end + official_shift, frame_seconds)
    live_times = official_times - (official_shift - live_shift)
    oi = np.rint(official_times / frame_seconds).astype(int)
    li = np.rint(live_times / frame_seconds).astype(int)
    valid = (oi >= 0) & (oi < len(official["f0"])) & (li >= 0) & (li < len(live["f0"]))
    oi, li = oi[valid], li[valid]
    rows = {}
    for threshold in (0.25, 0.35, 0.45):
        o_f0, l_f0 = official["f0"][oi], live["f0"][li]
        mask = (official["voiced_probability"][oi] >= threshold)
        mask &= live["voiced_probability"][li] >= threshold
        mask &= official["voiced"][oi] & live["voiced"][li]
        mask &= np.isfinite(o_f0) & np.isfinite(l_f0) & (o_f0 > 0) & (l_f0 > 0)
        cents = 1200 * np.log2(l_f0[mask] / o_f0[mask])
        # Prototype pitch kernel. Here official F0 is only a melody proxy;
        # this is a pitch-contour diagnostic, not an all-around quality score.
        deviation = np.maximum(0.0, np.abs(cents) - 25.0) / 75.0
        pitch_index = 100.0 * np.exp(-np.minimum(deviation, 40.0) ** 2).mean() if len(cents) else None
        rows[str(threshold)] = {
            "matched_frames": int(len(cents)),
            "observable_seconds": round(float(len(cents) * frame_seconds), 3),
            "frame_coverage": round(float(mask.mean()), 4) if len(mask) else 0.0,
            "median_absolute_cents": round(float(np.median(np.abs(cents))), 1) if len(cents) else None,
            "within_50_cents_percent": round(float(np.mean(np.abs(cents) <= 50) * 100), 1) if len(cents) else None,
            "pitch_only_index": round(float(pitch_index), 2) if pitch_index is not None else None,
        }
    # Four phrase boundaries are approximate, based on the four vocal lines
    # occupying the selected 14.13-second excerpt. The matched-frame pitch
    # statistics below are diagnostics only: averaging only detected frames
    # creates selection bias when most of the Live vocal is not observable.
    line_bounds = ((0.0, 3.5), (3.5, 6.5), (6.5, 9.5), (9.5, 13.533))
    line_rows = []
    for line_number, (line_start, line_end) in enumerate(line_bounds, start=1):
        official_times = np.arange(start + line_start + official_shift,
                                   start + line_end + official_shift, frame_seconds)
        live_times = official_times - (official_shift - live_shift)
        oi = np.rint(official_times / frame_seconds).astype(int)
        li = np.rint(live_times / frame_seconds).astype(int)
        valid = (oi >= 0) & (oi < len(official["f0"]))
        valid &= (li >= 0) & (li < len(live["f0"]))
        oi, li = oi[valid], li[valid]
        o_f0, l_f0 = official["f0"][oi], live["f0"][li]
        mask = official["voiced_probability"][oi] >= 0.15
        mask &= live["voiced_probability"][li] >= 0.15
        mask &= official["voiced"][oi] & live["voiced"][li]
        mask &= np.isfinite(o_f0) & np.isfinite(l_f0) & (o_f0 > 0) & (l_f0 > 0)
        cents = 1200 * np.log2(l_f0[mask] / o_f0[mask])
        line_rows.append({
            "line": line_number,
            "approx_start_seconds": round(line_start, 3),
            "approx_end_seconds": round(line_end, 3),
            "matched_frames": int(len(cents)),
            "observable_seconds": round(float(len(cents) * frame_seconds), 3),
            "frame_coverage": round(float(len(cents) / len(oi)), 4) if len(oi) else 0.0,
            "median_absolute_cents": round(float(np.median(np.abs(cents))), 1) if len(cents) else None,
            "within_50_cents_percent": round(float(np.mean(np.abs(cents) <= 50) * 100), 1) if len(cents) else None,
        })
    return {
        "segment_id": SEGMENT_ID,
        "dance_role_member": "OA",
        "member_assignment": "user_confirmed_for_this_segment",
        "score_mode": "live_standalone_with_official_as_reference_only",
        "reference_role": ["confirm_phrase_alignment", "provide_pitch_targets", "define_100_point_level"],
        "official_start_seconds": round(float(clip["official_audio_start_seconds"]), 3),
        "live_start_seconds": round(float(clip["live_audio_start_seconds"]), 3),
        "duration_seconds": round(float(clip["snippet_duration_seconds"]), 3),
        "rubric_dimensions": ["音准", "音高控制", "音色"],
        "rubric_weights": RUBRIC_WEIGHTS,
        "phrase_seconds_used_for_pitch": round(end - start, 3),
        "pitch_diagnostic": rows,
        "line_pitch_diagnostics": line_rows,
        "line_pitch_diagnostic_method": "per-line same-time pYIN F0 diagnostics at probability >= 0.15; statistics describe matched frames only and are not scores",
        "overall_score": None,
        "overall_score_status": "live_only_score_not_yet_reliable_from_current_features",
        "overall_coverage_at_confidence_0_45": rows["0.45"]["frame_coverage"],
        "overall_observable_seconds_at_confidence_0_45": rows["0.45"]["observable_seconds"],
        "score_note": "This is a Live-only singing-quality score. The official recording is used only to confirm the corresponding phrase, provide candidate target notes, and define the 100-point reference level. Waveform or timbre similarity is not scored. The current pitch diagnostic is based on sparse paired F0 frames and is not an overall score; pitch control and timbre quality are not yet calibrated for this recording condition.",
    }


def build_sequence_video(meta: dict) -> Path:
    """Join the matched official/live excerpts and show a truthful score card at Live start."""
    destination = OUT / "oa_first_segment_official_then_live.mp4"
    official_start = meta["official_start_seconds"]
    live_start = meta["live_start_seconds"]
    duration = meta["duration_seconds"]
    font = "'/System/Library/Fonts/Hiragino Sans GB.ttc'"
    video_filters = (
        f"[0:v:0]trim=duration={duration},setpts=PTS-STARTPTS,fps=30,"
        f"drawtext=fontfile={font}:text='官方录音版｜OA唱段':fontcolor=white:fontsize=34:"
        "box=1:boxcolor=black@0.7:boxborderw=14:x=36:y=32:enable='lt(t,5)'[v0];"
        f"[0:a:0]atrim=duration={duration},asetpts=PTS-STARTPTS,aresample=48000[a0];"
        f"[1:v:0]trim=duration={duration},setpts=PTS-STARTPTS,fps=30,"
        f"drawtext=fontfile={font}:text='LIVE现场版｜OA唱段':fontcolor=white:fontsize=34:"
        "box=1:boxcolor=black@0.7:boxborderw=14:x=36:y=28,"
        f"drawtext=fontfile={font}:text='LIVE独立评分：待有效数据（音准 · 音高控制 · 音色）':"
        "fontcolor=white:fontsize=30:box=1:boxcolor=black@0.78:boxborderw=16:"
        "x=36:y=92[v1];"
        f"[1:a:0]atrim=duration={duration},asetpts=PTS-STARTPTS,aresample=48000[a1];"
        "[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]"
    )
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(official_start), "-i", str(OFFICIAL_VIDEO),
            "-ss", str(live_start), "-i", str(LIVE_VIDEO),
            "-filter_complex", video_filters,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "h264_videotoolbox", "-b:v", "5M",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(destination),
        ],
        check=True,
    )
    return destination


def build_review_page(meta: dict) -> Path:
    diagnostic = meta["pitch_diagnostic"]["0.45"]
    line_rows = "".join(
        "<tr>"
        f"<td>第{r['line']}句（约 {r['approx_start_seconds']:.1f}–{r['approx_end_seconds']:.1f}s）</td>"
        f"<td>{r['median_absolute_cents']:.0f} cents</td>"
        f"<td>{r['within_50_cents_percent']:.1f}%</td>"
        f"<td>{r['observable_seconds']:.2f}s / {r['frame_coverage']*100:.1f}%</td>"
        "</tr>"
        for r in meta["line_pitch_diagnostics"]
    )
    video_url = "oa_first_segment_official_then_live.mp4"
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OA 第一段｜官方版 → Live 版</title>
<style>
body{{margin:0;background:#10131a;color:#edf2fa;font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1050px;margin:auto;padding:24px}}h1{{margin:0 0 5px}}.muted{{color:#aab4c5}}video{{width:100%;display:block;background:#050608;border-radius:12px;margin:18px 0}}
.buttons{{display:flex;gap:10px;flex-wrap:wrap}}button{{border:1px solid #384255;border-radius:8px;background:#222a38;color:#edf2fa;padding:10px 14px;cursor:pointer}}button:hover{{border-color:#97f3cd}}
.card{{background:#191e29;border:1px solid #2a3242;border-radius:12px;padding:16px;margin:14px 0}}.score{{font-size:25px;color:#97f3cd;font-weight:700}}.warning{{border-color:#66562b;background:#292416;color:#f1dfaa}}
table{{width:100%;border-collapse:collapse;margin:10px 0}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #303849}}th{{color:#b8c5d8}}.small{{font-size:14px;color:#aab4c5}}
</style></head><body><main>
<a class="muted" href="frame_comparison.html">← 返回逐段截图</a><h1>OA 第一段：官方录音版 → Live 版</h1>
<p class="muted">已合并为一条视频：官方版 {meta['official_start_seconds']:.3f}s 起 14.13 秒，接现场版 {meta['live_start_seconds']:.3f}s 起 14.13 秒。Live 段评分说明会一直显示到这段结束。</p>
<video controls playsinline preload="metadata" src="{video_url}"></video>
<p><a href="{video_url}" download>下载合并后的 MP4</a></p>
<section class="card"><h2>Live 独立唱功分</h2><div class="score">四句综合分：当前算法不能可靠给出</div>
<p>旧版的 93.8 以及刚才列出的逐句高分都撤回。它们只平均了成功检测到的少量音高帧，把大部分未识别的现场人声排除在计算之外，结果会虚高。当前高可信音高检测只有 {diagnostic['matched_frames']} 帧，约 {diagnostic['observable_seconds']:.2f} 秒、占整段 {diagnostic['frame_coverage']*100:.1f}%；音高控制与音色也没有足够可靠的测量值。官方版只用于对齐乐句和提供候选目标音高，不按声音相似度加分。</p></section>
<section class="card"><h2>OA 四句：可核对的音准原始指标</h2>
<table><thead><tr><th>唱句</th><th>检测帧中位绝对偏差</th><th>检测帧落在 ±50 cents</th><th>有效配对人声</th></tr></thead><tbody>{line_rows}</tbody></table>
<p class="small">这些数只描述被检测到的片段，不能解释为句子得分；每句有效配对人声仅占约 14%–27%，未检测片段不能当作唱准或唱错。四句边界按 14.13 秒片段粗分。</p></section>
<section class="card warning"><b>本段成员：</b>按你的确认，以 OA 作为这段的演唱成员。分数仅对应这一个约 14.1 秒片段，不能代表整首歌。</section>
</main></body></html>"""
    path = OUT / "oa_first_segment.html"
    path.write_text(page, encoding="utf-8")
    return path


def main() -> None:
    meta = local_pitch_diagnostic()
    video = build_sequence_video(meta)
    meta["sequence_video"] = video.name
    page = build_review_page(meta)
    meta["review_page"] = page.name
    (OUT / "oa_first_segment_score.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"video": str(video), "review_page": str(page),
                      "pitch_only_index": meta["pitch_diagnostic"]["0.45"]["pitch_only_index"],
                      "coverage": meta["overall_coverage_at_confidence_0_45"],
                      "overall_score": meta["overall_score"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
