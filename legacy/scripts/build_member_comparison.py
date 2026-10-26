"""Build A/B vocal-stem excerpts grouped by the reviewed dance-role names.

These are listening/review excerpts, not automatic singer diarization or
performance scores. Duo dance intervals remain group evidence only.
"""
import html
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/neon-vocal-comparison"


def safe_rate(path):
    rate = sf.info(path).samplerate
    return rate


def crop(audio, rate, start, end):
    first, last = round(start * rate), round(end * rate)
    result = audio[max(0, first):min(len(audio), last)]
    return result if len(result) else np.zeros((1, audio.shape[1]), dtype=audio.dtype)


def main():
    alignment = json.loads((OUT / "alignment.json").read_text(encoding="utf-8"))
    prior = json.loads((ROOT / "annotations/neon_vocal_order_prior.json").read_text(encoding="utf-8"))
    stems = OUT / "separated/htdemucs"
    official, sr_official = sf.read(stems / "official_stereo/vocals.wav", dtype="float32", always_2d=True)
    live, sr_live = sf.read(stems / "live_stereo/vocals.wav", dtype="float32", always_2d=True)
    if sr_official != sr_live:
        raise ValueError("Separated tracks must use the same sample rate")
    sr = sr_official
    rows = {s["id"]: s for s in alignment["segments"]}
    out = OUT / "snippets"
    groups = {}
    summary = {}
    detail = []
    for seg in prior["segments"]:
        row = rows[seg["id"]]
        d0, d1 = seg["start_seconds"], seg["end_seconds"]
        # Local chroma alignment wins when it agrees with the global anchor.
        # Intro mismatch is marked for review and falls back to the global map.
        oi = row["official_local_alignment"]
        li = row["live_local_alignment"]
        oc = oi.get("offset_adjustment_seconds")
        lc = li.get("offset_adjustment_seconds")
        local_ok = all(x is not None and abs(x) <= 0.15 for x in (oc, lc))
        om = oi.get("best_offset_seconds") if local_ok else alignment["maps"]["dance_to_official"]["offset_seconds"]
        lm = li.get("best_offset_seconds") if local_ok else alignment["maps"]["dance_to_live"]["offset_seconds"]
        fit = min(oi.get("feature_fit") or 0, li.get("feature_fit") or 0)
        state = "provisional_aligned" if local_ok and fit >= 0.60 else "review_alignment"
        extra = 0.10
        starts = {"official": d0 + om - extra, "live": d0 + lm - extra}
        ends = {"official": d1 + om + extra, "live": d1 + lm + extra}
        names = seg["candidate_members"]
        group = names[0] if len(names) == 1 else "group_" + "_".join(names)
        directory = out / group
        directory.mkdir(parents=True, exist_ok=True)
        op = f"{group}/{seg['id']}_official.wav"
        lp = f"{group}/{seg['id']}_live.wav"
        oclip = crop(official, sr, max(0, starts["official"]), max(0, ends["official"]))
        lclip = crop(live, sr, max(0, starts["live"]), max(0, ends["live"]))
        sf.write(out / op, oclip, sr, subtype="PCM_16")
        sf.write(out / lp, lclip, sr, subtype="PCM_16")
        length = max(0.0, d1 - d0)
        row_out = {
            "id": seg["id"], "dance_role_candidates": names,
            "assignment_status": "dance_role_proxy_not_confirmed_singer",
            "dance_start_seconds": d0, "dance_end_seconds": d1,
            "official_audio_start_seconds": round(max(0, starts["official"]), 3),
            "live_audio_start_seconds": round(max(0, starts["live"]), 3),
            "snippet_duration_seconds": round(length + 2 * extra, 3),
            "local_chroma_fit": fit, "alignment_status": state,
            "official_vocal_excerpt": "snippets/" + op,
            "live_vocal_excerpt": "snippets/" + lp,
            "metrics_status": "not_scored_no_validated_audio_quality_model",
        }
        detail.append(row_out)
        groups.setdefault(group, []).append(row_out)
        if len(names) == 1:
            member = names[0]
            slot = summary.setdefault(member, {"candidate_solo_segments": 0, "candidate_solo_seconds": 0.0,
                                               "review_alignment_segments": 0, "candidate_duet_segments": 0})
            slot["candidate_solo_segments"] += 1
            slot["candidate_solo_seconds"] += length
            slot["review_alignment_segments"] += state != "provisional_aligned"
        else:
            for member in names:
                summary.setdefault(member, {"candidate_solo_segments": 0, "candidate_solo_seconds": 0.0,
                                            "review_alignment_segments": 0, "candidate_duet_segments": 0})["candidate_duet_segments"] += 1
    for member, item in summary.items():
        item["candidate_solo_seconds"] = round(item["candidate_solo_seconds"], 3)
        item["note"] = "Dance role only; singing assignment awaits audio review. Does not represent singing duration or score."
    result = {
        "schema_version": 1, "alignment_report": "alignment.json",
        "vocal_separation": {"model": "htdemucs", "outputs": "two-stem vocals", "sample_rate": sr,
                             "warning": "Source separation can alter vocal timbre; compare pitch/timing cautiously."},
        "member_assignment_source": str(ROOT / "annotations/neon_vocal_order_prior.json"),
        "member_assignment_warning": "Dance C/role candidates are not verified vocal diarization. Dual role sections are retained under a shared group and are not credited to either member.",
        "score_status": "no_total_score_produced_unvalidated_quality_model_and_unconfirmed_singer_mapping",
        "members": summary, "groups": {k: [{"id": r["id"], "duration": r["snippet_duration_seconds"],
                                            "alignment": r["alignment_status"]} for r in v]
                                       for k,v in groups.items()},
        "segments": detail,
    }
    (OUT / "member_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sections = []
    for group, items in groups.items():
        cards = []
        for r in items:
            cards.append(f"""<article><h3>{html.escape(r['id'])} · {html.escape(r['alignment_status'])}</h3>
<p>{r['dance_start_seconds']:.2f}–{r['dance_end_seconds']:.2f}s dance-role interval · local chroma fit {r['local_chroma_fit']:.2f} · singer identity unconfirmed</p>
<div class="pair"><label>Official<input type="range" min="0" max="1" value="0.85" step="0.01" oninput="this.nextElementSibling.volume=this.value"><audio controls preload="none" src="snippets/{html.escape(group)}/{html.escape(r['id'])}_official.wav"></audio></label>
<label>Live<input type="range" min="0" max="1" value="0.85" step="0.01" oninput="this.nextElementSibling.volume=this.value"><audio controls preload="none" src="snippets/{html.escape(group)}/{html.escape(r['id'])}_live.wav"></audio></label></div></article>""")
        sections.append(f"<section><h2>{html.escape(group)}</h2>" + "\n".join(cards) + "</section>")
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>NEON 逐段人声对比</title><style>
body{font:16px system-ui;max-width:1000px;margin:2em auto;padding:0 1em;color:#21252b}section{border-top:2px solid #ddd;margin-top:2em}article{border:1px solid #ddd;border-radius:8px;padding:1em;margin:1em 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:1em}audio{display:block;width:100%;margin-top:.4em}input{width:80px}small{color:#59636e}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style>
<h1>NEON · 逐段音轨对比</h1><p>先检查全局对齐，再试听每段。成员是舞蹈角色候选，不等于已确认开口演唱的人。双人段保留在组合项。对齐值是音频指纹，不是质量分；这份报告没有生成唱功总分。</p>""" + "\n".join(sections) + "</html>\n"
    (OUT / "comparison.html").write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(OUT), "segments": len(detail), "member_groups": sorted(summary),
                      "group_count": len(groups), "review_alignment": [r["id"] for r in detail if r["alignment_status"] != "provisional_aligned"],
                      "report": str(OUT / "comparison.html")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
