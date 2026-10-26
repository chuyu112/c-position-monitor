"""Integer-frame timing, stable rankings and auditable reports."""

from collections import Counter
import csv
import itertools
import json
from pathlib import Path


class FrameCounter:
    def __init__(self, members, fps):
        self.order = list(members)
        self.counts = Counter({name: 0 for name in members})
        self.fps = fps
        self.next_frame = 0
        self.active = []

    def step(self, row):
        names = row["members"]
        if row["frame"] != self.next_frame:
            raise ValueError("Counter frames must be consecutive from zero")
        if len(names) != len(row["boxes"]) or len(names) > 2 or len(set(names)) != len(names):
            raise ValueError("One distinct member is required for each C box")
        if any(n not in self.counts for n in names):
            raise ValueError("Unknown member")
        self.active = list(names)
        self.counts.update(names)
        self.order.sort(key=lambda n: -self.counts[n])
        self.next_frame += 1
        return {"frame": row["frame"], "active": self.active[:], "order": self.order[:],
                "cumulative_frames": dict(self.counts)}


def summarize(profile, rows, digest, check_baseline=True):
    counter = FrameCounter(profile["members"], profile["fps"])
    for row in rows:
        counter.step(row)
    cardinality = {str(n): sum(len(r["members"]) == n for r in rows) for n in range(3)}
    if check_baseline:
        if profile.get("expected_member_frames") is not None and dict(counter.counts) != profile["expected_member_frames"]:
            raise ValueError("Member frame totals differ from reviewed baseline")
        if profile.get("expected_c_count_frames") is not None and cardinality != profile["expected_c_count_frames"]:
            raise ValueError("C-count totals differ from reviewed baseline")
    fps = profile["fps"]
    return {"schema_version": 1, "profile": profile["name"], "annotations_sha256": digest,
            "baseline_checked": check_baseline, "frames": len(rows), "fps": fps,
            "duration_seconds": len(rows) / fps, "music_start_frame": profile["music_start_frame"],
            "annotation_size": profile["annotation_size"], "output_size": profile["output_size"],
            "member_c_frames": dict(counter.counts),
            "member_c_seconds": {n: counter.counts[n] / fps for n in profile["members"]},
            "c_count_frames": cardinality, "final_order": counter.order,
            "person_seconds": sum(counter.counts.values()) / fps,
            "timing_rule": "Each visible C box adds one frame to its supplied member; dual C counts both; ties keep previous order.",
            "identity_source": "Reviewed supplied names; no face recognition or automatic ID-to-name inference."}


def write_reports(directory, profile, rows, summary):
    directory = Path(directory)
    fps = profile["fps"]
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    intervals = []
    # Membership order only describes spatial slots, not a new timing interval.
    for names, group in itertools.groupby(rows, key=lambda r: tuple(sorted(r["members"]))):
        segment = list(group)
        start, end = segment[0]["frame"], segment[-1]["frame"] + 1
        intervals.append({"start_frame": start, "end_frame_exclusive": end,
                          "start_seconds": start / fps, "end_seconds": end / fps, "members": list(names)})
    (directory / "timing_intervals.json").write_text(json.dumps(intervals, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (directory / "member_times.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rank", "member", "chinese_name", "frames", "seconds"])
        for rank, name in enumerate(summary["final_order"], 1):
            writer.writerow([rank, name, profile.get("chinese_names", {}).get(name, ""),
                             summary["member_c_frames"][name], f'{summary["member_c_seconds"][name]:.6f}'])
    counter = FrameCounter(profile["members"], fps)
    with (directory / "ranking_frames.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(counter.step(row), ensure_ascii=False) + "\n")
