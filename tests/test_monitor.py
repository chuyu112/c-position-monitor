import copy
import json
from pathlib import Path
import tempfile
import unittest

from c_monitor.dataset import load_profile, load_rows
from c_monitor.timing import FrameCounter, summarize, write_reports


def row(index, names):
    return {"frame": index, "members": names, "boxes": [[1, 1, 5, 8] for _ in names]}


class TimingTests(unittest.TestCase):
    def test_double_c_empty_frames_and_stable_ties(self):
        counter = FrameCounter(["A", "B"], 30)
        counter.step(row(0, ["B"]))
        self.assertEqual(counter.order, ["B", "A"])
        counter.step(row(1, ["A"]))
        self.assertEqual(counter.order, ["B", "A"])
        counter.step(row(2, ["A", "B"]))
        counter.step(row(3, []))
        self.assertEqual(dict(counter.counts), {"A": 2, "B": 2})
        counter.step(row(4, ["A"]))
        self.assertEqual(counter.order, ["A", "B"])

    def test_rejects_skips_and_duplicate_identity(self):
        counter = FrameCounter(["A", "B"], 30)
        with self.assertRaises(ValueError):
            counter.step(row(1, ["A"]))
        with self.assertRaises(ValueError):
            counter.step(row(0, ["A", "A"]))
        self.assertEqual(counter.next_frame, 0)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.profile = {"fps": 30, "frame_count": 1, "annotation_size": [10, 10],
                        "music_start_frame": 0, "members": ["A", "B"]}

    def check_rows(self, rows, profile=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            return load_rows(profile or self.profile, path, check_baseline=False)

    def test_rejects_nonconsecutive_nonfinite_unknown_and_mismatched_boxes(self):
        bad_rows = [row(1, ["A"]), row(0, ["X"]), row(0, ["A", "A"]),
                    {"frame": 0, "members": ["A"], "boxes": []},
                    {"frame": 0, "members": ["A"], "boxes": [[1, 1, float("nan"), 5]]},
                    {"frame": 0, "members": ["A"], "boxes": [[1, 1, 20, 5]]}]
        for bad in bad_rows:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.check_rows([bad])

    def test_music_preroll_requires_empty_rows(self):
        profile = dict(self.profile, music_start_frame=1)
        with self.assertRaises(ValueError):
            self.check_rows([row(0, ["A"])], profile)


class ReviewedBaselineTests(unittest.TestCase):
    def test_full_portrait_and_landscape_match_final_reviewed_totals(self):
        expected = {
            "portrait": (6824, 1843, 1292, 299.1),
            "landscape": (6657, 1802, 1408, 294.8),
        }
        for name, (frames, aijing, xili, person_seconds) in expected.items():
            with self.subTest(profile=name):
                profile = load_profile(name)
                rows, digest = load_rows(profile)
                summary = summarize(profile, rows, digest)
                self.assertEqual(summary["frames"], frames)
                self.assertEqual(summary["member_c_frames"]["AIJING"], aijing)
                self.assertEqual(summary["member_c_frames"]["XILI"], xili)
                self.assertAlmostEqual(summary["person_seconds"], person_seconds)
                with tempfile.TemporaryDirectory() as folder:
                    write_reports(folder, profile, rows, summary)
                    ranks = [json.loads(line) for line in (Path(folder) / "ranking_frames.jsonl").read_text().splitlines()]
                    self.assertEqual(ranks[-1]["cumulative_frames"], summary["member_c_frames"])
                    self.assertEqual(len(ranks), frames)
                    intervals = json.loads((Path(folder) / "timing_intervals.json").read_text())
                    self.assertEqual(intervals[0]["start_frame"], 0)
                    self.assertEqual(intervals[-1]["end_frame_exclusive"], frames)
                    for previous, current in zip(intervals, intervals[1:]):
                        self.assertEqual(previous["end_frame_exclusive"], current["start_frame"])

    def test_latest_landscape_corrections_and_music_start(self):
        profile = load_profile("landscape")
        rows, _ = load_rows(profile)
        self.assertTrue(all(not r["members"] for r in rows[:60]))
        self.assertTrue(rows[60]["members"])
        self.assertTrue(all(r["members"] == ["AIJING"] for r in rows[4071:4085]))
        self.assertTrue(all(set(r["members"]) == {"MERCI", "OA"} for r in rows[2294:2311]))

    def test_hash_check_detects_annotation_tampering(self):
        profile = copy.deepcopy(load_profile("portrait"))
        profile["annotations_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            load_rows(profile)


if __name__ == "__main__":
    unittest.main()
