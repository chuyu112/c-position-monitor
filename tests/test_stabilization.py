import unittest

import numpy as np

from c_detect.stabilization import stabilize_display_boxes


class StabilizationTests(unittest.TestCase):
    def test_rejects_jitter_while_following_real_motion(self):
        boxes = [[(40 + i + (4 if i % 2 else -4), 100, 80 + i + (4 if i % 2 else -4), 700)] for i in range(100)]
        result, metrics = stabilize_display_boxes(boxes, [dict(start_frame=0, end_frame=100, count=1)], 432, 960)
        centers = [(row[0][0] + row[0][2]) / 2 for row in result]
        self.assertLess(metrics['mean_edge_acceleration_after_px_per_frame2'], metrics['mean_edge_acceleration_before_px_per_frame2'] / 4)
        self.assertLess(np.max(np.abs(np.array(centers[20:80]) - (60 + np.arange(20, 80)))), 2)

    def test_does_not_blend_unrelated_people_at_role_switch(self):
        boxes = [[(20, 100, 60, 700)]] * 30 + [[(300, 200, 340, 800)]] * 30
        segments = [dict(start_frame=0, end_frame=30, count=1), dict(start_frame=30, end_frame=60, count=1)]
        result, _ = stabilize_display_boxes(boxes, segments, 432, 960)
        self.assertEqual(result[29], boxes[29])
        self.assertEqual(result[30], boxes[30])

    def test_converging_pairs_stay_distinct_and_inside_frame(self):
        boxes = [[(max(0, 80 - i * 2), 100, max(0, 80 - i * 2) + 50, 700),
                  (max(0, 140 - i * 3), 120, max(0, 140 - i * 3) + 50, 720)] for i in range(60)]
        result, _ = stabilize_display_boxes(boxes, [dict(start_frame=0, end_frame=60, count=2)], 432, 960)
        for pair in result:
            a, b = sorted(pair)
            self.assertGreaterEqual(b[0] - a[2], 8)
            for x1, y1, x2, y2 in pair:
                self.assertTrue(0 <= x1 < x2 < 432 and 0 <= y1 < y2 < 960)


if __name__ == '__main__':
    unittest.main()
