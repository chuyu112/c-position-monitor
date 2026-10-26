import unittest

from c_detect.core import CenterSelector, Person


def person(track_id, x, foot=None):
    return Person(track_id, (x - 0.04, 0.1, x + 0.04, 0.9), foot_y=foot)


class CenterSelectorTests(unittest.TestCase):
    def test_uses_group_center_when_formation_is_offscreen_center(self):
        selector = CenterSelector(stable_seconds=0)
        result = selector.update([person(1, 0.1), person(2, 0.25), person(3, 0.4)], 0)
        self.assertEqual(result.selected_id, 2)

    def test_equal_double_centers_are_unknown(self):
        selector = CenterSelector(stable_seconds=0)
        result = selector.update([person(i, x) for i, x in enumerate((0.1, 0.4, 0.6, 0.9))], 0)
        self.assertIsNone(result.selected_id)
        self.assertEqual(result.status, "ambiguous")

    def test_front_center_beats_rear_center_when_ankles_are_visible(self):
        selector = CenterSelector(stable_seconds=0)
        result = selector.update([person(1, 0.2, 0.6), person(2, 0.5, 0.6), person(3, 0.5, 0.9), person(4, 0.8, 0.6)], 0)
        self.assertEqual(result.selected_id, 3)

    def test_short_crossing_does_not_confirm_new_center(self):
        selector = CenterSelector(stable_seconds=0.3, smoothing_seconds=0)
        original = [person(1, 0.1), person(2, 0.5), person(3, 0.9)]
        swapped = [person(1, 0.5), person(2, 0.1), person(3, 0.9)]
        self.assertIsNone(selector.update(original, 0).selected_id)
        self.assertEqual(selector.update(original, 0.3).selected_id, 2)
        self.assertIsNone(selector.update(swapped, 0.4).selected_id)
        self.assertEqual(selector.update(original, 0.5).selected_id, 2)
        self.assertIsNone(selector.update(swapped, 0.6).selected_id)
        self.assertEqual(selector.update(swapped, 0.9).selected_id, 1)

    def test_missing_person_never_remains_selected(self):
        selector = CenterSelector(stable_seconds=0.3, smoothing_seconds=0)
        original = [person(1, 0.1), person(2, 0.5), person(3, 0.9)]
        selector.update(original, 0)
        self.assertEqual(selector.update(original, 0.3).selected_id, 2)
        result = selector.update([person(1, 0.1), person(3, 0.5), person(4, 0.9)], 0.4)
        self.assertIsNone(result.selected_id)
        self.assertNotIn(2, result.scores)

    def test_low_confidence_detections_do_not_satisfy_group_size(self):
        selector = CenterSelector(stable_seconds=0)
        result = selector.update([person(1, 0.2), person(2, 0.5), Person(3, (0.7, 0, 0.9, 1), 0.1)], 0)
        self.assertEqual(result.status, "insufficient_people")

    def test_empty_frame_resets_pending_confirmation(self):
        selector = CenterSelector(stable_seconds=0.3)
        people = [person(1, 0.1), person(2, 0.5), person(3, 0.9)]
        selector.update(people, 0)
        selector.update([], 0.2)
        self.assertIsNone(selector.update(people, 0.4).selected_id)
        self.assertEqual(selector.update(people, 0.7).selected_id, 2)

    def test_long_gap_requires_reconfirmation(self):
        selector = CenterSelector(stable_seconds=0.3)
        people = [person(1, 0.1), person(2, 0.5), person(3, 0.9)]
        selector.update(people, 0)
        self.assertEqual(selector.update(people, 0.3).selected_id, 2)
        self.assertIsNone(selector.update(people, 3).selected_id)

    def test_relay_accepts_one_visible_lead(self):
        selector = CenterSelector(mode="relay", min_people=1, stable_seconds=0)
        self.assertEqual(selector.update([person(1, 0.5, 0.8)], 0).selected_id, 1)

    def test_relay_does_not_move_axis_to_outgoing_dancer(self):
        selector = CenterSelector(mode="relay", min_people=1, stable_seconds=0)
        self.assertIsNone(selector.update([person(1, 0.05, 0.8)], 0).selected_id)

    def test_relay_does_not_promote_occluded_torso_at_center(self):
        selector = CenterSelector(mode="relay", min_people=1, stable_seconds=0)
        lead = Person(1, (0.3, 0.3, 0.7, 0.85), torso_x=0.4, foot_y=0.8)
        hidden = Person(2, (0.4, 0.3, 0.6, 0.45), torso_x=0.5)
        self.assertEqual(selector.update([lead, hidden], 0).selected_id, 1)

    def test_relay_dual_front_leads_remain_ambiguous(self):
        selector = CenterSelector(mode="relay", min_people=1, stable_seconds=0)
        self.assertEqual(selector.update([person(1, 0.4, 0.8), person(2, 0.6, 0.8)], 0).status, "ambiguous")


if __name__ == "__main__":
    unittest.main()
