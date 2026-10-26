import json
from pathlib import Path
import unittest

from c_detect.core import Person
from c_detect.frontrow import FrontRowSelector


def dancer(i, x, foot, height=0.5):
    return Person(i, (x - 0.1, foot - height, x + 0.1, foot + 0.025), torso_x=x, foot_y=foot)


class FrontRowTests(unittest.TestCase):
    def test_two_sides_in_front_exclude_center_in_back(self):
        result = FrontRowSelector(stable_seconds=0).update([
            dancer(4, 0.2, 0.8), dancer(44, 0.5, 0.7), dancer(62, 0.85, 0.81)], 0)
        self.assertEqual(set(result.selected_ids), {4, 62})
        self.assertIsNone(result.selected_id)

    def test_one_front_person_is_single_c_even_off_center(self):
        result = FrontRowSelector(stable_seconds=0).update([
            dancer(1, 0.1, 0.85), dancer(2, 0.5, 0.7), dancer(3, 0.8, 0.69)], 0)
        self.assertEqual(result.selected_ids, [1])

    def test_horizontal_positions_do_not_change_front_row(self):
        for positions in ((0.1, 0.5, 0.9), (0.5, 0.1, 0.9), (0.95, 0.4, 0.2)):
            result = FrontRowSelector(stable_seconds=0).update([
                dancer(1, positions[0], 0.8), dancer(2, positions[1], 0.7), dancer(3, positions[2], 0.81)], 0)
            self.assertEqual(set(result.selected_ids), {1, 3})

    def test_three_indistinguishable_front_people_are_not_arbitrarily_ranked(self):
        result = FrontRowSelector(stable_seconds=0).update([
            dancer(1, 0.1, 0.8), dancer(2, 0.5, 0.8), dancer(3, 0.9, 0.8)], 0)
        self.assertEqual(result.selected_ids, [])
        self.assertEqual(result.status, "ambiguous_front_row")

    def test_brief_foot_lift_does_not_remove_double_c(self):
        selector = FrontRowSelector(stable_seconds=0)
        original = [dancer(1, 0.2, 0.8), dancer(2, 0.5, 0.7), dancer(3, 0.8, 0.8)]
        for i in range(20):
            selector.update(original, i / 30)
        lifted = [dancer(1, 0.2, 0.72), dancer(2, 0.5, 0.7), dancer(3, 0.8, 0.8)]
        self.assertEqual(set(selector.update(lifted, 20 / 30).selected_ids), {1, 3})

    def test_disappeared_person_is_never_selected(self):
        selector = FrontRowSelector(stable_seconds=0)
        selector.update([dancer(1, 0.2, 0.8), dancer(2, 0.8, 0.8)], 0)
        self.assertEqual(selector.update([dancer(2, 0.8, 0.8)], 0.1).selected_ids, [2])

    def test_real_user_labeled_frame(self):
        fixture = Path(__file__).parent / "fixtures" / "double_c_reference.json"
        data = json.loads(fixture.read_text())
        selector = FrontRowSelector()
        for row in data["frames"]:
            people = [Person(p["track_id"], tuple(p["box"]), p["detection_confidence"], p["torso_x"], p["foot_y"]) for p in row["people"]]
            result = selector.update(people, row["seconds"])
        self.assertEqual(set(result.selected_ids), set(data["expected_ids"]))


if __name__ == "__main__":
    unittest.main()
