"""Front-row selection: one or two people; horizontal centrality has no vote."""

from collections import deque
from math import isfinite
from statistics import median

from .core import Decision


class FrontRowSelector:
    def __init__(self, stable_seconds=0.25, row_tolerance=0.075, history_seconds=0.6):
        if not all(isfinite(v) and v >= 0 for v in (stable_seconds, row_tolerance, history_seconds)):
            raise ValueError("time and tolerance settings must be finite and nonnegative")
        if row_tolerance == 0 or history_seconds == 0:
            raise ValueError("row_tolerance and history_seconds must be positive")
        self.stable_seconds = stable_seconds
        self.row_tolerance = row_tolerance
        self.history_seconds = history_seconds
        self._last_time = None
        self._history = {}
        self._pending = ()
        self._pending_since = 0.0
        self._selected = ()

    def update(self, people, timestamp):
        if not isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and nonnegative")
        if self._last_time is not None and timestamp < self._last_time:
            raise ValueError("timestamps must not move backwards")
        if self._last_time is not None and timestamp - self._last_time > 1:
            self._history.clear()
            self._selected = self._pending = ()
        self._last_time = timestamp
        people = [p for p in people if p.confidence >= 0.25]
        ids = {p.track_id for p in people}
        if len(ids) != len(people):
            raise ValueError("duplicate track IDs in one frame")
        self._selected = tuple(i for i in self._selected if i in ids)
        self._history = {i: h for i, h in self._history.items() if h and timestamp - h[-1][0] <= self.history_seconds}
        if not people:
            self._pending = self._selected = ()
            return Decision("insufficient_people", None, None)

        heights = {p.track_id: p.box[3] - p.box[1] for p in people}
        tallest = max(heights.values())
        depths = {}
        for p in people:
            height = heights[p.track_id]
            history = self._history.setdefault(p.track_id, deque())
            while history and timestamp - history[0][0] > self.history_seconds:
                history.popleft()
            foot = p.foot_y
            # Reject hallucinated ankles well below a truncated torso box.
            reliable = foot is not None and isfinite(foot) and p.box[1] + 0.35 * height < foot <= min(1, p.box[3] + 0.035)
            if reliable:
                observation = foot
            elif height >= 0.7 * tallest and p.confidence >= 0.5:
                # Full-body box fallback when feet are occluded. Subtract shoe height.
                observation = p.box[3] - 0.06 * height
            else:
                observation = None
            if observation is not None:
                history.append((timestamp, observation))
            if history and timestamp - history[-1][0] <= 0.25:
                values = sorted(v for _, v in history)
                # Recent high quantile approximates contact position during kicks/jumps.
                depths[p.track_id] = values[int((len(values) - 1) * 0.8)]
        if not depths:
            self._pending = self._selected = ()
            return Decision("depth_unavailable", None, None)

        scale = median(heights[i] for i in depths)
        band = max(0.018, min(0.05, self.row_tolerance * scale))
        front = max(depths.values())
        candidate_ids = [i for i, depth in depths.items() if front - depth <= band + 1e-9]
        # No center-based tie break: three indistinguishable front people stay unknown.
        if len(candidate_ids) > 2:
            self._pending = self._selected = ()
            return Decision("ambiguous_front_row", None, None, depths=depths)
        candidate = tuple(sorted(candidate_ids))
        if candidate == self._selected:
            self._pending = ()
            return Decision("selected", None, None, selected_ids=list(candidate), depths=depths)
        if candidate != self._pending:
            self._pending, self._pending_since = candidate, timestamp
        if timestamp - self._pending_since + 1e-9 < self.stable_seconds:
            return Decision("stabilizing", None, None, depths=depths)
        self._selected, self._pending = candidate, ()
        return Decision("selected", None, None, selected_ids=list(candidate), depths=depths)
