"""Geometry and temporal selection, independent of the vision runtime.

Scores are heuristic rankings, NOT probabilities of being the choreographic C.
All coordinates are normalized to the image size.
"""

from dataclasses import dataclass, field
from math import exp, isfinite
from statistics import median
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Person:
    track_id: int
    box: Tuple[float, float, float, float]
    confidence: float = 1.0
    torso_x: Optional[float] = None
    foot_y: Optional[float] = None

    @property
    def x(self) -> float:
        return self.torso_x if self.torso_x is not None else (self.box[0] + self.box[2]) / 2


@dataclass
class Decision:
    status: str
    selected_id: Optional[int]
    group_x: Optional[float]
    scores: Dict[int, float] = field(default_factory=dict)
    selected_ids: List[int] = field(default_factory=list)
    depths: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.selected_id is not None and not self.selected_ids:
            self.selected_ids = [self.selected_id]
        self.selected_id = self.selected_ids[0] if len(self.selected_ids) == 1 else None


class CenterSelector:
    def __init__(self, min_people=3, margin=0.08, stable_seconds=0.35, smoothing_seconds=0.18, mode="formation", axis_x=0.5):
        if mode not in ("formation", "relay") or not 0 <= axis_x <= 1:
            raise ValueError("mode must be formation/relay and axis_x must be in [0, 1]")
        if min_people < (1 if mode == "relay" else 2) or not 0 < margin <= 1:
            raise ValueError("min_people is too small for this mode; margin must be in (0, 1]")
        if not all(isfinite(x) and x >= 0 for x in (stable_seconds, smoothing_seconds)):
            raise ValueError("time settings must be finite and nonnegative")
        self.min_people = min_people
        self.margin = margin
        self.stable_seconds = stable_seconds
        self.smoothing_seconds = smoothing_seconds
        self.mode = mode
        self.axis_x = axis_x
        self._last_time = None
        self._clear()

    def _clear(self):
        self._scores = {}
        self._selected = None
        self._pending = None
        self._pending_since = 0.0

    def update(self, people: List[Person], timestamp: float) -> Decision:
        if not isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and nonnegative")
        dt = 0 if self._last_time is None else timestamp - self._last_time
        if dt < 0:
            raise ValueError("timestamps must not move backwards")
        self._last_time = timestamp
        if dt > 1:
            self._clear()
        people = [p for p in people if p.confidence >= 0.25]
        ids = {p.track_id for p in people}
        if len(ids) != len(people):
            raise ValueError("duplicate track IDs in one frame")
        if len(people) < self.min_people:
            self._clear()
            return Decision("insufficient_people", None, None)

        group_x = self.axis_x if self.mode == "relay" else median(p.x for p in people)
        span = max(0.1, max(p.x for p in people) - min(p.x for p in people))
        tallest = max(p.box[3] - p.box[1] for p in people)
        feet = [p.foot_y for p in people if p.foot_y is not None]
        depth_span = max(feet) - min(feet) if len(feet) >= 2 else 0
        alpha = 1 if self.smoothing_seconds == 0 else 1 - exp(-dt / self.smoothing_seconds)
        scores = {}
        for p in people:
            centrality = max(0.0, 1 - 2 * abs(p.x - group_x) / span)
            # Only a weak perspective cue. Kneeling/jumping can invalidate it.
            frontness = (p.foot_y - min(feet)) / depth_span if p.foot_y is not None and depth_span > 0.03 else 0.5
            if self.mode == "relay":
                centrality = max(0.0, 1 - abs(p.x - group_x) / 0.5)
                size = (p.box[3] - p.box[1]) / max(tallest, 1e-6)
                raw = 0.55 * centrality + 0.30 * frontness + 0.15 * size
                # A visible head/torso behind the lead is not itself a full lead.
                if size < 0.55:
                    raw *= size / 0.55
            else:
                raw = 0.8 * centrality + 0.2 * frontness
            scores[p.track_id] = raw if p.track_id not in self._scores else alpha * raw + (1 - alpha) * self._scores[p.track_id]
        self._scores = scores  # Missing tracks must never remain selectable.
        if self._selected not in ids:
            self._selected = None
        ranked = sorted(scores, key=lambda i: (-scores[i], i))
        best = ranked[0]
        runner_up = scores[ranked[1]] if len(ranked) > 1 else 0.0
        if scores[best] < 0.55 or scores[best] - runner_up < self.margin:
            self._pending = None
            return Decision("ambiguous", None, group_x, scores)
        if best == self._selected:
            self._pending = None
            return Decision("selected", best, group_x, scores)
        if best != self._pending:
            self._pending, self._pending_since = best, timestamp
        if timestamp - self._pending_since + 1e-9 < self.stable_seconds:
            return Decision("stabilizing", None, group_x, scores)
        self._selected, self._pending = best, None
        return Decision("selected", best, group_x, scores)
