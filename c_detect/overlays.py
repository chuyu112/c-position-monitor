"""Compact display markers; these do not alter detection or C-role boxes."""

from math import isfinite


def narrow_c_boxes(boxes, width, height, torso_xs=None, manual_pair=False, gap=8):
    """Keep full body height, narrow the width, and separate two C markers.

    Returned pixel rectangles are display markers, not training annotations.
    Arms, hair and spread legs may extend outside these deliberately narrow boxes.
    """
    if len(boxes) > 2:
        raise ValueError('C markers support at most two people')
    torso_xs = torso_xs or [None] * len(boxes)
    markers = []
    left_to_right = sorted(range(len(boxes)), key=lambda i: boxes[i][0] + boxes[i][2])
    for index, (box, torso) in enumerate(zip(boxes, torso_xs)):
        x1, y1, x2, y2 = box
        body_width = (x2 - x1) * width
        body_height = (y2 - y1) * height
        center = (x1 + x2) * width / 2
        if torso is not None and isfinite(torso) and x1 <= torso <= x2:
            # Limit the influence of a partly occluded shoulder/hip estimate.
            displacement = max(-.22 * body_width, min(.22 * body_width, torso * width - center))
            center += .8 * displacement
        elif manual_pair and len(boxes) == 2:
            # The reviewed foreground pair leans away from the middle.
            center += (-1 if index == left_to_right[0] else 1) * .25 * body_width
        marker_width = min(body_width * .52, body_height * .19, width * .22)
        marker_width = max(min(24, body_width), marker_width)
        center = max(marker_width / 2, min(width - 1 - marker_width / 2, center))
        markers.append([center, marker_width, max(0, round(y1 * height) - 2), min(height - 1, round(y2 * height) + 2)])
    return _marker_rectangles(markers, width, height, gap)


def constrain_display_boxes(boxes, width, height, gap=8):
    """Reapply frame bounds and pair separation after temporal smoothing."""
    if not 1 <= len(boxes) <= 2:
        raise ValueError('Expected one or two C markers')
    markers = [[(a + c) / 2, c - a, b, d] for a, b, c, d in boxes]
    return _marker_rectangles(markers, width, height, gap)


def _marker_rectangles(markers, width, height, gap):
    for marker in markers:
        marker[1] = max(2, min(width - 1, marker[1]))
        marker[0] = max(marker[1] / 2, min(width - 1 - marker[1] / 2, marker[0]))
    if len(markers) == 2:
        left, right = sorted(range(2), key=lambda i: markers[i][0])
        a, b = markers[left], markers[right]
        distance = b[0] - a[0]
        # Brief crossings still need two distinct markers. Move only the display
        # centers by the minimum amount needed to leave a visible gap.
        minimum_distance = min(28, a[1], b[1]) + gap + 2
        if distance < minimum_distance:
            midpoint = (a[0] + b[0]) / 2
            margin = minimum_distance / 2 + min(28, a[1], b[1]) / 2 + 1
            midpoint = max(margin, min(width - 1 - margin, midpoint))
            a[0], b[0] = midpoint - minimum_distance / 2, midpoint + minimum_distance / 2
            distance = minimum_distance
        maximum_width = max(2, distance - gap - 2)
        a[1], b[1] = min(a[1], maximum_width), min(b[1], maximum_width)
    rectangles = []
    for center, marker_width, top, bottom in markers:
        left = max(0, min(width - 2, round(center - marker_width / 2)))
        right = max(left + 1, min(width - 1, round(center + marker_width / 2)))
        top = max(0, min(height - 2, round(top)))
        bottom = max(top + 1, min(height - 1, round(bottom)))
        rectangles.append((left, top, right, bottom))
    if len(rectangles) == 2:
        a, b = sorted(rectangles)
        if b[0] - a[2] < gap:
            raise ValueError('Display rectangles must remain separated')
    return rectangles
