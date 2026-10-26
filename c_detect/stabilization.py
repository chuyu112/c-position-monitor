"""Offline stabilization of display markers within reviewed C-role intervals."""

import numpy as np

from .overlays import constrain_display_boxes


def _smooth(values, sigma, median_radius=2):
    """Reject short detection spikes, then use a centered filter without lag."""
    values = np.asarray(values, dtype=float)
    padded = np.pad(values, (median_radius, median_radius), mode='edge')
    robust = np.median(np.lib.stride_tricks.sliding_window_view(
        padded, 2 * median_radius + 1), axis=-1)
    radius = max(1, round(3 * sigma))
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    # Linear padding avoids slowing a moving marker at either end of a role.
    span = min(5, len(robust) - 1)
    if span:
        start_velocity = (robust[span] - robust[0]) / span
        end_velocity = (robust[-1] - robust[-1 - span]) / span
    else:
        start_velocity = end_velocity = 0
    extended = np.concatenate([
        robust[0] - start_velocity * np.arange(radius, 0, -1), robust,
        robust[-1] + end_velocity * np.arange(1, radius + 1)])
    return np.convolve(extended, weights, mode='valid')


def stabilize_display_boxes(displays, segments, width, height, fps=30, gap=8):
    """Smooth center/size separately and enforce non-overlap on the final boxes.

    Input slots must follow the same reviewed person within each segment.
    A role handoff starts a new filter: unrelated dancers are never averaged.
    Positions use a 0.10 s Gaussian sigma; width/height use 0.20/0.15 s.
    Full video access permits symmetric filtering, so there is no causal delay.
    """
    output = [None] * len(displays)
    raw_acceleration, stable_acceleration = [], []
    cursor = 0
    for segment in segments:
        start, end, count = (segment[k] for k in ('start_frame', 'end_frame', 'count'))
        if start != cursor or not start < end <= len(displays):
            raise ValueError('Role intervals must cover the video exactly once')
        raw = np.asarray(displays[start:end], dtype=float)
        if raw.shape != (end - start, count, 4) or not np.isfinite(raw).all():
            raise ValueError('Invalid or inconsistent marker geometry')
        center = (raw[:, :, :2] + raw[:, :, 2:]) / 2
        size = raw[:, :, 2:] - raw[:, :, :2]
        filtered_center, filtered_size = center.copy(), size.copy()
        for person in range(count):
            for axis in range(2):
                filtered_center[:, person, axis] = _smooth(center[:, person, axis], .10 * fps)
                filtered_size[:, person, axis] = _smooth(size[:, person, axis], (.20 if axis == 0 else .15) * fps)
        filtered_size = np.maximum(filtered_size, 2)
        smoothed = np.concatenate([filtered_center - filtered_size / 2,
                                   filtered_center + filtered_size / 2], axis=2)
        for offset, boxes in enumerate(smoothed):
            output[start + offset] = constrain_display_boxes(boxes, width, height, gap)
        if end - start >= 3:
            raw_acceleration.extend(np.abs(np.diff(raw, n=2, axis=0)).ravel())
            stable_acceleration.extend(np.abs(np.diff(np.asarray(output[start:end]), n=2, axis=0)).ravel())
        cursor = end
    if cursor != len(displays):
        raise ValueError('Missing role intervals')
    metrics = {
        'mean_edge_acceleration_before_px_per_frame2': float(np.mean(raw_acceleration)) if raw_acceleration else 0,
        'mean_edge_acceleration_after_px_per_frame2': float(np.mean(stable_acceleration)) if stable_acceleration else 0,
        'filter_resets_at_role_boundaries': True,
        'position_sigma_seconds': .10, 'width_sigma_seconds': .20, 'height_sigma_seconds': .15,
    }
    return output, metrics
