"""Conservative geometry and tracking helpers for experimental video metrics."""
import numpy as np
from src.core.pose import point, PoseFrame

SIDES = ('left', 'right')


def midpoint(sample, joint):
    a, b = (point(sample, side + '_' + joint) for side in SIDES)
    return None if a is None or b is None else (np.array(a) + b) / 2


def torso(sample):
    hip, shoulder = midpoint(sample, 'hip'), midpoint(sample, 'shoulder')
    return None if hip is None or shoulder is None else float(np.linalg.norm(hip - shoulder))


def sole(sample, side):
    points = [point(sample, f'{side}_{part}') for part in ('big_toe', 'small_toe', 'heel')]
    return None if any(p is None for p in points) else max(p[1] for p in points)


class AthleteTracker:
    """Reject ambiguous matches; never fill missing points or silently change athlete."""
    def __init__(self, box=None):
        self.box, self.previous, self.scale = box, None, None
        self.started = False

    def select(self, people):
        candidates = []
        for person in people:
            sample = PoseFrame(0, None, person)
            center, scale = midpoint(sample, 'hip'), torso(sample)
            if center is None or scale is None or scale < 10:
                continue
            if not self.started and self.box:
                b = self.box
                if not (b['x1'] <= center[0] <= b['x2'] and b['y1'] <= center[1] <= b['y2']):
                    continue
            if self.previous is not None:
                if np.linalg.norm(center - self.previous) > self.scale * 0.5:
                    continue
            candidates.append((person, center, scale))
        if len(candidates) != 1:
            return {}, 'Sporcu takibi eksik veya belirsiz.'
        person, self.previous, self.scale = candidates[0]
        self.started = True
        return person, None


def physical_times(samples, review):
    if not review.get('physical_time_confirmed'):
        return None
    factor = review.get('time_scale', 1.0)
    times = [s.time_seconds for s in samples]
    if not times or any(t is None or not np.isfinite(t) for t in times):
        return None
    if not np.isfinite(factor) or factor <= 0 or any(b <= a for a, b in zip(times, times[1:])):
        return None
    return (np.array(times) - times[0]) * factor


def flight_candidates(samples, times):
    """Draft foot-clearance events. They require manual contact confirmation."""
    if times is None or len(samples) < 10:
        return []
    scales = [torso(s) for s in samples]
    if any(v is None or v < 10 for v in scales):
        return []
    feet = [[sole(s, side) for side in SIDES] for s in samples]
    if any(v is None for row in feet for v in row):
        return []
    feet = np.array(feet)
    base = np.where(times <= 0.2)[0]
    if len(base) < 5 or np.ptp(feet[base], axis=0).max() > np.median(scales) * 0.04:
        return []
    baseline = np.median(feet[base], axis=0)
    air = np.all(baseline - feet > np.median(scales) * 0.025, axis=1)
    found, start = [], None
    for i, airborne in enumerate(air):
        if airborne and start is None:
            start = i
        elif not airborne and start is not None:
            if start > base[-1] and 0.1 <= times[i] - times[start] <= 1.2:
                found.append({'start_frame': samples[max(0, start-1)].frame,
                              'takeoff_frame': samples[start].frame, 'landing_frame': samples[i].frame,
                              'end_frame': samples[min(len(samples)-1, i+1)].frame})
            start = None
    return found
