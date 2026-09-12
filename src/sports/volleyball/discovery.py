"""Experimental motion candidates from image geometry, never physical measurements."""
import numpy as np
from src.core.pose import point
from src.sports.volleyball.signals import midpoint, sole, torso

VERSION = 'volleyball-discovery-1'
LABELS = {'jump': 'Sıçrama adayı', 'approach_jump': 'Yaklaşmalı sıçrama adayı',
          'stationary_jump': 'Yerinde sıçrama adayı', 'locomotion': 'Koşu / yer değiştirme adayı'}


def runs(mask):
    start = None
    for i, value in enumerate(list(mask) + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            yield start, i - 1
            start = None


def detect_movements(samples, fps):
    """Local foot clearance + pelvis excursion. Timing here is playback, not physics."""
    if len(samples) < 8:
        return []
    fps = float(fps) if fps and np.isfinite(fps) and fps > 0 else 30.
    valid = [not s.issue and midpoint(s, 'hip') is not None and torso(s) is not None
             and torso(s) >= 10 and all(sole(s, side) is not None for side in ('left', 'right')) for s in samples]
    events = []
    for begin, end in runs(valid):
        window = samples[begin:end+1]
        if len(window) < 8:
            continue
        scale = float(np.median([torso(s) for s in window]))
        hips = np.array([midpoint(s, 'hip') for s in window])
        feet = np.array([[sole(s, side) for side in ('left', 'right')] for s in window])
        # Median filter only observed points; no bridging a missing frame.
        smooth = lambda x: np.array([np.median(x[max(0,i-2):i+3], axis=0) for i in range(len(x))])
        feet, hips = smooth(feet), smooth(hips)
        radius = max(6, round(fps * 1.2))
        floor = np.array([np.quantile(feet[max(0,i-radius):i+radius+1], .9, axis=0) for i in range(len(feet))])
        clearance = np.min(floor - feet, axis=1)
        occupied = np.zeros(len(window), dtype=bool)
        for a, b in runs(clearance > .12*scale):
            if b-a+1 < max(3, round(.08*fps)) or a == 0 or b == len(window)-1:
                continue
            if b-a > fps*2.5:
                continue
            peak = a + int(np.argmax(clearance[a:b+1]))
            if clearance[peak] < .3*scale:
                continue
            # Both feet and pelvis must visibly rise and return; crouching alone is not a jump.
            if min(hips[a-1,1], hips[b+1,1]) - np.min(hips[a:b+1,1]) < .12*scale:
                continue
            before = max(0, a-round(.65*fps))
            after = min(len(window)-1, b+round(.15*fps))
            travel = abs(hips[a,0]-hips[before,0])/scale
            kind = 'approach_jump' if travel > .45 else 'stationary_jump' if travel < .15 else 'jump'
            # These are elevated-foot boundaries, not verified contact frames.
            event = dict(kind=kind, label=LABELS[kind], start_frame=window[before].frame,
                         end_frame=window[after].frame, takeoff_frame=window[a].frame,
                         peak_frame=window[peak].frame, landing_frame=window[b+1].frame,
                         status='candidate', metrics=[], contact_status='estimated',
                         reason='Görüntüde ayak ve pelvis yükselmesine dayalı aday; temaslar doğrulanmadı.')
            events.append(event)
            occupied[before:after+1] = True
        # Horizontal travel is deliberately not called a confirmed sprint.
        stride = max(2, round(.2*fps))
        moving = np.zeros(len(window), dtype=bool)
        separation = []
        for sample in window:
            left, right = point(sample, 'left_ankle'), point(sample, 'right_ankle')
            separation.append(np.nan if left is None or right is None else left[0]-right[0])
        for i in range(stride, len(window)-stride):
            displacement = abs(hips[i+stride,0]-hips[i-stride,0])/scale
            legs = np.array(separation[i-stride:i+stride+1])
            articulated = np.all(np.isfinite(legs)) and np.ptp(legs) > .3*scale
            moving[i] = displacement > .3 and articulated and not occupied[i]
        for a, b in runs(moving):
            if b-a+1 >= max(4, round(.18*fps)):
                events.append(dict(kind='locomotion', label=LABELS['locomotion'],
                                   start_frame=window[a].frame, end_frame=window[b].frame,
                                   status='candidate', metrics=[],
                                   reason='Yatay pelvis ilerlemesi ve bacak hareketi; yürüyüş/koşu ayrımı doğrulanmadı.'))
    merged = []
    by_frame = {s.frame: s for s in samples}
    for event in sorted(events, key=lambda e: e['start_frame']):
        if merged:
            previous = merged[-1]
            gap = event['start_frame'] - previous['end_frame']
            continuous = all(f in by_frame and not by_frame[f].issue
                             for f in range(previous['end_frame'], event['start_frame']+1))
            if (previous['kind'] == 'locomotion' and event['kind'] in ('locomotion', 'approach_jump')
                    and 0 <= gap <= max(1, round(.25*fps)) and continuous):
                event = dict(event, start_frame=previous['start_frame'])
                merged.pop()
        merged.append(event)
    return merged


def discover(samples, metadata, cuts, selections_needed):
    boundaries = sorted(set([0, *cuts, len(samples)]))
    segments, events = [], []
    for a, b in zip(boundaries, boundaries[1:]):
        chunk = samples[a:b]
        if not chunk:
            continue
        found = detect_movements(chunk, metadata.get('nominal_fps'))
        events.extend(dict(e, segment=len(segments)) for e in found)
        missing = sum(bool(s.issue) for s in chunk)
        segments.append(dict(start_frame=a, end_frame=b-1, missing_frames=missing,
                             candidate_count=len(found)))
    warnings = ['Hareketler deneysel adaylardır; fiziksel yükseklik ve hız bu taramada hesaplanmaz.']
    if any(s.issue for s in samples):
        warnings.append('Bazı karelerde takip belirsiz; bu boşluklardan hareket birleştirilmedi.')
    if not events:
        warnings.append('Desteklenen hareket için yeterli görsel kanıt bulunamadı.')
    return dict(mode='automatic', algorithm=VERSION, validation='unvalidated', segments=segments,
                events=events, selections_needed=selections_needed, warnings=warnings,
                metrics=[], candidates=[], speed_series=[])
