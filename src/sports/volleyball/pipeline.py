"""Frame-source orchestration. Measurements remain independent of model inference."""
from dataclasses import asdict
import json
import cv2
import numpy as np
from src.adapters.rtmpose_pose import RTMPoseRunner
from src.core.pose import PoseFrame
from src.sports.volleyball.signals import AthleteTracker
from src.sports.volleyball.measurements import analyze


def run_inference(source, review, metadata, pose_path, progress, runner_factory=RTMPoseRunner, automatic=False, athlete_choices=None, cached_pose=None, cached_model=None):
    start, end = (0, metadata['frame_count']-1) if automatic else (review['start_frame'], review['end_frame'])
    box = None if automatic else review.get('athlete_box')
    if box and box['frame'] != start:
        raise ValueError('Sporcu seçim kutusu inceleme başlangıç karesinde olmalı.')
    capture = cv2.VideoCapture(str(source))
    runner = None
    try:
        if not capture.isOpened():
            raise ValueError('Video açılamadı.')
        if cached_pose is None:
            runner = runner_factory()
        from src.adapters.video_review import ShotBoundaryDetector
        shot_detector = ShotBoundaryDetector()
        cached_rows = None
        if cached_pose is not None:
            cached_rows = iter(json.loads(row) for row in cached_pose.read_text().splitlines())
        tracker = AthleteTracker(box)
        samples, previous = [], None
        cuts, selections_needed = [], []
        waiting = False
        athlete_choices = athlete_choices or {}
        with pose_path.open('w', encoding='utf-8') as output:
            for index in range(end+1):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('Seçili aralık bitmeden video sona erdi.')
                if index < start:
                    continue
                tiny = cv2.resize(frame, (108, 192)).astype(float)
                cut = shot_detector.update(frame) if automatic else previous is not None and float(np.mean(np.abs(tiny-previous))) > 18
                previous = tiny
                if cached_rows is None:
                    people = runner.process(frame)
                else:
                    cached = next(cached_rows, None)
                    if cached is None or cached['frame'] != index:
                        raise ValueError('Kayıtlı pose kareleri kaynakla eşleşmiyor.')
                    people = cached['people']
                if automatic and cut:
                    cuts.append(index)
                    tracker = AthleteTracker()
                    waiting = False
                if automatic:
                    chosen = athlete_choices.get(str(index))
                    if chosen is not None:
                        if not isinstance(chosen, int) or not 0 <= chosen < len(people):
                            raise ValueError('Sporcu seçimi kaynak kareyle eşleşmiyor.')
                        from src.sports.volleyball.signals import midpoint, torso
                        selected = people[chosen]
                        reference = PoseFrame(index, None, selected)
                        center, scale = midpoint(reference, 'hip'), torso(reference)
                        if center is None or scale is None:
                            raise ValueError('Seçilen sporcunun gövdesi görünmüyor.')
                        tracker = AthleteTracker()
                        tracker.previous, tracker.scale, tracker.started = center, scale, True
                        waiting = False
                    if not tracker.started and len(people) > 1 and not waiting:
                        selections_needed.append(dict(frame=index, people=people))
                        waiting = True
                points, issue = ({}, 'Sporcu seçimi gerekli.') if automatic and waiting else tracker.select(people)
                if automatic and issue and not waiting:
                    tracker = AthleteTracker()
                if cut and not automatic:
                    issue = 'Olası sahne veya kamera değişimi.'
                source_times = metadata.get('timestamps', [])
                timestamp = source_times[index] if index < len(source_times) else None
                sample = PoseFrame(index, timestamp, points, issue)
                samples.append(sample)
                output.write(json.dumps({**asdict(sample), 'people': people, 'cut': bool(cut)}, allow_nan=False) + '\n')
                progress(index-start+1, end-start+1)
        if automatic:
            from src.sports.volleyball.discovery import discover
            return discover(samples, metadata, cuts, selections_needed), runner.provenance if runner else cached_model
        return analyze(samples, review, metadata), runner.provenance
    finally:
        capture.release()
        if runner is not None:
            runner.close()
