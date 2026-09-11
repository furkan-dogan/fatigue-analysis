"""Frame-source orchestration. Measurements remain independent of model inference."""
from dataclasses import asdict
import json
import cv2
import numpy as np
from src.adapters.rtmpose_pose import RTMPoseRunner
from src.core.pose import PoseFrame
from src.sports.volleyball.signals import AthleteTracker
from src.sports.volleyball.measurements import analyze


def run_inference(source, review, metadata, pose_path, progress, runner_factory=RTMPoseRunner):
    start, end = review['start_frame'], review['end_frame']
    box = review.get('athlete_box')
    if box and box['frame'] != start:
        raise ValueError('Sporcu seçim kutusu inceleme başlangıç karesinde olmalı.')
    capture = cv2.VideoCapture(str(source))
    runner = None
    try:
        if not capture.isOpened():
            raise ValueError('Video açılamadı.')
        runner = runner_factory()
        tracker = AthleteTracker(box)
        samples, previous = [], None
        with pose_path.open('w', encoding='utf-8') as output:
            for index in range(end+1):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('Seçili aralık bitmeden video sona erdi.')
                if index < start:
                    continue
                tiny = cv2.resize(frame, (108, 192)).astype(float)
                cut = previous is not None and float(np.mean(np.abs(tiny-previous))) > 18
                previous = tiny
                people = runner.process(frame)
                points, issue = tracker.select(people)
                if cut:
                    issue = 'Olası sahne veya kamera değişimi.'
                source_times = metadata.get('timestamps', [])
                timestamp = source_times[index] if index < len(source_times) else None
                sample = PoseFrame(index, timestamp, points, issue)
                samples.append(sample)
                output.write(json.dumps({**asdict(sample), 'people': people}, allow_nan=False) + '\n')
                progress(index-start+1, end-start+1)
        return analyze(samples, review, metadata), runner.provenance
    finally:
        capture.release()
        if runner is not None:
            runner.close()
