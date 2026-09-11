"""Offline model-selection experiment; never writes application performance metrics.

Optional research packages are installed separately from application requirements.
Run each candidate against the same source and sample manifest.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import time

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--samples', required=True, type=Path, help='JSON: source_sha256 and frame_indices')
    parser.add_argument('--model', required=True, choices=['mediapipe', 'yolo26x', 'rtmpose'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.samples.read_text())
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if manifest['source_sha256'] != source_hash:
        raise ValueError('Örnekleme listesi bu kaynak videoya ait değil.')
    selected = set(manifest['frame_indices'])
    if not selected or any(type(i) is not int or i < 0 for i in selected):
        raise ValueError('Örnek kare listesi geçersiz.')
    configuration = {'model': args.model, 'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
                     'platform': platform.platform(), 'device': 'cpu_requested', 'selected_indices': sorted(selected),
                     'protocol': 'seybering-review-v1', 'normalization': 'original_image_pixels',
                     'timing': 'wall-clock inference, excludes initialization and decoding',
                     'ground_truth': None, 'sample_manifest': manifest}
    if args.model == 'mediapipe':
        import mediapipe as mp
        runner = mp.solutions.pose.Pose(model_complexity=1, static_image_mode=False)
        configuration.update(package=version('mediapipe'), landmarks=33,
                             settings={'model_complexity': 1, 'static_image_mode': False})
        def predict(frame):
            result = runner.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.pose_landmarks:
                return []
            h, w = frame.shape[:2]
            return [[[p.x*w, p.y*h, p.visibility] for p in result.pose_landmarks.landmark]]
    elif args.model == 'yolo26x':
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        runner = YOLO(str(args.output / 'yolo26x-pose.pt'))
        configuration.update(package=version('ultralytics'), landmarks=17,
                             settings={'imgsz': 640, 'conf': 0.25, 'threads': 4})
        def predict(frame):
            result = runner.predict(frame, imgsz=640, conf=0.25, device='cpu', verbose=False)[0]
            return result.keypoints.data.cpu().numpy().tolist() if result.keypoints is not None else []
    else:
        from rtmlib import Wholebody
        pose_url = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip'
        runner = Wholebody(pose=pose_url, pose_input_size=(288, 384), mode='balanced',
                           to_openpose=False, backend='onnxruntime', device='cpu')
        configuration.update(package=version('rtmlib'), runtime=version('onnxruntime'), landmarks=133,
                             settings={'pose_url': pose_url, 'detector': runner.MODE['balanced']['det'],
                                       'pose_size': [288,384], 'detector_size': [640,640]})
        def predict(frame):
            points, scores = runner(frame)
            return np.concatenate([points, scores[...,None]], axis=-1).tolist() if len(points) else []
    # Warm-up on source frame 0; MediaPipe is then reset to avoid future-context leakage.
    cap = cv2.VideoCapture(str(args.source))
    ok, first = cap.read()
    if not ok:
        raise ValueError('Kaynak video okunamadı.')
    predict(first)
    if args.model == 'mediapipe':
        runner.close()
        runner = mp.solutions.pose.Pose(model_complexity=1, static_image_mode=False)
    cap.release()
    cap = cv2.VideoCapture(str(args.source))
    rows = []
    index = 0
    previous = None
    cuts = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tiny = cv2.resize(frame, (108,192)).astype(float)
            change = float(np.mean(np.abs(tiny-previous))) if previous is not None else 0
            if change > 18:
                cuts.append({'frame': index, 'mean_pixel_change': change})
            previous = tiny
            # MediaPipe receives every frame to preserve its actual tracking behavior.
            if args.model == 'mediapipe' or index in selected:
                begin = time.perf_counter()
                poses = predict(frame)
                elapsed = (time.perf_counter()-begin)*1000
                if index in selected:
                    rows.append({'frame': index, 'inference_ms': elapsed, 'poses': poses})
                    if len(rows) % 25 == 0:
                        print(args.model, len(rows), 'samples', flush=True)
            index += 1
    finally:
        cap.release()
        if args.model == 'mediapipe':
            runner.close()
    if any(i >= index for i in selected):
        raise ValueError('Örnek kare listesi kaynak sınırını aşıyor.')
    configuration['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)
    result = {'configuration': configuration, 'decoded_frames': index, 'candidate_cuts': cuts,
              'rows': rows, 'complete': True}
    (args.output / f'{args.model}.json').write_text(json.dumps(result, allow_nan=False))
    print(args.model, 'DONE', len(rows), flush=True)


if __name__ == '__main__':
    main()
