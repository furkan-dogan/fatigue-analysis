"""Pinned WholeBody inference with explicit COCO-WholeBody point mapping."""
import hashlib
from importlib.metadata import version
from math import isfinite
from pathlib import Path

# COCO-WholeBody: 17 body + 6 feet + face/hands. No invented landmarks.
POINT_NAMES = {5: 'left_shoulder', 6: 'right_shoulder', 11: 'left_hip', 12: 'right_hip',
               13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle',
               17: 'left_big_toe', 18: 'left_small_toe', 19: 'left_heel',
               20: 'right_big_toe', 21: 'right_small_toe', 22: 'right_heel'}
MODELS = {
    'detector': ('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
                 '3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23'),
    'pose': ('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip',
             '8cfecfc2226d8e14b510c2fd28442226c518b4d60690535753187877411d4005'),
}


def map_person(coordinates, scores, width, height):
    result = {}
    for index, name in POINT_NAMES.items():
        if index >= len(coordinates) or index >= len(scores):
            continue
        x, y = map(float, coordinates[index])
        score = float(scores[index])
        if all(isfinite(v) for v in (x, y, score)) and 0 <= x < width and 0 <= y < height:
            result[name] = (x, y, score)
    return result


class RTMPoseRunner:
    def __init__(self):
        from rtmlib import Wholebody
        from rtmlib.tools.file import download_checkpoint
        paths = {}
        for name, (url, expected) in MODELS.items():
            path = Path(download_checkpoint(url))
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Model bütünlük kontrolü başarısız: {name}')
            paths[name] = str(path)
        self.engine = Wholebody(det=paths['detector'], det_input_size=(640, 640),
                                pose=paths['pose'], pose_input_size=(288, 384),
                                to_openpose=False, backend='onnxruntime', device='cpu')
        self.provenance = {'model': 'rtmpose-l-dw-wholebody', 'schema': 'coco_wholebody_133',
                           'saved_landmarks': list(POINT_NAMES.values()), 'coordinates': 'image_pixels', 'score_type': 'simcc_score', 'device': 'cpu',
                           'models': {name: {'url': url, 'sha256': digest} for name, (url, digest) in MODELS.items()},
                           'packages': {p: version(p) for p in ('rtmlib', 'onnxruntime', 'numpy', 'opencv-python')}}

    def process(self, frame):
        points, scores = self.engine(frame)
        height, width = frame.shape[:2]
        return [map_person(p, s, width, height) for p, s in zip(points, scores)]

    def close(self):
        self.engine = None
