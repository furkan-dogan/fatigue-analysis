"""Persisted offline analysis and reconstruction of the existing dashboard result."""
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from uuid import uuid4

from src.adapters.analysis_store import AnalysisStore
from src.core.records import MovementEvent, MetricResult
from src.sports.taekwondo.pipeline import AnalysisResult, run_analysis
from src.sports.taekwondo.reporting import METRICS

PARAMETERS = dict(show_joint_labels=False, event_peak_prominence_norm=0.06,
                  event_min_distance_sec=0.25, event_min_duration_sec=0.10,
                  event_max_duration_sec=6.0, event_min_knee_rom_deg=12.0,
                  event_min_peak_kick_height_norm=-0.5, vel_assist_threshold=100.0)


def provenance():
    import mediapipe
    model = Path(mediapipe.__file__).parent / 'modules/pose_landmark/pose_landmark_full.tflite'
    detector = model.parent.parent / 'pose_detection/pose_detection.tflite'
    return {'schema_version': 1, 'protocol': 'taekwondo_pre_post', 'protocol_version': '1',
            'algorithm_version': 'taekwondo-2', 'model': 'mediapipe_pose_full',
            'model_complexity': 1, 'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
            'detector_sha256': hashlib.sha256(detector.read_bytes()).hexdigest(),
            'packages': {name: version(name) for name in ('mediapipe', 'opencv-python', 'numpy', 'scipy')},
            'parameters': PARAMETERS,
            'algorithm_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                 for name in ('pipeline.py', 'events.py', 'metrics.py', 'service.py')},
            'measurement_validation': 'unvalidated'}


def event_records(result, run):
    events, metrics = [], []
    for raw in result.events:
        event = MovementEvent(uuid4().hex, run.id, run.video_id, 'kick',
                              raw['start_time_sec'], raw['end_time_sec'])
        events.append(event)
        for key, (_, unit) in METRICS.items():
            value = raw.get(key)
            metrics.append(MetricResult(event.id, f'taekwondo.{key}', value, unit,
                                        f'taekwondo-2:{key}', 'missing' if value is None else 'unvalidated',
                                        'Gerekli hareket verisi yok.' if value is None else None))
    return events, metrics


def save_result(store, run, result):
    payload = asdict(result)
    hashes = {}
    for key in ('frame_csv_path', 'events_csv_path', 'output_video_path', 'pose_path'):
        if not payload.get(key):
            raise ValueError(f'Analiz çıktısı eksik: {key}')
        path = Path(payload[key]).resolve()
        relative = str(path.relative_to(store.root))
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        payload[key] = relative
    payload['artifact_sha256'] = hashes
    events, metrics = event_records(result, run)
    store.complete_run(run, payload, events, metrics)


def analyze_pair(pre_video: bytes, post_video: bytes, progress, *, store=None,
                 names=('pre.mp4', 'post.mp4'), label='Taekwondo analizi', revision_of=None):
    store = store or AnalysisStore()
    if not pre_video or not post_video:
        raise ValueError('İki video da dolu olmalı.')
    source_info = provenance()
    session = store.create_session('taekwondo', label, revision_of)
    videos = [store.add_video(session, role, name, content)
              for role, name, content in zip(('pre', 'post'), names, (pre_video, post_video))]
    runs, results = [], []
    for index, video in enumerate(videos):
        run = store.start_run(video, source_info)
        runs.append(run)
        directory = store.run_directory(run)

        def update(current, total):
            progress(index * 50 + int(min(current / max(total, 1), 1) * 50),
                     f'{video.role.upper()} — Kare {current}/{total}')

        try:
            result = run_analysis(store.path(video.path), directory / 'annotated.mp4',
                                  directory / 'frames.csv', directory / 'events.csv',
                                  progress_callback=update, **PARAMETERS)
            save_result(store, run, result)
            results.append(result)
        except Exception as exc:
            store.fail_run(run, exc)
            raise
    store.add_comparison(session, runs[0], runs[1])
    return results[0], results[1], store.path(f'analyses/{session.id}')


def load_pair(session_id, *, store=None):
    store = store or AnalysisStore()
    if session_id not in {s['id'] for s in store.sessions('taekwondo')}:
        raise ValueError('Taekwondo kaydı bulunamadı.')
    runs = store.runs(session_id)
    by_role = {row['role']: row for row in runs}
    if set(by_role) != {'pre', 'post'} or any(row['status'] != 'completed' for row in runs):
        raise ValueError('Bu oturumun iki analizi de tamamlanmamış; yeniden analiz edebilirsiniz.')
    results = []
    for role in ('pre', 'post'):
        row = by_role[role]
        store.source_bytes(json.loads(row['video']))
        payload = store.load_result(row)
        for relative, digest in payload.pop('artifact_sha256').items():
            if hashlib.sha256(store.path(relative).read_bytes()).hexdigest() != digest:
                raise ValueError('Analiz çıktısı değiştirilmiş veya bozulmuş.')
        for key in ('frame_csv_path', 'events_csv_path', 'output_video_path', 'pose_path'):
            payload[key] = str(store.path(payload[key]))
        results.append(AnalysisResult(**payload))
    return *results, store.path(f'analyses/{session_id}')


def retry_pair(session_id, progress, *, store=None):
    store = store or AnalysisStore()
    sessions = {s['id']: s for s in store.sessions('taekwondo')}
    if session_id not in sessions:
        raise ValueError('Taekwondo kaydı bulunamadı.')
    with store.connection() as db:
        videos = {row['role']: json.loads(row['payload']) for row in db.execute(
            'SELECT role,payload FROM videos WHERE session_id=?', (session_id,))}
    if set(videos) != {'pre', 'post'}:
        raise ValueError('Yeniden analiz için iki kaynak video gerekli.')
    return analyze_pair(*(store.source_bytes(videos[role]) for role in ('pre', 'post')), progress,
                        store=store, names=tuple(videos[role]['original_name'] for role in ('pre', 'post')),
                        label=sessions[session_id]['label'], revision_of=session_id)
