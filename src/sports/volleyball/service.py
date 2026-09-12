"""Persist manual reviews separately from future model analysis runs."""
from dataclasses import asdict
from src.adapters.analysis_store import AnalysisStore
from src.adapters.video_review import inspect_video
from src.sports.volleyball.review import default_review, validate_review


def create_review(content, name, *, store=None, athlete="", capture_group="", capture_notes=""):
    store = store or AnalysisStore()
    session = store.create_session('volleyball', name)
    video = store.add_video(session, 'source', name, content)
    run = store.start_run(video, {'operation': 'manual_video_review', 'version': 1, 'model': None})
    try:
        metadata = inspect_video(store.path(video.path))
        review = default_review(metadata)
        review.update(athlete=athlete.strip(), capture_group=capture_group.strip(), capture_notes=capture_notes.strip())
        result = dict(kind='manual_video_review', schema_version=1, video=asdict(video),
                      metadata=metadata, review=review, metrics=[])
        store.complete_run(run, result, [], [])
    except Exception as exc:
        store.fail_run(run, exc)
        raise
    return load_review(session.id, store=store)


def load_review(session_id, *, store=None):
    store = store or AnalysisStore()
    if session_id not in {s['id'] for s in store.sessions('volleyball')}:
        raise ValueError('Voleybol kaydı bulunamadı.')
    runs = store.runs(session_id)
    if len(runs) != 1:
        raise ValueError('İnceleme kaydı tamamlanmamış.')
    result = store.load_result(runs[0])
    if result.get('kind') not in ('manual_video_review', 'volleyball_analysis') or result.get('schema_version') != 1:
        raise ValueError('Desteklenmeyen inceleme kaydı.')
    store.source_bytes(result['video'])
    if result.get('kind') == 'volleyball_analysis':
        import hashlib
        if hashlib.sha256(store.path(result['pose_path']).read_bytes()).hexdigest() != result['pose_sha256']:
            raise ValueError('Pose dosyasının bütünlük kontrolü başarısız.')
    validate_review(result['review'], result['metadata'])
    return dict(session_id=session_id, result=result, source_path=str(store.path(result['video']['path'])))


def save_revision(current, review, *, store=None):
    store = store or AnalysisStore()
    original = load_review(current['session_id'], store=store)
    result = original['result']
    validate_review(review, result['metadata'])
    session = store.create_session('volleyball', review['athlete'] or result['video']['original_name'], current['session_id'])
    video = store.add_video(session, 'source', result['video']['original_name'], store.source_bytes(result['video']))
    run = store.start_run(video, {'operation': 'manual_video_review', 'version': 1, 'model': None})
    try:
        store.complete_run(run, dict(kind='manual_video_review', schema_version=1, metadata=result['metadata'],
                                    video=asdict(video), review=review, metrics=[]), [], [])
    except Exception as exc:
        store.fail_run(run, exc)
        raise
    return load_review(session.id, store=store)


def analyze_review(current, progress, *, store=None, runner_factory=None):
    """Create a separate immutable analysis revision from a saved review."""
    import hashlib
    from pathlib import Path
    from uuid import uuid4
    from src.core.records import MovementEvent, MetricResult
    from src.sports.volleyball.pipeline import run_inference
    from src.sports.volleyball.measurements import VERSION
    store = store or AnalysisStore()
    original = load_review(current['session_id'], store=store)
    result = original['result']
    review = result['review']
    session = store.create_session('volleyball', review['athlete'] or result['video']['original_name'], current['session_id'])
    video = store.add_video(session, 'source', result['video']['original_name'], store.source_bytes(result['video']))
    provenance = {'operation': 'volleyball_analysis', 'algorithm': VERSION, 'validation': 'unvalidated',
                  'review_session': current['session_id'], 'review': review,
                  'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    run = store.start_run(video, provenance)
    pose_path = store.run_directory(run) / 'pose.jsonl'
    try:
        options = {'runner_factory': runner_factory} if runner_factory else {}
        analysis, model_info = run_inference(store.path(video.path), review, result['metadata'], pose_path, progress, **options)
        provenance['model'] = model_info
        # Provenance becomes final while the run is still running, before completion.
        with store.connection() as db:
            from src.adapters.analysis_store import encode
            db.execute('UPDATE runs SET provenance=? WHERE id=? AND status=?', (encode(provenance), run.id, 'running'))
        events, metrics = [], []
        pts = result['metadata'].get('timestamps', [])
        for event in analysis['events']:
            a, b = event['start_frame'], event['end_frame']
            if len(pts) <= b or pts[a] is None or pts[b] is None:
                continue
            saved = MovementEvent(uuid4().hex, run.id, video.id, review['test'], pts[a], pts[b], 'source_pts_seconds')
            events.append(saved)
            for value in event['metrics']:
                metrics.append(MetricResult(saved.id, 'volleyball.'+value['key'], value['value'], value['unit'],
                                            value['method'], value['quality'], value['reason']))
        payload = {k: result[k] for k in ('schema_version', 'metadata', 'review')}
        payload.update(kind='volleyball_analysis', video=asdict(video), metrics=[], analysis=analysis,
                       model=model_info, pose_path=str(pose_path.relative_to(store.root)),
                       pose_sha256=hashlib.sha256(pose_path.read_bytes()).hexdigest())
        store.complete_run(run, payload, events, metrics)
    except Exception as exc:
        store.fail_run(run, exc)
        raise
    return load_review(session.id, store=store)
