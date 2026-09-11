"""Persist manual reviews separately from future model analysis runs."""
from dataclasses import asdict
from src.adapters.analysis_store import AnalysisStore
from src.adapters.video_review import inspect_video
from src.sports.volleyball.review import default_review, validate_review


def create_review(content, name, *, store=None):
    store = store or AnalysisStore()
    session = store.create_session('volleyball', name)
    video = store.add_video(session, 'source', name, content)
    run = store.start_run(video, {'operation': 'manual_video_review', 'version': 1, 'model': None})
    try:
        metadata = inspect_video(store.path(video.path))
        result = dict(kind='manual_video_review', schema_version=1, video=asdict(video),
                      metadata=metadata, review=default_review(metadata), metrics=[])
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
    if result.get('kind') != 'manual_video_review' or result.get('schema_version') != 1:
        raise ValueError('Desteklenmeyen inceleme kaydı.')
    store.source_bytes(result['video'])
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
        store.complete_run(run, {**result, 'video': asdict(video), 'review': review}, [], [])
    except Exception as exc:
        store.fail_run(run, exc)
        raise
    return load_review(session.id, store=store)
