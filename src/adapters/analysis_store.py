"""Small local SQLite catalog with immutable run artifacts under data/analyses."""
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from src.core.records import SCHEMA_VERSION, Session, VideoAsset, AnalysisRun, Comparison

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / 'data'


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(encode(value), encoding='utf-8')
    temporary.replace(path)


class AnalysisStore:
    def __init__(self, root=None):
        self.root = Path(root or DEFAULT_ROOT).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / 'analyses.sqlite3'
        with self.connection() as db:
            version = db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise ValueError('Desteklenmeyen kayıt şeması; veritabanı değiştirilmedi.')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, sport TEXT NOT NULL, label TEXT NOT NULL,
                    created_at TEXT NOT NULL, revision_of TEXT REFERENCES sessions(id));
                CREATE TABLE IF NOT EXISTS videos (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                    role TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(session_id, role));
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                    video_id TEXT NOT NULL REFERENCES videos(id),
                    status TEXT NOT NULL CHECK(status IN ('running','completed','failed','interrupted')),
                    provenance TEXT NOT NULL, result_path TEXT, result_sha256 TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS metrics (
                    event_id TEXT NOT NULL REFERENCES events(id), key TEXT NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY(event_id, key));
                CREATE TABLE IF NOT EXISTS comparisons (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                    before_run_id TEXT NOT NULL REFERENCES runs(id),
                    after_run_id TEXT NOT NULL REFERENCES runs(id), payload TEXT NOT NULL);
            ''')
            db.execute(f'PRAGMA user_version={SCHEMA_VERSION}')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.database)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Kayıt yolu veri klasörü dışında.')
        return path

    def create_session(self, sport, label, revision_of=None):
        session = Session(uuid4().hex, sport, label.strip() or 'Video analizi',
                          datetime.now(timezone.utc).isoformat(), revision_of)
        with self.connection() as db:
            if revision_of:
                parent = db.execute('SELECT sport FROM sessions WHERE id=?', (revision_of,)).fetchone()
                if parent is None or parent['sport'] != sport:
                    raise ValueError('Revizyon aynı branştaki bir oturuma bağlı olmalı.')
            db.execute('INSERT INTO sessions VALUES (?,?,?,?,?)', tuple(asdict(session).values()))
        return session

    def add_video(self, session, role, name, content):
        if not content:
            raise ValueError('Video dosyası boş.')
        video_id = uuid4().hex
        suffix = Path(name).suffix.lower()
        if suffix not in {'.mp4', '.mov', '.avi'}:
            raise ValueError('Desteklenmeyen video uzantısı.')
        relative = f'analyses/{session.id}/sources/{video_id}{suffix}'
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        video = VideoAsset(video_id, session.id, role, Path(name).name, relative,
                           hashlib.sha256(content).hexdigest(), len(content))
        try:
            with self.connection() as db:
                db.execute('INSERT INTO videos VALUES (?,?,?,?)',
                           (video.id, session.id, role, encode(asdict(video))))
        except Exception:
            path.unlink()
            raise
        return video

    def start_run(self, video, provenance):
        run = AnalysisRun(uuid4().hex, video.session_id, video.id, 'running', provenance)
        with self.connection() as db:
            parent = db.execute('SELECT session_id FROM videos WHERE id=?', (video.id,)).fetchone()
            if parent is None or parent['session_id'] != video.session_id:
                raise ValueError('Video ve oturum eşleşmiyor.')
            db.execute('INSERT INTO runs(id,session_id,video_id,status,provenance) VALUES (?,?,?,?,?)',
                       (run.id, run.session_id, run.video_id, run.status, encode(provenance)))
        self.run_directory(run).mkdir(parents=True, exist_ok=True)
        return run

    def run_directory(self, run):
        return self.path(f'analyses/{run.session_id}/runs/{run.id}')

    def complete_run(self, run, result, events, metrics):
        relative = f'analyses/{run.session_id}/runs/{run.id}/result.json'
        with self.connection() as db:
            current = db.execute('SELECT status FROM runs WHERE id=?', (run.id,)).fetchone()
            if current is None or current['status'] != 'running':
                raise ValueError('Yalnızca çalışan analiz tamamlanabilir; eski sonuç değiştirilemez.')
            for event in events:
                if event.run_id != run.id or event.video_id != run.video_id:
                    raise ValueError('Olay kaynağı analizle eşleşmiyor.')
                db.execute('INSERT INTO events VALUES (?,?,?)', (event.id, run.id, encode(asdict(event))))
            event_ids = {event.id for event in events}
            for metric in metrics:
                if metric.event_id not in event_ids:
                    raise ValueError('Metrik bu analizin olayına bağlı olmalı.')
                # A metric key must keep its unit across runs of the same method.
                prior = db.execute('SELECT payload FROM metrics WHERE key=? LIMIT 1', (metric.key,)).fetchone()
                if prior and json.loads(prior['payload'])['unit'] != metric.unit:
                    raise ValueError('Aynı metrik kodunda farklı birim kullanılamaz.')
                db.execute('INSERT INTO metrics VALUES (?,?,?)',
                           (metric.event_id, metric.key, encode(asdict(metric))))
            atomic_json(self.path(relative), result)
            digest = hashlib.sha256(self.path(relative).read_bytes()).hexdigest()
            db.execute("UPDATE runs SET status='completed', result_path=?, result_sha256=? WHERE id=?",
                       (relative, digest, run.id))

    def fail_run(self, run, error):
        with self.connection() as db:
            db.execute("UPDATE runs SET status='failed', error=? WHERE id=? AND status='running'",
                       (str(error), run.id))

    def recover_interrupted(self):
        """Explicit recovery only: opening a second connection must not fail active work."""
        with self.connection() as db:
            return db.execute("UPDATE runs SET status='interrupted', error='Önceki çalışma tamamlanmadan kesildi.' WHERE status='running'").rowcount

    def add_comparison(self, session, before, after):
        comparison = Comparison(uuid4().hex, session.id, before.id, after.id)
        with self.connection() as db:
            for run in (before, after):
                row = db.execute('SELECT status, session_id FROM runs WHERE id=?', (run.id,)).fetchone()
                if row is None or row['status'] != 'completed' or row['session_id'] != session.id:
                    raise ValueError('Karşılaştırma aynı oturumun tamamlanmış analizlerini gerektirir.')
            db.execute('INSERT INTO comparisons VALUES (?,?,?,?,?)',
                       (comparison.id, session.id, before.id, after.id, encode(asdict(comparison))))
        return comparison

    def sessions(self, sport):
        with self.connection() as db:
            return [dict(row) for row in db.execute('''
                SELECT s.*, (SELECT count(*) FROM runs r WHERE r.session_id=s.id AND r.status='completed') completed,
                (SELECT count(*) FROM runs r WHERE r.session_id=s.id AND r.status!='completed') unfinished
                FROM sessions s WHERE sport=? ORDER BY created_at DESC''', (sport,))]

    def runs(self, session_id):
        with self.connection() as db:
            return [dict(row) for row in db.execute('''SELECT r.*, v.role, v.payload video
                FROM runs r JOIN videos v ON v.id=r.video_id WHERE r.session_id=? ORDER BY v.role''', (session_id,))]

    def load_result(self, row):
        if row['status'] != 'completed':
            raise ValueError('Analiz tamamlanmamış.')
        data = self.path(row['result_path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['result_sha256']:
            raise ValueError('Analiz dosyasının bütünlük kontrolü başarısız.')
        return json.loads(data)

    def source_bytes(self, video):
        data = self.path(video['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != video['sha256']:
            raise ValueError('Orijinal video değiştirilmiş veya bozulmuş.')
        return data
