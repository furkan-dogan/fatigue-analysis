"""Versioned analysis contracts, independent of storage, models and presentation."""
from dataclasses import dataclass, field
from math import isfinite

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Session:
    id: str
    sport: str
    label: str
    created_at: str
    revision_of: str | None = None


@dataclass(frozen=True)
class VideoAsset:
    id: str
    session_id: str
    role: str
    original_name: str
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class AnalysisRun:
    id: str
    session_id: str
    video_id: str
    status: str
    provenance: dict
    result_path: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class MovementEvent:
    id: str
    run_id: str
    video_id: str
    kind: str
    start_seconds: float
    end_seconds: float
    time_basis: str = 'frame_index/nominal_fps'

    def __post_init__(self):
        if not (isfinite(self.start_seconds) and isfinite(self.end_seconds)
                and 0 <= self.start_seconds <= self.end_seconds):
            raise ValueError('Geçersiz olay zaman aralığı.')


@dataclass(frozen=True)
class MetricResult:
    event_id: str
    key: str
    value: float | None
    unit: str
    method: str
    quality: str
    reason: str | None = None

    def __post_init__(self):
        if not self.unit or not self.method:
            raise ValueError('Metrik birimi ve yöntemi zorunlu.')
        if self.value is not None and not isfinite(self.value):
            raise ValueError('Metrik sonlu olmalı; eksik değer için None kullanın.')
        if self.quality not in {'unvalidated', 'missing', 'rejected'}:
            raise ValueError('Bu sürüm doğrulanmış ölçüm üretmez.')
        if self.value is None and (self.quality not in {'missing', 'rejected'} or not self.reason):
            raise ValueError('Eksik metriğin gerekçesi zorunlu.')


@dataclass(frozen=True)
class Comparison:
    id: str
    session_id: str
    before_run_id: str
    after_run_id: str
    compatibility: str = 'unverified'
    notes: list[str] = field(default_factory=lambda: ['Çekim ve protokol uyumu kullanıcı tarafından doğrulanmadı.'])
