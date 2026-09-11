"""Display contracts. Values are prepared by the caller, not computed by widgets."""

from dataclasses import dataclass
import math
from typing import Literal


@dataclass(frozen=True)
class PanelState:
    status: Literal['ready', 'empty', 'loading', 'error'] = 'ready'
    message: str = ''


@dataclass(frozen=True)
class MetricCard:
    label: str
    value: str | float | int | None
    unit: str = ''
    delta: str | None = None
    help: str | None = None
    delta_color: Literal['normal', 'inverse', 'off'] = 'off'


@dataclass(frozen=True)
class QualityNotice:
    message: str
    level: Literal['info', 'warning', 'error', 'success'] = 'info'


@dataclass(frozen=True)
class TimelineEvent:
    id: str
    label: str
    start_seconds: float
    end_seconds: float
    peak_seconds: float | None = None

    def __post_init__(self):
        if not self.id:
            raise ValueError('Olay kimliği boş olamaz.')
        if not all(math.isfinite(v) for v in (self.start_seconds, self.end_seconds)):
            raise ValueError('Olay zamanları sonlu olmalı.')
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise ValueError('Olay zaman aralığı geçersiz.')
        if self.peak_seconds is not None and not (
            math.isfinite(self.peak_seconds) and self.start_seconds <= self.peak_seconds <= self.end_seconds
        ):
            raise ValueError('Tepe zamanı olay aralığında olmalı.')
