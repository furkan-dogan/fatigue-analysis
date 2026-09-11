"""Model-independent image-space pose samples. Scores are not accuracy."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class PoseFrame:
    frame: int
    time_seconds: float | None
    points: dict[str, tuple[float, float, float]]
    issue: str | None = None


def point(sample, name, threshold=0.5):
    value = sample.points.get(name)
    if sample.issue or value is None or len(value) != 3:
        return None
    if not all(isfinite(v) for v in value) or value[2] < threshold:
        return None
    return value[:2]
