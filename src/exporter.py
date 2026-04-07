"""CSV export helpers for frame-level and event-level outputs."""

from __future__ import annotations

import csv
from pathlib import Path


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    # Keep stable column order from first row, then append any extras.
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_frame_metrics_csv(path: str | Path, rows: list[dict]) -> None:
    _write_csv(Path(path), rows)


def write_event_metrics_csv(path: str | Path, rows: list[dict]) -> None:
    _write_csv(Path(path), rows)
