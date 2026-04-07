"""Compare pre-fatigue and post-fatigue event CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRIC_LABELS = {
    "kick_count": ("Kick Sayisi", "Kick Count"),
    "mean_duration_sec": ("Ortalama Sure (sn)", "Mean Duration (sec)"),
    "std_duration_sec": ("Sure Std (sn)", "Duration Std (sec)"),
    "mean_peak_kick_height_norm": ("Ortalama Peak Kick Yuksekligi (norm)", "Mean Peak Kick Height (norm)"),
    "mean_active_peak_knee_angle_deg": ("Ortalama Aktif Peak Diz Acisi (deg)", "Mean Active Peak Knee Angle (deg)"),
    "mean_active_knee_rom_deg": ("Ortalama Aktif Diz ROM (deg)", "Mean Active Knee ROM (deg)"),
    "mean_R_KNEE_rom": ("Ortalama Sag Diz ROM (deg)", "Mean Right Knee ROM (deg)"),
    "mean_L_KNEE_rom": ("Ortalama Sol Diz ROM (deg)", "Mean Left Knee ROM (deg)"),
    "mean_R_HIP_rom": ("Ortalama Sag Kalca ROM (deg)", "Mean Right Hip ROM (deg)"),
    "mean_L_HIP_rom": ("Ortalama Sol Kalca ROM (deg)", "Mean Left Hip ROM (deg)"),
    "mean_R_ANKLE_rom": ("Ortalama Sag Ayak Bilegi ROM (deg)", "Mean Right Ankle ROM (deg)"),
    "mean_L_ANKLE_rom": ("Ortalama Sol Ayak Bilegi ROM (deg)", "Mean Left Ankle ROM (deg)"),
    # Velocity metrics
    "mean_active_peak_knee_vel_deg_s": ("Ortalama Peak Diz Acısal Hizi (deg/s)", "Mean Peak Knee Angular Velocity (deg/s)"),
    "mean_active_mean_knee_vel_deg_s": ("Ortalama Diz Acısal Hizi (deg/s)", "Mean Knee Angular Velocity (deg/s)"),
    "mean_time_to_peak_knee_vel_sec": ("Ortalama Peak Hize Ulasma Suresi (sn)", "Mean Time to Peak Knee Velocity (sec)"),
    "mean_active_peak_foot_speed_norm": ("Ortalama Peak Ayak Hizi (norm)", "Mean Peak Foot Speed (norm torso/s)"),
    "mean_active_mean_foot_speed_norm": ("Ortalama Ayak Hizi (norm)", "Mean Foot Speed (norm torso/s)"),
}


METRIC_COLUMNS_MEAN = [
    "duration_sec",
    "peak_kick_height_norm",
    "active_peak_knee_angle_deg",
    "active_knee_rom_deg",
    "R_KNEE_rom",
    "L_KNEE_rom",
    "R_HIP_rom",
    "L_HIP_rom",
    "R_ANKLE_rom",
    "L_ANKLE_rom",
    # Velocity columns
    "active_peak_knee_vel_deg_s",
    "active_mean_knee_vel_deg_s",
    "time_to_peak_knee_vel_sec",
    "active_peak_foot_speed_norm",
    "active_mean_foot_speed_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pre/post movement event metrics")
    parser.add_argument("--pre-events", required=True, help="Pre-fatigue events CSV path")
    parser.add_argument("--post-events", required=True, help="Post-fatigue events CSV path")
    parser.add_argument("--output", default="output/pre_post_comparison.csv", help="Output comparison CSV path")
    parser.add_argument("--delimiter", default=";", help="CSV delimiter (default: ';')")
    return parser.parse_args()


def _read_events(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV bulunamadı: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(rows: list[dict[str, str]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        raw = row.get(key, "")
        if raw in ("", "None", None):
            continue
        out.append(float(raw))
    return out


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = _mean(values)
    assert m is not None
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return var**0.5


def _format(v: float | None) -> str:
    return "" if v is None else f"{v:.3f}"


def _build_metrics(rows: list[dict[str, str]]) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "kick_count": float(len(rows)),
    }

    duration_vals = _to_float(rows, "duration_sec")
    metrics["mean_duration_sec"] = _mean(duration_vals)
    metrics["std_duration_sec"] = _std(duration_vals)

    for col in METRIC_COLUMNS_MEAN:
        vals = _to_float(rows, col)
        metrics[f"mean_{col}"] = _mean(vals)

    return metrics


def main() -> None:
    args = parse_args()
    pre_rows = _read_events(Path(args.pre_events))
    post_rows = _read_events(Path(args.post_events))

    pre_metrics = _build_metrics(pre_rows)
    post_metrics = _build_metrics(post_rows)

    all_metric_keys = list(METRIC_LABELS.keys())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metrik_kodu",
                "metrik_tr",
                "metric_en",
                "pre",
                "post",
                "fark_post_eksi_pre",
                "degisim_yuzde",
            ],
            delimiter=args.delimiter,
        )
        writer.writeheader()

        for key in all_metric_keys:
            pre_val = pre_metrics.get(key)
            post_val = post_metrics.get(key)
            if pre_val is None or post_val is None:
                delta = None
                pct = None
            else:
                delta = post_val - pre_val
                pct = (delta / abs(pre_val) * 100.0) if abs(pre_val) > 1e-8 else None
            tr_label, en_label = METRIC_LABELS[key]
            writer.writerow(
                {
                    "metrik_kodu": key,
                    "metrik_tr": tr_label,
                    "metric_en": en_label,
                    "pre": _format(pre_val),
                    "post": _format(post_val),
                    "fark_post_eksi_pre": _format(delta),
                    "degisim_yuzde": _format(pct),
                }
            )

    print("Pre/Post karşılaştırma tamamlandı.")
    print(f"Pre events: {len(pre_rows)}")
    print(f"Post events: {len(post_rows)}")
    print(f"Çıktı: {output_path}")


if __name__ == "__main__":
    main()
