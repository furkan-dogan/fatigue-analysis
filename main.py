"""CLI entry point for taekwondo movement analysis."""

from __future__ import annotations

import argparse

from src.pipeline import run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taekwondo movement analysis")
    parser.add_argument("--input", default="videos/sample.mp4", help="Input video path")
    parser.add_argument("--output", default="output/annotated.mp4", help="Output annotated video path")
    parser.add_argument("--show-joint-labels", action="store_true", help="Overlay joint labels on video")
    parser.add_argument("--frame-csv", default="output/frame_metrics.csv")
    parser.add_argument("--events-csv", default="output/kick_events.csv")
    parser.add_argument("--event-peak-prominence-norm", type=float, default=0.06)
    parser.add_argument("--event-min-distance-sec", type=float, default=0.4)
    parser.add_argument("--event-min-duration-sec", type=float, default=0.15)
    parser.add_argument("--event-max-duration-sec", type=float, default=6.0)
    parser.add_argument("--event-min-knee-rom-deg", type=float, default=20.0, help="Min knee ROM for a valid kick (deg)")
    parser.add_argument("--event-min-peak-height", type=float, default=-0.3, help="Min normalized kick height at peak")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = run_analysis(
        input_path=args.input,
        output_path=args.output,
        frame_csv_path=args.frame_csv,
        events_csv_path=args.events_csv,
        show_joint_labels=args.show_joint_labels,
        event_peak_prominence_norm=args.event_peak_prominence_norm,
        event_min_distance_sec=args.event_min_distance_sec,
        event_min_duration_sec=args.event_min_duration_sec,
        event_max_duration_sec=args.event_max_duration_sec,
        event_min_knee_rom_deg=args.event_min_knee_rom_deg,
        event_min_peak_kick_height_norm=args.event_min_peak_height,
    )

    print(f"Analiz tamam. Toplam frame: {result.total_frames}  FPS: {result.fps:.1f}")

    if result.knee_summary:
        print(f"Max R diz açısı: {result.knee_summary['max']:.2f}°")
        print(f"Min R diz açısı: {result.knee_summary['min']:.2f}°")
    else:
        print("Diz açısı hesaplanamadı (pose bulunamadı).")

    events = result.events
    print(f"Tespit edilen tekme sayısı: {len(events)}")
    if events:
        mean_dur = sum(float(e["duration_sec"]) for e in events) / len(events)
        mean_rom = sum(float(e["active_knee_rom_deg"]) for e in events) / len(events)
        r_count = sum(1 for e in events if e.get("active_leg") == "R")
        l_count = sum(1 for e in events if e.get("active_leg") == "L")

        peak_vel_vals = [float(e["active_peak_knee_vel_deg_s"]) for e in events if e.get("active_peak_knee_vel_deg_s") is not None]
        peak_spd_vals = [float(e["active_peak_foot_speed_norm"]) for e in events if e.get("active_peak_foot_speed_norm") is not None]

        print(f"Ort. süre: {mean_dur:.3f}s  |  Ort. aktif diz ROM: {mean_rom:.1f}°")
        print(f"Aktif bacak: R={r_count}, L={l_count}")
        if peak_vel_vals:
            print(f"Ort. peak diz açısal hızı: {sum(peak_vel_vals)/len(peak_vel_vals):.1f} deg/s")
        if peak_spd_vals:
            print(f"Ort. peak ayak hızı (norm): {sum(peak_spd_vals)/len(peak_spd_vals):.3f} torso/s")

    print(f"\nFrame CSV  : {result.frame_csv_path}")
    print(f"Events CSV : {result.events_csv_path}")
    print(f"Video      : {result.output_video_path}")


if __name__ == "__main__":
    main()
