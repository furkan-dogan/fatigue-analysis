"""Legacy sensor summaries, kept outside rendering components."""

import numpy as np

def sensor_stats(emg_rows: list[dict], nirs_rows: list[dict], win: int) -> dict:
    """Compute summary stats for a single session's EMG + NIRS rows."""
    freq = [r["EMG_median_freq_Hz"] for r in emg_rows]
    smo2 = [r["SmO2"]              for r in nirs_rows]
    rms  = [r["EMG_RMS_mV"]        for r in emg_rows]
    return {
        "freq_start": float(np.mean(freq[:win])),
        "freq_end":   float(np.mean(freq[-win:])),
        "freq_arr":   freq,
        "smo2_start": float(np.mean(smo2[:win])),
        "smo2_end":   float(np.mean(smo2[-win:])),
        "smo2_min":   float(min(smo2)),
        "smo2_arr":   smo2,
        "rms_arr":    rms,
        "rms2_arr":   [r["EMG_CH2_RMS_mV"] for r in emg_rows],
        "thb_arr":    [r["THb"] for r in nirs_rows],
        "t_arr":      [r["time_sec"] for r in emg_rows],
        "t_nirs":     [r["time_sec"] for r in nirs_rows],
    }
