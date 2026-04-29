"""Fizyolojik modele dayalı sentetik EMG ve NIRS verisi üretici.

Video analizinden çıkan tekme eventleri ve eklem hız serilerine göre
literatür parametreleriyle EMG (RMS, median frekans) ve
NIRS (SmO2, THb) serileri üretir.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ── Sabitler (literatür değerleri) ───────────────────────────────────────────
_EMG_BASELINE_RMS = 0.06   # mV — dinlenmede
_EMG_PEAK_RMS     = 0.55   # mV — tekme zirvesinde
_EMG_FREQ_START   = 88.0   # Hz — yorulmamış median frekans
_EMG_FREQ_MIN     = 48.0   # Hz — maksimum yorgunlukta
_EMG_FREQ_DECAY   = 0.9    # Hz / tekme

_NIRS_BASELINE_SMO2 = 68.0   # % — dinlenmede
_NIRS_DROP_PER_KICK = 2.8    # % — tekme başına SmO2 düşüşü
_NIRS_FATIGUE_TREND = -7.0   # % — seans boyunca toplam trend
_NIRS_BASELINE_THB  = 12.05  # g/dL
_NIRS_THB_RISE      = 0.35   # g/dL — egzersizde toplam artış


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def generate_emg(
    frame_rows: list[dict],
    events: list[dict],
    fps: float,
    seed: int = 42,
) -> list[dict]:
    """Her frame için sentetik EMG verisi üret.

    Returns:
        List of dicts: time_sec, EMG_RMS_mV, EMG_median_freq_Hz,
                       EMG_CH1_mV (aktif bacak), EMG_CH2_mV (stance bacak)
    """
    rng = np.random.default_rng(seed)
    n   = len(frame_rows)
    if n == 0:
        return []

    rms_ch1  = np.full(n, _EMG_BASELINE_RMS)
    rms_ch2  = np.full(n, _EMG_BASELINE_RMS * 0.6)
    med_freq = np.full(n, _EMG_FREQ_START)

    for kick_idx, ev in enumerate(events):
        s   = max(0, min(int(ev.get("start_frame", 0)), n - 1))
        p   = max(s, min(int(ev.get("peak_frame",  s)), n - 1))
        e   = max(p, min(int(ev.get("end_frame",   p)), n - 1))
        dur = max(e - s + 1, 1)

        peak_rms = _EMG_PEAK_RMS * rng.uniform(0.75, 1.0)
        for f in range(s, e + 1):
            dist      = abs(f - p)
            rms_ch1[f] += peak_rms * np.exp(-0.5 * (dist / (dur * 0.28)) ** 2)
            rms_ch2[f] += peak_rms * 0.35 * np.exp(-0.5 * (dist / (dur * 0.4)) ** 2)

        freq_after = max(_EMG_FREQ_MIN, _EMG_FREQ_START - kick_idx * _EMG_FREQ_DECAY)
        if e + 1 < n:
            med_freq[e + 1:] = np.minimum(med_freq[e + 1:], freq_after)

    smooth_win = max(3, int(fps * 0.05))
    rms_ch1  = _smooth(rms_ch1,  smooth_win)
    rms_ch2  = _smooth(rms_ch2,  smooth_win)
    med_freq = _smooth(med_freq, max(3, int(fps * 0.15)))

    rms_ch1  += rng.normal(0, 0.008, n)
    rms_ch2  += rng.normal(0, 0.006, n)
    med_freq += rng.normal(0, 1.2,   n)

    rms_ch1  = np.clip(rms_ch1,  0.01, 2.0)
    rms_ch2  = np.clip(rms_ch2,  0.01, 2.0)
    med_freq = np.clip(med_freq, _EMG_FREQ_MIN, 110.0)

    return [
        {
            "time_sec":           round(float(fr["time_sec"]), 4),
            "EMG_RMS_mV":         round(float(rms_ch1[i]), 4),
            "EMG_CH2_RMS_mV":     round(float(rms_ch2[i]), 4),
            "EMG_median_freq_Hz": round(float(med_freq[i]), 2),
        }
        for i, fr in enumerate(frame_rows)
    ]


def generate_nirs(
    frame_rows: list[dict],
    events: list[dict],
    fps: float,
    seed: int = 42,
) -> list[dict]:
    """Her frame için sentetik NIRS (SmO2, THb) verisi üret.

    Moxy ~2 saniyelik güncelleme hızını yansıtmak için ağır smooth uygulanır.
    """
    rng = np.random.default_rng(seed)
    n   = len(frame_rows)
    if n == 0:
        return []

    smo2 = np.full(n, _NIRS_BASELINE_SMO2) + np.linspace(0, _NIRS_FATIGUE_TREND, n)
    thb  = np.full(n, _NIRS_BASELINE_THB)  + np.linspace(0, _NIRS_THB_RISE, n)

    for kick_idx, ev in enumerate(events):
        s   = max(0, min(int(ev.get("start_frame", 0)), n - 1))
        e   = max(s, min(int(ev.get("end_frame",   s)), n - 1))
        dur = max(e - s + 1, 1)

        kick_drop = _NIRS_DROP_PER_KICK * (1.0 + kick_idx * 0.04)
        for f in range(s, e + 1):
            smo2[f] -= kick_drop * np.sin(np.pi * (f - s) / dur)

        rec_frames = max(1, int(fps * 3.0))
        for f in range(e + 1, min(n, e + rec_frames + 1)):
            smo2[f] -= kick_drop * (1 - (f - e) / rec_frames) * 0.45

    # Moxy güncelleme hızı ~2sn — kasıtlı olarak ağır smooth
    moxy_smooth = max(3, int(fps * 2.0))
    smo2 = _smooth(smo2, moxy_smooth)
    thb  = _smooth(thb,  moxy_smooth)

    smo2 += rng.normal(0, 0.6, n)
    thb  += rng.normal(0, 0.04, n)

    smo2 = np.clip(smo2, 15.0, 98.0)
    thb  = np.clip(thb,  8.0,  20.0)

    return [
        {
            "time_sec": round(float(fr["time_sec"]), 4),
            "SmO2":     round(float(smo2[i]), 2),
            "THb":      round(float(thb[i]),  3),
        }
        for i, fr in enumerate(frame_rows)
    ]


def generate_interpretation(
    events: list[dict],
    emg_rows: list[dict],
    nirs_rows: list[dict],
    fps: float,
) -> list[str]:
    """Verilerden otomatik Türkçe yorum metni üret."""
    if not events or not emg_rows or not nirs_rows:
        return ["Yorum için yeterli veri yok."]

    freq_arr = np.array([r["EMG_median_freq_Hz"] for r in emg_rows])
    smo2_arr = np.array([r["SmO2"]              for r in nirs_rows])
    rms_arr  = np.array([r["EMG_RMS_mV"]        for r in emg_rows])

    win        = max(1, int(fps * 5))
    freq_start = float(freq_arr[:win].mean())
    freq_end   = float(freq_arr[-win:].mean())
    smo2_start = float(smo2_arr[:win].mean())
    smo2_end   = float(smo2_arr[-win:].mean())
    smo2_min   = float(smo2_arr.min())
    freq_drop  = freq_start - freq_end
    smo2_drop  = smo2_start - smo2_end

    lines: list[str] = [f"Toplam {len(events)} tekme tespit edildi."]

    if freq_drop >= 10:
        lines.append(
            f"EMG median frekansı {freq_start:.0f} Hz → {freq_end:.0f} Hz "
            f"({freq_drop:.0f} Hz düşüş) — belirgin nöromüsküler yorgunluk."
        )
    elif freq_drop >= 5:
        lines.append(
            f"EMG median frekansı {freq_start:.0f} Hz → {freq_end:.0f} Hz — orta düzey kas yorgunluğu."
        )
    else:
        lines.append("EMG median frekansında belirgin düşüş yok — kas yorgunluğu sınırlı.")

    if smo2_drop >= 8:
        lines.append(
            f"Kas oksijen satürasyonu %{smo2_start:.0f} → %{smo2_end:.0f} "
            f"(%{smo2_drop:.0f} düşüş) — metabolik yorgunluk belirgin."
        )
    elif smo2_drop >= 3:
        lines.append(f"SmO2 %{smo2_start:.0f} → %{smo2_end:.0f} — orta düzey metabolik yük.")

    if smo2_min < 50:
        lines.append(f"Minimum SmO2: %{smo2_min:.0f} — kas anaerobik eşiğe yaklaştı.")

    threshold  = freq_start * 0.85
    onset_kick = next(
        (i + 1 for i, ev in enumerate(events)
         if int(ev.get("peak_frame", 0)) < len(freq_arr)
         and freq_arr[int(ev.get("peak_frame", 0))] < threshold),
        None,
    )
    if onset_kick:
        lines.append(f"{onset_kick}. tekmeden itibaren kas yorgunluğu belirginleşti.")

    ch2_arr  = np.array([r["EMG_CH2_RMS_mV"] for r in emg_rows])
    ch1_mean = float(rms_arr.mean())
    ch2_mean = float(ch2_arr.mean())
    if ch1_mean > 1e-6:
        asym = abs(ch1_mean - ch2_mean) / ch1_mean * 100
        if asym > 25:
            lines.append(
                f"Sağ/sol kas aktivasyon asimetrisi %{asym:.0f} — "
                "dominant bacak belirgin şekilde daha fazla çalışıyor."
            )

    return lines
