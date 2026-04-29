"""Fizyolojik modele dayalı sentetik EMG ve NIRS verisi üretici.

Video analizinden çıkan tekme eventleri ve eklem hız serilerine göre
literatür parametreleriyle gerçekçi EMG (RMS, median frekans) ve
NIRS (SmO2, THb) serileri üretir.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


# ── Sabitler (literatür değerleri) ───────────────────────────────────────────
_EMG_BASELINE_RMS    = 0.06   # mV — dinlenmede
_EMG_PEAK_RMS        = 0.55   # mV — tekme zirvesinde
_EMG_FREQ_START      = 88.0   # Hz — yorulmamış median frekans
_EMG_FREQ_MIN        = 48.0   # Hz — maksimum yorgunlukta
_EMG_FREQ_DECAY      = 0.9    # Hz / tekme

_NIRS_BASELINE_SMO2  = 68.0   # % — dinlenmede
_NIRS_DROP_PER_KICK  = 2.8    # % — tekme başına SmO2 düşüşü
_NIRS_FATIGUE_TREND  = -7.0   # % — seans boyunca toplam trend
_NIRS_BASELINE_THB   = 12.05  # g/dL
_NIRS_THB_RISE       = 0.35   # g/dL — egzersizde toplam artış


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def generate_synthetic_emg(
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
        s = int(ev.get("start_frame", 0))
        p = int(ev.get("peak_frame",  s))
        e = int(ev.get("end_frame",   p))
        leg = str(ev.get("active_leg", "R"))

        s = max(0, min(s, n - 1))
        p = max(s, min(p, n - 1))
        e = max(p, min(e, n - 1))

        dur = max(e - s + 1, 1)
        peak_rms = _EMG_PEAK_RMS * rng.uniform(0.75, 1.0)

        # RMS gaussian tepe
        for f in range(s, e + 1):
            dist = abs(f - p)
            rms_ch1[f] += peak_rms * np.exp(-0.5 * (dist / (dur * 0.28)) ** 2)
            # Stance bacak: izometrik aktivasyon, daha düşük
            rms_ch2[f] += peak_rms * 0.35 * np.exp(-0.5 * (dist / (dur * 0.4)) ** 2)

        # Median frekans: her tekmeden sonra kümülatif düşüş
        fatigue_drop = kick_idx * _EMG_FREQ_DECAY
        freq_after   = max(_EMG_FREQ_MIN, _EMG_FREQ_START - fatigue_drop)
        if e + 1 < n:
            med_freq[e + 1:] = np.minimum(med_freq[e + 1:], freq_after)

    # Hafif smooth (kas sinyali)
    smooth_win = max(3, int(fps * 0.05))
    rms_ch1  = _smooth(rms_ch1,  smooth_win)
    rms_ch2  = _smooth(rms_ch2,  smooth_win)
    med_freq = _smooth(med_freq, max(3, int(fps * 0.15)))

    # Gerçekçi gürültü
    rms_ch1  += rng.normal(0, 0.008, n)
    rms_ch2  += rng.normal(0, 0.006, n)
    med_freq += rng.normal(0, 1.2,   n)

    rms_ch1  = np.clip(rms_ch1,  0.01, 2.0)
    rms_ch2  = np.clip(rms_ch2,  0.01, 2.0)
    med_freq = np.clip(med_freq, _EMG_FREQ_MIN, 110.0)

    rows = []
    for i, fr in enumerate(frame_rows):
        rows.append({
            "time_sec":            round(float(fr["time_sec"]), 4),
            "EMG_RMS_mV":          round(float(rms_ch1[i]), 4),
            "EMG_CH2_RMS_mV":      round(float(rms_ch2[i]), 4),
            "EMG_median_freq_Hz":  round(float(med_freq[i]), 2),
        })
    return rows


def generate_synthetic_nirs(
    frame_rows: list[dict],
    events: list[dict],
    fps: float,
    seed: int = 42,
) -> list[dict]:
    """Her frame için sentetik NIRS (SmO2, THb) verisi üret.

    Moxy 2 saniyelik güncelleme hızını yansıtmak için ağır smooth uygulanır.
    """
    rng = np.random.default_rng(seed)
    n   = len(frame_rows)
    if n == 0:
        return []

    smo2 = np.full(n, _NIRS_BASELINE_SMO2)
    thb  = np.full(n, _NIRS_BASELINE_THB)

    # Zaman boyunca genel yorgunluk trendi
    smo2 += np.linspace(0, _NIRS_FATIGUE_TREND, n)
    thb  += np.linspace(0, _NIRS_THB_RISE, n)

    # Tekme anında SmO2 lokal düşüş + kısmi toparlanma
    for kick_idx, ev in enumerate(events):
        s = int(ev.get("start_frame", 0))
        e = int(ev.get("end_frame",   s))
        s = max(0, min(s, n - 1))
        e = max(s, min(e, n - 1))

        kick_drop = _NIRS_DROP_PER_KICK * (1.0 + kick_idx * 0.04)

        # Düşüş (tekme sırasında)
        dur = max(e - s + 1, 1)
        for f in range(s, e + 1):
            progress = (f - s) / dur
            smo2[f] -= kick_drop * np.sin(np.pi * progress)

        # Kısmi toparlanma (tekme sonrası 3sn)
        rec_frames = max(1, int(fps * 3.0))
        for f in range(e + 1, min(n, e + rec_frames + 1)):
            t = (f - e) / rec_frames
            smo2[f] -= kick_drop * (1 - t) * 0.45

    # Moxy güncelleme hızı ~2sn → ağır smooth
    moxy_smooth = max(3, int(fps * 2.0))
    smo2 = _smooth(smo2, moxy_smooth)
    thb  = _smooth(thb,  moxy_smooth)

    # Gürültü
    smo2 += rng.normal(0, 0.6, n)
    thb  += rng.normal(0, 0.04, n)

    smo2 = np.clip(smo2, 15.0, 98.0)
    thb  = np.clip(thb,  8.0,  20.0)

    rows = []
    for i, fr in enumerate(frame_rows):
        rows.append({
            "time_sec": round(float(fr["time_sec"]), 4),
            "SmO2":     round(float(smo2[i]), 2),
            "THb":      round(float(thb[i]),  3),
        })
    return rows


def generate_interpretation(
    events: list[dict],
    emg_rows: list[dict],
    nirs_rows: list[dict],
    fps: float,
) -> list[str]:
    """Verilerden otomatik Türkçe yorum metni üret."""
    if not events or not emg_rows or not nirs_rows:
        return ["Yorum için yeterli veri yok."]

    lines: list[str] = []
    n = len(emg_rows)

    freq_arr = np.array([r["EMG_median_freq_Hz"] for r in emg_rows])
    smo2_arr = np.array([r["SmO2"] for r in nirs_rows])
    rms_arr  = np.array([r["EMG_RMS_mV"] for r in emg_rows])

    win = max(1, int(fps * 5))
    freq_start = float(freq_arr[:win].mean())
    freq_end   = float(freq_arr[-win:].mean())
    smo2_start = float(smo2_arr[:win].mean())
    smo2_end   = float(smo2_arr[-win:].mean())
    smo2_min   = float(smo2_arr.min())
    freq_drop  = freq_start - freq_end
    smo2_drop  = smo2_start - smo2_end

    lines.append(f"Toplam {len(events)} tekme tespit edildi.")

    # EMG yorgunluk
    if freq_drop >= 10:
        lines.append(
            f"EMG median frekansı {freq_start:.0f} Hz → {freq_end:.0f} Hz "
            f"({freq_drop:.0f} Hz düşüş) — belirgin nöromüsküler yorgunluk."
        )
    elif freq_drop >= 5:
        lines.append(
            f"EMG median frekansı {freq_start:.0f} Hz → {freq_end:.0f} Hz "
            f"— orta düzey kas yorgunluğu."
        )
    else:
        lines.append("EMG median frekansında belirgin düşüş yok — kas yorgunluğu sınırlı.")

    # NIRS yorgunluk
    if smo2_drop >= 8:
        lines.append(
            f"Kas oksijen satürasyonu %{smo2_start:.0f} → %{smo2_end:.0f} "
            f"(%{smo2_drop:.0f} düşüş) — metabolik yorgunluk belirgin."
        )
    elif smo2_drop >= 3:
        lines.append(
            f"SmO2 %{smo2_start:.0f} → %{smo2_end:.0f} — orta düzey metabolik yük."
        )

    if smo2_min < 50:
        lines.append(f"Minimum SmO2: %{smo2_min:.0f} — kas anaerobik eşiğe yaklaştı.")

    # Yorgunluk başlangıç noktası (ilk frekans %15 düşüş)
    threshold = freq_start * 0.85
    onset_kick = None
    for i, ev in enumerate(events):
        p = int(ev.get("peak_frame", 0))
        if p < len(freq_arr) and freq_arr[p] < threshold:
            onset_kick = i + 1
            break
    if onset_kick:
        lines.append(f"{onset_kick}. tekmeden itibaren kas yorgunluğu belirginleşti.")

    # Asimetri (CH1 vs CH2)
    ch1_mean = float(rms_arr.mean())
    ch2_arr  = np.array([r["EMG_CH2_RMS_mV"] for r in emg_rows])
    ch2_mean = float(ch2_arr.mean())
    if ch1_mean > 1e-6:
        asym = abs(ch1_mean - ch2_mean) / ch1_mean * 100
        if asym > 25:
            lines.append(
                f"Sağ/sol kas aktivasyon asimetrisi %{asym:.0f} — "
                "dominant bacak belirgin şekilde daha fazla çalışıyor."
            )

    return lines
