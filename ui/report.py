"""Athlete report renderer — Section 10 of the Video Analysis page."""
from __future__ import annotations

import datetime

import pandas as pd
import streamlit as st

from src.utils import events_mean, pct_change, change_status


def _header_card(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div style="border:1px solid #334155;border-radius:8px;padding:16px 20px;'
        f'margin-bottom:16px;background:#0f172a">'
        f'<h3 style="margin:0 0 4px 0;color:#f1f5f9">{title}</h3>'
        f'<p style="margin:0;color:#94a3b8;font-size:13px">{subtitle}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _score_bar(score: float, color: str, label: str) -> None:
    st.markdown(
        f'<div style="border-left:4px solid {color};padding:10px 16px;'
        f'border-radius:4px;background:#1e293b;margin:8px 0">'
        f'<span style="font-size:22px;font-weight:700;color:{color}">{score:.1f} / 100</span>'
        f'<span style="color:#94a3b8;margin-left:12px">{label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _suggestion_card(priority: str, title: str, detail: str, source: str, color: str) -> None:
    st.markdown(
        f'<div style="border:1px solid #334155;border-radius:6px;padding:12px 16px;'
        f'margin:6px 0;background:#0f172a">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">'
        f'  <span style="background:{color};color:#fff;font-size:10px;font-weight:700;'
        f'padding:2px 8px;border-radius:3px">{priority}</span>'
        f'  <span style="color:#f1f5f9;font-weight:600;font-size:15px">{title}</span>'
        f'  <span style="color:#64748b;font-size:11px;margin-left:auto">📊 {source}</span>'
        f'</div>'
        f'<p style="margin:0;color:#cbd5e1;font-size:13px;line-height:1.6">{detail}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_athlete_report(
    pre_events: list[dict],
    post_events: list[dict],
    pre_res,
    post_res,
    fi: float,
    sensor_pre: dict,
    sensor_post: dict,
    pre_freq_drop: float,
    post_freq_drop: float,
    pre_smo2_drop: float,
    post_smo2_drop: float,
    sensor_ok: bool,
    real_sensor: bool = False,
) -> None:
    """Render the full athlete report tab."""

    st.subheader("Sporcu Analiz Raporu")
    if real_sensor:
        st.success("📡 Gerçek Sensör Verisi aktif — EMG ve NIRS grafikleri gerçek cihaz çıktısından üretilmiştir.")
    else:
        st.info("🧪 Sensör verisi simüle edilmiştir. Gerçek veri için **📡 Gerçek Sensör** sekmesini kullanın.")
    st.caption(
        "Biyomekanik (video/pose analizi) + EMG + NIRS birleşik değerlendirmesi. "
        "Her bulgu için ölçüm kaynağı belirtilmiştir. Çıktı alınabilir."
    )

    pmv   = events_mean(pre_events,  "active_peak_knee_vel_deg_s")
    pomv  = events_mean(post_events, "active_peak_knee_vel_deg_s")
    prm   = events_mean(pre_events,  "active_knee_rom_deg")
    porm  = events_mean(post_events, "active_knee_rom_deg")
    ph    = events_mean(pre_events,  "peak_kick_height_norm")
    poh   = events_mean(post_events, "peak_kick_height_norm")
    ptt   = events_mean(pre_events,  "time_to_peak_knee_vel_sec")
    pott  = events_mean(post_events, "time_to_peak_knee_vel_sec")
    pfs   = events_mean(pre_events,  "active_peak_foot_speed_norm")
    pofs  = events_mean(post_events, "active_peak_foot_speed_norm")
    pexv  = events_mean(pre_events,  "extension_peak_vel_deg_s")
    poexv = events_mean(post_events, "extension_peak_vel_deg_s")
    prv   = events_mean(pre_events,  "retraction_peak_vel_deg_s")
    porv  = events_mean(post_events, "retraction_peak_vel_deg_s")

    today    = datetime.date.today().strftime("%d.%m.%Y")
    pre_dur  = round(pre_res.total_frames  / pre_res.fps,  1) if pre_res.fps  else 0
    post_dur = round(post_res.total_frames / post_res.fps, 1) if post_res.fps else 0
    vel_pct  = pct_change(pmv, pomv)
    rom_pct  = pct_change(prm, porm)

    fi_color = "#ef4444" if fi >= 66 else ("#f59e0b" if fi >= 33 else "#22c55e")
    fi_label = (
        "Yüksek — Yoğun antrenman yükü. Toparlanma süreci kritik." if fi >= 66
        else "Orta — Yorgunluk birikimi mevcut, dikkatli yüklenme önerilir." if fi >= 33
        else "Düşük — İyi toparlanma. Yük artışı uygun."
    )

    _header_card(
        "Taekwondo Yorgunluk Analiz Raporu",
        f"Tarih: {today} &nbsp;|&nbsp; Pre: {len(pre_events)} tekme, {pre_dur} sn &nbsp;|&nbsp; "
        f"Post: {len(post_events)} tekme, {post_dur} sn &nbsp;|&nbsp; "
        f'Yorgunluk İndeksi: <b style="color:{fi_color}">{fi:.1f}/100</b>',
    )

    # ── 1. Biyomekanik ────────────────────────────────────────────────────────
    st.markdown("### 1. Biyomekanik Bulgular")
    st.caption("📌 Kaynak: MediaPipe pose modeli — frame bazlı eklem açısı, hız ve normalize yükseklik (Savitzky-Golay filtrelemeli).")

    bio_rows = []
    for lbl, pv, pov, direction in [
        ("Peak Diz Hızı (°/s)",      pmv,  pomv,  -1),
        ("Diz ROM (°)",              prm,  porm,  -1),
        ("Tekme Yüksekliği (norm.)", ph,   poh,   -1),
        ("Peak Hıza Süre (sn)",      ptt,  pott,  +1),
        ("Ayak Hızı (norm.)",        pfs,  pofs,  -1),
        ("Uzatma Hızı (°/s)",        pexv, poexv, -1),
        ("Geri Çekim Hızı (°/s)",    prv,  porv,  -1),
    ]:
        p = pct_change(pv, pov)
        bio_rows.append({
            "Metrik":      lbl,
            "Pre (ort.)":  f"{pv:.2f}"  if pv  is not None else "—",
            "Post (ort.)": f"{pov:.2f}" if pov is not None else "—",
            "Δ%":          f"{p:+.1f}%" if p   is not None else "—",
            "Durum":       change_status(p, direction),
        })
    st.dataframe(pd.DataFrame(bio_rows).set_index("Metrik"), use_container_width=True)

    bio_lines = []
    src = "*(📐 Video pose analizi)*"
    if vel_pct is not None:
        if vel_pct < -10:
            bio_lines.append(
                f"**Peak diz hızı** antrenman sonrası **%{abs(vel_pct):.0f} geriledi** "
                f"({pmv:.0f} → {pomv:.0f} °/s). Yorgunluğun hız üretme kapasitesini kısıtladığı görülmektedir. {src}"
            )
        elif vel_pct < -5:
            bio_lines.append(
                f"**Peak diz hızında hafif düşüş** (%{abs(vel_pct):.0f}): "
                f"{pmv:.0f} → {pomv:.0f} °/s. Yorgunluk etkisi erken dönemde başlamaktadır. {src}"
            )
        else:
            bio_lines.append(
                f"**Peak diz hızı stabil** ({pmv:.0f} → {pomv:.0f} °/s, Δ%{vel_pct:+.0f}). "
                f"Hız üretme kapasitesi antrenman yükü altında korunmuştur. {src}"
            )
    if rom_pct is not None:
        if rom_pct < -10:
            bio_lines.append(
                f"**Diz eklem hareket açıklığı %{abs(rom_pct):.0f} azaldı** "
                f"({prm:.1f}° → {porm:.1f}°). Kas sertliği veya yorgunluk kaynaklı eklem kısıtlanması. {src}"
            )
        elif rom_pct > 5:
            bio_lines.append(
                f"**Diz ROM artmış** (%{rom_pct:.0f}): {prm:.1f}° → {porm:.1f}°. "
                f"Isınma etkisiyle hareket serbestisi iyileşmiş. {src}"
            )
        else:
            bio_lines.append(f"**Diz ROM stabil**: {prm:.1f}° → {porm:.1f}° (Δ%{rom_pct:+.0f}). {src}")
    if ph is not None and poh is not None:
        h_pct = pct_change(ph, poh)
        if h_pct is not None and h_pct < -10:
            bio_lines.append(
                f"**Tekme yüksekliği %{abs(h_pct):.0f} düştü** ({ph:.2f} → {poh:.2f} norm.). "
                f"Yorgunlukla birlikte kalça fleksör gücü zayıflamaktadır. {src}"
            )
    for line in bio_lines:
        st.markdown(f"> {line}")

    # ── 2. Nöromüsküler ───────────────────────────────────────────────────────
    st.markdown("### 2. Nöromüsküler Bulgular *(EMG Simülasyonu)*")
    st.caption(
        "📌 Kaynak: K-Myo benzeri yüzey EMG simülasyonu — quadriceps + hamstring. "
        "Median frekans düşüşü kas yorgunluğunun nörofizyolojik göstergesidir (Tip II lif yorulması). "
        "NOT: Fizyolojik model tabanlı simülasyon; gerçek ölçüm için K-Myo cihazı gerekir."
    )
    emg_tbl = []
    if not sensor_ok:
        st.info("Sensör verisi mevcut değil.")
    else:
        ps, pos = sensor_pre, sensor_post
        emg_tbl = [
            {"Parametre": "Median Frekans Başlangıç — Pre",  "Değer": f"{ps['freq_start']:.1f} Hz",  "Açıklama": "Baseline nöromüsküler aktivasyon"},
            {"Parametre": "Median Frekans Bitiş — Pre",      "Değer": f"{ps['freq_end']:.1f} Hz",    "Açıklama": f"Δ = {ps['freq_end']-ps['freq_start']:+.1f} Hz  ({(ps['freq_end']-ps['freq_start'])/ps['freq_start']*100:+.1f}%)"},
            {"Parametre": "Median Frekans Başlangıç — Post", "Değer": f"{pos['freq_start']:.1f} Hz", "Açıklama": "Post-antrenman baseline"},
            {"Parametre": "Median Frekans Bitiş — Post",     "Değer": f"{pos['freq_end']:.1f} Hz",   "Açıklama": f"Δ = {pos['freq_end']-pos['freq_start']:+.1f} Hz  ({(pos['freq_end']-pos['freq_start'])/pos['freq_start']*100:+.1f}%)"},
            {"Parametre": "Post − Pre Düşüş Farkı",          "Değer": f"{post_freq_drop - pre_freq_drop:+.1f} Hz", "Açıklama": "Artı → Post'ta daha fazla nöromüsküler yorgunluk"},
        ]
        st.dataframe(pd.DataFrame(emg_tbl).set_index("Parametre"), use_container_width=True)

        if post_freq_drop > pre_freq_drop + 3:
            sev = "🔴 **Belirgin nöromüsküler yorgunluk**"
            txt = (
                f"Post oturumunda EMG median frekansı **{post_freq_drop:.0f} Hz** düştü "
                f"(pre'de {pre_freq_drop:.0f} Hz). Motor ünite aktivasyonu anlamlı düzeyde bozulmuş. "
                "Hızlı kasılan (Tip II) lifler yorulmuş ve kasılma verimliliği düşmüştür."
            )
        elif post_freq_drop > 8:
            sev = "🟡 **Orta düzey nöromüsküler yorgunluk**"
            txt = (
                f"Post EMG median frekansı {post_freq_drop:.0f} Hz geriledi. "
                "Nöromüsküler yorgunluk tespit edilmiş; kas dayanıklılığı geliştirilmesi önerilir."
            )
        else:
            sev = "🟢 **Nöromüsküler yorgunluk sınırlı**"
            txt = (
                f"Pre/Post frekans düşüşü benzer (pre: {pre_freq_drop:.0f} Hz, post: {post_freq_drop:.0f} Hz). "
                "Motor ünite kalıpları korunmuştur."
            )
        st.markdown(f"> {sev}: {txt}")

    # ── 3. Metabolik ──────────────────────────────────────────────────────────
    st.markdown("### 3. Metabolik Bulgular *(NIRS Simülasyonu)*")
    st.caption(
        "📌 Kaynak: Moxy Monitor benzeri NIRS simülasyonu — vastus lateralis kas oksijenlenmesi. "
        "SmO2 aerobik enerji sisteminin kapasitesini yansıtır. "
        "NOT: Fizyolojik model tabanlı simülasyon; gerçek ölçüm için Moxy Monitor cihazı gerekir."
    )
    nirs_tbl = []
    if not sensor_ok:
        st.info("Sensör verisi mevcut değil.")
    else:
        nirs_tbl = [
            {"Parametre": "SmO2 Başlangıç — Pre",  "Değer": f"%{ps['smo2_start']:.1f}",  "Açıklama": "Dinlenme kas oksijen satürasyonu (normal: %60-80)"},
            {"Parametre": "SmO2 Bitiş — Pre",      "Değer": f"%{ps['smo2_end']:.1f}",    "Açıklama": f"Δ = {ps['smo2_end']-ps['smo2_start']:+.1f}%  |  Pre toplam düşüş"},
            {"Parametre": "SmO2 Minimum — Pre",    "Değer": f"%{ps['smo2_min']:.1f}",    "Açıklama": "<%50 → anaerobik eşik; <%40 → kritik hipoksi"},
            {"Parametre": "SmO2 Başlangıç — Post", "Değer": f"%{pos['smo2_start']:.1f}", "Açıklama": "Post başlangıç"},
            {"Parametre": "SmO2 Bitiş — Post",     "Değer": f"%{pos['smo2_end']:.1f}",   "Açıklama": f"Δ = {pos['smo2_end']-pos['smo2_start']:+.1f}%  |  Post toplam düşüş"},
            {"Parametre": "SmO2 Minimum — Post",   "Değer": f"%{pos['smo2_min']:.1f}",   "Açıklama": "<%50 → anaerobik eşik; <%40 → kritik hipoksi"},
        ]
        st.dataframe(pd.DataFrame(nirs_tbl).set_index("Parametre"), use_container_width=True)

        if post_smo2_drop > pre_smo2_drop + 3:
            sev = "🔴 **Belirgin metabolik stres**"
            txt = (
                f"Post'ta SmO2 **%{post_smo2_drop:.0f}** düştü (pre'de %{pre_smo2_drop:.0f}). "
                "Oksidatif enerji sistemi yetersiz. Mitokondrial kapasite veya kardiyak debi sınırlayıcı."
            )
        elif post_smo2_drop > 5:
            sev = "🟡 **Orta metabolik yük**"
            txt = (
                f"Post SmO2 %{post_smo2_drop:.0f} geriledi. "
                "Aerobik kapasite sınırlarına yaklaşılmaktadır; zone 2 antrenman ile iyileştirilebilir."
            )
        else:
            sev = "🟢 **Metabolik sistem stabil**"
            txt = (
                f"SmO2 düşüşü sınırlı (post: %{post_smo2_drop:.0f}). "
                "Aerobik taban yüke yeterli."
            )
        if pos["smo2_min"] < 50:
            txt += (
                f" **Uyarı:** Min SmO2 %{pos['smo2_min']:.0f} — anaerobik eşiğe yaklaşılmıştır."
            )
        st.markdown(f"> {sev}: {txt}")

    # ── 4. Asimetri ───────────────────────────────────────────────────────────
    st.markdown("### 4. Bilateral Asimetri Bulguları")
    st.caption("📌 Kaynak: Video pose analizi — ASI = |(dominant − non-dominant) / max| × 100")

    pre_ka  = [float(e["knee_asi"]) for e in pre_events  if e.get("knee_asi") is not None]
    post_ka = [float(e["knee_asi"]) for e in post_events if e.get("knee_asi") is not None]
    pre_ha  = [float(e["hip_asi"])  for e in pre_events  if e.get("hip_asi")  is not None]
    post_ha = [float(e["hip_asi"])  for e in post_events if e.get("hip_asi")  is not None]

    if pre_ka or post_ka:
        asi_rows = []
        for albl, pal, poal in [("Diz ASI (%)", pre_ka, post_ka), ("Kalça ASI (%)", pre_ha, post_ha)]:
            if not pal and not poal:
                continue
            pa  = sum(pal)  / len(pal)  if pal  else None
            poa = sum(poal) / len(poal) if poal else None
            asi_rows.append({
                "Ölçüm":        albl,
                "Pre Ort. ASI": f"{pa:+.1f}%"  if pa  is not None else "—",
                "Post Ort. ASI":f"{poa:+.1f}%" if poa is not None else "—",
                "Klinik Eşik":  "|ASI| > 10% klinik anlamlı",
                "Durum":        ("⚠️ Asimetri" if poa is not None and abs(poa) > 10 else "✅ Normal"),
            })
        if asi_rows:
            st.dataframe(pd.DataFrame(asi_rows).set_index("Ölçüm"), use_container_width=True)
            if any(r["Durum"] == "⚠️ Asimetri" for r in asi_rows):
                st.markdown(
                    "> ⚠️ Dominant ve non-dominant bacak arasında yük dengesizliği. "
                    "Uzun vadede sakatlık riski artmaktadır. Unilateral güç antrenmanı önerilir."
                )
            else:
                st.markdown("> ✅ Bilateral asimetri klinik eşiğin altında.")
    else:
        st.info("ASI hesaplaması için yeterli veri yok (sol ve sağ bacak tekmeleri gerekli).")

    # ── 5. Yorgunluk İndeksi ──────────────────────────────────────────────────
    st.markdown("### 5. Genel Yorgunluk Değerlendirmesi")
    st.caption(
        "📌 Kaynak: 7 biyomekanik metriğin ağırlıklı ortalaması — "
        "Diz ROM ×0.25 | Peak hız ×0.25 | Peak hıza süre ×0.15 | "
        "Tekme yüksekliği ×0.15 | Ayak hızı ×0.10 | Tekme süresi ×0.05 | Ort. hız ×0.05"
    )
    _score_bar(fi, fi_color, fi_label)

    fi_detail = []
    if vel_pct is not None:
        fi_detail.append(f"Peak diz hızı Δ%{vel_pct:+.0f} (ağırlık %25)")
    if rom_pct is not None:
        fi_detail.append(f"Diz ROM Δ%{rom_pct:+.0f} (ağırlık %25)")
    if sensor_ok:
        fi_detail.append(f"EMG freq düşüşü post: {post_freq_drop:.0f} Hz | pre: {pre_freq_drop:.0f} Hz")
        fi_detail.append(f"SmO2 düşüşü post: %{post_smo2_drop:.0f} | pre: %{pre_smo2_drop:.0f}")
    if fi_detail:
        st.markdown("**İndekse katkıda bulunan başlıca faktörler:**")
        for fd in fi_detail:
            st.markdown(f"- {fd}")

    # ── 6. Gelişim Önerileri ──────────────────────────────────────────────────
    st.markdown("### 6. Gelişim Önerileri")
    st.caption("Bulgulara göre öncelik sıralamasıyla üretilmiştir. Her önerinin kaynağı sağ üstte gösterilmektedir.")

    suggs_raw: list[tuple[int, str, str, str]] = []

    if sensor_ok and post_freq_drop > 10:
        suggs_raw.append((10, "Nöromüsküler Dayanıklılık",
            f"EMG median frekansı post'ta kritik düşüş gösterdi ({post_freq_drop:.0f} Hz Δ, pre: {pre_freq_drop:.0f} Hz). "
            "Pliometrik antrenman (squat jump, lunge jump, patlayıcı roundhouse) ve hız-kuvvet devresi "
            "(3×8 patlayıcı squat + 3×10 yavaş eksantrik) haftada 2 gün. "
            "Hedef: 8 haftada post frekans düşüşünü <8 Hz'e indirmek.",
            "EMG Simülasyonu"))
    elif sensor_ok and post_freq_drop > 6:
        suggs_raw.append((20, "Nöromüsküler Dayanıklılık (Orta Düzey)",
            f"Orta düzey EMG frekans düşüşü ({post_freq_drop:.0f} Hz Δ). "
            "Mevcut antrenmanına plyometrik komponent eklenmesi yeterli: haftada 1-2 seans 20 dk.",
            "EMG Simülasyonu"))

    if sensor_ok and post_smo2_drop > 8:
        suggs_raw.append((11, "Aerobik Kapasite Geliştirme",
            f"SmO2 post'ta kritik düşüş (%{post_smo2_drop:.0f} Δ, pre: %{pre_smo2_drop:.0f}). "
            "Zone 2 aerobik antrenman (KH 130-145 bpm, 30-45 dk, haftada 3 gün) "
            "oksidatif kapasiteyi 6-12 haftada anlamlı iyileştirir. "
            "Antrenman aralarında aktif toparlanma (hafif bisiklet, yürüyüş) önerilir.",
            "NIRS Simülasyonu"))
    elif sensor_ok and post_smo2_drop > 5:
        suggs_raw.append((30, "Aerobik Taban Güçlendirme",
            f"Orta metabolik yük (SmO2 Δ %{post_smo2_drop:.0f}). Zone 2 çalışma haftada 2 gün yeterli.",
            "NIRS Simülasyonu"))

    if vel_pct is not None and vel_pct < -10:
        suggs_raw.append((12, "Yorgunlukta Hız Koruması",
            f"Peak diz hızı %{abs(vel_pct):.0f} geriledi ({pmv:.0f} → {pomv:.0f} °/s). "
            "Antrenman önerisi: 5×(10 hızlı tekme + 20 sn dinlenme) devresi, direnç bandı ile. "
            "Shadow sparring'de bilinçli hız koruması hedefi belirlenmeli.",
            "Video Pose Analizi"))

    if rom_pct is not None and rom_pct < -10:
        suggs_raw.append((13, "Hareket Genişliği (ROM) Geliştirme",
            f"Diz ROM %{abs(rom_pct):.0f} azaldı ({prm:.1f}° → {porm:.1f}°). "
            "Önce: dinamik ısınma — leg swing, hip circle, lunge stretch (2×10). "
            "Sonra: statik esneme — hamstring, quadriceps, hip flexor (30 sn×3).",
            "Video Pose Analizi"))

    post_ka_mean = sum(post_ka) / len(post_ka) if post_ka else None
    if post_ka_mean is not None and abs(post_ka_mean) > 10:
        suggs_raw.append((40, "Bilateral Denge Geliştirme",
            f"Diz ASI {post_ka_mean:+.0f}% — dominant bacak aşırı yükleniyor. "
            "Unilateral antrenman: tek bacak squat, single-leg RDL (3×8 her bacak ayrı). "
            "Hedef: |ASI| < 10%.",
            "Video Pose Analizi"))

    if fi >= 50:
        suggs_raw.append((50, "Toparlanma Protokolü",
            f"Yorgunluk indeksi {fi:.0f}/100. "
            "Sonraki yüksek yoğunluklu seansa kadar 48-72 saat aktif toparlanma: "
            "hafif yürüyüş, esneme, soğuk-sıcak kontrast banyo (30 sn/90 sn, 3 döngü). "
            "Uyku: ≥8 saat. Protein: ≥1.8 g/kg/gün.",
            "Yorgunluk İndeksi"))

    if not suggs_raw:
        suggs_raw.append((99, "Tüm Göstergeler Normal",
            "Metriklerin tümü kabul edilebilir sınırlar içinde. Kademeli yük artışına hazır.",
            "Tüm sistemler"))

    suggs_raw.sort(key=lambda x: x[0])
    suggs = [(f"ÖNCELİK {i+1}", t, d, s) for i, (_, t, d, s) in enumerate(suggs_raw)]

    for pri, title, detail, src in suggs:
        pc = "#ef4444" if "1" in pri else ("#f59e0b" if "2" in pri else "#3b82f6")
        _suggestion_card(pri, title, detail, src, pc)

    # ── Dışa Aktar ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Raporu Dışa Aktar")

    txt_lines = [
        "TAEKWONDO YORGUNLUK ANALİZ RAPORU",
        "=" * 60,
        f"Tarih         : {today}",
        f"Pre           : {len(pre_events)} tekme, {pre_dur} sn",
        f"Post          : {len(post_events)} tekme, {post_dur} sn",
        f"Yorgunluk İnd.: {fi:.1f}/100 — {fi_label}",
        "",
        "1. BİYOMEKANİK BULGULAR (Kaynak: Video Pose Analizi)",
        "-" * 60,
    ]
    for r in bio_rows:
        txt_lines.append(f"  {r['Metrik']:<38} Pre: {r['Pre (ort.)']: <8} Post: {r['Post (ort.)']: <8} {r['Δ%']: <8} {r['Durum']}")
    for line in bio_lines:
        txt_lines.append(f"  > {line.replace('**', '').replace('*(', '(').replace(')*', ')')}")

    if sensor_ok:
        txt_lines += ["", "2. NÖROMÜSKÜler BULGULAR (Kaynak: EMG Simülasyonu)", "-" * 60]
        for r in emg_tbl:
            txt_lines.append(f"  {r['Parametre']:<45} {r['Değer']:<14} {r['Açıklama']}")
        txt_lines += ["", "3. METABOLİK BULGULAR (Kaynak: NIRS Simülasyonu)", "-" * 60]
        for r in nirs_tbl:
            txt_lines.append(f"  {r['Parametre']:<38} {r['Değer']:<14} {r['Açıklama']}")

    txt_lines += ["", "6. GELİŞİM ÖNERİLERİ", "-" * 60]
    for pri, title, detail, src in suggs:
        txt_lines.append(f"\n  [{pri}] {title}  (Kaynak: {src})")
        for chunk in [detail[i:i+90] for i in range(0, len(detail), 90)]:
            txt_lines.append(f"  {chunk}")

    txt_lines += [
        "", "─" * 60,
        "NOT: EMG ve NIRS verileri fizyolojik model tabanlı simülasyondur.",
        "Gerçek ölçüm: K-Myo (EMG) + Moxy Monitor (NIRS) cihazları ile yapılmalıdır.",
    ]

    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            "📥 Raporu TXT olarak indir",
            "\n".join(txt_lines).encode("utf-8"),
            f"sporcu_raporu_{today.replace('.', '-')}.txt",
            "text/plain",
        )
    with exp2:
        sum_rows = [
            {"Alan": "Tarih",          "Pre": today,          "Post": today},
            {"Alan": "Tekme",          "Pre": len(pre_events),"Post": len(post_events)},
            {"Alan": "Yorgunluk İnd.", "Pre": "—",            "Post": f"{fi:.1f}"},
        ]
        for r in bio_rows:
            sum_rows.append({"Alan": r["Metrik"], "Pre": r["Pre (ort.)"], "Post": r["Post (ort.)"]})
        if sensor_ok:
            sum_rows += [
                {"Alan": "EMG Freq Düşüşü (Hz)", "Pre": f"{pre_freq_drop:.1f}",   "Post": f"{post_freq_drop:.1f}"},
                {"Alan": "SmO2 Düşüşü (%)",      "Pre": f"{pre_smo2_drop:.1f}",   "Post": f"{post_smo2_drop:.1f}"},
                {"Alan": "Min SmO2 (%)",          "Pre": f"{ps['smo2_min']:.1f}",  "Post": f"{pos['smo2_min']:.1f}"},
            ]
        st.download_button(
            "📥 Özet CSV indir",
            pd.DataFrame(sum_rows).to_csv(index=False).encode("utf-8"),
            f"sporcu_ozet_{today.replace('.', '-')}.csv",
            "text/csv",
        )

    st.caption(
        "⚠️ EMG ve NIRS verileri fizyolojik model tabanlı simülasyondur. "
        "Gerçek ölçüm için K-Myo + Moxy Monitor kullanılmalıdır."
    )
