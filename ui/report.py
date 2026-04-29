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


def _supplement_card(name: str, dose: str, timing: str, reason: str, evidence: str, color: str) -> None:
    st.markdown(
        f'<div style="border:1px solid {color}55;border-radius:8px;padding:14px 18px;background:#0f172a;margin:8px 0">'
        f'<div style="font-weight:700;color:#f1f5f9;font-size:15px;margin-bottom:10px">{name}</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">'
        f'  <div><div style="color:#64748b;font-size:10px;font-weight:600;text-transform:uppercase;margin-bottom:2px">Doz</div>'
        f'  <div style="color:{color};font-size:13px;font-weight:600">{dose}</div></div>'
        f'  <div><div style="color:#64748b;font-size:10px;font-weight:600;text-transform:uppercase;margin-bottom:2px">Zamanlama</div>'
        f'  <div style="color:#e2e8f0;font-size:13px">{timing}</div></div>'
        f'</div>'
        f'<p style="margin:0 0 8px 0;color:#cbd5e1;font-size:13px;line-height:1.6">{reason}</p>'
        f'<div style="color:#64748b;font-size:11px;border-top:1px solid #1e293b;padding-top:6px">📚 Kanıt düzeyi: {evidence}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _risk_card(label: str, risk_level: str, detail: str, color: str) -> None:
    st.markdown(
        f'<div style="border-left:4px solid {color};padding:8px 14px;border-radius:0 6px 6px 0;background:#1e293b;margin:5px 0">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:3px">'
        f'  <span style="color:{color};font-weight:600;font-size:14px">{label}</span>'
        f'  <span style="background:{color}22;color:{color};padding:1px 7px;border-radius:3px;font-size:10px;font-weight:700">{risk_level}</span>'
        f'</div>'
        f'<p style="margin:0;color:#94a3b8;font-size:12px;line-height:1.5">{detail}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_injury_risk(
    fi: float,
    vel_pct: float | None,
    rom_pct: float | None,
    post_ka_mean: float | None,
    sensor_ok: bool,
    sensor_post: dict,
) -> int:
    st.markdown("### 7. Yaralanma Risk Profili")
    st.caption(
        "📌 Kaynak: Bilateral asimetri, yorgunluk indeksi, ROM kaybı ve metabolik stres birleşimi. "
        "Risk skoru 0–100 arasında hesaplanır; ≥50 = yüksek risk."
    )

    risk = 0
    factors: list[tuple[str, str, str, str]] = []

    if post_ka_mean is not None:
        if abs(post_ka_mean) > 15:
            risk += 25
            factors.append(("Yüksek bilateral asimetri",
                f"Diz ASI {post_ka_mean:+.0f}% → dominant bacak aşırı yükleniyor; tekrarlı stres yaralanması riski.",
                "#ef4444", "YÜKSEK"))
        elif abs(post_ka_mean) > 10:
            risk += 12
            factors.append(("Orta bilateral asimetri",
                f"Diz ASI {post_ka_mean:+.0f}% → klinik eşiğin (±10%) üzerinde; izleme gerektirir.",
                "#f59e0b", "ORTA"))

    if fi > 66:
        risk += 25
        factors.append(("Yüksek yorgunluk",
            f"Yorgunluk İndeksi {fi:.0f}/100 → yorgun kasta nöromüsküler kontrol bozulur.",
            "#ef4444", "YÜKSEK"))
    elif fi > 33:
        risk += 12
        factors.append(("Orta yorgunluk",
            f"Yorgunluk İndeksi {fi:.0f}/100 → birikimli yük takip edilmeli.",
            "#f59e0b", "ORTA"))

    if rom_pct is not None and rom_pct < -15:
        risk += 20
        factors.append(("Belirgin ROM kısıtlanması",
            f"Diz ROM %{abs(rom_pct):.0f} düştü → kas sertliği + eklem kısıtlanması → hamstring/quadriceps strain riski.",
            "#ef4444", "YÜKSEK"))
    elif rom_pct is not None and rom_pct < -8:
        risk += 10
        factors.append(("Hafif ROM azalması",
            f"Diz ROM %{abs(rom_pct):.0f} düştü → esneme protokolü önerilir.",
            "#f59e0b", "ORTA"))

    if vel_pct is not None and vel_pct < -15:
        risk += 15
        factors.append(("Belirgin hız kaybı",
            f"Peak hız %{abs(vel_pct):.0f} geriledi → kasın hızlı kasılma kapasitesi bozulmuş; ani yük altında risk.",
            "#f59e0b", "ORTA"))

    if sensor_ok and sensor_post:
        smo2_min = sensor_post.get("smo2_min", 100)
        if smo2_min < 45:
            risk += 12
            factors.append(("Kritik kas hipoksisi",
                f"Min SmO2 %{smo2_min:.0f} → anaerobik eşiğin belirgin altında; kas hasarı birikmekte.",
                "#ef4444", "YÜKSEK"))
        elif smo2_min < 50:
            risk += 6
            factors.append(("Kas hipoksisi sınırında",
                f"Min SmO2 %{smo2_min:.0f} → anaerobik eşiğe yakın.",
                "#f59e0b", "ORTA"))

    risk = min(100, risk)
    risk_color = "#ef4444" if risk >= 50 else ("#f59e0b" if risk >= 25 else "#22c55e")
    risk_label = "Yüksek Risk" if risk >= 50 else ("Orta Risk" if risk >= 25 else "Düşük Risk")

    rc1, rc2 = st.columns([1, 2])
    with rc1:
        st.markdown(
            f'<div style="border:2px solid {risk_color};border-radius:12px;padding:24px 16px;'
            f'text-align:center;background:#0f172a">'
            f'<div style="font-size:48px;font-weight:800;color:{risk_color}">{risk}</div>'
            f'<div style="color:{risk_color};font-size:12px;font-weight:600">/100</div>'
            f'<div style="color:#94a3b8;font-size:12px;margin-top:8px">Yaralanma Risk Skoru</div>'
            f'<div style="color:{risk_color};font-size:15px;font-weight:700;margin-top:6px">{risk_label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with rc2:
        if not factors:
            st.success("✅ Anlamlı risk faktörü tespit edilmedi. Mevcut yük profili düşük risk taşıyor.")
        else:
            for fname, fdesc, fcolor, flevel in factors:
                _risk_card(fname, flevel, fdesc, fcolor)

    if risk >= 25:
        st.markdown("**Taekwondo'ya Özel Uyarılar:**")
        if post_ka_mean is not None and abs(post_ka_mean) > 10:
            st.markdown(f"- 🦵 **Dominant bacak tekrarlı stres**: ASI {post_ka_mean:+.0f}% → tibia/fibula stres kırığı riski; yük dağılımını dengele")
        if rom_pct is not None and rom_pct < -10:
            st.markdown("- 🔴 **Hamstring/quadriceps gerilmesi**: ROM düşüşü + yüksek hız kombinasyonu → kas yırtığı riski; dinamik ısınma zorunlu")
        if fi > 50:
            st.markdown("- ⚠️ **ACL zorlanma riski**: Yorgun kasta nöromüsküler kontrol bozulur → ani yön değişimlerinde risk artar; teknik seans öncelikli")
        if sensor_ok and sensor_post and sensor_post.get("smo2_min", 100) < 50:
            st.markdown("- 💧 **Dehidrasyon / elektrolit eksikliği**: Düşük SmO2, kas perfüzyon sorununa işaret edebilir; sıvı alımını artır")

    return risk


def _render_nutrition(
    fi: float,
    vel_pct: float | None,
    rom_pct: float | None,
    post_freq_drop: float,
    post_smo2_drop: float,
    sensor_ok: bool,
    sensor_post: dict,
) -> list[dict]:
    st.markdown("### 8. Beslenme & Takviye Önerileri")
    st.caption(
        "📌 Kaynak: Analiz sonuçlarından türetilmiş, bireysel metrik değerlere göre önceliklendirilmiş öneriler. "
        "Doz bilgileri ISSN (International Society of Sports Nutrition) kılavuzlarına dayanmaktadır."
    )
    st.warning("⚕️ Bu öneriler genel bilgi amaçlıdır. Uygulama öncesinde spor hekimi veya diyetisyen ile görüşünüz.")

    supps: list[dict] = []

    neuro_fatigued = (sensor_ok and post_freq_drop > 8) or (not sensor_ok and vel_pct is not None and vel_pct < -10)
    if neuro_fatigued:
        reason_k = (
            f"EMG median frekansı post'ta {post_freq_drop:.0f} Hz düştü" if sensor_ok
            else f"Peak diz hızı %{abs(vel_pct):.0f} geriledi"
        )
        supps.append({
            "priority": 1, "color": "#ef4444",
            "name": "💪 Kreatin Monohidrat",
            "dose": "3–5 g/gün (yükleme: 20 g/gün × 5 gün)",
            "timing": "Antrenman sonrası karbonhidrat veya protein ile",
            "reason": (
                f"{reason_k}. Kreatin, kasın ATP-PC sistemini hızla yeniler, Tip II hızlı kasılan liflerin "
                "patlayıcı güç üretimini ve toparlanmasını destekler. Taekwondo'nun tekrarlı patlayıcı tekmeleri "
                "için en kanıtlı ergojenik takviyedir."
            ),
            "evidence": "A — ISSN Grade A (Rawson & Volek, 2003; meta-analizlerle destekli)",
        })

    aerob_stressed = (sensor_ok and post_smo2_drop > 5) or (not sensor_ok and fi > 50)
    if aerob_stressed:
        reason_b = (
            f"SmO2 post'ta %{post_smo2_drop:.0f} düştü → kas tampon kapasitesi yetersiz" if sensor_ok
            else f"Yorgunluk indeksi {fi:.0f}/100 → tekrarlı anaerobik çalışma"
        )
        supps.append({
            "priority": 1 if (sensor_ok and post_smo2_drop > 8) else 2, "color": "#f59e0b",
            "name": "⚡ Beta-Alanin",
            "dose": "3.2–6.4 g/gün (bölünmüş dozlarda)",
            "timing": "Günde 2–4 kez yemekle (karıncalanma yan etkisi azalır)",
            "reason": (
                f"{reason_b}. Beta-alanin, kas karnosin depolarını artırarak laktik asit birikimini tamponlar. "
                "Tekrarlı yüksek yoğunluklu tekme performansı 4–6 haftalık kullanımda iyileşir."
            ),
            "evidence": "A — Hobson et al. (2012); Saunders et al. (2017) meta-analiz",
        })

    if fi > 33 or (vel_pct is not None and vel_pct < -5):
        supps.append({
            "priority": 2, "color": "#3b82f6",
            "name": "🥩 Whey Protein / BCAA",
            "dose": "20–40 g whey protein veya 5–10 g BCAA (lösin ağırlıklı)",
            "timing": "Antrenman bitiminden sonra 30 dakika içinde",
            "reason": (
                f"Yorgunluk indeksi {fi:.0f}/100 — yoğun antrenman sonrası kas protein sentezi artar; "
                "toparlanma penceresi 30–60 dakikadır. Lösin (2.5–3 g) mTOR aktivasyonunu tetikler, "
                "kas protein yıkımını sınırlar."
            ),
            "evidence": "A — ISSN pozisyon bildirisi (Stokes et al., 2018)",
        })

    if sensor_ok and post_smo2_drop > 8:
        supps.append({
            "priority": 1, "color": "#8b5cf6",
            "name": "🍌 Karbonhidrat Stratejisi",
            "dose": "Öncesi: 1–4 g/kg (3–4 saat önce) | Sırası: 30–60 g/saat",
            "timing": "Antrenman 3–4 saat öncesi ve sırası",
            "reason": (
                f"SmO2 %{post_smo2_drop:.0f} düştü → glikojen depolarının yetersizliğine işaret. "
                "Yeterli karbonhidrat yüklenmesi aerobik sistemi korur ve SmO2 düşüşünü yavaşlatır. "
                "Glikojen ressentezi için antrenman sonrası ilk 2 saatte 1–1.2 g/kg karbonhidrat kritik."
            ),
            "evidence": "A — Burke et al. (2011) Journal of Sports Sciences",
        })

    if fi > 40:
        supps.append({
            "priority": 3, "color": "#22c55e",
            "name": "🧘 Magnezyum (Bisglisinat/Malat)",
            "dose": "300–400 mg/gün",
            "timing": "Gece yatmadan 30–60 dakika önce",
            "reason": (
                f"Yorgunluk indeksi {fi:.0f}/100 — yoğun antrenmanda terlemeyle magnezyum kaybı olur. "
                "Kas gevşemesi, uyku kalitesi (derin uyku süresi) ve sinir iletiminde kritik rol; "
                "eksikliği kramplara ve toparlanma gecikmesine yol açar. Bisglisinat formu en iyi emilimi sağlar."
            ),
            "evidence": "B — Abbasi et al. (2012); Setaro et al. (2014)",
        })

    if vel_pct is not None and vel_pct < -8:
        supps.append({
            "priority": 2, "color": "#f97316",
            "name": "☕ Kafein",
            "dose": "3–6 mg/kg vücut ağırlığı (70 kg için ~210–420 mg)",
            "timing": "Antrenman 30–60 dakika öncesi",
            "reason": (
                f"Peak hız %{abs(vel_pct):.0f} geriledi — kafein adenozin antagonizması ile yorgunluk algısını "
                "azaltır, nöromüsküler iletimi hızlandırır, patlayıcı güç ve reaksiyon süresini iyileştirir. "
                "Gece çalışmalarında uyku kalitesini bozabileceği için zamanlama önemlidir."
            ),
            "evidence": "A — Grgic et al. (2021) British Journal of Sports Medicine meta-analiz",
        })

    if fi > 33 or (rom_pct is not None and rom_pct < -8):
        supps.append({
            "priority": 3, "color": "#06b6d4",
            "name": "🐟 Omega-3 + D Vitamini",
            "dose": "EPA+DHA: 2–4 g/gün | D Vitamini: 2000–4000 IU/gün",
            "timing": "Yemekle (sabah veya öğle, her gün)",
            "reason": (
                "Yoğun antrenman sonrası inflamasyonu azaltır (Omega-3: IL-6, TNF-α baskısı). "
                "D vitamini kas gücü ve immün fonksiyon için kritik; Türkiye'deki sporcularda D vitamini "
                "eksikliği yaygındır. ROM kısıtlanmasının inflamatuar bileşenini azaltmada destekleyici."
            ),
            "evidence": "B — Smith et al. (2011); Owens et al. (2015); Pilz et al. (2019)",
        })

    if not supps:
        st.info("Mevcut verilere göre acil takviye ihtiyacı tespit edilmedi. Temel protein ve karbonhidrat gereksinimlerini karşılamak yeterlidir.")
        return []

    p1 = [s for s in supps if s["priority"] == 1]
    p2 = [s for s in supps if s["priority"] == 2]
    p3 = [s for s in supps if s["priority"] == 3]

    if p1:
        st.markdown("#### 🔴 Yüksek Öncelik")
        for s in p1:
            _supplement_card(s["name"], s["dose"], s["timing"], s["reason"], s["evidence"], s["color"])
    if p2:
        st.markdown("#### 🟡 Orta Öncelik")
        for s in p2:
            _supplement_card(s["name"], s["dose"], s["timing"], s["reason"], s["evidence"], s["color"])
    if p3:
        st.markdown("#### 🟢 Destekleyici")
        for s in p3:
            _supplement_card(s["name"], s["dose"], s["timing"], s["reason"], s["evidence"], s["color"])

    return supps


def _render_recovery_timeline(
    fi: float,
    post_freq_drop: float,
    post_smo2_drop: float,
    sensor_ok: bool,
    risk_score: int,
) -> None:
    st.markdown("### 9. Toparlanma Takvimi")
    st.caption(
        f"📌 Yorgunluk İndeksi {fi:.0f}/100 + Risk skoru {risk_score}/100 temel alınarak kişiselleştirilmiştir."
    )

    cold = "10 dk buz banyosu (12–15°C) veya kriyoterapi" if fi > 50 else "Soğuk duş 3–5 dk (15–18°C)"
    protein_note = "≥2.0 g/kg/gün" if fi > 50 else "1.6–1.8 g/kg/gün"
    sleep_h = "≥9 saat (toparlanma kritik)" if fi > 50 else "≥8 saat"
    next_hi = "72 saat" if fi > 66 else ("48 saat" if fi > 33 else "24 saat")
    sparring = "❌ Sparring önerilmez" if fi > 66 else ("⚠️ Kontrollü sparring" if fi > 33 else "✅ Normal sparring")

    cols = st.columns(4)
    frames = [
        ("0 – 2 Saat", "⚡", [
            "20–40 g whey protein + 40–60 g karbonhidrat (toparlanma penceresi)",
            cold,
            "Statik esneme: quadriceps, hamstring, kalça fleksörü (3×30 sn)",
            "500–750 ml su + elektrolit (sodyum + potasyum)",
            "Bacakları yukarı kaldır: 10–15 dk (venöz dönüşü kolaylaştırır)",
        ], "#ef4444"),
        ("2 – 24 Saat", "🌙", [
            f"Uyku: {sleep_h} — büyüme hormonu salınımının %70'i derin uykuda",
            f"Protein: {protein_note} (öğünlere yayılmış)",
            "Alkol ve kafein kaçın (uyku kalitesini bozar, kortizolü artırır)",
            "Soğuk-sıcak kontrast: 90 sn sıcak / 30 sn soğuk, 3 döngü" if fi > 33 else "Ilık duş yeterli",
            "Hidrasyon: idrar rengi açık sarı olana kadar su iç",
        ], "#f59e0b"),
        ("24 – 48 Saat", "🔄", [
            "Mobilite çalışması: eklem hareket açıklığı, yoga / pilates 30 dk",
            "Hafif teknik antrenman: shadow, yavaş tempo tekme tekniği",
            "Karbonhidrat yüklemesi: 5–7 g/kg/gün (glikojen depolarını tamamla)",
            "Masaj veya foam roller: vastus lateralis, hamstring, IT band (15 dk)",
            f"Bir sonraki yüksek yoğunluklu seans: {next_hi} sonra",
        ], "#3b82f6"),
        ("48 – 72 Saat", "💪", [
            f"{'Teknik odaklı düşük yoğunluklu seans' if fi > 50 else 'Normal yüke kademeli dönüş'}",
            sparring,
            "Güç antrenmanı: hafif direnç + yüksek tekrar (patlayıcı lunge, squat jump)",
            "Mental hazırlık: video analizi, taktik çalışma",
            "Performans testi: birkaç tekme ile hız/ROM kontrolü",
        ], "#22c55e"),
    ]

    for col, (tf, icon, acts, clr) in zip(cols, frames):
        items_html = "".join(
            f'<li style="margin:4px 0;color:#cbd5e1;font-size:12px;line-height:1.5">{a}</li>'
            for a in acts
        )
        col.markdown(
            f'<div style="border:1px solid {clr}44;border-radius:8px;padding:12px 14px;background:#0f172a">'
            f'<div style="color:{clr};font-weight:700;font-size:13px;margin-bottom:8px">{icon} {tf}</div>'
            f'<ul style="margin:0;padding-left:18px;line-height:1.6">{items_html}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _generate_weekly_plan(fi: float, has_neuro: bool, has_aerob: bool, has_rom: bool, has_asi: bool) -> list[dict]:
    inten = "Düşük" if fi > 66 else ("Orta" if fi > 33 else "Normal")
    return [
        {"Gün": "Pazartesi",
         "Odak": "Aerobik Kapasite" if has_aerob else "Teknik",
         "İçerik": "30–40 dk Zone 2 koşu + teknik drill (roundhouse serisi)" if has_aerob else "Shadow boksing + teknik analiz",
         "Yoğunluk": inten},
        {"Gün": "Salı",
         "Odak": "Nöromüsküler Güç" if has_neuro else "Kombine",
         "İçerik": "Patlayıcı squat jump 4×8 + elastik bant tekme 3×10" if has_neuro else "Kombine tekme-yumruk drill",
         "Yoğunluk": inten},
        {"Gün": "Çarşamba",
         "Odak": "Mobilite & Toparlanma",
         "İçerik": "Foam roller 15 dk + esneme + hafif shadow",
         "Yoğunluk": "Düşük"},
        {"Gün": "Perşembe",
         "Odak": "Bilateral Denge" if has_asi else "Güç",
         "İçerik": "Tek bacak squat 3×8 + single-leg RDL 3×10" if has_asi else "Plyometrik devre + hız tekmeleri",
         "Yoğunluk": inten},
        {"Gün": "Cuma",
         "Odak": "Teknik / Sparring",
         "İçerik": "Shadow + video analizi (yüksek FI)" if fi > 50 else "Kontrollü sparring + teknik analiz",
         "Yoğunluk": "Orta" if fi > 50 else inten},
        {"Gün": "Cumartesi",
         "Odak": "Aktif Toparlanma",
         "İçerik": "Yürüyüş 30 dk + statik esneme",
         "Yoğunluk": "Çok Düşük"},
        {"Gün": "Pazar",
         "Odak": "Dinlenme",
         "İçerik": "Tam dinlenme veya hafif yürüyüş",
         "Yoğunluk": "—"},
    ]


def _render_next_session(
    fi: float,
    vel_pct: float | None,
    rom_pct: float | None,
    post_ka_mean: float | None,
    post_freq_drop: float,
    post_smo2_drop: float,
    sensor_ok: bool,
    pre_events: list[dict],
    post_events: list[dict],
    risk_score: int,
) -> None:
    st.markdown("### 10. Sonraki Antrenman Haftası Önerisi")
    st.caption(
        f"📌 Yorgunluk İndeksi {fi:.0f}/100, Risk Skoru {risk_score}/100 ve "
        "tespit edilen zayıf noktalara göre kişiselleştirilmiştir."
    )

    if fi > 66:
        vol_adj, int_adj, vol_color = "Hacmi %30–40 azalt", "Düşük yoğunluk (MKH %60–70)", "#ef4444"
    elif fi > 33:
        vol_adj, int_adj, vol_color = "Hacmi koru / %10–15 azalt", "Orta yoğunluk (MKH %70–80)", "#f59e0b"
    else:
        vol_adj, int_adj, vol_color = "Hacmi %10–15 artırabilirsin", "Yüksek yoğunluk uygun (MKH %80–90)", "#22c55e"

    vc1, vc2, vc3 = st.columns(3)
    for col, lbl, val in [
        (vc1, "HAFTALık HACİM", vol_adj),
        (vc2, "YOğUNLUK", int_adj),
        (vc3, "SPARRING", "❌ Önerilmez (FI > 66)" if fi > 66 else ("⚠️ Kontrollü" if fi > 33 else "✅ Normal")),
    ]:
        col.markdown(
            f'<div style="border-left:4px solid {vol_color};padding:10px 16px;background:#1e293b;border-radius:4px">'
            f'<div style="color:#64748b;font-size:10px;font-weight:600;text-transform:uppercase">{lbl}</div>'
            f'<div style="color:{vol_color};font-size:14px;font-weight:700;margin-top:4px">{val}</div>'
            f'</div>', unsafe_allow_html=True
        )

    st.markdown("#### Odak Alanları")

    focus: list[dict] = []
    has_neuro = (sensor_ok and post_freq_drop > 8) or (not sensor_ok and vel_pct is not None and vel_pct < -10)
    has_aerob = (sensor_ok and post_smo2_drop > 5) or (not sensor_ok and fi > 50)
    has_rom   = rom_pct is not None and rom_pct < -8
    has_asi   = post_ka_mean is not None and abs(post_ka_mean) > 10

    if has_neuro:
        drop_str = f"{post_freq_drop:.0f} Hz EMG freq düşüşü" if sensor_ok else f"%{abs(vel_pct):.0f} hız kaybı"
        focus.append({
            "priority": 1, "color": "#ef4444",
            "title": "🔴 Nöromüsküler Patlayıcı Güç",
            "trigger": drop_str,
            "frequency": "Haftada 2 gün (Salı + Perşembe)",
            "drills": [
                "Patlayıcı squat jump: 4×8 set (max hız, 90 sn tam dinlenme)",
                "Elastik bant ile hızlı roundhouse: 3×10 her bacak",
                "Kum torbası kombinasyon: roundhouse + yan tekme, 5×(6 tekme), 45 sn ara",
                "CMJ (countermovement jump) tek bacak: 3×5 her bacak",
                "Sprint merdiven: 3×5 (patlayıcı ilk adım, taekwondo çıkış pozisyonu)",
            ],
        })

    if has_aerob:
        smo2_str = f"SmO2 %{post_smo2_drop:.0f} düşüşü" if sensor_ok else f"FI {fi:.0f}/100"
        focus.append({
            "priority": 1 if (sensor_ok and post_smo2_drop > 8) else 2, "color": "#3b82f6",
            "title": "🔵 Aerobik Kapasite (Zone 2)",
            "trigger": smo2_str,
            "frequency": "Haftada 2–3 gün (Pazartesi + Çarşamba + opsiyonel Cumartesi)",
            "drills": [
                "Sürekli koşu 30–40 dk, MKH %65–75 (konuşabilme eşiği)",
                "Merdiven/basamak protokolü: 20 dk kesintisiz tempo",
                "Shadow boksing: 5×3 dk (60 sn ara) teknik odaklı, düşük hız",
                "Bisiklet ergometre (düşük direnç): 30 dk Zone 2",
                "Uzun mesafe atlama ipi: 3×5 dk (70 rpm sabit tempo)",
            ],
        })

    if has_rom:
        focus.append({
            "priority": 2, "color": "#22c55e",
            "title": "🟢 Hareket Genişliği & Mobilite",
            "trigger": f"Diz ROM %{abs(rom_pct):.0f} azalması",
            "frequency": "Her antrenman öncesi 15 dk + haftada 1 gün sadece mobilite",
            "drills": [
                "Dinamik ısınma: leg swing (öne-arkaya-yana) 2×15 tekrar",
                "Hip flexor lunge stretch: 3×30 sn her taraf",
                "Hamstring PNF gerdirme: 6×(10 sn kasılma + 30 sn pasif germe)",
                "Foam roller: IT band, vastus lateralis, hamstring — 2×60 sn her bölge",
                "Yoga: pigeon pose + lizard pose, 2×45 sn her taraf",
            ],
        })

    if has_asi:
        side = "sol" if (post_ka_mean is not None and post_ka_mean > 0) else "sağ"
        focus.append({
            "priority": 2, "color": "#8b5cf6",
            "title": "🟣 Bilateral Denge & Simetri",
            "trigger": f"Diz ASI {post_ka_mean:+.0f}%",
            "frequency": "Haftada 2 gün (güç günleri başında, 20 dk)",
            "drills": [
                f"Tek bacak squat ({side} taraf önce): 3×8 yavaş tempo",
                "Single-leg Romanian deadlift: 3×10 her bacak ayrı",
                "Lateral band walk: 3×20 adım her yön (mini bant)",
                "Tek bacak denge: 3×30 sn gözler kapalı (her bacak)",
                f"Non-dominant ({side}) bacakla tekme tekniği: hacim +25%",
            ],
        })

    focus.append({
        "priority": 3, "color": "#f59e0b",
        "title": "🟡 Teknik Kalite & Taktik",
        "trigger": "Temel bileşen (her zaman)",
        "frequency": "Her seans sonunda 15–20 dk",
        "drills": [
            "Yavaş tempo tekme analizi: ayna / video kaydı, faz kontrolü",
            "Hedef kombinasyonları: jiryo + dolyo tekme serisi (düşük hız, yüksek teknik)",
            f"{'Shadow sparring (gerçek sparring önerilmez, FI yüksek)' if fi > 50 else 'Kontrollü sparring: teknik odaklı, kontrollü yoğunluk'}",
            "Video geri bildirim: bugünkü pre/post analiz karşılaştırması",
            "Reaksiyon drill: partner komutla tekme (kognitif yük + teknik)",
        ],
    })

    focus.sort(key=lambda x: x["priority"])

    for item in focus:
        drills_html = "".join(
            f'<li style="margin:3px 0;color:#cbd5e1;font-size:13px">{d}</li>'
            for d in item["drills"]
        )
        st.markdown(
            f'<div style="border:1px solid {item["color"]}44;border-radius:8px;padding:14px 18px;background:#0f172a;margin:8px 0">'
            f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px">'
            f'  <span style="font-weight:700;color:#f1f5f9;font-size:15px">{item["title"]}</span>'
            f'  <span style="background:{item["color"]}22;color:{item["color"]};padding:2px 8px;border-radius:3px;font-size:11px">'
            f'    Tetikleyen: {item["trigger"]}</span>'
            f'  <span style="color:#64748b;font-size:11px;margin-left:auto">{item["frequency"]}</span>'
            f'</div>'
            f'<ul style="margin:0;padding-left:20px;line-height:1.7">{drills_html}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### 📅 Örnek Haftalık Plan")
    weekly = _generate_weekly_plan(fi, has_neuro, has_aerob, has_rom, has_asi)
    st.dataframe(pd.DataFrame(weekly).set_index("Gün"), use_container_width=True)


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

    # ── 7. Yaralanma Risk ─────────────────────────────────────────────────────
    risk_score = _render_injury_risk(fi, vel_pct, rom_pct, post_ka_mean, sensor_ok, sensor_post if sensor_ok else {})

    # ── 8. Beslenme & Takviye ─────────────────────────────────────────────────
    _render_nutrition(fi, vel_pct, rom_pct, post_freq_drop, post_smo2_drop, sensor_ok, sensor_post if sensor_ok else {})

    # ── 9. Toparlanma Takvimi ─────────────────────────────────────────────────
    _render_recovery_timeline(fi, post_freq_drop, post_smo2_drop, sensor_ok, risk_score)

    # ── 10. Sonraki Antrenman ─────────────────────────────────────────────────
    _render_next_session(
        fi, vel_pct, rom_pct, post_ka_mean,
        post_freq_drop, post_smo2_drop,
        sensor_ok, pre_events, post_events, risk_score,
    )

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
