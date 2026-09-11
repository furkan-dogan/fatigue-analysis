"""Video Analizi sayfası — pre/post karşılaştırma."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.sports.taekwondo.fatigue import FATIGUE_METRICS, compute_fatigue
from src.core.numeric import events_mean
from ui.sports.taekwondo.presentation import (
    EMG_CH1_GROUP,
    EMG_CH1_MUSCLE,
    EMG_CH2_GROUP,
    EMG_CH2_MUSCLE,
    EMG_PLACEMENT,
    PRIMARY_METRICS,
    action_recommendations,
    comparison_rows,
    emg_device_rows,
    emg_summary_text,
    fmt_value,
    metric_info,
    movement_summary_rows,
    readable_kick_rows,
    readable_emg_frequency,
    readable_emg_rms,
    readable_findings,
    render_data_quality,
    render_metric_help,
    session_summary_rows,
    top_findings,
)
from ui.sports.taekwondo.charts import gauge, overlay_chart, per_kick_trend
from ui.components.charts import readable_bar_comparison
from ui.sports.taekwondo.components import kick_video_section, phase_bars
from src.sports.taekwondo.sensor_summary import sensor_stats
from ui.components.video_player import video_player
from ui.paths import SAMPLE_DATA_DIR
from ui.sports.taekwondo.report import render_athlete_report


def _read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _metric_cards(cards: list[tuple[str, str]]) -> None:
    cols = st.columns(len(cards))
    for col, (label, value) in zip(cols, cards):
        with col:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.markdown(f"### {value}")


def _fatigue_metric_rows(fatigue_data: dict) -> list[dict]:
    rows_ft = []
    for key, m in fatigue_data["metrics"].items():
        if m["pre"] is None or m["post"] is None or m["pct"] is None:
            continue
        info = metric_info(key)
        is_fatigue = (m["direction"] * m["pct"]) > 0
        fatigue_text = "🔴 Evet" if is_fatigue else "🟢 Hayır"
        contribution_text = f"{'🔴' if m['fatigue_contribution'] > 0 else '🟢'} {m['fatigue_contribution']:+.1f}"
        rows_ft.append({
            "Metrik":      info.label,
            "Pre":         fmt_value(m["pre"], key),
            "Post":        fmt_value(m["post"], key),
            "Değişim":     f"{m['pct']:+.1f}%",
            "Yorgunluk Bulgusu": fatigue_text,
            "Katkı Puanı": contribution_text,
            "Yorum":       "Bu değişim yorgunluk yönünde." if is_fatigue else "Bu değişim yorgunluk yönünde değil veya iyileşme gösteriyor.",
        })
    return rows_ft


def _render_fatigue_metric_table(fatigue_data: dict, title: str = "Metrik Bazlı Yorgunluk Bulguları") -> None:
    st.markdown(f"#### {title}")
    st.info(
        "Bu tablo yorgunluk sonucunu hangi ölçümlerin etkilediğini gösterir. "
        "'Yorgunluk Bulgusu: Evet' yazıyorsa post-antrenmanda ilgili metrik yorgunluk yönünde değişmiştir. "
        "Katkı puanı pozitifse yorgunluğu artırır; negatifse performans korunmuş veya iyileşmiş olabilir."
    )
    rows_ft = _fatigue_metric_rows(fatigue_data)
    st.dataframe(pd.DataFrame(rows_ft), use_container_width=True, hide_index=True)
    with st.expander("Bu tablo nasıl hesaplandı ve nasıl okunur?", expanded=False):
        st.markdown("**Pre / Post:** Antrenman öncesi ve sonrası videolardan hesaplanan ortalama değerlerdir.")
        st.markdown("**Değişim:** Post değerin pre değere göre yüzde olarak ne kadar değiştiğini gösterir.")
        st.markdown("**Yorgunluk Bulgusu:** Değişimin yorgunluk yönünde olup olmadığını söyler. Örneğin hız düşerse yorgunluk bulgusudur; peak hıza ulaşma süresi artarsa yine yorgunluk bulgusudur.")
        st.markdown("**Katkı Puanı:** O metriğin genel yorgunluk skorunu ne kadar artırdığını veya azalttığını gösterir.")
        st.markdown("Tek bir satır tek başına kesin karar verdirmez; video, EMG, takip güveni ve diğer hareket ölçümleriyle birlikte okunmalıdır.")
    render_metric_help([k for k in fatigue_data["metrics"] if fatigue_data["metrics"][k].get("pre") is not None], "Metrikler tek tek nasıl hesaplandı?")


def _render_general_summary(
    pre_events: list[dict],
    post_events: list[dict],
    pre_res,
    post_res,
    fatigue_data: dict,
    fi: float,
    fi_label: str,
    pre_emg_rows: list[dict],
    post_emg_rows: list[dict],
) -> None:
    st.subheader("Genel Sonuç")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Genel Yorgunluk", fi_label, f"{fi:.1f}/100", delta_color="off")
    c2.metric("Pre Tekme", len(pre_events))
    c3.metric("Post Tekme", len(post_events))
    conf_vals = [
        float(e["pose_confidence"])
        for e in pre_events + post_events
        if e.get("pose_confidence") is not None
    ]
    avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else None
    c4.metric("Takip Güveni", fmt_value(avg_conf, "pose_confidence"))

    emg_rows = (
        emg_device_rows(pre_events, pre_emg_rows, "Pre")
        + emg_device_rows(post_events, post_emg_rows, "Post")
    )
    fatigue_count = sum(1 for r in emg_rows if r["fatigue_flag"] != "Yorgunluk sınırlı")
    rectus_vals = [float(r["rectus_femoris_rms_mv"]) for r in emg_rows]
    biceps_vals = [float(r["biceps_femoris_rms_mv"]) for r in emg_rows]

    st.markdown("#### Hızlı Karar")
    d1, d2, d3 = st.columns(3)
    if fi >= 66:
        fatigue_status = "Yüksek dikkat"
        fatigue_text = "Genel yorgunluk belirgin; teknik kalite ve toparlanma birlikte izlenmeli."
    elif fi >= 33:
        fatigue_status = "Orta dikkat"
        fatigue_text = "Yorgunluk orta düzeyde; yüksek şiddet yerine kontrollü teknik tekrar daha uygun."
    else:
        fatigue_status = "Kontrollü"
        fatigue_text = "Genel yorgunluk düşük; hız ve teknik kalite korunarak yüklenme sürdürülebilir."
    d1.metric("Teknik Performans", fatigue_status, f"{fi:.1f}/100", delta_color="off")
    d1.caption(fatigue_text)

    if rectus_vals and biceps_vals:
        rectus_avg = sum(rectus_vals) / len(rectus_vals)
        biceps_avg = sum(biceps_vals) / len(biceps_vals)
        ratio = rectus_avg / max(biceps_avg, 1e-6)
        if ratio >= 1.35:
            muscle_status = "Rectus baskın"
            muscle_text = "Vuruşta bacağı kaldırma ve diz açma fazı daha çok yük alıyor."
        elif ratio <= 0.85:
            muscle_status = "Biceps baskın"
            muscle_text = "Geri çekme ve frenleme fazı daha çok yük alıyor."
        else:
            muscle_status = "Dengeli"
            muscle_text = "Ön ve arka uyluk aktivasyonu birbirine yakın görünüyor."
        d2.metric("Kas Aktivasyonu", muscle_status, f"CH1/CH2 {ratio:.2f}", delta_color="off")
        d2.caption(muscle_text)
    else:
        d2.metric("Kas Aktivasyonu", "Veri yok")
        d2.caption("EMG satırı bulunamadığı için kas aktivasyonu okunamadı.")

    if emg_rows:
        emg_status = "Belirti var" if fatigue_count else "Belirti sınırlı"
        d3.metric("EMG Yorgunluğu", emg_status, f"{fatigue_count}/{len(emg_rows)} tekme", delta_color="off")
        d3.caption("Median frekans düşen tekmeler kas yorgunluğu açısından işaretlenir.")
    else:
        d3.metric("EMG Yorgunluğu", "Veri yok")
        d3.caption("EMG çıktısı üretilemedi.")

    render_data_quality(pre_events, post_events)
    with st.expander("Takip Güveni ne demek?", expanded=False):
        st.markdown(
            "Takip güveni, sistemin videoda vücut noktalarını ne kadar net yakalayabildiğini gösterir. "
            "Diz, kalça, ayak bileği gibi noktalar net takip edilirse açı, hız ve tekme yüksekliği hesapları daha güvenilir olur."
        )
        st.markdown("**1.00'a yakın değer:** vücut noktaları net takip edilmiş, ölçümler daha güvenilir.")
        st.markdown("**0.60 altı değer:** kamera açısı, örtüşme, hızlı hareket veya bulanıklık nedeniyle ölçümler daha dikkatli yorumlanmalı.")
        st.markdown("Bu değer performans skoru değildir; ölçümün ne kadar güvenilir olduğunu anlatır.")

    st.markdown("#### Analiz Videoları")
    vc1, vc2 = st.columns(2)
    with vc1:
        st.markdown("**Pre-antrenman video çıktısı**")
        pre_vid = Path(pre_res.output_video_path)
        if pre_vid.exists():
            video_player(pre_vid)
        else:
            st.warning("Pre video çıktısı bulunamadı.")
    with vc2:
        st.markdown("**Post-antrenman video çıktısı**")
        post_vid = Path(post_res.output_video_path)
        if post_vid.exists():
            video_player(post_vid)
        else:
            st.warning("Post video çıktısı bulunamadı.")
    st.caption("Videolardaki iskelet çizimi ve açı paneli, sistemin hangi vücut noktalarını takip ettiğini görsel olarak kontrol etmek içindir.")

    with st.expander("Kamera açısı: tekmeler kameraya doğru atıldığında ne olur?", expanded=False):
        st.markdown(
            "Tekme kameraya doğru atıldığında video iki boyutlu olduğu için ileri-geri derinliği santimetre olarak net vermez. "
            "Bu yüzden sistem gerçek mesafe yerine gövdeye göre oran, zaman, açı, hız ve faz değişimlerini kullanır."
        )
        st.markdown(
            "Bu açı tekme sayımı, tekme zamanı, diz fleksiyonu/ekstansiyonu, ROM, faz süresi ve EMG senkronu için kullanılabilir. "
            "En dikkatli okunması gereken alan tekme yüksekliği ve ayak hızının görüntü düzlemine bağlı normalize yorumudur."
        )
        st.markdown(
            "Ek algoritma şu aşamada şart değil. Daha ileri doğruluk istenirse en temiz çözüm yan kamera, çift kamera veya kamera kalibrasyonudur; "
            "mevcut demo için normalize metriklerle okumak daha stabil ve anlaşılırdır."
        )

    st.markdown("#### Özet Nasıl Okunur?")
    st.markdown("- **Video analizi** tekme yüksekliği, diz bükülmesi, diz açılması, hareket açıklığı, hız ve süreyi ölçer.")
    st.markdown("- **EMG RMS (mV)** kas aktivasyonunun büyüklüğüdür; değer arttıkça ilgili kas daha fazla devreye girer.")
    st.markdown("- **EMG median frekans (Hz)** kas yorgunluğunu okumaya yardım eder; tekrarlar ilerledikçe düşmesi yorgunlukla uyumludur.")
    st.markdown("- **Rectus femoris** bacağı kaldırma ve diz açma fazını, **biceps femoris** diz bükme, geri çekme ve frenleme fazını gösterir.")
    st.markdown("- **Takip güveni** performans skoru değildir; videodaki vücut noktalarının ne kadar net yakalandığını anlatır.")

    st.markdown("#### En Önemli Bulgular")
    st.dataframe(pd.DataFrame(readable_findings(pre_events, post_events, fatigue_data, fi)), use_container_width=True, hide_index=True)
    with st.expander("Gövde oranı ve diğer ifadeler nasıl okunur?", expanded=False):
        st.markdown(
            "**Gövde oranı**, ayağın yükselmesini sporcunun gövde uzunluğuna göre anlatır. "
            "Kamera kalibrasyonu olmadan gerçek cm vermek güvenilir olmadığı için oran kullanılır."
        )
        st.markdown("**Düşük/orta/yüksek seviye**, bu oranın pratik yorumudur; sayı parantez içinde teknik referans olarak kalır.")
        st.markdown("**Derece (°)** dizin bükülme veya açılma açısıdır. Örneğin 180°'ye yakın değer dizin daha çok açıldığını gösterir.")
        st.markdown("**°/sn** hareket hızıdır. Değer yükseldikçe diz veya ayak daha hızlı hareket etmiş kabul edilir.")

    st.markdown("#### Ne Yapılmalı?")
    for rec in action_recommendations(pre_events, post_events, fatigue_data, fi):
        st.markdown(f"- {rec}")

    _render_fatigue_metric_table(fatigue_data)

    st.markdown("#### EMG Cihaz Yerleşimi")
    placement_rows = [
        {"Kanal": "CH1", "Takılan Kas": EMG_CH1_MUSCLE, "Kas Grubu": EMG_CH1_GROUP, "Ne İşe Yarar?": "Dizi açma, bacağı kaldırma ve vuruş fazındaki ön uyluk aktivasyonunu gösterir."},
        {"Kanal": "CH2", "Takılan Kas": EMG_CH2_MUSCLE, "Kas Grubu": EMG_CH2_GROUP, "Ne İşe Yarar?": "Dizi bükme, bacağı geri çekme ve hareketi frenleme fazındaki arka uyluk aktivasyonunu gösterir."},
    ]
    st.dataframe(pd.DataFrame(placement_rows), use_container_width=True, hide_index=True)
    st.info(EMG_PLACEMENT)
    with st.expander("Bu kaslar neden önemli?", expanded=False):
        st.markdown("**Rectus femoris**, quadriceps grubunun parçasıdır; kalçayı bükmeye ve dizi açmaya yardım eder. Bandal chagi sırasında bacağın hedefe doğru hızlanması ve dizin açılmasıyla ilişkilidir.")
        st.markdown("**Biceps femoris**, hamstring grubunun parçasıdır; dizin bükülmesi, bacağın geri çekilmesi ve vuruş sonrası frenleme kontrolünde önemlidir.")
        st.markdown("Bu iki kas birlikte okununca sadece 'tekme güçlü mü?' sorusu değil, vuruş ve geri çekiş fazlarında ön-arka uyluk yük dağılımı da anlaşılır.")

    st.markdown("#### Video Analizinden Toplanan Hareket Verileri")
    st.dataframe(pd.DataFrame(movement_summary_rows(pre_events, post_events)), use_container_width=True, hide_index=True)
    with st.expander("Fleksiyon, ekstansiyon ve ROM ne demek?", expanded=False):
        st.markdown("**Diz fleksiyonu:** Tekme öncesinde dizin bükülmesidir. Açı küçüldükçe diz daha fazla bükülmüş kabul edilir.")
        st.markdown("**Diz ekstansiyonu:** Tekme sırasında dizin açılmasıdır. Açı 180 dereceye yaklaştıkça diz daha fazla açılmış kabul edilir.")
        st.markdown("**Diz ROM:** Fleksiyon ve ekstansiyon arasındaki toplam hareket farkıdır. Tekme hareketinin ne kadar geniş yapıldığını gösterir.")

    st.markdown("#### Tekme Fazları ve Sağ-Sol Denge")
    phase_rows = comparison_rows(
        pre_events,
        post_events,
        ["extension_dur_sec", "retraction_dur_sec", "extension_peak_vel_deg_s", "retraction_peak_vel_deg_s", "knee_asi", "hip_asi"],
    )
    st.dataframe(pd.DataFrame(phase_rows), use_container_width=True, hide_index=True)
    with st.expander("Faz ve asimetri nasıl okunur?", expanded=False):
        st.markdown("**Uzatma fazı:** Bacağın hedefe doğru açıldığı bölümdür. Süre uzarsa veya hız düşerse tekme daha yavaş çıkıyor olabilir.")
        st.markdown("**Geri çekim fazı:** Vuruştan sonra bacağın geri alınmasıdır. Bu faz uzarsa savunmaya dönüş gecikebilir.")
        st.markdown("**ASI:** Sağ-sol taraf farkını yüzde olarak gösterir. 10% üzeri farklar yük dağılımı açısından dikkat gerektirir.")

    st.markdown("#### EMG Cihaz Çıktısı")
    if emg_rows:
        emg_df = pd.DataFrame(emg_rows)
        freq_vals = [float(r["ortalama_emg_median_freq_hz"]) for r in emg_rows]
        _metric_cards(
            [
                ("Rectus Femoris", readable_emg_rms(sum(rectus_vals) / len(rectus_vals))),
                ("Biceps Femoris", readable_emg_rms(sum(biceps_vals) / len(biceps_vals))),
                ("Kas Yorgunluğu", readable_emg_frequency(sum(freq_vals) / len(freq_vals))),
                ("Yorgunluk İşaretli Tekme", f"{fatigue_count}/{len(emg_rows)}"),
            ]
        )
        st.info(emg_summary_text(emg_rows))
        st.markdown("##### EMG - Video Senkron Kontrolü")
        sync_cols = [
            "Oturum",
            "kick_id",
            "peak_time_sec",
            "CH1 Kas",
            "rectus_femoris_rms_mv",
            "rectus_femoris_median_freq_hz",
            "CH2 Kas",
            "biceps_femoris_rms_mv",
            "biceps_femoris_median_freq_hz",
            "Kas Dengesi",
            "fatigue_flag",
        ]
        st.dataframe(emg_df[sync_cols], use_container_width=True, hide_index=True)
        st.caption("Her satır videodaki bir tekmenin peak zamanına bağlanır; bu yüzden EMG değeri video çıktısıyla aynı tekme penceresinde okunur.")
        with st.expander("EMG cihaz çıktısını göster", expanded=False):
            st.dataframe(emg_df, use_container_width=True, hide_index=True)
        with st.expander("EMG sütunları ne anlama geliyor?", expanded=False):
            st.markdown("**rectus_femoris_rms_mv:** Rectus femoris kasının ortalama aktivasyon büyüklüğüdür. Diz açma ve bacağı kaldırma fazını okumaya yardım eder.")
            st.markdown("**rectus_femoris_peak_rms_mv:** Tekme penceresinde rectus femoris için görülen en yüksek aktivasyondur.")
            st.markdown("**biceps_femoris_rms_mv:** Biceps femoris kasının ortalama aktivasyon büyüklüğüdür. Diz bükme, geri çekme ve frenleme fazını okumaya yardım eder.")
            st.markdown("**biceps_femoris_peak_rms_mv:** Tekme penceresinde biceps femoris için görülen en yüksek aktivasyondur.")
            st.markdown("**rectus_femoris_median_freq_hz:** Rectus femoris kanalının median frekansıdır. Tekmeler ilerledikçe düşmesi ön uyluk yorgunluğuyla uyumludur.")
            st.markdown("**biceps_femoris_median_freq_hz:** Biceps femoris kanalının median frekansıdır. Tekmeler ilerledikçe düşmesi arka uyluk yorgunluğuyla uyumludur.")
            st.markdown("**ortalama_emg_median_freq_hz:** İki kanalın ortalama median frekansıdır; genel kas yorgunluğu kartında kullanılır.")
            st.markdown("**Kas Dengesi:** Rectus femoris ve biceps femoris RMS ortalamalarının birbirine göre baskınlığını gösterir.")
            st.markdown("**Aktivasyon Yorumu:** RMS değerine göre düşük, orta veya yüksek aktivasyon sınıflamasıdır.")
            st.markdown("**fatigue_flag:** Median frekans düşüşüne göre yorgunluk yorumudur.")
        with st.expander("CSV çıktıları ekranda göster", expanded=False):
            pre_df = _read_csv(pre_res.frame_csv_path)
            post_df = _read_csv(post_res.frame_csv_path)
            pre_emg_frame_df = pd.DataFrame(_emg_frame_export_rows(pre_df, pre_emg_rows, pre_events, "Pre"))
            post_emg_frame_df = pd.DataFrame(_emg_frame_export_rows(post_df, post_emg_rows, post_events, "Post"))
            pre_emg_kick_df = pd.DataFrame(emg_device_rows(pre_events, pre_emg_rows, "Pre"))
            post_emg_kick_df = pd.DataFrame(emg_device_rows(post_events, post_emg_rows, "Post"))
            with st.expander("pre_emg_frame_sync.csv", expanded=False):
                st.dataframe(pre_emg_frame_df, use_container_width=True, hide_index=True)
            with st.expander("post_emg_frame_sync.csv", expanded=False):
                st.dataframe(post_emg_frame_df, use_container_width=True, hide_index=True)
            with st.expander("pre_emg_kick_summary.csv", expanded=False):
                st.dataframe(pre_emg_kick_df, use_container_width=True, hide_index=True)
            with st.expander("post_emg_kick_summary.csv", expanded=False):
                st.dataframe(post_emg_kick_df, use_container_width=True, hide_index=True)
    else:
        st.warning("EMG çıktısı üretilemedi. Analiz sonucunda tekme veya EMG satırı bulunamadı.")

    st.markdown("#### Oturum Özeti")
    rows = (
        session_summary_rows(pre_events, "Pre-antrenman", pre_res.total_frames, pre_res.fps)
        + session_summary_rows(post_events, "Post-antrenman", post_res.total_frames, post_res.fps)
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Pre/Post Temel Performans Karşılaştırması")
    st.info(
        "Bu bölüm pre-antrenman ve post-antrenman ortalamalarını yan yana gösterir. "
        "Mavi değer pre, kırmızı değer post sonucudur. Post değeri düşüyorsa bu genelde yorgunluk veya teknik kalite kaybı anlamına gelebilir; "
        "tekme süresi gibi metrikler ise hız ve hareket genişliğiyle birlikte yorumlanmalıdır."
    )
    cmp_rows = comparison_rows(pre_events, post_events, PRIMARY_METRICS)
    st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
    with st.expander("Bu karşılaştırmayı nasıl okumalıyım?", expanded=False):
        st.markdown("- **Tekme yüksekliği:** Post düşükse bacak antrenman sonrası daha az yükselmiş olabilir.")
        st.markdown("- **Diz hareket açıklığı:** Post düşükse tekme daha dar yapılmış, diz bükülme-açılma aralığı azalmış olabilir.")
        st.markdown("- **Maksimum diz hızı:** Post düşükse patlayıcı hız üretimi azalmış olabilir.")
        st.markdown("- **Ayak hızı:** Post düşükse tekmenin uç hızı azalmış olabilir.")
        st.markdown("- **Peak hıza ulaşma süresi:** Post artarsa sporcu maksimum hıza daha geç ulaşıyor olabilir.")
        st.markdown("- **Tekme süresi:** Tek başına iyi/kötü değildir; hız ve hareket açıklığıyla birlikte okunmalıdır.")
    render_metric_help(PRIMARY_METRICS, "Metrikler nasıl hesaplandı?")

    labels, pre_vals, post_vals = [], [], []
    for key in ["active_peak_knee_vel_deg_s", "active_knee_rom_deg", "duration_sec"]:
        pre = events_mean(pre_events, key)
        post = events_mean(post_events, key)
        if pre is not None and post is not None:
            labels.append(metric_info(key).label)
            pre_vals.append(float(pre))
            post_vals.append(float(post))
    if labels:
        st.plotly_chart(
            readable_bar_comparison(
                labels,
                pre_vals,
                post_vals,
                "Pre/Post Temel Performans Karşılaştırması",
                "Ortalama değer",
                "Grafikte mavi pre-antrenman, kırmızı post-antrenman ortalamasıdır. Kırmızı barın düşmesi performans kaybı anlamına gelebilir.",
            ),
            use_container_width=True,
        )


def _render_fatigue_tab(fatigue_data: dict, fi: float, fi_label: str) -> None:
    g1, g2 = st.columns([1, 2])
    with g1:
        st.plotly_chart(gauge(fi, "Yorgunluk İndeksi"), use_container_width=True)
        fi_color = "🟢" if fi < 33 else ("🟡" if fi < 66 else "🔴")
        st.markdown(f"**{fi_color} {fi_label} yorgunluk** — {fi:.1f}/100")
        st.info(
            "Yorgunluk indeksi 0-100 arası bileşik skordur. 0-33 düşük, 33-66 orta, "
            "66-100 yüksek yorgunluk olarak yorumlanır. Skor, pre ve post videolardaki "
            "hız, diz ROM, tekme yüksekliği, ayak hızı, tekme süresi ve peak hıza ulaşma süresi değişimlerinden hesaplanır."
        )
    with g2:
        _render_fatigue_metric_table(fatigue_data)

    st.subheader("Yorgunluk Katkı Grafiği")
    st.caption("Sağa giden kırmızı barlar yorgunluğu artıran değişimi, sola giden yeşil barlar korunmuş/iyileşmiş performansı gösterir.")
    labels, vals, colors = [], [], []
    for m in fatigue_data["metrics"].values():
        if m.get("fatigue_contribution") is None:
            continue
        labels.append(m["label"])
        vals.append(round(m["fatigue_contribution"], 1))
        colors.append("#ef4444" if m["fatigue_contribution"] > 0 else "#22c55e")

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}" for v in vals], textposition="outside",
    ))
    fig.add_vline(x=0, line_color="#888", line_width=1)
    fig.update_layout(
        height=420, xaxis_title="Yorgunluk katkı puanı (+ yorgunluk, - iyileşme)",
        margin=dict(l=220, r=80, t=35, b=45),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa", size=13),
        xaxis=dict(gridcolor="#333", range=[-110, 110]),
        yaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_per_kick_tab(
    pre_events: list[dict],
    post_events: list[dict],
    tmp: Path | None,
) -> None:
    st.caption("Her tekme ayrı satırda gösterilir. Bu bölüm hangi tekmenin analizi güçlendirdiğini veya zayıflattığını görmek içindir.")
    pk_col = st.selectbox(
        "Grafikte gösterilecek metrik",
        [k for k in FATIGUE_METRICS if any(e.get(k) is not None for e in pre_events + post_events)],
        format_func=lambda k: FATIGUE_METRICS[k][0],
        key="dv_pk_col",
        help="Seçilen metrik her tekme için pre ve post barlarıyla çizilir. Tekme numarası sıra numarasıdır; aynı numara bire bir aynı teknik tekrarını garanti etmez.",
    )
    if pk_col:
        st.plotly_chart(
            per_kick_trend(pre_events, post_events, pk_col, FATIGUE_METRICS[pk_col][0]),
            use_container_width=True,
        )
        render_metric_help([pk_col], "Seçilen metrik ne anlama geliyor?")

    pre_tbl = pd.DataFrame(readable_kick_rows(pre_events))
    post_tbl = pd.DataFrame(readable_kick_rows(post_events))
    st.markdown("#### Pre/Post Tekme Tablosu")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("**Pre-antrenman tekmeleri**")
        if not pre_tbl.empty:
            st.dataframe(pre_tbl, use_container_width=True, hide_index=True)
    with tc2:
        st.markdown("**Post-antrenman tekmeleri**")
        if not post_tbl.empty:
            st.dataframe(post_tbl, use_container_width=True, hide_index=True)

    st.divider()
    kc1, kc2 = st.columns(2)
    with kc1:
        kick_video_section(
            pre_events,
            str(tmp / "pre_annotated.mp4") if tmp else "",
            "Pre Tekmeleri", tmp,
        )
    with kc2:
        kick_video_section(
            post_events,
            str(tmp / "post_annotated.mp4") if tmp else "",
            "Post Tekmeleri", tmp,
        )


def _render_phase_tab(pre_events: list[dict], post_events: list[dict]) -> None:
    st.subheader("Tekme Faz Analizi")
    st.info(
        "Her tekme üç faza ayrılır: yüklenme, uzatma ve geri çekim. "
        "Yorgunlukta özellikle uzatma hızı düşebilir ve geri çekim süresi uzayabilir; bu durum sporcunun savunmaya dönüşünü geciktirir."
    )
    render_metric_help(["extension_dur_sec", "retraction_dur_sec", "extension_peak_vel_deg_s", "retraction_peak_vel_deg_s"])

    phase_rows = comparison_rows(
        pre_events,
        post_events,
        ["extension_dur_sec", "retraction_dur_sec", "extension_peak_vel_deg_s", "retraction_peak_vel_deg_s"],
    )
    st.dataframe(pd.DataFrame(phase_rows), use_container_width=True, hide_index=True)

    fc1, fc2 = st.columns(2)
    with fc1:
        phase_bars(pre_events,  "Pre — Faz Süreleri", "#3b82f6")
    with fc2:
        phase_bars(post_events, "Post — Faz Süreleri", "#ef4444")

    st.subheader("Faz Hızları — Pre vs Post")
    for pk, plbl in [
        ("extension_peak_vel_deg_s",  "Uzatma Peak Hızı (°/s)"),
        ("retraction_peak_vel_deg_s", "Geri Çekim Peak Hızı (°/s)"),
        ("loading_peak_vel_deg_s",    "Yüklenme Peak Hızı (°/s)"),
    ]:
        if any(e.get(pk) is not None for e in pre_events + post_events):
            st.plotly_chart(per_kick_trend(pre_events, post_events, pk, plbl), use_container_width=True)

    st.subheader("Geri Çekim / Uzatma Oranı")
    st.caption("Yorgunlukla geri çekim yavaşlar → oran artar. >1.5 = belirgin yavaşlama.")
    for label, evs in [("Pre", pre_events), ("Post", post_events)]:
        ratios = []
        for ev in evs:
            ext = ev.get("extension_dur_sec")
            ret = ev.get("retraction_dur_sec")
            if ext and ret and float(ext) > 0.001:
                ratios.append(round(float(ret) / float(ext), 3))
        if ratios:
            mean_ratio = sum(ratios) / len(ratios)
            st.metric(
                f"{label} — ort. geri çekim/uzatma oranı", f"{mean_ratio:.2f}",
                delta="⚠️ yavaş geri çekim" if mean_ratio > 1.5 else "✅ normal",
                delta_color="off",
            )


def _render_asymmetry_tab(pre_events: list[dict], post_events: list[dict]) -> None:
    st.subheader("Bilateral Asimetri İndeksi (ASI)")
    st.info(
        "ASI sağ ve sol taraf arasındaki yüzde farkı gösterir. Pozitif değer sağ tarafın, negatif değer sol tarafın daha baskın olduğunu belirtir. "
        "Mutlak değerin 10% üzerine çıkması klinik ve sportif performans açısından dikkat gerektirir."
    )
    render_metric_help(["knee_asi", "hip_asi"], "ASI nasıl hesaplandı ve nasıl yorumlanır?")

    for asi_key, asi_lbl in [("knee_asi", "Diz ASI (%)"), ("hip_asi", "Kalça ASI (%)")]:
        if not any(e.get(asi_key) is not None for e in pre_events + post_events):
            continue
        fig = go.Figure()
        for evs, clr, lbl in [(pre_events, "#3b82f6", "Pre"), (post_events, "#ef4444", "Post")]:
            vals  = [float(e[asi_key]) if e.get(asi_key) is not None else None for e in evs]
            x_lbl = [f"T{int(e['kick_id'])}" for e in evs]
            fig.add_trace(go.Bar(name=lbl, x=x_lbl, y=vals, marker_color=clr))
        fig.add_hline(y=10,  line_dash="dash", line_color="orange", annotation_text="+10% eşik")
        fig.add_hline(y=-10, line_dash="dash", line_color="orange", annotation_text="-10% eşik")
        fig.add_hline(y=0,   line_color="#555", line_width=1)
        fig.update_layout(
            barmode="group", height=280,
            title=dict(text=asi_lbl, font=dict(color="#fafafa", size=13)),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#fafafa"),
            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333", title="%"),
            margin=dict(l=40, r=10, t=35, b=30),
            legend=dict(orientation="h", y=-0.35),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Oturum Ortalama ASI")
    ac1, ac2, ac3, ac4 = st.columns(4)
    for col_w, label, evs in [
        (ac1, "Pre Diz",    pre_events), (ac2, "Post Diz",    post_events),
        (ac3, "Pre Kalça",  pre_events), (ac4, "Post Kalça",  post_events),
    ]:
        k    = "knee_asi" if "Diz" in label else "hip_asi"
        vals = [float(e[k]) for e in evs if e.get(k) is not None]
        if vals:
            mean_asi = sum(vals) / len(vals)
            col_w.metric(label, f"{mean_asi:+.1f}%",
                         delta="⚠️" if abs(mean_asi) > 10 else "✅", delta_color="off")


def _render_stats_tab(pre_events: list[dict], post_events: list[dict]) -> None:
    st.subheader("İstatistiksel Karşılaştırma")
    st.info(
        "Bu bölüm ortalama farkın yanında etki büyüklüğünü gösterir. Cohen's d mutlak değeri 0.2 küçük, 0.5 orta, 0.8 ve üstü büyük etki olarak yorumlanır. "
        "Tekme sayısı düşükse güven aralıkları genişler; bu durumda sonuç ön bulgu olarak raporlanmalıdır."
    )

    from src.core.statistics import compare_metric

    STAT_METRICS = {
        "active_knee_rom_deg":         "Aktif Diz ROM (°)",
        "active_peak_knee_vel_deg_s":  "Peak Diz Hızı (°/s)",
        "active_mean_knee_vel_deg_s":  "Ort. Diz Hızı (°/s)",
        "time_to_peak_knee_vel_sec":   "Peak Hıza Süre (sn)",
        "peak_kick_height_norm":       "Tekme Yüksekliği",
        "extension_peak_vel_deg_s":    "Uzatma Hızı (°/s)",
        "retraction_peak_vel_deg_s":   "Geri Çekim Hızı (°/s)",
        "retraction_dur_sec":          "Geri Çekim Süresi (sn)",
        "knee_asi":                    "Diz ASI (%)",
        "duration_sec":                "Tekme Süresi (sn)",
    }

    stat_rows = []
    for col_k, col_lbl in STAT_METRICS.items():
        pre_v  = [float(e[col_k]) for e in pre_events  if e.get(col_k) is not None]
        post_v = [float(e[col_k]) for e in post_events if e.get(col_k) is not None]
        if not pre_v or not post_v:
            continue
        res    = compare_metric(pre_v, post_v)
        ci_pre = res["pre_ci"]
        ci_post= res["post_ci"]
        d      = res["cohens_d"]
        stat_rows.append({
            "Metrik":          col_lbl,
            "Pre ort. ± std":  f"{res['pre_mean']:.2f} ± {res['pre_std']:.2f}" if res["pre_std"] else f"{res['pre_mean']:.2f}",
            "Post ort. ± std": f"{res['post_mean']:.2f} ± {res['post_std']:.2f}" if res["post_std"] else f"{res['post_mean']:.2f}",
            "%95 CI (Pre)":    f"[{ci_pre[0]:.2f}, {ci_pre[1]:.2f}]"  if ci_pre  else "—",
            "%95 CI (Post)":   f"[{ci_post[0]:.2f}, {ci_post[1]:.2f}]" if ci_post else "—",
            "Δ":               f"{res['delta']:+.2f}" if res["delta"] is not None else "—",
            "% Değişim":       f"{res['pct_change']:+.1f}%" if res["pct_change"] is not None else "—",
            "Cohen's d":       f"{d:.2f}" if d is not None else "—",
            "Etki Büyüklüğü":  res["effect_label"],
            "n (pre/post)":    f"{res['n_pre']} / {res['n_post']}",
            "Sonuç Yorumu":    "Güçlü değişim" if d is not None and abs(d) >= 0.8 and res["n_pre"] >= 5 and res["n_post"] >= 5 else ("Ön bulgu; tekme sayısı artırılmalı" if res["n_pre"] < 5 or res["n_post"] < 5 else "Destekleyici bulgu"),
        })

    if stat_rows:
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

        d_labels = [r["Metrik"]     for r in stat_rows if r["Cohen's d"] != "—"]
        d_vals   = [float(r["Cohen's d"]) for r in stat_rows if r["Cohen's d"] != "—"]
        d_colors = ["#ef4444" if v < 0 else "#22c55e" for v in d_vals]
        if d_vals:
            fig = go.Figure(go.Bar(
                x=d_vals, y=d_labels, orientation="h",
                marker_color=d_colors,
                text=[f"{v:+.2f}" for v in d_vals], textposition="outside",
            ))
            fig.add_vline(x=0,    line_color="#888",   line_width=1)
            fig.add_vline(x=0.8,  line_dash="dot", line_color="#f59e0b", annotation_text="büyük")
            fig.add_vline(x=-0.8, line_dash="dot", line_color="#f59e0b")
            fig.add_vline(x=0.5,  line_dash="dot", line_color="#64748b", annotation_text="orta")
            fig.add_vline(x=-0.5, line_dash="dot", line_color="#64748b")
            fig.update_layout(
                height=380, xaxis_title="Cohen's d  (negatif = post < pre)",
                margin=dict(l=200, r=80, t=20, b=30),
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#fafafa"),
                xaxis=dict(gridcolor="#333", range=[-3, 3]),
                yaxis=dict(gridcolor="#333"),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Güven Uyarıları")
    low_pre  = [e for e in pre_events  if e.get("confidence_flag") == "low"]
    low_post = [e for e in post_events if e.get("confidence_flag") == "low"]

    if not low_pre and not low_post:
        st.success("Tüm tekmelerde pose güveni yeterli (≥0.60)")
    else:
        if low_pre:
            ids = ", ".join(f"T{int(e['kick_id'])} ({e.get('pose_confidence','?'):.2f})" for e in low_pre)
            st.warning(f"**Pre** — düşük güvenli tekmeler: {ids}")
        if low_post:
            ids = ", ".join(f"T{int(e['kick_id'])} ({e.get('pose_confidence','?'):.2f})" for e in low_post)
            st.warning(f"**Post** — düşük güvenli tekmeler: {ids}")
        st.caption("Düşük güvenli tekmelerin metrikleri gürültülü olabilir — karşılaştırmada dikkate alın.")

    if len(pre_events) < 5 or len(post_events) < 5:
        st.info(
            f"Pre: {len(pre_events)} tekme, Post: {len(post_events)} tekme. "
            "İstatistiksel karşılaştırma için her oturumda en az 5 tekme önerilir. "
            "Cohen's d ve CI değerleri düşük n'de geniş belirsizlik taşır."
        )


def _emg_frame_export_rows(
    frame_df: pd.DataFrame,
    emg_rows: list[dict],
    events: list[dict],
    session_label: str,
) -> list[dict]:
    if frame_df.empty or not emg_rows:
        return []

    event_by_frame: dict[int, dict] = {}
    phase_by_frame: dict[int, str] = {}
    for ev in events:
        start = int(ev.get("start_frame", 0))
        peak = int(ev.get("peak_frame", start))
        end = int(ev.get("end_frame", peak))
        chamber = int(ev.get("chamber_frame", start))
        extension = int(ev.get("extension_frame", peak))
        for frame_idx in range(max(0, start), max(start, end) + 1):
            event_by_frame[frame_idx] = ev
            if frame_idx < chamber:
                phase_by_frame[frame_idx] = "hazırlık"
            elif frame_idx < extension:
                phase_by_frame[frame_idx] = "bacağı kaldırma"
            elif frame_idx < peak:
                phase_by_frame[frame_idx] = "vuruşa hızlanma"
            elif frame_idx == peak:
                phase_by_frame[frame_idx] = "vuruş zirvesi"
            else:
                phase_by_frame[frame_idx] = "geri çekme"

    rows: list[dict] = []
    frame_count = min(len(frame_df), len(emg_rows))
    for frame_idx in range(frame_count):
        frame = frame_df.iloc[frame_idx]
        emg = emg_rows[frame_idx]
        ev = event_by_frame.get(frame_idx)
        rows.append(
            {
                "Oturum": session_label,
                "frame_idx": frame_idx,
                "video_time_sec": round(float(frame.get("time_sec", emg.get("time_sec", 0))), 4),
                "emg_time_sec": round(float(emg.get("time_sec", frame.get("time_sec", 0))), 4),
                "kick_id": "" if ev is None else f"T{int(ev['kick_id'])}",
                "kick_phase": "" if ev is None else phase_by_frame.get(frame_idx, "tekme penceresi"),
                "kick_start_time_sec": "" if ev is None else float(ev.get("start_time_sec", 0)),
                "kick_peak_time_sec": "" if ev is None else float(ev.get("peak_time_sec", 0)),
                "kick_end_time_sec": "" if ev is None else float(ev.get("end_time_sec", 0)),
                "CH1_Kas": EMG_CH1_MUSCLE,
                "CH1_RectusFemoris_mV": round(float(emg["EMG_RMS_mV"]), 4),
                "CH1_RectusFemoris_median_freq_Hz": round(float(emg.get("EMG_CH1_median_freq_Hz", emg["EMG_median_freq_Hz"])), 2),
                "CH2_Kas": EMG_CH2_MUSCLE,
                "CH2_BicepsFemoris_mV": round(float(emg["EMG_CH2_RMS_mV"]), 4),
                "CH2_BicepsFemoris_median_freq_Hz": round(float(emg.get("EMG_CH2_median_freq_Hz", emg["EMG_median_freq_Hz"])), 2),
                "EMG_ortalama_median_freq_Hz": round(float(emg["EMG_median_freq_Hz"]), 2),
                "active_knee_angle_deg": frame.get("active_knee_angle_deg", ""),
                "active_knee_vel_deg_s": frame.get("active_knee_vel_deg_s", ""),
                "active_foot_y_norm": frame.get("active_foot_y_norm", ""),
            }
        )
    return rows


def _render_export_tab(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    pre_events: list[dict],
    post_events: list[dict],
    fatigue_data: dict,
    fi: float,
    pre_emg_rows: list[dict],
    post_emg_rows: list[dict],
) -> None:
    st.subheader("Dışa Aktar")
    st.info(
        "Ham CSV dosyaları tüm ölçümleri içerir. Normal kullanım için 'okunabilir özet' dosyalarını kullanmak daha uygundur; "
        "bu dosyalarda metrik adı, pre/post değeri, değişim yüzdesi ve bilimsel yorum birlikte yer alır."
    )
    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        st.markdown("**Pre Frame Metrikleri**")
        if not pre_df.empty:
            st.download_button(
                "📥 pre_frame_metrics.csv",
                pre_df.to_csv(index=False).encode("utf-8"),
                "pre_frame_metrics.csv", "text/csv",
            )
    with exp2:
        st.markdown("**Post Frame Metrikleri**")
        if not post_df.empty:
            st.download_button(
                "📥 post_frame_metrics.csv",
                post_df.to_csv(index=False).encode("utf-8"),
                "post_frame_metrics.csv", "text/csv",
            )
    with exp3:
        st.markdown("**Kick Events**")
        if pre_events:
            st.download_button(
                "📥 pre_kick_events.csv",
                pd.DataFrame(pre_events).to_csv(index=False).encode("utf-8"),
                "pre_kick_events.csv", "text/csv",
            )
        if post_events:
            st.download_button(
                "📥 post_kick_events.csv",
                pd.DataFrame(post_events).to_csv(index=False).encode("utf-8"),
                "post_kick_events.csv", "text/csv",
            )

    st.markdown("**Yorgunluk Raporu**")
    report_rows = comparison_rows(pre_events, post_events, PRIMARY_METRICS)
    for key, m in fatigue_data["metrics"].items():
        if m["pre"] is None or key in PRIMARY_METRICS:
            continue
        info = metric_info(key)
        report_rows.append({
            "Metrik": info.label,
            "Pre Ortalama": fmt_value(m["pre"], key),
            "Post Ortalama": fmt_value(m["post"], key),
            "Değişim": f"{m['pct']:+.1f}%" if m["pct"] is not None else "-",
            "Sonuç": "Yorgunluk yönünde" if m["fatigue_contribution"] and m["fatigue_contribution"] > 0 else "Yorgunluk yönünde değil",
            "Bu Ne Anlama Geliyor?": info.interpretation,
            "Nasıl Hesaplandı": info.calculation,
        })
    report_rows.append({
        "Metrik": "Yorgunluk İndeksi",
        "Pre Ortalama": "-",
        "Post Ortalama": f"{fi:.1f}/100",
        "Değişim": "-",
        "Sonuç": "Düşük" if fi < 33 else ("Orta" if fi < 66 else "Yüksek"),
        "Bu Ne Anlama Geliyor?": "Pre ve post biyomekanik metriklerin ağırlıklı değişiminden hesaplanan bileşik yorgunluk skorudur.",
        "Nasıl Hesaplandı": "Diz ROM, peak diz hızı, peak hıza süre, tekme yüksekliği, ayak hızı, tekme süresi ve ortalama hız ağırlıklı olarak birleştirilir.",
    })
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)
    st.download_button(
        "📥 okunabilir_yorgunluk_raporu.csv",
        pd.DataFrame(report_rows).to_csv(index=False).encode("utf-8"),
        "okunabilir_yorgunluk_raporu.csv", "text/csv",
    )

    st.markdown("**EMG Cihaz Çıktıları**")
    st.caption(
        "Bu dosyalar video zamanıyla aynı eksendedir. Frame çıktısında her satır bir video frame'ine, tekme çıktısında her satır bir tekme penceresine karşılık gelir."
    )
    emg_exp1, emg_exp2 = st.columns(2)
    pre_emg_frame_df = pd.DataFrame(_emg_frame_export_rows(pre_df, pre_emg_rows, pre_events, "Pre"))
    post_emg_frame_df = pd.DataFrame(_emg_frame_export_rows(post_df, post_emg_rows, post_events, "Post"))
    pre_emg_kick_df = pd.DataFrame(emg_device_rows(pre_events, pre_emg_rows, "Pre"))
    post_emg_kick_df = pd.DataFrame(emg_device_rows(post_events, post_emg_rows, "Post"))
    with emg_exp1:
        st.markdown("**Frame senkron EMG**")
        if not pre_emg_frame_df.empty:
            st.download_button(
                "📥 pre_emg_frame_sync.csv",
                pre_emg_frame_df.to_csv(index=False).encode("utf-8"),
                "pre_emg_frame_sync.csv", "text/csv",
            )
        if not post_emg_frame_df.empty:
            st.download_button(
                "📥 post_emg_frame_sync.csv",
                post_emg_frame_df.to_csv(index=False).encode("utf-8"),
                "post_emg_frame_sync.csv", "text/csv",
            )
    with emg_exp2:
        st.markdown("**Tekme bazlı EMG**")
        if not pre_emg_kick_df.empty:
            st.download_button(
                "📥 pre_emg_kick_summary.csv",
                pre_emg_kick_df.to_csv(index=False).encode("utf-8"),
                "pre_emg_kick_summary.csv", "text/csv",
            )
        if not post_emg_kick_df.empty:
            st.download_button(
                "📥 post_emg_kick_summary.csv",
                post_emg_kick_df.to_csv(index=False).encode("utf-8"),
                "post_emg_kick_summary.csv", "text/csv",
            )
    with st.expander("EMG CSV kolonları nasıl okunur?", expanded=False):
        st.markdown("**frame_idx / video_time_sec:** EMG satırının hangi video frame'i ve saniyesiyle eşleştiğini gösterir.")
        st.markdown("**kick_id:** Satır bir tekme penceresindeyse tekme numarasıdır; boşsa tekme dışında kalan zamandır.")
        st.markdown("**kick_phase:** Frame'in tekmenin hazırlık, vuruşa yükselme, vuruş anı veya geri çekme bölümünde olduğunu gösterir.")
        st.markdown("**CH1_RectusFemoris_mV:** Rectus femoris kanalının mV cinsinden aktivasyon değeridir.")
        st.markdown("**CH2_BicepsFemoris_mV:** Biceps femoris kanalının mV cinsinden aktivasyon değeridir.")
        st.markdown("**CH1/CH2 median freq Hz:** Her kas kanalının frekans bilgisidir; tekrarlar ilerledikçe düşmesi yorgunlukla uyumludur.")


def _render_sensor_tab(
    sensor_ok: bool,
    ps: dict,
    pos: dict,
    pre_events: list[dict],
    post_events: list[dict],
    pre_freq_drop: float,
    post_freq_drop: float,
) -> None:
    st.subheader("EMG Detayları")
    st.info(
        "Pre → Post arası EMG değişimi nöromüsküler yorgunluk perspektifinden gösterilir. "
        "Rectus femoris ve biceps femoris RMS değerleri kas aktivasyonunu, median frekans düşüşü kas yorgunluğu eğilimini anlatır."
    )

    if not sensor_ok:
        st.warning("Sensör verisi üretilemedi. Analizi yeniden çalıştırın.")
        return

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("EMG Frekans — Pre sonu",  f"{ps['freq_end']:.0f} Hz",
               delta=f"{ps['freq_end'] - ps['freq_start']:.0f} Hz",  delta_color="inverse")
    sc2.metric("EMG Frekans — Post sonu", f"{pos['freq_end']:.0f} Hz",
               delta=f"{pos['freq_end'] - pos['freq_start']:.0f} Hz", delta_color="inverse")
    sc3.metric("SmO2 — Pre sonu",  f"%{ps['smo2_end']:.0f}",
               delta=f"{ps['smo2_end'] - ps['smo2_start']:.0f}%",  delta_color="inverse")
    sc4.metric("SmO2 — Post sonu", f"%{pos['smo2_end']:.0f}",
               delta=f"{pos['smo2_end'] - pos['smo2_start']:.0f}%", delta_color="inverse")
    with st.expander("EMG ve NIRS değerleri nasıl yorumlanır?", expanded=False):
        st.markdown(
            "**EMG median frekans:** Kas kasılması sırasında güç spektrumunun orta frekansıdır. "
            "Yorgunlukta motor ünite iletimi ve hızlı kasılan lif katkısı azaldığı için frekans düşebilir."
        )
        st.markdown(
            "**EMG RMS:** Kas aktivasyon genliğini gösterir. Artış daha fazla kas aktivasyonu veya kompansasyon anlamına gelebilir; tek başına yorgunluk bulgusu değildir."
        )
        st.markdown(
            "**SmO2:** Kas oksijen satürasyonudur. Düşüş, çalışan kasın oksijen tüketiminin arttığını veya oksijenlenmenin yetersiz kaldığını gösterir."
        )

    st.markdown("---")

    # EMG median frequency overlay
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Scatter(x=ps["t_arr"],  y=ps["freq_arr"],
        name="Pre — EMG Median Frekans",  line=dict(color="#3b82f6", width=2, dash="dot")))
    fig_freq.add_trace(go.Scatter(x=pos["t_arr"], y=pos["freq_arr"],
        name="Post — EMG Median Frekans", line=dict(color="#ef4444", width=2)))
    for ev in pre_events:
        fig_freq.add_vline(x=float(ev["peak_time_sec"]),
                           line_dash="dot", line_color="rgba(59,130,246,0.3)", line_width=1)
    for ev in post_events:
        fig_freq.add_vline(x=float(ev["peak_time_sec"]),
                           line_dash="dot", line_color="rgba(239,68,68,0.3)", line_width=1)
    fig_freq.update_layout(
        title="EMG Median Frekans — Nöromüsküler Yorgunluk Trendi",
        height=270, xaxis_title="Zaman (sn)",
        yaxis=dict(title="Frekans (Hz)", gridcolor="#333"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(l=50, r=20, t=40, b=45),
        legend=dict(orientation="h", y=-0.4),
    )
    st.plotly_chart(fig_freq, use_container_width=True)

    # SmO2 overlay
    fig_smo2 = go.Figure()
    fig_smo2.add_trace(go.Scatter(x=ps["t_nirs"],  y=ps["smo2_arr"],
        name="Pre — SmO2 (%)",  line=dict(color="#22c55e", width=2, dash="dot")))
    fig_smo2.add_trace(go.Scatter(x=pos["t_nirs"], y=pos["smo2_arr"],
        name="Post — SmO2 (%)", line=dict(color="#f97316", width=2)))
    for ev in pre_events:
        fig_smo2.add_vline(x=float(ev["peak_time_sec"]),
                           line_dash="dot", line_color="rgba(34,197,94,0.3)", line_width=1)
    for ev in post_events:
        fig_smo2.add_vline(x=float(ev["peak_time_sec"]),
                           line_dash="dot", line_color="rgba(249,115,22,0.3)", line_width=1)
    fig_smo2.update_layout(
        title="NIRS — Kas Oksijen Satürasyonu (SmO2)",
        height=270, xaxis_title="Zaman (sn)",
        yaxis=dict(title="SmO2 (%)", gridcolor="#333", range=[0, 100]),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(l=50, r=20, t=40, b=45),
        legend=dict(orientation="h", y=-0.4),
    )
    st.plotly_chart(fig_smo2, use_container_width=True)

    # EMG RMS overlay
    fig_rms = go.Figure()
    fig_rms.add_trace(go.Scatter(x=ps["t_arr"],   y=ps["rms_arr"],
        name="Pre CH1 — Rectus femoris",   line=dict(color="#3b82f6", width=1.5, dash="dot")))
    fig_rms.add_trace(go.Scatter(x=pos["t_arr"],  y=pos["rms_arr"],
        name="Post CH1 — Rectus femoris",  line=dict(color="#ef4444", width=1.5)))
    fig_rms.add_trace(go.Scatter(x=ps["t_arr"],   y=ps["rms2_arr"],
        name="Pre CH2 — Biceps femoris",  line=dict(color="#60a5fa", width=1, dash="dot")))
    fig_rms.add_trace(go.Scatter(x=pos["t_arr"],  y=pos["rms2_arr"],
        name="Post CH2 — Biceps femoris", line=dict(color="#fca5a5", width=1)))
    fig_rms.update_layout(
        title="EMG RMS — Kas Aktivasyon Büyüklüğü",
        height=260, xaxis_title="Zaman (sn)",
        yaxis=dict(title="RMS (mV)", gridcolor="#333"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(l=50, r=20, t=40, b=45),
        legend=dict(orientation="h", y=-0.5, font=dict(size=10)),
    )
    st.plotly_chart(fig_rms, use_container_width=True)
    st.caption("Detaylı yorum için → 📄 Sporcu Raporu sekmesi.")


def _render_real_sensor_tab(
    pre_res,
    post_res,
    pre_events: list[dict],
    post_events: list[dict],
) -> bool:
    """EMG+NIRS CSV yükleme, senkron ve görselleştirme. CSV yüklüyse True döner."""
    import tempfile
    from src.sports.taekwondo.sensor_sync import (
        load_emg_csv, load_nirs_csv,
        resample_to_video_times, resample_nirs_to_video_times,
        compute_rms_per_kick, compute_nirs_per_kick,
    )

    st.subheader("Sensör CSV Verisi — EMG + NIRS")
    st.info(
        "K-Myo / Delsys EMG ve Moxy NIRS cihazlarından alınan CSV çıktılarını yükleyin. "
        "Sistem sensör zaman serisini video frame zamanlarına yeniden örnekler, tekme pencereleriyle eşleştirir ve her tekme için RMS/SmO2 özeti üretir."
    )

    sample_emg  = SAMPLE_DATA_DIR / "emg_sample.csv"
    sample_nirs = SAMPLE_DATA_DIR / "nirs_moxy_sample.csv"
    dl1, dl2, _ = st.columns([1, 1, 3])
    if sample_emg.exists():
        dl1.download_button(
            "📥 Örnek EMG CSV", sample_emg.read_bytes(),
            "emg_sample.csv", "text/csv", key="dl_emg_sample",
        )
    if sample_nirs.exists():
        dl2.download_button(
            "📥 Örnek NIRS CSV", sample_nirs.read_bytes(),
            "nirs_moxy_sample.csv", "text/csv", key="dl_nirs_sample",
        )

    st.divider()
    pre_col, post_col = st.columns(2)
    with pre_col:
        st.markdown("**Pre Oturum**")
        pre_emg_f  = st.file_uploader("EMG CSV",         type=["csv"], key="rs_pre_emg")
        pre_nirs_f = st.file_uploader("NIRS CSV (Moxy)",  type=["csv"], key="rs_pre_nirs")
        pre_off    = st.slider("Sync gecikmesi — Pre (sn)", -30.0, 30.0, 0.0, 0.5, key="rs_pre_off",
                               help="EMG/NIRS kaydı videodan kaç saniye sonra başladı.")
    with post_col:
        st.markdown("**Post Oturum**")
        post_emg_f  = st.file_uploader("EMG CSV",         type=["csv"], key="rs_post_emg")
        post_nirs_f = st.file_uploader("NIRS CSV (Moxy)",  type=["csv"], key="rs_post_nirs")
        post_off    = st.slider("Sync gecikmesi — Post (sn)", -30.0, 30.0, 0.0, 0.5, key="rs_post_off",
                                help="EMG/NIRS kaydı videodan kaç saniye sonra başladı.")

    any_uploaded = any([pre_emg_f, pre_nirs_f, post_emg_f, post_nirs_f])
    if not any_uploaded:
        st.info("Sensör CSV dosyalarını yükleyin veya yukarıdaki örnek dosyaları indirip deneyin.")
        return False

    def _save(uf) -> Path:
        p = Path(tempfile.mktemp(suffix=".csv"))
        p.write_bytes(uf.read())
        return p

    color_map  = {"Pre": "#3b82f6", "Post": "#ef4444"}
    ev_map     = {"Pre": pre_events, "Post": post_events}
    sessions   = [
        ("Pre",  pre_emg_f,  pre_nirs_f,  pre_events,  pre_res.fps,  pre_res.total_frames,  pre_off),
        ("Post", post_emg_f, post_nirs_f, post_events, post_res.fps, post_res.total_frames, post_off),
    ]

    results: dict[str, dict] = {}
    for lbl, emg_f, nirs_f, events, fps, total_frames, offset in sessions:
        if not emg_f and not nirs_f:
            continue
        vtimes = [i / fps for i in range(total_frames)]
        sr: dict = {"t": vtimes, "offset": offset}

        if emg_f:
            try:
                emg_data          = load_emg_csv(_save(emg_f))
                emg_res           = resample_to_video_times(emg_data, vtimes, offset)
                sr["emg"]         = emg_data
                sr["emg_res"]     = emg_res
                sr["kick_emg"]    = compute_rms_per_kick(emg_res, events, fps)
            except Exception as exc:
                st.error(f"{lbl} EMG yüklenemedi: {exc}")

        if nirs_f:
            try:
                nirs_data         = load_nirs_csv(_save(nirs_f))
                nirs_res          = resample_nirs_to_video_times(nirs_data, vtimes, offset)
                sr["nirs"]        = nirs_data
                sr["nirs_res"]    = nirs_res
                sr["kick_nirs"]   = compute_nirs_per_kick(nirs_res, events)
            except Exception as exc:
                st.error(f"{lbl} NIRS yüklenemedi: {exc}")

        if len(sr) > 2:
            results[lbl] = sr

    if not results:
        return False

    # ── EMG grafiği ────────────────────────────────────────────────────────────
    if any("emg" in v for v in results.values()):
        st.subheader("EMG — Kas Aktivasyonu")
        fig_emg = go.Figure()
        for lbl, sr in results.items():
            if "emg_res" not in sr:
                continue
            ch_name = list(sr["emg_res"].keys())[0]
            vals = [v if v is not None else float("nan") for v in sr["emg_res"][ch_name]]
            fig_emg.add_trace(go.Scatter(
                x=sr["t"], y=vals, name=f"{lbl} — {ch_name}",
                line=dict(color=color_map[lbl], width=1.2), opacity=0.85,
            ))
        for ev in pre_events:
            fig_emg.add_vline(x=float(ev["peak_time_sec"]),
                              line_dash="dot", line_color="rgba(59,130,246,0.35)", line_width=1)
        for ev in post_events:
            fig_emg.add_vline(x=float(ev["peak_time_sec"]),
                              line_dash="dot", line_color="rgba(239,68,68,0.35)", line_width=1)
        fig_emg.update_layout(
            height=260, xaxis_title="Zaman (sn)", yaxis_title="mV",
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#fafafa"),
            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
            margin=dict(l=50, r=20, t=30, b=40), legend=dict(orientation="h", y=-0.4),
        )
        st.plotly_chart(fig_emg, use_container_width=True)

    # ── NIRS grafiği ───────────────────────────────────────────────────────────
    if any("nirs" in v for v in results.values()):
        st.subheader("NIRS — Kas Oksijen Satürasyonu (SmO2)")
        fig_nirs = go.Figure()
        for lbl, sr in results.items():
            if "nirs" not in sr:
                continue
            nd = sr["nirs"]
            adj_t = [t + sr["offset"] for t in nd["time_s"]]
            fig_nirs.add_trace(go.Scatter(
                x=adj_t, y=nd["smo2"], name=f"{lbl} — SmO2",
                line=dict(color=color_map[lbl], width=2),
            ))
        for ev in pre_events:
            fig_nirs.add_vline(x=float(ev["peak_time_sec"]),
                               line_dash="dot", line_color="rgba(59,130,246,0.35)", line_width=1)
        for ev in post_events:
            fig_nirs.add_vline(x=float(ev["peak_time_sec"]),
                               line_dash="dot", line_color="rgba(239,68,68,0.35)", line_width=1)
        fig_nirs.update_layout(
            height=260, xaxis_title="Zaman (sn)", yaxis_title="SmO2 (%)",
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#fafafa"),
            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333", range=[30, 100]),
            margin=dict(l=50, r=20, t=30, b=40), legend=dict(orientation="h", y=-0.4),
        )
        st.plotly_chart(fig_nirs, use_container_width=True)

    # ── Per-kick tablo ─────────────────────────────────────────────────────────
    st.subheader("Tekme Bazlı Sensör Özeti")
    for lbl, sr in results.items():
        st.markdown(f"**{lbl} Oturum**")
        rows = []
        for ev in ev_map[lbl]:
            kid = int(ev["kick_id"])
            row: dict = {"Tekme": f"T{kid}", "Süre (sn)": round(float(ev.get("duration_sec", 0)), 2)}
            if "kick_emg" in sr:
                ke = next((k for k in sr["kick_emg"] if k["kick_id"] == kid), {})
                for ck in [k for k in ke if k.endswith("_rms")][:2]:
                    row[ck.replace("_rms", " RMS (mV)")] = ke.get(ck)
            if "kick_nirs" in sr:
                kn = next((k for k in sr["kick_nirs"] if k["kick_id"] == kid), {})
                row["Ort. SmO2 (%)"] = kn.get("mean_smo2")
                if kn.get("mean_thb") is not None:
                    row["Ort. THb"] = kn.get("mean_thb")
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Tekme"), use_container_width=True)

    st.session_state["rs_real_sensor_ok"] = True
    return True


def render() -> None:
    st.title("Video Analizi — Yorgunluk Değerlendirmesi")
    st.info(
        "Pre ve post antrenman videolarını yükle, analiz et, "
        "yorgunluk metriklerini ve EMG değerlerini incele. "
        "Ekrandaki her tablo/grafik, sporcu veya antrenörün kolay okuyacağı şekilde 'ne ölçüldü, nasıl hesaplandı, nasıl yorumlanır' açıklamalarıyla verilir."
    )

    # Fixed analysis parameters keep demo and repeated analyses consistent.
    dv_show_labels = False
    dv_prominence = 0.06
    dv_min_dist = 0.25
    dv_min_dur = 0.10
    dv_max_dur = 6.0
    dv_min_rom = 12
    dv_min_height = -0.5
    dv_vel_assist = 100

    # ── Upload ─────────────────────────────────────────────────────────────────
    col_pre, col_post = st.columns(2)
    with col_pre:
        st.markdown("#### Pre-antrenman")
        pre_upload = st.file_uploader("Pre video", type=["mp4", "avi", "mov"], key="dv_pre")
    with col_post:
        st.markdown("#### Post-antrenman")
        post_upload = st.file_uploader("Post video", type=["mp4", "avi", "mov"], key="dv_post")

    both_ready = pre_upload is not None and post_upload is not None
    run_dual   = st.button("▶ Her İkisini Analiz Et", disabled=not both_ready, type="primary")

    # ── Session state ──────────────────────────────────────────────────────────
    if "dv_pre_result" not in st.session_state:
        st.session_state.update({
            "dv_pre_result":  None,
            "dv_post_result": None,
            "dv_pre_df":      None,
            "dv_post_df":     None,
            "dv_tmp":         None,
        })

    if run_dual and both_ready:
        from src.sports.taekwondo.pipeline import run_analysis

        tmp = Path(tempfile.mkdtemp())
        pre_in,   post_in   = tmp / "pre_input.mp4",     tmp / "post_input.mp4"
        pre_out,  post_out  = tmp / "pre_annotated.mp4", tmp / "post_annotated.mp4"
        pre_fcsv, post_fcsv = tmp / "pre_frames.csv",    tmp / "post_frames.csv"
        pre_ecsv, post_ecsv = tmp / "pre_events.csv",    tmp / "post_events.csv"

        pre_in.write_bytes(pre_upload.read())
        post_in.write_bytes(post_upload.read())

        prog = st.progress(0, text="Pre video analiz ediliyor…")

        def _prog_pre(cur, total):
            prog.progress(int(min(cur / max(total, 1), 1.0) * 50), text=f"PRE — Frame {cur}/{total}")

        def _prog_post(cur, total):
            prog.progress(50 + int(min(cur / max(total, 1), 1.0) * 50), text=f"POST — Frame {cur}/{total}")

        try:
            kw = dict(
                show_joint_labels=dv_show_labels,
                event_peak_prominence_norm=dv_prominence,
                event_min_distance_sec=dv_min_dist,
                event_min_duration_sec=dv_min_dur,
                event_max_duration_sec=dv_max_dur,
                event_min_knee_rom_deg=float(dv_min_rom),
                event_min_peak_kick_height_norm=float(dv_min_height),
                vel_assist_threshold=float(dv_vel_assist),
            )
            pre_res  = run_analysis(pre_in,  pre_out,  pre_fcsv,  pre_ecsv,  progress_callback=_prog_pre,  **kw)
            post_res = run_analysis(post_in, post_out, post_fcsv, post_ecsv, progress_callback=_prog_post, **kw)
            prog.progress(100, text="Tamamlandı!")
        except Exception as exc:
            st.error(f"Analiz hatası: {exc}")
            st.stop()

        st.session_state.update({
            "dv_pre_result":  pre_res,
            "dv_post_result": post_res,
            "dv_pre_df":      _read_csv(str(pre_fcsv)),
            "dv_post_df":     _read_csv(str(post_fcsv)),
            "dv_tmp":         tmp,
        })

    pre_res  = st.session_state.get("dv_pre_result")
    post_res = st.session_state.get("dv_post_result")
    pre_df   = st.session_state.get("dv_pre_df",  pd.DataFrame())
    post_df  = st.session_state.get("dv_post_df", pd.DataFrame())

    if pre_res is None or post_res is None:
        st.info("Her iki videoyu yükleyip analiz başlatın.")
        st.stop()

    pre_events, post_events = pre_res.events, post_res.events

    # ── Sensor stats ───────────────────────────────────────────────────────────
    _pre_emg   = pre_res.synthetic_emg_rows   or []
    _pre_nirs  = pre_res.synthetic_nirs_rows  or []
    _post_emg  = post_res.synthetic_emg_rows  or []
    _post_nirs = post_res.synthetic_nirs_rows or []
    _sensor_ok = bool(_pre_emg and _post_emg)

    if _sensor_ok:
        _win5           = max(1, int(pre_res.fps * 5))
        _ps             = sensor_stats(_pre_emg,  _pre_nirs,  _win5)
        _pos            = sensor_stats(_post_emg, _post_nirs, _win5)
        _pre_freq_drop  = _ps["freq_start"]  - _ps["freq_end"]
        _post_freq_drop = _pos["freq_start"] - _pos["freq_end"]
        _pre_smo2_drop  = _ps["smo2_start"]  - _ps["smo2_end"]
        _post_smo2_drop = _pos["smo2_start"] - _pos["smo2_end"]
    else:
        _ps = _pos = {}
        _pre_freq_drop = _post_freq_drop = _pre_smo2_drop = _post_smo2_drop = 0.0

    # ── Summary header ─────────────────────────────────────────────────────────
    fatigue_data = compute_fatigue(pre_events, post_events)
    fi = fatigue_data["fatigue_index"]

    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Pre — Tespit Edilen Tekme", len(pre_events))
    h1.metric("Pre — Toplam Frame",        pre_res.total_frames)
    h2.metric("Post — Tespit Edilen Tekme", len(post_events))
    h2.metric("Post — Toplam Frame",        post_res.total_frames)

    pre_mean_vel  = events_mean(pre_events,  "active_peak_knee_vel_deg_s")
    post_mean_vel = events_mean(post_events, "active_peak_knee_vel_deg_s")
    if pre_mean_vel and post_mean_vel:
        h3.metric("Ort. Peak Diz Hızı — Pre",  f"{pre_mean_vel:.0f} °/s")
        h3.metric("Ort. Peak Diz Hızı — Post", f"{post_mean_vel:.0f} °/s",
                  delta=f"{post_mean_vel - pre_mean_vel:+.0f} °/s", delta_color="inverse")

    pre_mean_rom  = events_mean(pre_events,  "active_knee_rom_deg")
    post_mean_rom = events_mean(post_events, "active_knee_rom_deg")
    if pre_mean_rom and post_mean_rom:
        h4.metric("Ort. Diz ROM — Pre",  f"{pre_mean_rom:.1f}°")
        h4.metric("Ort. Diz ROM — Post", f"{post_mean_rom:.1f}°",
                  delta=f"{post_mean_rom - pre_mean_rom:+.1f}°", delta_color="inverse")

    fi_label = "Düşük" if fi < 33 else ("Orta" if fi < 66 else "Yüksek")
    h5.metric("Yorgunluk İndeksi", f"{fi:.1f}/100", delta=fi_label, delta_color="off")
    st.divider()

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "Özet", "📐 Açı Karşılaştırma", "⚡ Hız Karşılaştırma",
        "🔥 Yorgunluk Analizi", "📊 Tekme Bazlı", "🏃 Faz Analizi",
        "📏 Asimetri", "🔬 İstatistik", "💾 Export",
        "🧪 EMG Detayları", "📡 Sensör CSV", "📄 Sporcu Raporu",
    ])

    with tabs[0]:
        _render_general_summary(
            pre_events,
            post_events,
            pre_res,
            post_res,
            fatigue_data,
            fi,
            fi_label,
            _pre_emg,
            _post_emg,
        )

    with tabs[1]:
        st.info(
            "Açı grafikleri eklemlerin zaman içindeki konum değişimini gösterir. "
            "Diz ve kalça açılarındaki post düşüşleri hareket genişliği ve teknik stabilite açısından değerlendirilir."
        )
        render_metric_help(["active_knee_rom_deg"], "Açı grafiklerini nasıl okumalıyım?")
        for col_name, lbl in [
            ("R_KNEE",  "Sağ Diz Açısı"),       ("L_KNEE",  "Sol Diz Açısı"),
            ("R_HIP",   "Sağ Kalça Açısı"),      ("L_HIP",   "Sol Kalça Açısı"),
            ("R_ANKLE", "Sağ Ayak Bileği Açısı"),("L_ANKLE", "Sol Ayak Bileği Açısı"),
        ]:
            if col_name in pre_df.columns or col_name in post_df.columns:
                st.plotly_chart(
                    overlay_chart(pre_df, post_df, col_name, lbl, pre_events, post_events),
                    use_container_width=True,
                )

    with tabs[2]:
        st.info(
            "Hız grafikleri yorgunluk etkisini en hızlı gösteren bölümdür. "
            "Post çizgisinde peak hızların azalması, sporcunun patlayıcı hareket üretimini koruyamadığını gösterebilir."
        )
        render_metric_help(["active_peak_knee_vel_deg_s", "active_peak_foot_speed_norm"], "Hız metrikleri nasıl hesaplandı?")
        for col_name, lbl in [
            ("R_KNEE_vel_deg_s",  "Sağ Diz Açısal Hızı"),
            ("L_KNEE_vel_deg_s",  "Sol Diz Açısal Hızı"),
            ("R_HIP_vel_deg_s",   "Sağ Kalça Açısal Hızı"),
            ("R_FOOT_speed_norm", "Sağ Ayak Hızı (normalize)"),
            ("L_FOOT_speed_norm", "Sol Ayak Hızı (normalize)"),
        ]:
            if col_name in pre_df.columns or col_name in post_df.columns:
                st.plotly_chart(
                    overlay_chart(pre_df, post_df, col_name, lbl, pre_events, post_events),
                    use_container_width=True,
                )

    with tabs[3]:
        _render_fatigue_tab(fatigue_data, fi, fi_label)

    with tabs[4]:
        _render_per_kick_tab(pre_events, post_events, st.session_state.get("dv_tmp"))

    with tabs[5]:
        _render_phase_tab(pre_events, post_events)

    with tabs[6]:
        _render_asymmetry_tab(pre_events, post_events)

    with tabs[7]:
        _render_stats_tab(pre_events, post_events)

    with tabs[8]:
        _render_export_tab(pre_df, post_df, pre_events, post_events, fatigue_data, fi, _pre_emg, _post_emg)

    with tabs[9]:
        _render_sensor_tab(_sensor_ok, _ps, _pos, pre_events, post_events,
                           _pre_freq_drop, _post_freq_drop)

    with tabs[10]:
        _real_sensor_loaded = _render_real_sensor_tab(pre_res, post_res, pre_events, post_events)

    with tabs[11]:
        _real_ok = st.session_state.get("rs_real_sensor_ok", False)
        render_athlete_report(
            pre_events, post_events, pre_res, post_res, fi,
            _ps, _pos,
            _pre_freq_drop, _post_freq_drop,
            _pre_smo2_drop, _post_smo2_drop,
            _sensor_ok,
            real_sensor=_real_ok,
        )
