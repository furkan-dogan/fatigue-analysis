from __future__ import annotations
from src.adapters.csv_read import read_optional_csv as _read_csv

from pathlib import Path
import pandas as pd
import streamlit as st
from src.core.numeric import events_mean
from ui.sports.taekwondo.presentation import EMG_CH1_GROUP, EMG_CH1_MUSCLE, EMG_CH2_GROUP, EMG_CH2_MUSCLE, EMG_PLACEMENT, PRIMARY_METRICS, action_recommendations, comparison_rows, emg_device_rows, emg_summary_text, fmt_value, metric_info, movement_summary_rows, readable_emg_frequency, readable_emg_rms, readable_findings, render_data_quality, render_metric_help, session_summary_rows
from ui.components.charts import readable_bar_comparison
from ui.components.video_player import video_player
from ui.components.comparison import comparison_panel
from src.sports.taekwondo.reporting.tables import _emg_frame_export_rows
from ui.sports.taekwondo.tabs.fatigue import _render_fatigue_metric_table
from ui.components.metrics import metric_cards
from ui.components.models import MetricCard


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
    comparison_panel(phase_rows)
    with st.expander("Faz ve asimetri nasıl okunur?", expanded=False):
        st.markdown("**Uzatma fazı:** Bacağın hedefe doğru açıldığı bölümdür. Süre uzarsa veya hız düşerse tekme daha yavaş çıkıyor olabilir.")
        st.markdown("**Geri çekim fazı:** Vuruştan sonra bacağın geri alınmasıdır. Bu faz uzarsa savunmaya dönüş gecikebilir.")
        st.markdown("**ASI:** Sağ-sol taraf farkını yüzde olarak gösterir. 10% üzeri farklar yük dağılımı açısından dikkat gerektirir.")

    st.markdown("#### EMG Cihaz Çıktısı")
    if emg_rows:
        emg_df = pd.DataFrame(emg_rows)
        freq_vals = [float(r["ortalama_emg_median_freq_hz"]) for r in emg_rows]
        metric_cards([MetricCard(label, value) for label, value in [('Rectus Femoris', readable_emg_rms(sum(rectus_vals) / len(rectus_vals))), ('Biceps Femoris', readable_emg_rms(sum(biceps_vals) / len(biceps_vals))), ('Kas Yorgunluğu', readable_emg_frequency(sum(freq_vals) / len(freq_vals))), ('Yorgunluk İşaretli Tekme', f'{fatigue_count}/{len(emg_rows)}')]])
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
    comparison_panel(cmp_rows)
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
