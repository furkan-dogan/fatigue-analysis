from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui.sports.taekwondo.presentation import render_metric_help
from ui.sports.taekwondo.charts import gauge
from ui.components.comparison import comparison_panel
from src.sports.taekwondo.reporting.tables import _fatigue_metric_rows


def _render_fatigue_metric_table(fatigue_data: dict, title: str = "Metrik Bazlı Yorgunluk Bulguları") -> None:
    st.markdown(f"#### {title}")
    st.info(
        "Bu tablo yorgunluk sonucunu hangi ölçümlerin etkilediğini gösterir. "
        "'Yorgunluk Bulgusu: Evet' yazıyorsa post-antrenmanda ilgili metrik yorgunluk yönünde değişmiştir. "
        "Katkı puanı pozitifse yorgunluğu artırır; negatifse performans korunmuş veya iyileşmiş olabilir."
    )
    rows_ft = _fatigue_metric_rows(fatigue_data)
    comparison_panel(rows_ft)
    with st.expander("Bu tablo nasıl hesaplandı ve nasıl okunur?", expanded=False):
        st.markdown("**Pre / Post:** Antrenman öncesi ve sonrası videolardan hesaplanan ortalama değerlerdir.")
        st.markdown("**Değişim:** Post değerin pre değere göre yüzde olarak ne kadar değiştiğini gösterir.")
        st.markdown("**Yorgunluk Bulgusu:** Değişimin yorgunluk yönünde olup olmadığını söyler. Örneğin hız düşerse yorgunluk bulgusudur; peak hıza ulaşma süresi artarsa yine yorgunluk bulgusudur.")
        st.markdown("**Katkı Puanı:** O metriğin genel yorgunluk skorunu ne kadar artırdığını veya azalttığını gösterir.")
        st.markdown("Tek bir satır tek başına kesin karar verdirmez; video, EMG, takip güveni ve diğer hareket ölçümleriyle birlikte okunmalıdır.")
    render_metric_help([k for k in fatigue_data["metrics"] if fatigue_data["metrics"][k].get("pre") is not None], "Metrikler tek tek nasıl hesaplandı?")


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
