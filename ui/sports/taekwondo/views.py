"""Taekwondo views composed from branch-independent UI components."""
import plotly.graph_objects as go
import streamlit as st
from src.sports.taekwondo.reporting import comparison_rows, statistics_rows, quality_notices
from ui.components.models import MetricCard, QualityNotice, TimelineEvent
from ui.components.metrics import metric_cards
from ui.components.quality import quality_panel
from ui.components.comparison import comparison_panel
from ui.components.timeline import event_timeline
from ui.components.video_player import video_player


def summary(pre, post):
    metric_cards([MetricCard('Önce — Tekme', len(pre.events)), MetricCard('Sonra — Tekme', len(post.events)),
                  MetricCard('Önce — Kare', pre.total_frames), MetricCard('Sonra — Kare', post.total_frames)])
    quality_panel([QualityNotice(text, level) for level, text in quality_notices(pre.events, post.events)])
    comparison_panel(comparison_rows(pre.events, post.events))
    for column, result, label in zip(st.columns(2), (pre, post), ('Önce', 'Sonra')):
        with column:
            st.caption(label)
            video_player(result.output_video_path)


def signals(pre_df, post_df):
    st.caption('Açılar görüntü düzlemindedir; açısal hız sporcu sprint hızı değildir.')
    for key, label in [('R_KNEE', 'Sağ diz (°)'), ('L_KNEE', 'Sol diz (°)'),
                       ('R_HIP', 'Sağ kalça (°)'), ('L_HIP', 'Sol kalça (°)'),
                       ('R_ANKLE', 'Sağ ayak bileği (°)'), ('L_ANKLE', 'Sol ayak bileği (°)'),
                       ('R_KNEE_vel_deg_s', 'Sağ diz açısal hızı (°/s)'),
                       ('L_KNEE_vel_deg_s', 'Sol diz açısal hızı (°/s)')]:
        fig = go.Figure()
        for frame, name in [(pre_df, 'Önce'), (post_df, 'Sonra')]:
            if frame is not None and key in frame and 'time_sec' in frame:
                fig.add_trace(go.Scatter(x=frame['time_sec'], y=frame[key], name=name))
        if fig.data:
            fig.update_layout(title=label, xaxis_title='Zaman (s)', yaxis_title=label)
            st.plotly_chart(fig, use_container_width=True)


def events(pre, post):
    for column, result, key in zip(st.columns(2), (pre, post), ('pre', 'post')):
        with column:
            st.caption('Önce' if key == 'pre' else 'Sonra')
            timeline = []
            for i, event in enumerate(result.events):
                try:
                    timeline.append(TimelineEvent(str(i), f'Tekme {i+1}', float(event['start_time_sec']),
                                                  float(event['end_time_sec'])))
                except (KeyError, TypeError, ValueError):
                    continue
            selected = event_timeline(timeline, key=f'taekwondo_{key}_event')
            if selected:
                video_player(result.output_video_path, start_time=selected.start_seconds)
            comparison_panel(result.events)


def movement(pre, post):
    rows = comparison_rows(pre.events, post.events)
    comparison_panel(rows[4:])
    st.caption('Açı asimetrisi kuvvet farkını veya yaralanma riskini kanıtlamaz.')


def statistics(pre, post):
    comparison_panel(statistics_rows(pre.events, post.events))
    st.caption('En az iki geçerli tekrar bulunan metrikler gösterilir. Aynı oturumdaki tekrarlar bağımsız sporcu örnekleri değildir.')
