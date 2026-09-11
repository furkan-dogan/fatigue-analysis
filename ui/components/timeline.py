"""Select timed events independently of the sport or playback implementation."""

from collections.abc import Sequence

import plotly.graph_objects as go
import streamlit as st

from ui.components.models import PanelState, TimelineEvent
from ui.components.state import render_state


def event_timeline(events: Sequence[TimelineEvent], *, key: str,
                   state: PanelState = PanelState()) -> TimelineEvent | None:
    if not render_state(state):
        return None
    if not events:
        render_state(PanelState('empty', 'Zaman çizelgesinde gösterilecek olay yok.'))
        return None
    by_id = {event.id: event for event in events}
    if len(by_id) != len(events):
        raise ValueError('Zaman çizelgesindeki olay kimlikleri benzersiz olmalı.')
    figure = go.Figure(go.Bar(
        x=[event.end_seconds - event.start_seconds for event in events],
        base=[event.start_seconds for event in events],
        y=[event.label for event in events], orientation='h',
        hovertemplate='%{y}<br>Başlangıç: %{base:.3f} s<br>Süre: %{x:.3f} s<extra></extra>',
    ))
    peaks = [event for event in events if event.peak_seconds is not None]
    if peaks:
        figure.add_trace(go.Scatter(x=[event.peak_seconds for event in peaks],
                                   y=[event.label for event in peaks], mode='markers', name='Tepe'))
    figure.update_layout(xaxis_title='Video zamanı (s)', height=min(500, 150 + 25 * len(events)),
                         showlegend=False, margin=dict(l=20, r=20, t=20, b=40))
    st.plotly_chart(figure, use_container_width=True, key=f'{key}_chart')
    selected = st.selectbox('İncelenecek olay', list(by_id), key=key,
                            format_func=lambda event_id: by_id[event_id].label)
    return by_id.get(selected)
