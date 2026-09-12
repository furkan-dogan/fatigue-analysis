"""Sport-independent metric cards; missing values remain distinct from zero."""

from collections.abc import Sequence
import math

import streamlit as st

from ui.components.models import MetricCard, PanelState
from ui.components.state import render_state


def metric_cards(cards: Sequence[MetricCard], *, state: PanelState = PanelState(), columns: int = 4) -> None:
    if not render_state(state):
        return
    if not cards:
        render_state(PanelState('empty', 'Gösterilecek metrik yok.'))
        return
    for start in range(0, len(cards), columns):
        batch = cards[start:start + columns]
        for column, card in zip(st.columns(len(batch)), batch):
            missing = card.value is None or (isinstance(card.value, float) and not math.isfinite(card.value))
            value = '—' if missing else f'{card.value}{" " + card.unit if card.unit else ""}'
            with column, st.container(border=True):
                st.metric(card.label, value, delta=None if missing else card.delta,
                          delta_color=card.delta_color, help=card.help)
