"""Render prepared comparison rows; no sport, unit conversion or scoring rules."""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import streamlit as st

from ui.components.models import PanelState
from ui.components.state import render_state


def comparison_panel(rows: Sequence[Mapping[str, Any]], *, state: PanelState = PanelState()) -> None:
    if not render_state(state):
        return
    if not rows:
        render_state(PanelState('empty', 'Karşılaştırılacak veri yok.'))
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
