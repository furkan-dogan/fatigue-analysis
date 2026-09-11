"""Application shell with explicit, lazily loaded sport entry points."""
from importlib import import_module
import streamlit as st
from ui.theme import configure_page

SPORTS = {'volleyball': 'Voleybol', 'taekwondo': 'Taekwondo', 'basketball': 'Basketbol'}


def main() -> None:
    configure_page()
    sport = st.radio('Branş', list(SPORTS), format_func=SPORTS.get,
                     horizontal=True, key='active_sport')
    import_module(f'ui.sports.{sport}.page').render()
