"""Compact event selection independent of sport and detection method."""
import streamlit as st


def movement_list(events, *, key):
    if not events:
        return None
    selected = st.selectbox('Bulunan hareket', range(len(events)),
                            format_func=lambda i: f"{i+1}. {events[i]['label']}", key=key)
    event = events[selected]
    st.caption(event.get('reason', ''))
    return event
