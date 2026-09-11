"""Video measurements and raw-data downloads."""
import pandas as pd
import streamlit as st
from src.sports.taekwondo.reporting import comparison_rows
from ui.components.comparison import comparison_panel


def render_report(pre, post, pre_df, post_df):
    rows = comparison_rows(pre.events, post.events)
    comparison_panel(rows)
    st.caption('Boş değer ölçüm yok anlamına gelir. Karşılaştırma, aynı protokol ve uyumlu kamera koşulları gerektirir.')
    exports = {'karsilastirma': pd.DataFrame(rows), 'once_kareler': pre_df,
               'sonra_kareler': post_df, 'once_tekmeler': pd.DataFrame(pre.events),
               'sonra_tekmeler': pd.DataFrame(post.events)}
    for name, frame in exports.items():
        if frame is not None and not frame.empty:
            st.download_button(name.replace('_', ' ').capitalize(), frame.to_csv(index=False).encode('utf-8-sig'),
                               f'{name}.csv', 'text/csv', key=f'taekwondo_export_{name}')
