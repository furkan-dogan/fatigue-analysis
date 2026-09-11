"""Source-frame inspection with optional pixel-coordinate reference overlays."""
import cv2
import streamlit as st
from src.adapters.video_review import read_frame


@st.cache_data(max_entries=8, show_spinner=False)
def source_frame(path, index):
    return read_frame(path, index)


def frame_inspector(path, count, *, key, boxes=(), lines=()):
    index = int(st.number_input('İncelenecek kare (0 başlangıç)', min_value=0, max_value=count - 1,
                                value=0, step=1, key=key))
    try:
        frame = source_frame(path, index).copy()
        for box in boxes:
            if box and box['frame'] == index:
                cv2.rectangle(frame, (int(box['x1']), int(box['y1'])), (int(box['x2']), int(box['y2'])), (0, 220, 0), 2)
        for line in lines:
            if line and line['frame'] == index:
                cv2.line(frame, (int(line['x1']), int(line['y1'])), (int(line['x2']), int(line['y2'])), (255, 180, 0), 2)
        st.image(frame, caption=f'Kaynak kare {index} · Koordinat başlangıcı sol üst köşe (0, 0).', use_container_width=True)
    except (OSError, ValueError) as exc:
        st.error(str(exc))
    return index
