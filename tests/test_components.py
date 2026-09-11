"""Exercise reusable widgets with empty, missing, failed and populated inputs."""

import unittest

from streamlit.testing.v1 import AppTest

from ui.components.models import TimelineEvent


def render_components(status='ready', empty=False):
    import streamlit as st
    from ui.components.models import MetricCard, PanelState, QualityNotice, TimelineEvent
    from ui.components.metrics import metric_cards
    from ui.components.comparison import comparison_panel
    from ui.components.quality import quality_panel
    from ui.components.timeline import event_timeline
    from ui.components.upload import video_uploader

    state = PanelState(status)
    metric_cards([] if empty else [MetricCard('Sıfır', 0, 'cm'), MetricCard('Eksik', None),
                                   MetricCard('Sonlu değil', float('nan'))], state=state)
    comparison_panel([] if empty else [{'Ölçüm': 'Süre', 'Önce': 0, 'Sonra': None}], state=state)
    quality_panel([] if empty else [QualityNotice('Ölçüm doğrulanmadı.', 'warning')], state=state)
    selected = event_timeline([] if empty else [TimelineEvent('a', 'Birinci', 1, 2, 1.5),
                                               TimelineEvent('b', 'İkinci', 3, 4)], key='events', state=state)
    if selected:
        st.caption(f'Seçilen: {selected.id} / {selected.start_seconds}')
    video_uploader('Video', key='video', disabled=empty, state=state)


class ComponentTest(unittest.TestCase):
    def test_zero_missing_and_comparison_values(self):
        app = AppTest.from_function(render_components).run()
        self.assertFalse(list(app.exception))
        self.assertEqual([metric.value for metric in app.metric], ['0 cm', '—', '—'])
        self.assertEqual(app.dataframe[0].value.iloc[0]['Önce'], 0)
        self.assertEqual(len(app.warning), 1)
        self.assertEqual(len(app.get('file_uploader')), 1)

    def test_timeline_selection_survives_rerun(self):
        app = AppTest.from_function(render_components).run()
        app.selectbox(key='events').set_value('b').run()
        self.assertFalse(list(app.exception))
        self.assertIn('Seçilen: b / 3', [caption.value for caption in app.caption])

    def test_empty_ready_data_does_not_create_invalid_widgets(self):
        app = AppTest.from_function(render_components, kwargs={'empty': True}).run()
        self.assertFalse(list(app.exception))
        self.assertFalse(list(app.metric))
        self.assertFalse(list(app.selectbox))
        self.assertEqual(len(app.info), 4)
        self.assertTrue(app.get('file_uploader')[0].proto.disabled)

    def test_nonready_states_suppress_content(self):
        for state in ('empty', 'loading', 'error'):
            with self.subTest(state=state):
                app = AppTest.from_function(render_components, args=(state,)).run()
                self.assertFalse(list(app.exception))
                self.assertFalse(list(app.metric))
                self.assertFalse(list(app.dataframe))
                self.assertFalse(list(app.get('file_uploader')))
                self.assertEqual(len(app.error if state == 'error' else app.info), 5)

    def test_invalid_event_ranges_are_rejected(self):
        for start, end, peak in ((2, 1, None), (-1, 1, None), (0, float('nan'), None), (0, 1, 2)):
            with self.subTest(start=start, end=end, peak=peak), self.assertRaises(ValueError):
                TimelineEvent('id', 'Olay', start, end, peak)
