import unittest
from src.sports.taekwondo.reporting import comparison_rows, quality_notices, statistics_rows

class VideoReportTest(unittest.TestCase):
    def test_zero_and_missing_values(self):
        rows = comparison_rows([{'duration_sec': 0}], [{'duration_sec': 2}])
        self.assertEqual(rows[0]['Önce'], 0)
        self.assertEqual(rows[0]['Fark'], 2)
        self.assertIsNone(rows[0]['Değişim (%)'])
        self.assertIsNone(rows[1]['Önce'])

    def test_empty_and_nonfinite_values(self):
        rows = comparison_rows([{'duration_sec': float('nan')}], [])
        self.assertIsNone(rows[0]['Önce'])
        self.assertEqual(statistics_rows([], []), [])
        self.assertTrue(any(level == 'warning' for level, _ in quality_notices([], [])))
        self.assertFalse(any(level == 'success' for level, _ in quality_notices([{}]*6, [{}]*6)))

    def test_video_player_uses_managed_media_and_seek(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from unittest.mock import patch
        from ui.components.video_player import video_player
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'clip.mp4'
            path.write_bytes(b'placeholder')
            with patch('ui.components.video_player.st.video') as render:
                video_player(path, start_time=1.25)
                render.assert_called_once_with(str(path), start_time=1.25)
