import unittest
from src.sports.taekwondo.metrics import compute_foot_speed


class MetricUnitsTest(unittest.TestCase):
    def test_missing_scale_never_returns_pixel_speed(self):
        for scale in (None, 0, -1, float('nan'), float('inf')):
            self.assertEqual(compute_foot_speed([(0, 0), (10, 0)], 10, [scale, scale]), [None, None])

    def test_known_scale_keeps_normalized_unit(self):
        self.assertEqual(compute_foot_speed([(0, 0), (10, 0)], 10, [100, 100]), [1, 1])
