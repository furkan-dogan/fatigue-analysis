"""Freeze legacy behavior for a structural move, including known limitations."""

import json
from pathlib import Path
import unittest

from tests.regression.taekwondo_case import legacy_snapshot


class TaekwondoMigrationTest(unittest.TestCase):
    def test_legacy_metrics_events_and_simulations_are_preserved(self):
        expected = json.loads(Path(__file__).with_name('legacy_snapshot.json').read_text())
        self.assertEqual(legacy_snapshot(), expected)
