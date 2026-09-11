"""Keep shared code independent from presentation and individual sports."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def imports_in(path: Path) -> set[str]:
    imports = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


class ArchitectureTest(unittest.TestCase):
    def assert_no_dependencies(self, folder, forbidden):
        for path in (ROOT / folder).rglob('*.py'):
            for dependency in imports_in(path):
                with self.subTest(file=str(path.relative_to(ROOT)), dependency=dependency):
                    self.assertFalse(any(
                        dependency == prefix or dependency.startswith(prefix + '.')
                        for prefix in forbidden
                    ))

    def test_core_is_independent_of_backends_sports_and_ui(self):
        self.assert_no_dependencies('src/core', ('src.adapters', 'src.sports', 'ui', 'streamlit', 'mediapipe', 'cv2'))

    def test_analysis_does_not_import_ui(self):
        self.assert_no_dependencies('src', ('ui', 'streamlit'))

    def test_shared_components_do_not_import_sports(self):
        self.assert_no_dependencies('ui/components', ('src.sports', 'ui.sports'))

    def test_adapters_do_not_import_sports(self):
        self.assert_no_dependencies('src/adapters', ('src.sports',))

    def test_sports_do_not_import_each_other(self):
        for sport in ('volleyball', 'taekwondo', 'basketball'):
            others = tuple('src.sports.' + other for other in ('volleyball', 'taekwondo', 'basketball') if other != sport)
            self.assert_no_dependencies('src/sports/' + sport, others)
