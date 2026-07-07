import importlib
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class ProjectScaffoldTests(unittest.TestCase):
    def test_pyproject_declares_src_package(self):
        pyproject = ROOT / "pyproject.toml"
        self.assertTrue(pyproject.exists())

        content = pyproject.read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.10,<3.13"', content)
        self.assertIn('package-dir = { "" = "src" }', content)
        self.assertIn('where = ["src"]', content)
        self.assertIn("[project.optional-dependencies]", content)
        self.assertIn("web =", content)
        self.assertIn("separation =", content)
        self.assertIn("lyrics =", content)
        self.assertIn("dev =", content)

    def test_lightweight_package_imports(self):
        remixmate = importlib.import_module("remixmate")
        config = importlib.import_module("remixmate.config")

        self.assertIsNotNone(remixmate)
        self.assertEqual(config.SAMPLE_RATE, 44100)
        self.assertTrue(pathlib.Path(config.BASE_DIR).exists())

    def test_app_dependency_check_is_non_installing(self):
        app = importlib.import_module("app")

        self.assertTrue(hasattr(app, "check_runtime_environment"))
        self.assertFalse(hasattr(app, "check_and_install_dependencies"))
        self.assertFalse(hasattr(app, "subprocess"))


if __name__ == "__main__":
    unittest.main()
