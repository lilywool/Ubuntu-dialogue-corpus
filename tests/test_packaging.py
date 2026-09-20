import tomllib
import unittest
from pathlib import Path


class ProjectPackagingTests(unittest.TestCase):
    def test_databricks_worker_package_includes_code_and_runtime_data(self):
        root = Path(__file__).resolve().parents[1]
        config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(config["project"]["name"], "ubuntu-dialogue-pipeline")
        self.assertEqual(config["project"]["requires-python"], ">=3.11,<3.13")
        self.assertEqual(
            set(config["tool"]["setuptools"]["packages"]["find"]["include"]),
            {
                "databricks_integration*",
                "lexicons_and_templates*",
                "pipeline*",
                "reviewed_overlays*",
            },
        )
        package_data = config["tool"]["setuptools"]["package-data"]
        self.assertIn(
            "sms_slang_emoticons_dictionary_filled.csv",
            package_data["lexicons_and_templates"],
        )
        self.assertIn(
            "still_unclassified_defaults_to_NONWORD.csv",
            package_data["reviewed_overlays"],
        )


if __name__ == "__main__":
    unittest.main()
