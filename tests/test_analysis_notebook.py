import json
import unittest
from pathlib import Path


class AnalysisNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path("notebooks/Ubuntu_step7-12.ipynb")
        cls.notebook = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_remaining_steps_are_present(self):
        for heading in (
            "Step 9: Statistics & Correlation",
            "Step 10: Visualization",
            "Step 11: Machine Learning",
            "Step 12: Evaluation",
        ):
            self.assertIn(heading, self.markdown)

    def test_all_code_cells_compile(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile(
                    "".join(cell.get("source", [])),
                    f"{self.path}:cell-{index}",
                    "exec",
                )

    def test_response_model_contract_is_leakage_safe(self):
        self.assertIn("first_messages", self.code)
        self.assertIn("was_answered", self.code)
        self.assertIn("DummyClassifier", self.code)
        self.assertIn("RandomizedSearchCV", self.code)
        self.assertIn("excluded_as_leakage", self.code)
        self.assertIn("conversation_duration_mins", self.code)
        self.assertNotIn("numeric_feature_candidates = [\n    'conversation_message_count'", self.code)

    def test_dashboard_exports_exclude_raw_text_and_usernames(self):
        self.assertIn("from pipeline.dashboard import export_dashboard_bundle", self.code)
        self.assertIn("export_dashboard_bundle(", self.code)

    def test_step_12_covers_evaluation_and_documentation(self):
        self.assertIn("business_impact_assessment", self.code)
        self.assertIn("evaluation_manifest.json", self.code)
        self.assertIn("conversation_response_model_card.md", self.code)
        self.assertIn("Downstream dashboard handoff (after Step 12)", self.markdown)


if __name__ == "__main__":
    unittest.main()
