import unittest
from pathlib import Path

from databricks_integration.scripts.bronze_to_silver_ubuntu import (
    build_residual_plan,
    residual_stage_policy,
)


class DatabricksAdapterPolicyTests(unittest.TestCase):
    def test_no_key_resolves_to_manual_review(self):
        self.assertEqual(
            residual_stage_policy(use_api=True, api_key=None),
            "manual_review",
        )

    def test_manual_review_plan_does_not_claim_nonword_mutation(self):
        plan = build_residual_plan(use_api=False)

        self.assertEqual(plan["status"], "manual_review")
        self.assertIn("unchanged", plan["message"])
        self.assertFalse(plan["api_enabled"])

    def test_gold_job_is_spark_delta_not_pandas_csv(self):
        source = Path(
            "databricks_integration/scripts/silver_to_gold_ubuntu.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("pd.read_csv", source)
        self.assertNotIn("to_csv", source)
        self.assertIn('write.format("delta")', source)
        self.assertIn("saveAsTable", source)
        self.assertIn("validate_gold_spark", source)
        self.assertIn("F.xxhash64", source)
        self.assertNotIn("F.rand(", source)


if __name__ == "__main__":
    unittest.main()
