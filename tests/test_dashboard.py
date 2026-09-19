import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipeline.dashboard import (
    build_topic_priorities,
    export_dashboard_bundle,
    filter_conversations,
    load_dashboard_bundle,
    validate_dashboard_tables,
)


def _dashboard_tables():
    return {
        "conversation_summary": pd.DataFrame({
            "conversation_id": [1, 2],
            "message_count": [3, 1],
            "conversation_was_answered": [True, False],
            "conversation_start": ["2020-01-01", "2020-02-01"],
            "most_common_topic": ["packages", "display"],
            "avg_vader_compound": [0.2, -0.4],
        }),
        "monthly_summary": pd.DataFrame({
            "date_period": ["2020-01-01", "2020-02-01"],
            "message_count": [3, 1],
        }),
        "correlation_matrix": pd.DataFrame({
            "feature": ["word_count"], "word_count": [1.0]
        }),
        "statistical_summary": pd.DataFrame({
            "feature": ["word_count"], "mean": [4.0]
        }),
        "hypothesis_tests": pd.DataFrame({
            "hypothesis": ["example"],
            "test": ["Mann-Whitney U"],
            "p_value": [0.5],
            "p_value_holm": [0.5],
        }),
        "model_selection": pd.DataFrame({
            "model": ["Dummy baseline"], "average_precision": [0.5]
        }),
        "model_test_metrics": pd.DataFrame({
            "model": ["Logistic regression"], "roc_auc": [0.7]
        }),
        "model_feature_importance": pd.DataFrame({
            "feature": ["word_count"], "importance": [0.2]
        }),
    }


class DashboardPipelineTests(unittest.TestCase):
    def test_bundle_round_trip_is_portable_and_validated(self):
        tables = _dashboard_tables()
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_dashboard_bundle(
                tables, directory, metadata={"analysis_seed": 42}
            )
            self.assertEqual(
                manifest["dashboard_tables"]["conversation_summary"],
                "conversation_summary.csv",
            )
            loaded, loaded_manifest = load_dashboard_bundle(directory)

        self.assertEqual(len(loaded), len(tables))
        self.assertTrue(loaded_manifest["validation"]["passed"])
        self.assertEqual(
            loaded["conversation_summary"]["conversation_was_answered"].tolist(),
            [True, False],
        )

    def test_privacy_columns_fail_closed(self):
        tables = _dashboard_tables()
        tables["conversation_summary"]["text_cleaned"] = ["secret", "secret"]
        with self.assertRaisesRegex(ValueError, "exposes"):
            validate_dashboard_tables(tables)

    def test_filters_apply_date_outcome_and_topic(self):
        conversations = _dashboard_tables()["conversation_summary"]
        conversations["conversation_start"] = pd.to_datetime(
            conversations["conversation_start"], utc=True
        )
        result = filter_conversations(
            conversations,
            start_date="2020-01-01",
            end_date="2020-01-31",
            answered=True,
            topic="packages",
        )
        self.assertEqual(result["conversation_id"].tolist(), [1])

    def test_topic_priority_rewards_volume_and_low_answer_rate(self):
        topics = pd.DataFrame({
            "topic_label": ["low", "high", "tiny"],
            "conversations": [100, 100, 2],
            "answer_rate": [0.2, 0.8, 0.0],
        })
        result = build_topic_priorities(topics, minimum_conversations=25)
        self.assertEqual(result["topic_label"].tolist(), ["low", "high"])

    def test_streamlit_app_has_all_four_views(self):
        source = Path("dashboard/app.py").read_text(encoding="utf-8")
        compile(source, "dashboard/app.py", "exec")
        for view in ("Descriptive", "Diagnostic", "Predictive", "Prescriptive"):
            self.assertIn(f'"{view}"', source)
        self.assertIn("load_dashboard_bundle", source)


if __name__ == "__main__":
    unittest.main()
