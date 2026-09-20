import json
import unittest
from datetime import datetime

from databricks_integration.run_metrics import (
    RUN_METRIC_COLUMNS,
    append_stage_metric,
    build_run_metric_context,
    detect_worker_count,
)
from pipeline.pipeline import PipelineConfig


class RunMetricsTests(unittest.TestCase):
    def test_worker_count_uses_spark_conf_without_spark_context(self):
        class FakeConf:
            def get(self, key):
                if key == "spark.databricks.clusterUsageTags.clusterWorkers":
                    return "4"
                raise KeyError(key)

        class FakeSpark:
            conf = FakeConf()

        self.assertEqual(detect_worker_count(FakeSpark()), 4)

    def test_context_is_allow_listed_and_secret_free(self):
        secret = "sk-fixture-never-serialize"
        context = build_run_metric_context(
            run_id="a" * 32,
            pipeline_layer="bronze_to_silver",
            config=PipelineConfig(
                api_key=secret,
                residual_policy="api",
                sample_size=10_000,
                sample_seed=42,
            ),
            materialization_mode="delta",
            compute_mode="spark_connect_or_serverless",
        )

        self.assertNotIn("api_key", context)
        self.assertNotIn(secret, json.dumps(context))
        self.assertEqual(context["sample_size"], 10_000)
        self.assertEqual(context["sample_seed"], 42)

    def test_stage_metrics_calculate_throughput_without_rounding(self):
        records = []
        context = build_run_metric_context(
            run_id="b" * 32,
            pipeline_layer="silver_to_gold",
            materialization_mode="persist",
            compute_mode="classic",
            gold_level="conversation",
        )

        record = append_stage_metric(
            records,
            context,
            stage="gold_aggregation",
            duration_seconds=2.5,
            rows_in=1_000,
            rows_out=100,
        )

        self.assertEqual(tuple(record), RUN_METRIC_COLUMNS)
        self.assertEqual(record["stage_sequence"], 1)
        self.assertEqual(record["rows_per_second"], 400.0)
        self.assertEqual(record["duration_per_row_ms"], 2.5)
        self.assertEqual(record["gold_level"], "conversation")
        self.assertIsInstance(record["completed_at_utc"], datetime)

    def test_zero_row_stage_has_no_per_row_metric(self):
        records = []
        context = build_run_metric_context(
            run_id="c" * 32,
            pipeline_layer="bronze_to_silver",
        )

        record = append_stage_metric(
            records,
            context,
            stage="bronze_read",
            duration_seconds=0.25,
            rows_out=0,
        )

        self.assertEqual(record["rows_per_second"], 0.0)
        self.assertIsNone(record["duration_per_row_ms"])


if __name__ == "__main__":
    unittest.main()
