import ast
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from databricks_integration.scripts.bronze_to_silver_ubuntu import (
    _normalize_match_lists,
    _qualified_scratch_table,
    _resolve_materialization_mode,
    build_residual_labels_from_counts,
    build_residual_plan,
    expected_feature_columns,
    residual_stage_policy,
    transform_core_partition,
    transform_enrichment_partition,
)
from pipeline.pipeline import PipelineConfig, run_pipeline


class DatabricksAdapterPolicyTests(unittest.TestCase):
    def test_runtime_csv_inputs_are_not_git_lfs_filtered(self):
        attributes = Path(".gitattributes").read_text(encoding="utf-8")

        self.assertNotIn("*.csv filter=lfs", attributes)
        for path in (
            Path("lexicons_and_templates/sms_slang_emoticons_dictionary_filled.csv"),
            Path("reviewed_overlays/residual_words_for_classification.csv"),
            Path("reviewed_overlays/still_unclassified_defaults_to_NONWORD.csv"),
        ):
            self.assertFalse(
                path.read_text(encoding="utf-8").startswith(
                    "version https://git-lfs.github.com/spec/v1"
                ),
                f"{path} must be available to Databricks as ordinary Git content",
            )

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

    def test_bronze_to_silver_job_is_partitioned_spark_delta(self):
        source = Path(
            "databricks_integration/scripts/bronze_to_silver_ubuntu.py"
        ).read_text(encoding="utf-8")

        self.assertIn("mapInPandas", source)
        self.assertIn("fit_global_topic_model_spark", source)
        self.assertIn("spark.sparkContext.broadcast", source)
        self.assertIn('write.format("delta")', source)
        self.assertIn("validate_silver_spark", source)
        self.assertNotIn("bronze_df.toPandas()", source)

    def test_spark_job_avoids_python_chained_comparisons(self):
        source = Path(
            "databricks_integration/scripts/bronze_to_silver_ubuntu.py"
        ).read_text(encoding="utf-8")
        chained = [
            node.lineno
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Compare) and len(node.ops) > 1
        ]

        self.assertEqual(
            chained,
            [],
            "Spark Column comparisons must use individually parenthesized &/| clauses",
        )

    def test_partition_functions_match_local_vader_pipeline(self):
        frame = pd.DataFrame({
            "message_id": ["m1", "m2", "m3"],
            "conversation_id": ["c1", "c1", "c2"],
            "text": [
                "Ubuntu works great!",
                "I cannot connect to 192.168.1.5 :(",
                "sudo apt-get update fixed it",
            ],
        })
        with TemporaryDirectory() as output_dir:
            config = PipelineConfig(
                residual_policy="manual_review",
                sentiment_mode="vader",
                advanced_nlp=False,
                parallel=False,
                workers=1,
                output_dir=output_dir,
            )
            local = run_pipeline(frame.copy(), config=config)
            partitioned = transform_enrichment_partition(
                transform_core_partition(frame.copy()),
                config=config,
                residual_labels={},
            )

        comparable = sorted(expected_feature_columns(config))
        self.assertEqual(set(comparable).difference(local.columns), set())
        self.assertEqual(set(comparable).difference(partitioned.columns), set())
        for column in comparable:
            left = local[column].reset_index(drop=True)
            right = partitioned[column].reset_index(drop=True)
            if pd.api.types.is_numeric_dtype(left.dtype):
                np.testing.assert_allclose(
                    left.to_numpy(dtype=float),
                    right.to_numpy(dtype=float),
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=True,
                    err_msg=column,
                )
            else:
                self.assertEqual(left.tolist(), right.tolist(), column)

    def test_reviewed_policy_is_deterministic_and_source_labeled(self):
        labels, sources = build_residual_labels_from_counts(
            {"unrecognizedfixturetoken": 3, "12345": 1},
            policy="reviewed",
        )

        self.assertEqual(labels["unrecognizedfixturetoken"], "NONWORD")
        self.assertEqual(labels["12345"], "UNCERTAIN")
        self.assertEqual(set(sources.values()), {"reviewed"})

    def test_api_policy_without_key_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "requires an API key"):
            build_residual_labels_from_counts(
                {"unrecognizedfixturetoken": 1}, policy="api", api_key=None
            )

    def test_partition_topic_assignment_requires_global_model(self):
        frame = transform_core_partition(pd.DataFrame({"text": ["ubuntu help"]}))
        config = PipelineConfig(
            sentiment_mode="none",
            advanced_nlp=True,
            run_spacy=False,
            topic_mode="nmf",
            parallel=False,
            workers=1,
        )

        with self.assertRaisesRegex(ValueError, "globally fitted topic model"):
            transform_enrichment_partition(frame, config=config, topic_model=None)

    def test_arrow_array_values_are_normalized_for_spark_output(self):
        frame = pd.DataFrame({
            "structural_matches": [np.array([["host", "domain"]], dtype=object)],
            "tech_lexicon_matches": [np.array([["apt", "package"]], dtype=object)],
            "slang_matches": [np.array([], dtype=object)],
            "glued_matches": [np.array([{
                "term": "aptget", "category": "package", "glue": "prefix",
                "run": "apt",
            }], dtype=object)],
            "residual_words": [np.array(["please", "help"], dtype=object)],
        })

        normalized = _normalize_match_lists(frame)

        self.assertEqual(normalized.at[0, "structural_matches"], [["host", "domain"]])
        self.assertEqual(normalized.at[0, "tech_lexicon_matches"], [["apt", "package"]])
        self.assertEqual(normalized.at[0, "slang_matches"], [])
        self.assertEqual(normalized.at[0, "glued_matches"][0]["term"], "aptget")
        self.assertEqual(normalized.at[0, "residual_words"], ["please", "help"])

    def test_enrichment_normalizes_arrow_arrays_before_core_validation(self):
        frame = pd.DataFrame({
            "message_id": ["m1"],
            "conversation_id": ["c1"],
            "text": ["sudo apt update works great"],
        })
        core = transform_core_partition(frame)
        for column in (
            "structural_matches",
            "tech_lexicon_matches",
            "slang_matches",
            "glued_matches",
            "residual_words",
        ):
            core[column] = core[column].map(
                lambda values: np.array(values, dtype=object)
            )

        enriched = transform_enrichment_partition(
            core,
            config=PipelineConfig(
                sentiment_mode="none",
                advanced_nlp=False,
                parallel=False,
            ),
        )

        for column in (
            "structural_matches",
            "tech_lexicon_matches",
            "slang_matches",
            "glued_matches",
            "residual_words",
        ):
            self.assertIsInstance(enriched.at[0, column], list)

    def test_serverless_delta_materialization_names_are_validated(self):
        run_id = "a" * 32

        self.assertEqual(_resolve_materialization_mode("delta"), "delta")
        self.assertEqual(
            _qualified_scratch_table("workspace.default", "core", run_id),
            f"workspace.default._ubuntu_core_{run_id}",
        )
        with self.assertRaisesRegex(ValueError, "materialization_schema"):
            _qualified_scratch_table("workspace.bad-name", "core", run_id)
        with self.assertRaisesRegex(ValueError, "materialization_mode"):
            _resolve_materialization_mode("cache")


if __name__ == "__main__":
    unittest.main()
