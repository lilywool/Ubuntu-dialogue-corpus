import unittest
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from pipeline.apply_lexicons import anonymize_structural
from pipeline.data_preparation import optimize_dtypes, prepare_text_column
from pipeline.parallel_execution import recommended_workers
from pipeline.pipeline import PipelineConfig, run_pipeline
from pipeline.residual_stage import (
    apply_api_labels,
    classify_residuals,
    extract_residual_vocabulary,
)
from pipeline.audit import append_run_audit
from pipeline.sentiment_analysis import (
    SENTIMENT_GENERATED_COLUMNS,
    VADER_PROBABILITY_ATOL,
    _features_from_pipeline_result,
    analyze_sentiment,
    score_transformer_series,
    validate_sentiment,
)
from pipeline.advanced_nlp import annotate_messages, validate_advanced_nlp
from pipeline.emotion_analysis import _emotion_features
from pipeline.evaluation import evaluate_sentiment_labels
from pipeline.glued_terms import find_glued_matches
from pipeline.spacy_features import annotate_spacy
from pipeline.topic_modeling import annotate_topics, fit_topic_model


def _residual_frame(text: str) -> pd.DataFrame:
    return pd.DataFrame({
        "text_cleaned": [text],
        "tech_lexicon_matches": [[]],
        "slang_matches": [[]],
        "glued_matches": [[]],
        "structural_matches": [[]],
    })


class DataPreparationTests(unittest.TestCase):
    def test_raw_text_is_copied_and_normalized(self):
        frame = pd.DataFrame({"text": ["dont panic", None]})

        result = prepare_text_column(frame)

        self.assertEqual(result.loc[0, "text_cleaned"], "don't panic")
        self.assertEqual(result.loc[1, "text_cleaned"], "")

    def test_hashed_identifiers_are_not_coerced_to_numeric(self):
        frame = pd.DataFrame({
            "conversation_id": ["a" * 64, "b" * 64],
            "message_id": ["c" * 64, "d" * 64],
            "folder": ["7", "8"],
            "turn_count": ["1", "2"],
        })

        with patch(
            "pipeline.data_preparation.pd.to_numeric",
            wraps=pd.to_numeric,
        ) as converter:
            result = optimize_dtypes(frame)

        converted_columns = {
            call.args[0].name
            for call in converter.call_args_list
            if getattr(call.args[0], "name", None) is not None
        }
        self.assertNotIn("conversation_id", converted_columns)
        self.assertNotIn("message_id", converted_columns)
        self.assertEqual(result["conversation_id"].tolist(), ["a" * 64, "b" * 64])
        self.assertEqual(result["message_id"].tolist(), ["c" * 64, "d" * 64])
        self.assertTrue(pd.api.types.is_unsigned_integer_dtype(result["folder"]))
        self.assertTrue(pd.api.types.is_unsigned_integer_dtype(result["turn_count"]))

    def test_url_containing_email_like_path_has_final_placeholder_counts(self):
        text = pd.Series(
            [
                "http://www.mail-archive.com/lug@linux.or.ug/msg14772.html",
                "contact person@example.com",
            ],
            dtype="string",
        )

        scrubbed, counts = anonymize_structural(text)

        self.assertEqual(scrubbed.iloc[0], "WEBSITEDOMAIN")
        self.assertEqual(counts.loc[0, "domain_count"], 1)
        self.assertEqual(counts.loc[0, "email_count"], 0)
        self.assertEqual(scrubbed.iloc[1], "contact EMAILADDRESS")
        self.assertEqual(counts.loc[1, "domain_count"], 0)
        self.assertEqual(counts.loc[1, "email_count"], 1)


class ResidualStageTests(unittest.TestCase):
    def test_glued_matcher_ignores_structural_placeholder_tokens(self):
        self.assertEqual(
            find_glued_matches("EMAILADDRESS KEYBOARDSHORTCUT WEBSITEDOMAIN"),
            [],
        )

    def test_reserved_labels_and_structural_placeholders_are_not_residuals(self):
        frame = _residual_frame(
            "EMAILADDRESS SPANISH NONWORD unrecognizedfixturetoken"
        )

        result, counts = extract_residual_vocabulary(frame)

        self.assertEqual(
            result.loc[0, "residual_words"], ["unrecognizedfixturetoken"]
        )
        self.assertEqual(counts, Counter({"unrecognizedfixturetoken": 1}))

    def test_multiword_known_match_excludes_each_component(self):
        frame = _residual_frame("video card unrecognizedfixturetoken")
        frame.at[0, "tech_lexicon_matches"] = [("video card", "hardware")]

        _, counts = extract_residual_vocabulary(frame)

        self.assertEqual(counts, Counter({"unrecognizedfixturetoken": 1}))

    def test_parallel_residual_extraction_uses_shared_backend(self):
        frame = pd.concat(
            [
                _residual_frame("firstfixturetoken sharedfixturetoken"),
                _residual_frame("secondfixturetoken sharedfixturetoken"),
            ],
            ignore_index=True,
        )

        result, counts = extract_residual_vocabulary(frame, workers=2)

        self.assertEqual(
            result.loc[0, "residual_words"],
            ["firstfixturetoken", "sharedfixturetoken"],
        )
        self.assertEqual(
            result.loc[1, "residual_words"],
            ["secondfixturetoken", "sharedfixturetoken"],
        )
        self.assertEqual(
            counts,
            Counter({
                "sharedfixturetoken": 2,
                "firstfixturetoken": 1,
                "secondfixturetoken": 1,
            }),
        )

    def test_known_english_is_not_sent_to_residual_classification(self):
        frame = _residual_frame("the delete command works thanks")

        result, counts = extract_residual_vocabulary(frame)

        self.assertEqual(result.loc[0, "residual_words"], [])
        self.assertEqual(counts, Counter())

    def test_digit_only_tokens_are_not_residual_vocabulary(self):
        frame = _residual_frame("version 2 or 64 bit with ntfs-3g")

        result, counts = extract_residual_vocabulary(frame)

        self.assertNotIn("2", result.loc[0, "residual_words"])
        self.assertNotIn("64", result.loc[0, "residual_words"])
        self.assertNotIn("2", counts)
        self.assertNotIn("64", counts)

    def test_manual_review_policy_does_not_mutate_text(self):
        frame = _residual_frame("mystery")

        result, labels, sources = classify_residuals(
            frame, Counter({"mystery": 1}), policy="manual_review"
        )

        self.assertEqual(result.loc[0, "text_cleaned"], "mystery")
        self.assertEqual(labels, {})
        self.assertEqual(sources, {})

    def test_api_labels_preserve_jargon_and_uncertain_tokens(self):
        frame = _residual_frame("kernel maybe basura hola")

        result = apply_api_labels(frame, {
            "kernel": "JARGON",
            "maybe": "UNCERTAIN",
            "basura": "NONWORD",
            "hola": "SPANISH",
        })

        self.assertEqual(
            result.loc[0, "text_cleaned"],
            "kernel maybe NONWORD SPANISH",
        )

    def test_api_policy_requires_a_key(self):
        with self.assertRaisesRegex(ValueError, "requires an API key"):
            classify_residuals(
                _residual_frame("mystery"),
                Counter({"mystery": 1}),
                policy="api",
            )

    def test_reviewed_policy_applies_language_and_nonword_labels(self):
        frame = _residual_frame("hola res unrecognizedfixturetoken")

        result, labels, sources = classify_residuals(
            frame,
            Counter({"hola": 1, "res": 1, "unrecognizedfixturetoken": 1}),
            policy="reviewed",
        )

        self.assertEqual(
            result.loc[0, "text_cleaned"],
            "SPANISH NONWORD unrecognizedfixturetoken",
        )
        self.assertEqual(labels, {"hola": "SPANISH", "res": "NONWORD"})
        self.assertEqual(sources, {"hola": "reviewed", "res": "reviewed"})

    def test_reviewed_pipeline_rejects_excessive_nonword_replacement(self):
        frame = pd.DataFrame({
            "message_id": range(100),
            "text": ["res"] * 100,
        })

        with TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(AssertionError, "NONWORD token rate"):
                run_pipeline(
                    frame,
                    config=PipelineConfig(
                        residual_policy="reviewed",
                        sentiment_mode="none",
                        parallel=False,
                        workers=1,
                        output_dir=output_dir,
                    ),
                )

    def test_reviewed_pipeline_preserves_reported_databricks_english(self):
        texts = [
            "i'm running my old laptop as a server... haha",
            "what is the delete command?",
            "of course that doesnt invalidate it just no experience with it",
            "thanks",
        ]
        frame = pd.DataFrame({"message_id": range(len(texts)), "text": texts})

        with TemporaryDirectory() as output_dir:
            result = run_pipeline(
                frame,
                config=PipelineConfig(
                    residual_policy="reviewed",
                    sentiment_mode="none",
                    parallel=False,
                    workers=1,
                    output_dir=output_dir,
                ),
            )

        self.assertFalse(result["text_cleaned"].str.contains("NONWORD").any())
        self.assertIn("delete command", result.loc[1, "text_cleaned"])
        self.assertEqual(result.loc[3, "text_cleaned"], "thanks")


class ParallelExecutionTests(unittest.TestCase):
    def test_default_worker_count_is_capped(self):
        self.assertLessEqual(recommended_workers(8_600_000), 4)


class SentimentTests(unittest.TestCase):
    def test_vader_validation_accepts_documented_rounding_error(self):
        frame = pd.DataFrame({
            "text_cleaned": ["ordinary message"],
            "vader_negative": [0.333],
            "vader_neutral": [0.333],
            "vader_positive": [0.335],
            "vader_compound": [0.0],
            "vader_label": ["NEUTRAL"],
        })

        self.assertEqual(VADER_PROBABILITY_ATOL, 0.002)
        self.assertTrue(validate_sentiment(frame, mode="vader")["passed"])

    def test_vader_scores_text_and_skips_placeholder_only_rows(self):
        frame = pd.DataFrame({
            "text_cleaned": ["great fix", "terrible failure", "NONWORD"],
        })

        result = analyze_sentiment(frame, mode="vader")

        self.assertEqual(result.loc[0, "vader_label"], "POSITIVE")
        self.assertEqual(result.loc[1, "vader_label"], "NEGATIVE")
        self.assertTrue(pd.isna(result.loc[2, "vader_compound"]))
        self.assertTrue(pd.isna(result.loc[2, "vader_label"]))
        self.assertAlmostEqual(
            result.loc[0, [
                "vader_negative", "vader_neutral", "vader_positive"
            ]].astype(float).sum(),
            1.0,
            places=3,
        )
        self.assertTrue(result.attrs["sentiment_validation"]["passed"])
        self.assertIn("vader", result.attrs["sentiment_provenance"])

    def test_transformer_probability_features_preserve_neutral_and_uncertainty(self):
        features = _features_from_pipeline_result([
            {"label": "negative", "score": 0.1},
            {"label": "neutral", "score": 0.7},
            {"label": "positive", "score": 0.2},
        ])

        self.assertEqual(features["transformer_label"], "NEUTRAL")
        self.assertAlmostEqual(features["transformer_score"], 0.7)
        self.assertAlmostEqual(features["transformer_expected_sentiment"], 0.1)
        self.assertAlmostEqual(features["transformer_margin"], 0.5)
        self.assertGreater(features["transformer_normalized_entropy"], 0)
        self.assertLess(features["transformer_normalized_entropy"], 1)

    def test_rerun_removes_columns_owned_by_the_other_sentiment_mode(self):
        frame = pd.DataFrame({"text_cleaned": ["great fix"]})
        for column in SENTIMENT_GENERATED_COLUMNS:
            frame[column] = "stale"

        result = analyze_sentiment(frame, mode="vader")

        self.assertIn("vader_positive", result)
        self.assertNotIn("transformer_score", result)

    def test_transformer_scoring_obeys_inference_chunk_bound(self):
        call_sizes = []

        def fake_classifier(texts, **_kwargs):
            call_sizes.append(len(texts))
            return [[
                {"label": "negative", "score": 0.1},
                {"label": "neutral", "score": 0.2},
                {"label": "positive", "score": 0.7},
            ] for _ in texts]

        with patch(
            "pipeline.sentiment_analysis.get_transformer_pipeline",
            return_value=fake_classifier,
        ):
            result = score_transformer_series(
                pd.Series([f"message {index}" for index in range(10)]),
                inference_chunk_size=3,
                batch_size=2,
            )

        self.assertEqual(call_sizes, [3, 3, 3, 1])
        self.assertEqual(result["transformer_label"].tolist(), ["POSITIVE"] * 10)
        self.assertTrue((result["transformer_expected_sentiment"] == 0.6).all())

    def test_sentiment_validation_rejects_inconsistent_probability_sum(self):
        frame = pd.DataFrame({
            "text_cleaned": ["message"],
            "transformer_negative": [0.4],
            "transformer_neutral": [0.4],
            "transformer_positive": [0.4],
            "transformer_label": ["NEGATIVE"],
            "transformer_score": [0.4],
            "transformer_expected_sentiment": [0.0],
            "transformer_entropy": [1.0],
            "transformer_normalized_entropy": [0.9],
            "transformer_margin": [0.0],
        })

        with self.assertRaisesRegex(AssertionError, "sum to 1"):
            validate_sentiment(frame, mode="transformer")

    def test_gold_label_evaluation_reports_classification_and_calibration(self):
        frame = pd.DataFrame({
            "gold": ["NEGATIVE", "NEUTRAL", "POSITIVE"],
            "transformer_label": ["NEGATIVE", "POSITIVE", "POSITIVE"],
            "transformer_negative": [0.8, 0.1, 0.1],
            "transformer_neutral": [0.1, 0.3, 0.1],
            "transformer_positive": [0.1, 0.6, 0.8],
        })

        metrics = evaluate_sentiment_labels(
            frame, truth_col="gold", backend="transformer"
        )

        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["scored_rows"], 3)
        self.assertGreater(metrics["log_loss"], 0)
        self.assertIn("NEUTRAL", metrics["per_class"])


class AdvancedNLPTests(unittest.TestCase):
    def test_spacy_parallel_path_adds_pos_and_named_entity_features(self):
        frame = pd.DataFrame({
            "text_cleaned": [
                "The video card failed in Ubuntu.",
                "Restart NetworkManager from the terminal.",
                "",
            ],
        })

        result = annotate_spacy(frame, workers=2, batch_size=2)

        pos_counts = json.loads(result.loc[0, "nlp_pos_counts"])
        self.assertEqual(sum(pos_counts.values()), result.loc[0, "nlp_token_count"])
        self.assertEqual(result.loc[2, "nlp_token_count"], 0)
        self.assertGreaterEqual(result.loc[0, "nlp_sentence_count"], 1)
        self.assertGreater(result.loc[0, "nlp_lexical_diversity"], 0)
        self.assertAlmostEqual(
            result.loc[0, "noun_ratio"],
            result.loc[0, "noun_count"] / result.loc[0, "nlp_token_count"],
        )
        metrics = validate_advanced_nlp(result, topic_mode="none")
        self.assertTrue(metrics["passed"])

    def test_topic_model_is_fitted_once_then_reused_for_assignment(self):
        frame = pd.DataFrame({
            "text_cleaned": [
                "wireless driver network adapter",
                "wifi connection network timeout",
                "package install apt repository",
                "apt package dependency update",
                "display driver graphics card",
                "monitor resolution graphics display",
            ],
        })

        bundle = fit_topic_model(frame["text_cleaned"], n_topics=3)
        result = annotate_topics(frame, bundle, batch_size=2)

        self.assertEqual(result["topic_score"].notna().sum(), len(frame))
        self.assertLessEqual(result["topic_id"].nunique(), 3)
        metrics = validate_advanced_nlp(
            result,
            run_spacy=False,
            topic_mode="nmf",
        )
        self.assertTrue(metrics["passed"])
        self.assertTrue(result["topic_score"].between(0, 1).all())
        self.assertTrue(result["topic_entropy"].between(0, 1).all())

    def test_topic_model_does_not_force_oov_text_into_topic_zero(self):
        bundle = fit_topic_model(
            ["wireless driver network", "package install repository"],
            n_topics=2,
        )
        frame = annotate_topics(
            pd.DataFrame({"text_cleaned": ["wireless driver", "zzzxxyy"]}),
            bundle,
        )

        self.assertGreater(frame.loc[0, "topic_vocabulary_terms"], 0)
        self.assertEqual(frame.loc[1, "topic_vocabulary_terms"], 0)
        self.assertTrue(pd.isna(frame.loc[1, "topic_id"]))
        self.assertTrue(pd.isna(frame.loc[1, "topic_score"]))

    def test_emotion_features_retain_all_classes_and_uncertainty(self):
        scores = [0.05, 0.05, 0.10, 0.50, 0.10, 0.10, 0.10]
        result = _emotion_features([
            {"label": label, "score": score}
            for label, score in zip(
                ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"),
                scores,
            )
        ])

        self.assertEqual(result["emotion_label"], "JOY")
        self.assertAlmostEqual(result["emotion_score"], 0.5)
        self.assertAlmostEqual(result["emotion_margin"], 0.4)
        self.assertAlmostEqual(sum(result[f"emotion_{label}"] for label in (
            "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
        )), 1.0)

    def test_validation_rejects_frozen_spacy_output(self):
        frame = annotate_spacy(pd.DataFrame({
            "text_cleaned": [f"message {index}" for index in range(20)],
        }))

        with self.assertRaisesRegex(AssertionError, "frozen"):
            validate_advanced_nlp(frame)

    def test_validation_rejects_collapsed_transformer_sentiment(self):
        frame = pd.DataFrame({
            "text_cleaned": [f"different message number {index}" for index in range(100)],
            "transformer_label": ["POSITIVE"] * 100,
            "transformer_score": [1.0] * 100,
            "transformer_negative": [0.0] * 100,
            "transformer_neutral": [0.0] * 100,
            "transformer_positive": [1.0] * 100,
            "transformer_expected_sentiment": [1.0] * 100,
            "transformer_entropy": [0.0] * 100,
            "transformer_normalized_entropy": [0.0] * 100,
            "transformer_margin": [1.0] * 100,
        })

        with self.assertRaisesRegex(AssertionError, "low diversity"):
            validate_advanced_nlp(
                frame,
                sentiment_mode="transformer",
                run_spacy=False,
            )

    def test_advanced_rerun_drops_outputs_from_disabled_stages(self):
        frame = pd.DataFrame({
            "text_cleaned": ["wireless driver failure"],
            "topic_id": [3],
            "topic_label": ["stale"],
            "topic_score": [0.9],
            "emotion_label": ["JOY"],
            "emotion_score": [0.9],
        })

        result = annotate_messages(
            frame,
            run_spacy=False,
            topic_mode="none",
            emotion_mode="none",
        )

        self.assertNotIn("topic_id", result)
        self.assertNotIn("emotion_label", result)


class PipelineIntegrationTests(unittest.TestCase):
    def test_pipeline_sample_is_seeded_order_preserving_and_auditable(self):
        frame = pd.DataFrame({
            "message_id": list(range(10)),
            "text": [f"message {index}" for index in range(10)],
        })
        with TemporaryDirectory() as output_dir:
            first = run_pipeline(
                frame,
                config=PipelineConfig(
                    sentiment_mode="none", workers=1, sample_size=4,
                    sample_seed=19, output_dir=output_dir,
                ),
            )
            second = run_pipeline(
                frame,
                config=PipelineConfig(
                    sentiment_mode="none", workers=1, sample_size=4,
                    sample_seed=19, output_dir=output_dir,
                ),
            )

        self.assertEqual(first["message_id"].tolist(), second["message_id"].tolist())
        self.assertEqual(first["message_id"].tolist(), sorted(first["message_id"].tolist()))
        self.assertEqual(first.attrs["sample_provenance"]["source_rows"], 10)
        self.assertTrue(first.attrs["sample_provenance"]["sampled"])

    def test_raw_frame_runs_without_mutating_manual_review_residuals(self):
        frame = pd.DataFrame({
            "message_id": [1, 2, 3],
            "text": [
                "dont email me at person@example.com",
                "hola mystery",
                "",
            ],
        })
        with TemporaryDirectory() as output_dir:
            result = run_pipeline(
                frame,
                config=PipelineConfig(
                    residual_policy="manual_review",
                    sentiment_mode="none",
                    parallel=True,
                    workers=2,
                    output_dir=output_dir,
                ),
            )

            self.assertEqual(len(result), 3)
            self.assertTrue(result.loc[0, "text_cleaned"].startswith("don't"))
            self.assertIn("EMAILADDRESS", result.loc[0, "text_cleaned"])
            self.assertNotIn("EMAILADDRESS", result.loc[0, "residual_words"])
            self.assertTrue(
                result["tech_lexicon_matches"].str.len().equals(
                    result["tech_lexicon_match_count"]
                )
            )
            self.assertEqual(result.loc[1, "text_cleaned"], "hola mystery")
            self.assertTrue(Path(result.attrs["residual_output"]).exists())
            self.assertEqual(result.attrs["parallel_workers"], 2)

    def test_audit_is_append_only_and_requires_a_validated_existing_output(self):
        frame = pd.DataFrame({"text_cleaned": ["great fix"]})
        result = analyze_sentiment(frame, mode="vader")
        with TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "result.csv"
            log_path = Path(output_dir) / "pipeline_run_log.jsonl"
            result.to_csv(output_path, index=False)
            for _ in range(2):
                append_run_audit(
                    log_path,
                    df=result,
                    stage="test",
                    output_path=output_path,
                    validation=result.attrs["sentiment_validation"],
                )

            records = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["rows"], 1)
            self.assertTrue(records[0]["validation"]["passed"])


if __name__ == "__main__":
    unittest.main()
