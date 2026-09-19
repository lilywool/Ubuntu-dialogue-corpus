import unittest

import pandas as pd

from pipeline.aggregation import (
    aggregate_conversations,
    aggregate_entities,
    aggregate_languages,
    aggregate_residuals,
    aggregate_technical_terms,
    aggregate_users,
    run_silver_to_gold,
    sample_messages,
)
from pipeline.feature_engineering import engineer_features


def _silver_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "message_id": [10, 11, 12],
        "conversation_id": [1, 1, 2],
        "date": [
            "2020-01-01 00:02:00", "2020-01-01 00:00:00", "2020-01-02 00:00:00",
        ],
        "from": ["alice", "bob", "alice"],
        "to": ["bob", "alice", "carol"],
        "channel": ["support", "support", "install"],
        "text_cleaned": ["great fix", "bad failure SPANISH", "neutral question"],
        "word_count": [2, 2, 2],
        "text_length": [9, 11, 16],
        "vader_compound": [0.8, -0.6, 0.0],
        "vader_label": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
        "transformer_expected_sentiment": [0.9, -0.9, 0.0],
        "transformer_score": [0.99, 0.99, 0.7],
        "transformer_label": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
        "transformer_negative": [0.02, 0.94, 0.15],
        "transformer_neutral": [0.03, 0.02, 0.70],
        "transformer_positive": [0.95, 0.04, 0.15],
        "days_since_release_bucket": ["0-30", "0-30", "31-90"],
        "days_until_release_bucket": ["31-90", "31-90", "0-30"],
        "topic_id": [0, 1, pd.NA],
        "topic_label": ["packages", "kernel", pd.NA],
        "tech_lexicon_matches": [
            [("apt", "package_management")],
            [("kernel", "operating_system")],
            [],
        ],
        "tech_lexicon_match_count": [1, 1, 0],
        "residual_words": [["mystery"], ["failure"], []],
        "named_entities": [
            '[{"text":"Ubuntu","label":"PRODUCT"}]', "[]", "[]",
        ],
    })


class SamplingTests(unittest.TestCase):
    def test_seeded_sample_is_reproducible_and_preserves_source_order(self):
        frame = pd.DataFrame({"value": list(range(20))})
        first = sample_messages(frame, 6, random_state=42)
        second = sample_messages(frame, 6, random_state=42)

        self.assertEqual(first["value"].tolist(), second["value"].tolist())
        self.assertEqual(first["value"].tolist(), sorted(first["value"].tolist()))

    def test_full_size_sample_does_not_shuffle(self):
        frame = pd.DataFrame({"value": list(range(5))})
        result = sample_messages(frame, 5, random_state=42)
        self.assertEqual(result["value"].tolist(), frame["value"].tolist())


class FeatureEngineeringTests(unittest.TestCase):
    def test_string_dates_are_parsed_and_sequence_features_are_chronological(self):
        frame = pd.DataFrame({
            "conversation_id": [1, 1, 1],
            "from": ["alice", "bob", "alice"],
            "date": [
                "2020-01-01 00:02", "2020-01-01 00:00", "2020-01-01 00:01",
            ],
        }, index=[7, 3, 5])

        result = engineer_features(frame)

        self.assertEqual(result.index.tolist(), [7, 3, 5])
        self.assertEqual(result["turn_count"].tolist(), [2, 0, 1])
        self.assertEqual(result["user_message_count"].tolist(), [2, 1, 1])
        self.assertAlmostEqual(result.loc[5, "response_gap_mins_between_speakers"], 1.0)
        self.assertTrue(pd.isna(result.loc[7, "response_gap_mins_between_speakers"]))

    def test_user_counts_do_not_require_conversation_ids(self):
        frame = pd.DataFrame({
            "from": ["alice", "alice", "bob"],
            "date": ["2020-01-02", "2020-01-01", "2020-01-03"],
        })
        result = engineer_features(frame)
        self.assertEqual(result["user_message_count"].tolist(), [2, 1, 1])

    def test_recipient_alone_does_not_count_as_an_answer(self):
        frame = pd.DataFrame({
            "conversation_id": [1],
            "from": ["alice"],
            "to": ["bob"],
            "date": ["2020-01-01"],
        })
        result = engineer_features(frame)
        self.assertEqual(result.loc[0, "conversation_participant_count"], 2)
        self.assertFalse(result.loc[0, "conversation_was_answered"])


class GoldAggregationTests(unittest.TestCase):
    def test_conversation_gold_works_without_entity_columns(self):
        frame = _silver_frame().drop(columns=["named_entities", "tech_lexicon_matches"])
        result = aggregate_conversations(frame, engineer=True)

        self.assertEqual(result["message_count"].sum(), len(frame))
        self.assertIn("conversation_duration_mins", result)
        answered = result.set_index("conversation_id")["conversation_was_answered"]
        self.assertTrue(answered.loc[1])
        self.assertFalse(answered.loc[2])

    def test_directional_sentiment_uses_expected_score_not_confidence(self):
        result = run_silver_to_gold(
            _silver_frame(), gold_level="conversation", engineer=False
        ).set_index("conversation_id")

        self.assertAlmostEqual(result.loc[1, "avg_transformer_expected_sentiment"], 0.0)
        self.assertAlmostEqual(result.loc[1, "avg_transformer_confidence"], 0.99)

    def test_user_gold_separates_sent_and_received_metrics(self):
        result = aggregate_users(_silver_frame(), engineer=False).set_index("user")

        self.assertAlmostEqual(
            result.loc["alice", "sent_avg_transformer_expected_sentiment"], 0.45
        )
        self.assertAlmostEqual(
            result.loc["alice", "received_avg_transformer_expected_sentiment"], -0.9
        )
        self.assertEqual(result.loc["alice", "sent_message_count"], 2)
        self.assertEqual(result.loc["alice", "received_message_count"], 1)

    def test_technical_and_named_entities_use_canonical_columns(self):
        frame = _silver_frame()
        technical = aggregate_technical_terms(frame, engineer=False)
        entities = aggregate_entities(frame, engineer=False)

        self.assertEqual(set(technical["technical_term"]), {"apt", "kernel"})
        self.assertEqual(
            set(entities["entity_source"]), {"technical_lexicon", "spacy_ner"}
        )

    def test_malformed_collection_fails_closed(self):
        frame = _silver_frame()
        frame.loc[0, "residual_words"] = "not valid ["
        with self.assertRaisesRegex(ValueError, "malformed"):
            aggregate_residuals(frame, engineer=False)

    def test_language_gold_includes_unlabeled_messages(self):
        result = aggregate_languages(_silver_frame(), engineer=False)
        self.assertEqual(set(result["language_label"]), {"SPANISH", "UNLABELED"})
        self.assertEqual(result["message_count"].sum(), len(_silver_frame()))

    def test_all_gold_routes_validate_and_record_provenance(self):
        frame = _silver_frame()
        levels = (
            "conversation", "user", "date", "channel", "release", "language",
            "residual", "technical", "topic", "entity", "sample",
        )
        for level in levels:
            with self.subTest(level=level):
                result = run_silver_to_gold(
                    frame,
                    gold_level=level,
                    sample_size=2 if level == "sample" else None,
                    engineer=False,
                )
                self.assertTrue(result.attrs["gold_validation"]["passed"])
                self.assertEqual(result.attrs["gold_provenance"]["gold_level"], level)


if __name__ == "__main__":
    unittest.main()
