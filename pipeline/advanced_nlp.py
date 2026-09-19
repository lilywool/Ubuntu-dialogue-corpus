"""Orchestrate optional message-level advanced NLP stages."""

from __future__ import annotations

import importlib.metadata
import json
from typing import Any

import numpy as np
import pandas as pd

from pipeline.emotion_analysis import (
    DEFAULT_EMOTION_MODEL,
    DEFAULT_EMOTION_REVISION,
    EMOTION_COLUMNS,
    EMOTION_LABELS,
    EMOTION_PROBABILITY_COLUMNS,
)
from pipeline.sentiment_analysis import (
    is_effectively_empty,
    strip_label_tokens_series,
    validate_sentiment,
)
from pipeline.spacy_features import DEFAULT_SPACY_MODEL
from pipeline.topic_modeling import TopicModelBundle
from pipeline.schema import FEATURE_SCHEMA_VERSION

SPACY_COLUMNS = (
    "nlp_pos_counts", "named_entities", "named_entity_label_counts",
    "nlp_token_count", "nlp_alpha_token_count", "nlp_unique_alpha_token_count",
    "nlp_lexical_diversity", "nlp_sentence_count", "nlp_avg_sentence_tokens",
    "nlp_stopword_count", "nlp_negation_count", "nlp_punctuation_count",
    "nlp_exclamation_count", "nlp_question_count", "nlp_uppercase_token_count",
    "noun_count", "noun_ratio", "proper_noun_count", "proper_noun_ratio",
    "verb_count", "verb_ratio", "adjective_count", "adjective_ratio",
    "adverb_count", "adverb_ratio", "named_entity_count", "person_entity_count",
    "location_entity_count", "organization_entity_count", "product_entity_count",
    "nlp_tokens", "nlp_pos_tags",
)
TOPIC_COLUMNS = (
    "topic_id", "topic_label", "topic_weight", "topic_score",
    "topic_entropy", "topic_margin", "topic_vocabulary_terms",
)
ADVANCED_NLP_GENERATED_COLUMNS = SPACY_COLUMNS + TOPIC_COLUMNS + EMOTION_COLUMNS


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def clear_advanced_nlp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove stage-owned outputs so mode changes cannot leave stale data."""
    existing = [column for column in ADVANCED_NLP_GENERATED_COLUMNS if column in df]
    if existing:
        df.drop(columns=existing, inplace=True)
    df.attrs.pop("topic_model_metadata", None)
    df.attrs.pop("advanced_nlp_provenance", None)
    df.attrs.pop("advanced_nlp_validation", None)
    return df


def annotate_messages(
    df: pd.DataFrame,
    *,
    text_col: str = "text_cleaned",
    run_spacy: bool = True,
    spacy_model: str = DEFAULT_SPACY_MODEL,
    spacy_batch_size: int = 128,
    workers: int = 1,
    include_token_details: bool = False,
    topic_mode: str = "none",
    topic_model: TopicModelBundle | None = None,
    n_topics: int = 12,
    topic_max_features: int = 20_000,
    topic_fit_sample_size: int = 100_000,
    topic_batch_size: int = 50_000,
    topic_random_state: int = 42,
    emotion_mode: str = "none",
    emotion_model: str = DEFAULT_EMOTION_MODEL,
    emotion_revision: str = DEFAULT_EMOTION_REVISION,
    emotion_batch_size: int = 32,
    emotion_inference_chunk_size: int = 4096,
    emotion_device: int | None = None,
    emotion_dtype: str = "float32",
) -> pd.DataFrame:
    """Add selected features while preserving one row per message.

    Topic fitting is separated from assignment. If ``topic_model`` is omitted
    locally, one deterministic bounded-sample model is fitted. Databricks
    callers should use ``fit_topic_model`` once on a representative sample and
    reuse/broadcast that bundle for every partition.
    """
    if text_col not in df:
        raise KeyError(f"missing advanced NLP text column: {text_col}")
    if topic_mode not in {"none", "nmf"}:
        raise ValueError("topic_mode must be 'none' or 'nmf'")
    if emotion_mode not in {"none", "transformer"}:
        raise ValueError("emotion_mode must be 'none' or 'transformer'")
    clear_advanced_nlp_columns(df)
    original_index = df.index.copy()
    analysis_text = strip_label_tokens_series(df[text_col].fillna("").astype(str))
    analysis_frame = pd.DataFrame({text_col: analysis_text}, index=df.index)

    if run_spacy:
        from pipeline.spacy_features import annotate_spacy

        analysis_frame = annotate_spacy(
            analysis_frame,
            text_col=text_col,
            model_name=spacy_model,
            batch_size=spacy_batch_size,
            workers=workers,
            include_token_details=include_token_details,
        )
        for column in analysis_frame.columns.difference([text_col]):
            df[column] = analysis_frame[column]

    if topic_mode == "nmf":
        from pipeline.topic_modeling import annotate_topics, fit_topic_model

        if topic_model is None:
            topic_model = fit_topic_model(
                analysis_text,
                n_topics=n_topics,
                max_features=topic_max_features,
                fit_sample_size=topic_fit_sample_size,
                random_state=topic_random_state,
            )
        analysis_frame = annotate_topics(
            analysis_frame,
            topic_model,
            text_col=text_col,
            batch_size=topic_batch_size,
        )
        for column in TOPIC_COLUMNS:
            df[column] = analysis_frame[column]
        df.attrs["topic_model_metadata"] = {
            "labels": topic_model.labels,
            "fit_rows": topic_model.fit_rows,
            "random_state": topic_model.random_state,
        }

    if emotion_mode == "transformer":
        from pipeline.emotion_analysis import emotion_runtime_metadata, score_emotion_series

        emotion = score_emotion_series(
            df[text_col],
            model_name=emotion_model,
            revision=emotion_revision,
            batch_size=emotion_batch_size,
            inference_chunk_size=emotion_inference_chunk_size,
            device=emotion_device,
            dtype=emotion_dtype,
        )
        for column in EMOTION_COLUMNS:
            df[column] = emotion[column]
        resolved_emotion_runtime = emotion_runtime_metadata()
    else:
        resolved_emotion_runtime = {}

    df.attrs["advanced_nlp_provenance"] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "text_column": text_col,
        "spacy": {
            "enabled": run_spacy,
            "model": spacy_model if run_spacy else None,
            "batch_size": spacy_batch_size if run_spacy else None,
            "workers": workers if run_spacy else None,
            "token_details": include_token_details if run_spacy else None,
            "spacy_version": _package_version("spacy") if run_spacy else None,
            "model_package_version": (
                _package_version(spacy_model.replace("_", "-")) if run_spacy else None
            ),
        },
        "topics": {
            "mode": topic_mode,
            "random_state": topic_random_state if topic_mode == "nmf" else None,
            "fit_sample_size": topic_fit_sample_size if topic_mode == "nmf" else None,
            "scikit_learn_version": (
                _package_version("scikit-learn") if topic_mode == "nmf" else None
            ),
        },
        "emotion": {
            "mode": emotion_mode,
            "model": emotion_model if emotion_mode == "transformer" else None,
            "revision": emotion_revision if emotion_mode == "transformer" else None,
            "dtype": emotion_dtype if emotion_mode == "transformer" else None,
            "batch_size": emotion_batch_size if emotion_mode == "transformer" else None,
            "inference_chunk_size": (
                emotion_inference_chunk_size if emotion_mode == "transformer" else None
            ),
            **resolved_emotion_runtime,
            "transformers_version": (
                _package_version("transformers") if emotion_mode == "transformer" else None
            ),
            "torch_version": (
                _package_version("torch") if emotion_mode == "transformer" else None
            ),
        },
    }

    if not df.index.equals(original_index):
        raise RuntimeError("advanced NLP changed the DataFrame index")
    return df


def validate_advanced_nlp(
    df: pd.DataFrame,
    *,
    text_col: str = "text_cleaned",
    sentiment_mode: str = "none",
    run_spacy: bool = True,
    topic_mode: str = "none",
    emotion_mode: str = "none",
    require_technical_lexicon: bool = False,
    minimum_topic_vocabulary_coverage: float = 0.50,
) -> dict[str, Any]:
    """Fail closed on missing, malformed, or frozen advanced-NLP output."""
    failures: list[str] = []
    metrics: dict[str, Any] = {"rows": len(df)}
    if sentiment_mode not in {"none", "vader", "transformer", "both"}:
        raise ValueError("invalid sentiment_mode for validation")
    if not 0 <= minimum_topic_vocabulary_coverage <= 1:
        raise ValueError("minimum_topic_vocabulary_coverage must be in [0, 1]")
    stripped = strip_label_tokens_series(df[text_col].fillna("").astype(str))
    scoreable = ~stripped.map(is_effectively_empty)

    if sentiment_mode != "none":
        try:
            sentiment_metrics = validate_sentiment(
                df, text_col=text_col, mode=sentiment_mode
            )
        except AssertionError as exc:
            failures.append(str(exc))
        else:
            metrics.update(sentiment_metrics)

    if require_technical_lexicon:
        required = {"tech_lexicon_matches", "tech_lexicon_match_count"}
        missing = sorted(required.difference(df.columns))
        if missing:
            failures.append(f"missing canonical technical-lexicon columns: {missing}")
        else:
            actual_counts = df["tech_lexicon_matches"].str.len()
            if not actual_counts.equals(df["tech_lexicon_match_count"].astype(int)):
                failures.append("tech_lexicon_match_count does not match canonical matches")

    if run_spacy:
        required = {
            "nlp_pos_counts", "named_entities", "nlp_token_count",
            "named_entity_label_counts", "named_entity_count",
            "nlp_alpha_token_count", "nlp_unique_alpha_token_count",
            "nlp_lexical_diversity", "nlp_sentence_count",
            "noun_count", "noun_ratio", "proper_noun_count", "proper_noun_ratio",
            "verb_count", "verb_ratio", "adjective_count", "adjective_ratio",
            "adverb_count", "adverb_ratio", "person_entity_count",
            "location_entity_count", "organization_entity_count",
            "product_entity_count",
        }
        missing = sorted(required.difference(df.columns))
        if missing:
            failures.append(f"missing spaCy columns: {missing}")
        else:
            try:
                parsed_counts = df["nlp_pos_counts"].map(
                    lambda value: sum(json.loads(value).values())
                )
            except (TypeError, json.JSONDecodeError) as exc:
                failures.append(f"invalid nlp_pos_counts JSON: {exc}")
            else:
                if not parsed_counts.equals(df["nlp_token_count"].astype(int)):
                    failures.append("nlp_token_count does not match POS-count JSON")
                try:
                    entity_counts = df["named_entity_label_counts"].map(json.loads)
                except (TypeError, json.JSONDecodeError) as exc:
                    failures.append(f"invalid named_entity_label_counts JSON: {exc}")
                else:
                    entity_totals = entity_counts.map(lambda value: sum(value.values()))
                    if not entity_totals.equals(df["named_entity_count"].astype(int)):
                        failures.append(
                            "named_entity_count does not match entity-label JSON"
                        )
                    expected_entity_counts = {
                        "person_entity_count": entity_counts.map(
                            lambda value: value.get("PERSON", 0)
                        ),
                        "location_entity_count": entity_counts.map(
                            lambda value: sum(value.get(label, 0) for label in ("GPE", "LOC", "FAC"))
                        ),
                        "organization_entity_count": entity_counts.map(
                            lambda value: value.get("ORG", 0)
                        ),
                        "product_entity_count": entity_counts.map(
                            lambda value: value.get("PRODUCT", 0)
                        ),
                    }
                    for column, expected in expected_entity_counts.items():
                        if not expected.equals(df[column].astype(int)):
                            failures.append(f"{column} does not match entity-label JSON")
                token_counts = df["nlp_token_count"].to_numpy(dtype=float)
                for count_column, ratio_column in (
                    ("noun_count", "noun_ratio"),
                    ("proper_noun_count", "proper_noun_ratio"),
                    ("verb_count", "verb_ratio"),
                    ("adjective_count", "adjective_ratio"),
                    ("adverb_count", "adverb_ratio"),
                ):
                    counts = df[count_column].to_numpy(dtype=float)
                    expected = np.divide(
                        counts,
                        token_counts,
                        out=np.zeros_like(counts),
                        where=token_counts != 0,
                    )
                    if not np.allclose(df[ratio_column], expected, atol=1e-6, rtol=0.0):
                        failures.append(f"{ratio_column} does not match {count_column}")
                alpha = df["nlp_alpha_token_count"]
                unique_alpha = df["nlp_unique_alpha_token_count"]
                if (alpha > df["nlp_token_count"]).any() or (unique_alpha > alpha).any():
                    failures.append("spaCy alpha/unique token counts are inconsistent")
                diversity = df["nlp_lexical_diversity"]
                if ((diversity < 0) | (diversity > 1) | ~np.isfinite(diversity)).any():
                    failures.append("nlp_lexical_diversity falls outside [0, 1]")
                if scoreable.any() and df.loc[scoreable, "nlp_sentence_count"].lt(1).any():
                    failures.append("scoreable rows are missing spaCy sentence boundaries")
                coverage = (
                    float(df.loc[scoreable, "nlp_token_count"].gt(0).mean())
                    if scoreable.any()
                    else 1.0
                )
                metrics["spacy_nonempty_coverage"] = coverage
                if coverage < 0.99:
                    failures.append(f"spaCy token coverage is only {coverage:.3f}")
                unique_counts = int(df.loc[scoreable, "nlp_token_count"].nunique())
                metrics["spacy_token_count_unique"] = unique_counts
                if scoreable.sum() >= 20 and unique_counts <= 1:
                    failures.append("nlp_token_count is frozen at one value")

    if topic_mode == "nmf":
        missing = sorted(set(TOPIC_COLUMNS).difference(df.columns))
        if missing:
            failures.append(f"missing topic columns: {missing}")
        else:
            vocabulary_counts = df["topic_vocabulary_terms"].astype(int)
            if (vocabulary_counts < 0).any():
                failures.append("topic_vocabulary_terms contains negative values")
            in_vocabulary = scoreable & vocabulary_counts.gt(0)
            valid_scores = df.loc[in_vocabulary, "topic_score"].dropna()
            metrics["topic_rows_scored"] = int(valid_scores.size)
            coverage = float(in_vocabulary.sum() / scoreable.sum()) if scoreable.any() else 1.0
            metrics["topic_vocabulary_coverage"] = coverage
            if coverage < minimum_topic_vocabulary_coverage:
                failures.append(f"topic vocabulary coverage is only {coverage:.3f}")
            if valid_scores.size != int(in_vocabulary.sum()):
                failures.append("some in-vocabulary rows have no topic score")
            if df.loc[scoreable & ~in_vocabulary, [
                "topic_id", "topic_label", "topic_weight", "topic_score",
                "topic_entropy", "topic_margin",
            ]].notna().any(axis=None):
                failures.append("out-of-vocabulary rows have topic assignments")
            if not valid_scores.empty:
                if ((valid_scores < 0) | (valid_scores > 1) | ~np.isfinite(valid_scores)).any():
                    failures.append("normalized topic scores fall outside [0, 1]")
                for column in ("topic_entropy", "topic_margin"):
                    values = df.loc[in_vocabulary, column]
                    if ((values < 0) | (values > 1) | ~np.isfinite(values)).any():
                        failures.append(f"{column} falls outside [0, 1]")
                weights = df.loc[in_vocabulary, "topic_weight"]
                if ((weights < 0) | ~np.isfinite(weights)).any():
                    failures.append("topic weights contain invalid values")
            unique_topics = int(df.loc[in_vocabulary, "topic_id"].nunique())
            metrics["topic_id_unique"] = unique_topics
            if in_vocabulary.sum() >= 20 and unique_topics <= 1:
                failures.append("topic assignments are frozen at one topic")

    if emotion_mode == "transformer":
        missing = sorted(set(EMOTION_COLUMNS).difference(df.columns))
        if missing:
            failures.append(f"missing emotion columns: {missing}")
        else:
            if df.loc[~scoreable, list(EMOTION_COLUMNS)].notna().any(axis=None):
                failures.append("unscoreable rows contain emotion output")
            scored = df.loc[scoreable]
            probabilities = scored[list(EMOTION_PROBABILITY_COLUMNS)]
            complete = probabilities.notna().all(axis=1)
            valid_scores = scored.loc[complete, "emotion_score"]
            metrics["emotion_rows_scored"] = int(valid_scores.size)
            if not complete.all():
                failures.append("some non-empty rows have incomplete emotion probabilities")
            if not valid_scores.empty:
                values = probabilities.loc[complete].to_numpy(dtype=float)
                if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                    failures.append("emotion probabilities contain invalid values")
                if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
                    failures.append("emotion probabilities do not sum to 1")
                labels = np.asarray(EMOTION_LABELS, dtype=object)[values.argmax(axis=1)]
                labels = np.char.upper(labels.astype(str))
                if not np.array_equal(scored.loc[complete, "emotion_label"].to_numpy(), labels):
                    failures.append("emotion labels do not match probability argmax")
                if not np.allclose(valid_scores, values.max(axis=1), atol=1e-6, rtol=0.0):
                    failures.append("emotion scores do not match maximum probability")
                normalized_entropy = scored.loc[complete, "emotion_normalized_entropy"]
                margin = scored.loc[complete, "emotion_margin"]
                if ((normalized_entropy < 0) | (normalized_entropy > 1)).any():
                    failures.append("emotion normalized entropy falls outside [0, 1]")
                if ((margin < 0) | (margin > 1)).any():
                    failures.append("emotion margin falls outside [0, 1]")
                unique_scores = int(valid_scores.nunique())
                metrics["emotion_score_unique"] = unique_scores
                if scoreable.sum() >= 20 and unique_scores <= 1:
                    failures.append("emotion scores are frozen at one value")

    metrics["passed"] = not failures
    metrics["failures"] = failures
    if failures:
        raise AssertionError("advanced NLP validation failed: " + "; ".join(failures))
    return metrics
