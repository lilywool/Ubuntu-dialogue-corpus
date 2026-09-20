"""Spark/Delta bronze-to-silver job for the Ubuntu dialogue corpus.

PySpark imports stay lazy so the repository remains importable and testable in
the isolated local pandas environment. The distributed path deliberately uses
two ``mapInPandas`` stages:

1. deterministic normalization, anonymization, lexicons, and residual extraction;
2. residual-label application, sentiment, spaCy, global-topic assignment, and
   optional transformer emotion scoring.

The separation allows the driver to build one global residual vocabulary and
fit one NMF model before broadcasting immutable artifacts to every partition.
Transformer and spaCy loaders already use module-level caches, so each Python
worker loads a requested model once and reuses it for subsequent Arrow batches.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping
from uuid import uuid4

import pandas as pd

# Workspace Python-script tasks are executed through ``exec`` and expose the
# absolute path as ``filename`` rather than ``__file__``. Normal Python uses
# ``__file__``. Resolve either form without changing global ``PYTHONPATH``.
def _resolve_repository_root(
    script_file: str | None,
    execution_filename: str | None,
) -> Path:
    for raw_path in (script_file, execution_filename):
        if not raw_path:
            continue
        resolved = Path(raw_path).resolve()
        for parent in resolved.parents:
            if (parent / "pipeline").is_dir() and (
                parent / "databricks_integration"
            ).is_dir():
                return parent
    current = Path.cwd().resolve()
    for parent in (current, *current.parents):
        if (parent / "pipeline").is_dir() and (
            parent / "databricks_integration"
        ).is_dir():
            return parent
    raise RuntimeError("could not locate the Ubuntu dialogue repository root")


_REPOSITORY_ROOT = _resolve_repository_root(
    globals().get("__file__"),
    locals().get("filename"),
)
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from pipeline.pipeline import PipelineConfig, run_pipeline
from pipeline.schema import FEATURE_SCHEMA_VERSION
from pipeline.sentiment_analysis import VADER_PROBABILITY_ATOL
from databricks_integration.run_metrics import (
    append_run_metrics_delta,
    append_stage_metric,
    build_run_metric_context,
    detect_worker_count,
)


UBUNTU_RELEASE_DATES = (
    "2004-10-26", "2005-04-08", "2005-10-12", "2006-06-01",
    "2006-10-26", "2007-04-19", "2007-10-18", "2008-04-24",
    "2008-10-30", "2009-04-23", "2009-10-29", "2010-04-29",
    "2010-10-10", "2011-04-28", "2011-10-13", "2012-04-26",
    "2012-10-18", "2013-04-25", "2013-10-17", "2014-04-17",
    "2014-10-23", "2015-04-23", "2015-10-22", "2016-04-21",
)

_CORE_LIST_COLUMNS = (
    "structural_matches", "tech_lexicon_matches", "slang_matches",
    "glued_matches", "residual_words",
)
_CORE_COUNT_COLUMNS = (
    "email_count", "domain_count", "phone_count", "ssn_count",
    "ipv4_count", "ipv6_count", "menu_path_count",
    "keyboard_shortcut_count", "tech_lexicon_match_count",
    "slang_match_count", "glued_match_count",
)
_SPACY_JSON_COLUMNS = (
    "nlp_pos_counts", "named_entities", "named_entity_label_counts",
    "nlp_tokens", "nlp_pos_tags",
)
_SPACY_COUNT_COLUMNS = (
    "nlp_token_count", "nlp_alpha_token_count",
    "nlp_unique_alpha_token_count", "nlp_sentence_count",
    "nlp_stopword_count", "nlp_negation_count", "nlp_punctuation_count",
    "nlp_exclamation_count", "nlp_question_count",
    "nlp_uppercase_token_count", "noun_count", "proper_noun_count",
    "verb_count", "adjective_count", "adverb_count", "named_entity_count",
    "person_entity_count", "location_entity_count",
    "organization_entity_count", "product_entity_count",
)
_SPACY_FLOAT_COLUMNS = (
    "nlp_lexical_diversity", "nlp_avg_sentence_tokens", "noun_ratio",
    "proper_noun_ratio", "verb_ratio", "adjective_ratio", "adverb_ratio",
)


def _spark_modules():
    try:
        from pyspark.sql import SparkSession, Window, functions as F, types as T
    except ImportError as exc:
        raise ImportError(
            "The distributed bronze-to-silver job requires a Databricks/Spark "
            "runtime; do not install PySpark globally for the local project."
        ) from exc
    return SparkSession, Window, F, T


def residual_stage_policy(use_api: bool = False, api_key: str | None = None) -> str:
    """Resolve the legacy API toggle without silently enabling a network call."""
    if not use_api:
        return "manual_review"
    return "api" if api_key or os.getenv("OPENAI_API_KEY") else "manual_review"


def build_residual_plan(use_api: bool = False, api_key: str | None = None) -> dict:
    """Describe whether residuals remain manual or use the opt-in API."""
    policy = residual_stage_policy(use_api=use_api, api_key=api_key)
    if policy == "manual_review":
        return {
            "status": "manual_review",
            "message": "Residual words are unchanged and ready for manual review.",
            "api_enabled": False,
        }
    return {
        "status": "api",
        "message": "One global residual vocabulary will be classified on the driver.",
        "api_enabled": True,
        "api_key_present": bool(api_key or os.getenv("OPENAI_API_KEY")),
    }


def run_bronze_to_silver(
    bronze_df: pd.DataFrame,
    *,
    text_col: str = "text_cleaned",
    source_text_col: str = "text",
    residual_policy: str = "manual_review",
    maximum_nonword_token_rate: float = 0.25,
    api_key: str | None = None,
    residual_api_model: str = "gpt-4o-mini-2024-07-18",
    residual_api_batch_size: int = 500,
    residual_api_timeout_seconds: float = 120.0,
    residual_api_max_retries: int = 2,
    sentiment_mode: str = "vader",
    transformer_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    transformer_revision: str = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7",
    transformer_batch_size: int = 32,
    transformer_inference_chunk_size: int = 4096,
    transformer_device: int | None = None,
    transformer_dtype: str = "float32",
    advanced_nlp: bool = False,
    run_spacy: bool = True,
    spacy_batch_size: int = 128,
    include_token_details: bool = False,
    topic_mode: str = "none",
    topic_fit_sample_size: int = 100_000,
    topic_batch_size: int = 50_000,
    topic_minimum_vocabulary_coverage: float = 0.50,
    emotion_mode: str = "none",
    spacy_model: str = "en_core_web_sm",
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
    emotion_revision: str = "cea2f78f197f0337186a0faa93e00ef93811f6eb",
    emotion_batch_size: int = 32,
    emotion_inference_chunk_size: int = 4096,
    emotion_device: int | None = None,
    emotion_dtype: str = "float32",
    validate_advanced_nlp: bool = True,
    workers: int | None = None,
    sample_size: int | None = None,
    sample_seed: int = 0,
) -> pd.DataFrame:
    """Keep the original local adapter for notebooks and local parity tests."""
    config = PipelineConfig(
        residual_policy=residual_policy,
        maximum_nonword_token_rate=maximum_nonword_token_rate,
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        residual_api_model=residual_api_model,
        residual_api_batch_size=residual_api_batch_size,
        residual_api_timeout_seconds=residual_api_timeout_seconds,
        residual_api_max_retries=residual_api_max_retries,
        sentiment_mode=sentiment_mode,
        transformer_model=transformer_model,
        transformer_revision=transformer_revision,
        transformer_batch_size=transformer_batch_size,
        transformer_inference_chunk_size=transformer_inference_chunk_size,
        transformer_device=transformer_device,
        transformer_dtype=transformer_dtype,
        advanced_nlp=advanced_nlp,
        run_spacy=run_spacy,
        spacy_batch_size=spacy_batch_size,
        include_token_details=include_token_details,
        topic_mode=topic_mode,
        topic_fit_sample_size=topic_fit_sample_size,
        topic_batch_size=topic_batch_size,
        topic_minimum_vocabulary_coverage=topic_minimum_vocabulary_coverage,
        emotion_mode=emotion_mode,
        spacy_model=spacy_model,
        emotion_model=emotion_model,
        emotion_revision=emotion_revision,
        emotion_batch_size=emotion_batch_size,
        emotion_inference_chunk_size=emotion_inference_chunk_size,
        emotion_device=emotion_device,
        emotion_dtype=emotion_dtype,
        validate_advanced_nlp=validate_advanced_nlp,
        parallel=False,
        workers=workers,
        sample_size=sample_size,
        sample_seed=sample_seed,
        output_dir="databricks_integration/outputs",
    )
    return run_pipeline(
        bronze_df, config=config, text_col=text_col,
        source_text_col=source_text_col,
    )


def read_bronze_spark(
    spark,
    *,
    input_table: str | None = None,
    input_delta_path: str | None = None,
    input_csv_path: str | None = None,
):
    """Read exactly one supported bronze source without collecting it."""
    sources = [input_table, input_delta_path, input_csv_path]
    if sum(value is not None for value in sources) != 1:
        raise ValueError(
            "provide exactly one of input_table, input_delta_path, or input_csv_path"
        )
    if input_table:
        return spark.table(input_table)
    if input_delta_path:
        return spark.read.format("delta").load(input_delta_path)
    return (
        spark.read.option("header", True)
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        .option("mode", "FAILFAST")
        .csv(input_csv_path)
    )


def _require_columns(frame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{context} requires columns: {missing}")


def _normalized_column_name(column: str) -> str:
    return re.sub(r"\s+", "_", str(column).strip().lower())


def prepare_bronze_spark(bronze_df):
    """Create stable message/conversation keys and notebook-compatible features."""
    _SparkSession, Window, F, _T = _spark_modules()
    normalized_names = [_normalized_column_name(column) for column in bronze_df.columns]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("bronze column normalization would create duplicate names")
    frame = bronze_df
    for original, normalized in zip(bronze_df.columns, normalized_names):
        if original != normalized:
            frame = frame.withColumnRenamed(original, normalized)
    if "dialogueid" in frame.columns and "dialogue_id" not in frame.columns:
        frame = frame.withColumnRenamed("dialogueid", "dialogue_id")

    _require_columns(frame, ["date", "from", "text"], "bronze ingestion")
    frame = (
        frame.withColumn("text", F.col("text").cast("string"))
        .withColumn("from", F.when(F.trim(F.col("from")) == "", None).otherwise(F.col("from")))
    )
    if "to" not in frame.columns:
        frame = frame.withColumn("to", F.lit(None).cast("string"))
    else:
        frame = frame.withColumn(
            "to", F.when(F.trim(F.col("to")) == "", None).otherwise(F.col("to"))
        )
    frame = frame.withColumn(
        "date",
        F.coalesce(
            F.to_timestamp("date", "yyyy-MM-dd'T'HH:mm:ss.SSSX"),
            F.to_timestamp("date"),
        ),
    ).where(F.col("text").isNotNull() & F.col("date").isNotNull())
    frame = frame.dropDuplicates(["date", "from", "text"])

    if "dialogue_id" in frame.columns:
        frame = frame.withColumn(
            "dialogue_id",
            F.regexp_replace(F.col("dialogue_id").cast("string"), r"\.tsv$", "").cast("long"),
        )
    if "folder" in frame.columns:
        frame = frame.withColumn("folder", F.col("folder").cast("long"))

    if "conversation_id" not in frame.columns:
        _require_columns(frame, ["folder", "dialogue_id"], "conversation identity")
        frame = frame.withColumn(
            "conversation_id",
            F.sha2(
                F.concat_ws(
                    "::",
                    F.coalesce(F.col("folder").cast("string"), F.lit("<NULL>")),
                    F.coalesce(F.col("dialogue_id").cast("string"), F.lit("<NULL>")),
                ),
                256,
            ),
        )

    participants = frame.groupBy("conversation_id").agg(
        F.collect_set("from").alias("_known_senders")
    )
    frame = frame.join(participants, "conversation_id", "left")
    possible_sender = F.filter(
        F.col("_known_senders"), lambda value: value != F.col("to")
    )
    frame = frame.withColumn(
        "from",
        F.when(
            F.col("from").isNull()
            & F.col("to").isNotNull()
            & (F.size(F.col("_known_senders")) == 2)
            & (F.size(possible_sender) == 1),
            F.element_at(possible_sender, 1),
        ).otherwise(F.col("from")),
    ).where(F.col("from").isNotNull()).drop("_known_senders")

    senders = frame.groupBy("conversation_id").agg(
        F.collect_set("from").alias("_senders"),
        F.countDistinct("from").alias("_sender_count"),
    )
    frame = frame.join(senders, "conversation_id", "left")
    frame = frame.withColumn(
        "_tie_breaker",
        F.xxhash64(
            F.coalesce(F.col("from"), F.lit("")),
            F.coalesce(F.col("to"), F.lit("")),
            F.coalesce(F.col("text"), F.lit("")),
        ),
    )
    conversation_window = Window.partitionBy("conversation_id").orderBy(
        F.col("date").asc(), F.col("_tie_breaker").asc()
    )
    frame = frame.withColumn("turn_count", (F.row_number().over(conversation_window) - 1).cast("long"))
    other_sender = F.filter(F.col("_senders"), lambda value: value != F.col("from"))
    frame = frame.withColumn(
        "to",
        F.when(F.col("to").isNotNull(), F.col("to"))
        .when(F.col("_sender_count") == 1, F.lit("__UNANSWERED__"))
        .when(F.col("turn_count") == 0, F.lit("__ALL__"))
        .when(F.size(other_sender) == 1, F.element_at(other_sender, 1))
        .otherwise(F.lit("__ALL__")),
    )

    frame = frame.withColumn(
        "message_id",
        F.sha2(
            F.concat_ws(
                "::",
                F.col("conversation_id").cast("string"),
                F.col("date").cast("string"),
                F.coalesce(F.col("from"), F.lit("<NULL>")),
                F.coalesce(F.col("to"), F.lit("<NULL>")),
                F.coalesce(F.col("text"), F.lit("<NULL>")),
            ),
            256,
        ),
    )
    full_conversation = conversation_window.rowsBetween(
        Window.unboundedPreceding, Window.unboundedFollowing
    )
    frame = frame.withColumn(
        "is_op", F.col("from") == F.first("from", ignorenulls=True).over(full_conversation)
    )
    prior_date = F.lag("date").over(conversation_window)
    prior_sender = F.lag("from").over(conversation_window)
    frame = frame.withColumn(
        "response_gap_mins",
        (F.col("date").cast("long") - prior_date.cast("long")) / F.lit(60.0),
    ).withColumn(
        "response_gap_mins_between_speakers",
        F.when(F.col("from") != prior_sender, F.col("response_gap_mins")),
    )
    user_window = Window.partitionBy("from").orderBy(
        F.col("date"), F.col("message_id")
    )
    frame = frame.withColumn(
        "user_message_count", F.row_number().over(user_window).cast("long")
    )
    trimmed_text = F.trim(F.col("text"))
    frame = frame.withColumn("text_length", F.length("text").cast("long")).withColumn(
        "word_count",
        F.when(trimmed_text == "", F.lit(0)).otherwise(
            F.size(F.split(trimmed_text, r"\s+"))
        ).cast("long"),
    )
    frame = frame.withColumn("year", F.year("date").cast("int")).withColumn(
        "month", F.month("date").cast("int")
    ).withColumn("day", F.dayofmonth("date").cast("int")).withColumn(
        "day_of_week", F.date_format("date", "EEEE")
    ).withColumn("hour", F.hour("date").cast("int"))

    monthly_window = Window.partitionBy("from", "year", "month")
    frame = frame.withColumn(
        "user_messages_this_month", F.count(F.lit(1)).over(monthly_window).cast("long")
    ).withColumn(
        "user_tier",
        F.when(F.col("user_messages_this_month") <= 10, "occasional")
        .when(F.col("user_messages_this_month") <= 50, "regular")
        .when(F.col("user_messages_this_month") <= 150, "active")
        .when(F.col("user_messages_this_month") <= 500, "frequent")
        .otherwise("power_user"),
    )

    release_array = F.array(*[F.to_date(F.lit(value)) for value in UBUNTU_RELEASE_DATES])
    message_day = F.to_date("date")
    prior_release = F.array_max(F.filter(release_array, lambda value: value <= message_day))
    next_release = F.array_min(F.filter(release_array, lambda value: value >= message_day))
    frame = frame.withColumn(
        "days_since_release", F.datediff(message_day, prior_release).cast("int")
    ).withColumn(
        "days_until_release", F.datediff(next_release, message_day).cast("int")
    )

    def release_bucket(column):
        return (
            F.when(column.isNull(), None)
            .when(column <= 7, "0-1wk").when(column <= 14, "1-2wk")
            .when(column <= 21, "2-3wk").when(column <= 28, "3-4wk")
            .when(column <= 42, "4-6wk").when(column <= 56, "6-8wk")
            .otherwise("8wk+")
        )

    frame = frame.withColumn(
        "days_since_release_bucket", release_bucket(F.col("days_since_release"))
    ).withColumn(
        "days_until_release_bucket", release_bucket(F.col("days_until_release"))
    ).withColumn(
        "text_length_bucket",
        F.when(F.col("text_length") <= 20, "very_short")
        .when(F.col("text_length") <= 50, "short")
        .when(F.col("text_length") <= 100, "medium")
        .when(F.col("text_length") <= 200, "long").otherwise("very_long"),
    ).withColumn(
        "word_count_bucket",
        F.when(F.col("word_count") <= 5, "very_short")
        .when(F.col("word_count") <= 10, "short")
        .when(F.col("word_count") <= 20, "medium")
        .when(F.col("word_count") <= 40, "long").otherwise("very_long"),
    ).withColumn(
        "response_gap_bucket",
        F.when(F.col("response_gap_mins_between_speakers").isNull(), None)
        .when(F.col("response_gap_mins_between_speakers") <= 5, "<5min")
        .when(F.col("response_gap_mins_between_speakers") <= 30, "5-30min")
        .when(F.col("response_gap_mins_between_speakers") <= 120, "30min-2hr")
        .when(F.col("response_gap_mins_between_speakers") <= 1440, "2hr-1day")
        .otherwise("1day+"),
    )

    conversation_summary = frame.groupBy("conversation_id").agg(
        F.min("date").alias("conversation_start"),
        F.max("date").alias("conversation_end"),
        F.count(F.lit(1)).alias("conversation_message_count"),
        F.countDistinct("from").alias("_conversation_sender_count"),
    )
    participants = frame.select(
        "conversation_id", F.col("from").alias("_participant")
    ).unionByName(frame.select(
        "conversation_id", F.col("to").alias("_participant")
    )).where(
        F.col("_participant").isNotNull()
        & ~F.col("_participant").isin("__ALL__", "__UNANSWERED__")
    )
    participant_counts = participants.groupBy("conversation_id").agg(
        F.countDistinct("_participant").alias("conversation_participant_count")
    )
    conversation_summary = conversation_summary.join(
        participant_counts, "conversation_id", "left"
    ).withColumn(
        "conversation_duration_mins",
        (F.col("conversation_end").cast("long") - F.col("conversation_start").cast("long"))
        / F.lit(60.0),
    ).withColumn(
        "conversation_was_answered", F.col("_conversation_sender_count") > 1
    ).drop("_conversation_sender_count")
    return frame.join(conversation_summary, "conversation_id", "left").drop(
        "_senders", "_sender_count", "_tie_breaker"
    )


def sample_bronze_spark(frame, *, sample_size: int | None, seed: int):
    """Select stable message IDs without partition-local random-number state."""
    if sample_size is None:
        return frame
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    _SparkSession, _Window, F, _T = _spark_modules()
    _require_columns(frame, ["message_id"], "distributed sample")
    return (
        frame.withColumn(
            "_sample_hash", F.xxhash64(F.lit(int(seed)), F.col("message_id").cast("string"))
        )
        .orderBy("_sample_hash", "message_id")
        .limit(sample_size)
        .drop("_sample_hash")
    )


def transform_core_partition(
    frame: pd.DataFrame,
    *,
    text_col: str = "text_cleaned",
    source_text_col: str = "text",
) -> pd.DataFrame:
    """Run partition-safe deterministic stages with local worker count fixed at one."""
    from pipeline.apply_lexicons import apply_lexicons
    from pipeline.data_preparation import optimize_dtypes, prepare_text_column
    from pipeline.residual_stage import extract_residual_vocabulary
    from pipeline.validation import validate_core_pipeline

    result = optimize_dtypes(frame.copy())
    result = prepare_text_column(
        result, text_col=text_col, source_text_col=source_text_col
    )
    result, _matched, _tally = apply_lexicons(
        result, text_col=text_col, workers=1, show_progress=False
    )
    result, _counts = extract_residual_vocabulary(
        result, text_col=text_col, workers=1
    )
    validate_core_pipeline(
        result, text_col=text_col, expected_rows=len(frame),
        expected_index=frame.index,
    )
    return result


def build_residual_labels_from_counts(
    counts: Mapping[str, int],
    *,
    policy: str,
    api_key: str | None = None,
    api_options: Mapping[str, Any] | None = None,
    api_provenance: dict[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve one global vocabulary; API calls occur only on the driver."""
    from pipeline.residual_stage import _classify_residual_words_api, _reviewed_labels

    if policy not in {"manual_review", "reviewed", "api"}:
        raise ValueError("residual policy must be 'manual_review', 'reviewed', or 'api'")
    normalized_counts: Counter[str] = Counter()
    for word, count in counts.items():
        normalized_counts[str(word).lower()] += int(count)
    words = sorted(normalized_counts)
    if policy == "manual_review" or not words:
        return {}, {}
    if policy == "reviewed":
        labels = _reviewed_labels(words)
        return labels, {word: "reviewed" for word in labels}
    if not api_key:
        raise ValueError("residual policy 'api' requires an API key")
    api_labels = _classify_residual_words_api(
        words,
        normalized_counts,
        api_key=api_key,
        provenance=api_provenance,
        **dict(api_options or {}),
    )
    return api_labels, {word: "api_candidate" for word in api_labels}


def collect_global_residual_labels(
    core_df,
    *,
    policy: str,
    api_key: str | None = None,
    maximum_vocabulary: int = 500_000,
    api_options: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Aggregate residual counts in Spark, then collect only unique vocabulary."""
    _SparkSession, _Window, F, _T = _spark_modules()
    if policy == "manual_review":
        return {}, {
            "policy": policy,
            "vocabulary_size": None,
            "labeled_terms": 0,
            "sources": {},
            "note": "residuals remain on message rows; no driver collection performed",
        }
    counts_df = (
        core_df.select(F.explode_outer("residual_words").alias("word"))
        .where(F.col("word").isNotNull())
        .groupBy("word").count()
    )
    vocabulary_size = counts_df.count()
    if vocabulary_size > maximum_vocabulary:
        raise MemoryError(
            f"residual vocabulary has {vocabulary_size:,} terms; configured driver "
            f"limit is {maximum_vocabulary:,}"
        )
    counts = {row["word"]: int(row["count"]) for row in counts_df.collect()}
    api_provenance: dict[str, Any] = {}
    labels, sources = build_residual_labels_from_counts(
        counts,
        policy=policy,
        api_key=api_key,
        api_options=api_options,
        api_provenance=api_provenance,
    )
    source_counts = Counter(sources.values())
    metadata = {
        "policy": policy,
        "vocabulary_size": vocabulary_size,
        "labeled_terms": len(labels),
        "sources": dict(sorted(source_counts.items())),
    }
    if api_provenance:
        metadata["api_provenance"] = api_provenance
    return labels, metadata


def fit_global_topic_model_spark(
    core_df,
    *,
    config: PipelineConfig,
    residual_labels: Mapping[str, str],
    text_col: str,
):
    """Fit one deterministic driver-side NMF bundle from a stable Spark sample."""
    if config.topic_mode != "nmf":
        return None
    _SparkSession, _Window, F, _T = _spark_modules()
    from pipeline.residual_stage import apply_api_labels
    from pipeline.sentiment_analysis import strip_label_tokens_series
    from pipeline.topic_modeling import fit_topic_model

    _require_columns(core_df, ["message_id", text_col], "topic fitting")
    sample = (
        core_df.withColumn(
            "_topic_hash",
            F.xxhash64(F.lit(int(config.sample_seed)), F.col("message_id").cast("string")),
        )
        .orderBy("_topic_hash", "message_id")
        .limit(config.topic_fit_sample_size)
        .select(text_col)
        .toPandas()
    )
    if residual_labels:
        sample = apply_api_labels(sample, dict(residual_labels), text_col=text_col)
    texts = strip_label_tokens_series(sample[text_col].fillna("").astype(str))
    return fit_topic_model(
        texts,
        fit_sample_size=config.topic_fit_sample_size,
        random_state=config.sample_seed,
    )


def transform_enrichment_partition(
    frame: pd.DataFrame,
    *,
    config: PipelineConfig,
    residual_labels: Mapping[str, str] | None = None,
    topic_model=None,
    text_col: str = "text_cleaned",
) -> pd.DataFrame:
    """Apply broadcast artifacts and optional NLP without partition-local fitting."""
    from pipeline.residual_stage import apply_api_labels
    from pipeline.validation import validate_core_pipeline

    # Spark/Arrow commonly returns array columns to pandas as numpy arrays.
    # Normalize only that distributed representation; retain the canonical
    # local list/tuple records so local and distributed parity remains exact.
    result = frame.copy()
    if any(
        column in result
        and not result[column].map(lambda value: isinstance(value, list)).all()
        for column in _CORE_LIST_COLUMNS
    ):
        result = _normalize_match_lists(result)
    if residual_labels:
        result = apply_api_labels(result, dict(residual_labels), text_col=text_col)
    validate_core_pipeline(
        result, text_col=text_col, expected_rows=len(frame),
        expected_index=frame.index,
    )
    if config.sentiment_mode != "none":
        from pipeline.sentiment_analysis import analyze_sentiment

        result = analyze_sentiment(
            result,
            text_column=text_col,
            mode=config.sentiment_mode,
            transformer_model=config.transformer_model,
            transformer_revision=config.transformer_revision,
            transformer_batch_size=config.transformer_batch_size,
            transformer_inference_chunk_size=config.transformer_inference_chunk_size,
            transformer_device=config.transformer_device,
            transformer_dtype=config.transformer_dtype,
        )
    if config.advanced_nlp:
        if config.topic_mode == "nmf" and topic_model is None:
            raise ValueError(
                "distributed topic assignment requires one globally fitted topic model"
            )
        from pipeline.advanced_nlp import annotate_messages, validate_advanced_nlp

        result = annotate_messages(
            result,
            text_col=text_col,
            run_spacy=config.run_spacy,
            spacy_model=config.spacy_model,
            spacy_batch_size=config.spacy_batch_size,
            workers=1,
            include_token_details=config.include_token_details,
            topic_mode=config.topic_mode,
            topic_model=topic_model,
            topic_fit_sample_size=config.topic_fit_sample_size,
            topic_batch_size=config.topic_batch_size,
            topic_random_state=config.sample_seed,
            emotion_mode=config.emotion_mode,
            emotion_model=config.emotion_model,
            emotion_revision=config.emotion_revision,
            emotion_batch_size=config.emotion_batch_size,
            emotion_inference_chunk_size=config.emotion_inference_chunk_size,
            emotion_device=config.emotion_device,
            emotion_dtype=config.emotion_dtype,
        )
        if config.validate_advanced_nlp:
            validate_advanced_nlp(
                result,
                text_col=text_col,
                sentiment_mode=config.sentiment_mode,
                run_spacy=config.run_spacy,
                topic_mode=config.topic_mode,
                emotion_mode=config.emotion_mode,
                require_technical_lexicon=True,
                minimum_topic_vocabulary_coverage=config.topic_minimum_vocabulary_coverage,
            )
    return result


def expected_feature_columns(config: PipelineConfig) -> set[str]:
    """Return the stage-owned columns expected from a configured partition."""
    from pipeline.advanced_nlp import EMOTION_COLUMNS, SPACY_COLUMNS, TOPIC_COLUMNS
    from pipeline.sentiment_analysis import TRANSFORMER_COLUMNS, VADER_COLUMNS

    columns = {
        "text_cleaned", *_CORE_LIST_COLUMNS, *_CORE_COUNT_COLUMNS,
        "has_lexicon_match", "has_residual",
    }
    if config.sentiment_mode in {"vader", "both"}:
        columns.update(VADER_COLUMNS)
    if config.sentiment_mode in {"transformer", "both"}:
        columns.update(TRANSFORMER_COLUMNS)
    if config.advanced_nlp and config.run_spacy:
        columns.update(
            column for column in SPACY_COLUMNS
            if config.include_token_details or column not in {"nlp_tokens", "nlp_pos_tags"}
        )
    if config.advanced_nlp and config.topic_mode == "nmf":
        columns.update(TOPIC_COLUMNS)
    if config.advanced_nlp and config.emotion_mode == "transformer":
        columns.update(EMOTION_COLUMNS)
    return columns


def _schema_with_features(input_schema, config: PipelineConfig, *, core_only: bool):
    _SparkSession, _Window, _F, T = _spark_modules()
    specs: dict[str, Any] = {
        "text_cleaned": T.StringType(),
        "structural_matches": T.ArrayType(T.ArrayType(T.StringType())),
        "tech_lexicon_matches": T.ArrayType(T.ArrayType(T.StringType())),
        "slang_matches": T.ArrayType(T.ArrayType(T.StringType())),
        "glued_matches": T.ArrayType(T.StructType([
            T.StructField("term", T.StringType()),
            T.StructField("category", T.StringType()),
            T.StructField("glue", T.StringType()),
            T.StructField("run", T.StringType()),
        ])),
        "residual_words": T.ArrayType(T.StringType()),
        "has_lexicon_match": T.BooleanType(),
        "has_residual": T.BooleanType(),
    }
    specs.update({column: T.LongType() for column in _CORE_COUNT_COLUMNS})
    if not core_only:
        from pipeline.advanced_nlp import EMOTION_COLUMNS
        from pipeline.sentiment_analysis import TRANSFORMER_COLUMNS, VADER_COLUMNS

        if config.sentiment_mode in {"vader", "both"}:
            specs.update({
                column: T.FloatType()
                for column in VADER_COLUMNS if column != "vader_label"
            })
            specs["vader_label"] = T.StringType()
        if config.sentiment_mode in {"transformer", "both"}:
            specs.update({
                column: T.FloatType()
                for column in TRANSFORMER_COLUMNS if column != "transformer_label"
            })
            specs["transformer_label"] = T.StringType()
        if config.advanced_nlp and config.run_spacy:
            json_columns = (
                _SPACY_JSON_COLUMNS if config.include_token_details
                else tuple(
                    column for column in _SPACY_JSON_COLUMNS
                    if column not in {"nlp_tokens", "nlp_pos_tags"}
                )
            )
            specs.update({column: T.StringType() for column in json_columns})
            specs.update({column: T.LongType() for column in _SPACY_COUNT_COLUMNS})
            specs.update({column: T.FloatType() for column in _SPACY_FLOAT_COLUMNS})
        if config.advanced_nlp and config.topic_mode == "nmf":
            specs.update({
                "topic_id": T.IntegerType(), "topic_label": T.StringType(),
                "topic_weight": T.FloatType(), "topic_score": T.FloatType(),
                "topic_entropy": T.FloatType(), "topic_margin": T.FloatType(),
                "topic_vocabulary_terms": T.LongType(),
            })
        if config.advanced_nlp and config.emotion_mode == "transformer":
            specs.update({
                column: T.FloatType()
                for column in EMOTION_COLUMNS if column != "emotion_label"
            })
            specs["emotion_label"] = T.StringType()

    fields = []
    seen = set()
    for field in input_schema.fields:
        fields.append(T.StructField(
            field.name, specs.get(field.name, field.dataType), True,
            field.metadata,
        ))
        seen.add(field.name)
    for name, data_type in specs.items():
        if name not in seen:
            fields.append(T.StructField(name, data_type, True))
    return T.StructType(fields)


def _normalize_match_lists(frame: pd.DataFrame) -> pd.DataFrame:
    def items(value: Any) -> list[Any]:
        if value is None or value is pd.NA:
            return []
        if isinstance(value, float) and pd.isna(value):
            return []
        return list(value)

    def glued_record(value: Any) -> dict[str, str | None]:
        if hasattr(value, "asDict"):
            value = value.asDict(recursive=True)
        if not isinstance(value, Mapping):
            raise TypeError(
                "glued_matches must contain mappings or Spark Row values"
            )
        return {
            "term": str(value.get("term", "")),
            "category": (
                None if value.get("category") is None else str(value["category"])
            ),
            "glue": str(value.get("glue", "")),
            "run": str(value.get("run", "")),
        }

    result = frame.copy()
    for column in ("structural_matches", "tech_lexicon_matches", "slang_matches"):
        if column in result:
            result[column] = result[column].map(
                lambda values: [
                    [None if item is None else str(item) for item in value]
                    for value in items(values)
                ]
            )
    if "glued_matches" in result:
        result["glued_matches"] = result["glued_matches"].map(
            lambda values: [glued_record(value) for value in items(values)]
        )
    if "residual_words" in result:
        result["residual_words"] = result["residual_words"].map(
            lambda values: [str(value) for value in items(values)]
        )
    return result


def _align_partition_output(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = _normalize_match_lists(frame)
    for column in columns:
        if column not in result:
            result[column] = None
    for column in result.select_dtypes(include=["category"]).columns:
        converted = result[column].astype("string").astype(object)
        result[column] = converted.where(converted.notna(), None)
    return result.loc[:, columns]


def _core_iterator(text_col: str, source_text_col: str, columns: list[str]):
    def transform(iterator: Iterable[pd.DataFrame]):
        for frame in iterator:
            yield _align_partition_output(
                transform_core_partition(
                    frame, text_col=text_col, source_text_col=source_text_col
                ),
                columns,
            )
    return transform


def _transform_enrichment_iterator(
    iterator: Iterable[pd.DataFrame],
    *,
    config: PipelineConfig,
    residual_payload,
    topic_payload,
    text_col: str,
    columns: list[str],
):
    """Run enrichment with an executor-safe, secret-free configuration."""
    labels = (
        residual_payload.value
        if hasattr(residual_payload, "value")
        else residual_payload or {}
    )
    topic_model = (
        topic_payload.value
        if hasattr(topic_payload, "value")
        else topic_payload
    )
    for frame in iterator:
        yield _align_partition_output(
            transform_enrichment_partition(
                frame,
                config=config,
                residual_labels=labels,
                topic_model=topic_model,
                text_col=text_col,
            ),
            columns,
        )


def _enrichment_iterator(
    config: PipelineConfig,
    residual_payload,
    topic_payload,
    text_col: str,
    columns: list[str],
):
    """Build the callable serialized by Spark without driver-only secrets."""
    executor_config = replace(config, api_key=None)
    return partial(
        _transform_enrichment_iterator,
        config=executor_config,
        residual_payload=residual_payload,
        topic_payload=topic_payload,
        text_col=text_col,
        columns=columns,
    )


def _resolve_materialization_mode(mode: str) -> str:
    """Choose a reusable intermediate strategy compatible with the Spark API."""
    if mode not in {"auto", "persist", "delta", "none"}:
        raise ValueError(
            "materialization_mode must be 'auto', 'persist', 'delta', or 'none'"
        )
    if mode != "auto":
        return mode
    try:
        from pyspark.sql.utils import is_remote

        return "delta" if is_remote() else "persist"
    except (ImportError, TypeError):
        return "persist"


def _qualified_scratch_table(schema: str, stage: str, run_id: str) -> str:
    """Return a validated managed-table name for serverless intermediates."""
    parts = str(schema).split(".")
    if len(parts) not in {1, 2} or any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) for part in parts
    ):
        raise ValueError(
            "materialization_schema must be a simple schema or catalog.schema name"
        )
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("materialization run_id must be a UUID hex value")
    return ".".join([*parts, f"_ubuntu_{stage}_{run_id}"])


def _drop_managed_table(spark, table: str) -> None:
    quoted = ".".join(f"`{part}`" for part in table.split("."))
    spark.sql(f"DROP TABLE IF EXISTS {quoted}")


def release_silver_resources(spark, silver_df, metadata: Mapping[str, Any]) -> None:
    """Release classic cache state or serverless scratch Delta tables."""
    materialization = dict(metadata.get("materialization") or {})
    mode = materialization.get("mode")
    if mode == "persist":
        silver_df.unpersist()
    for table in materialization.get("temporary_tables", []):
        _drop_managed_table(spark, str(table))


def validate_silver_spark(
    silver_df,
    *,
    expected_rows: int,
    config: PipelineConfig,
    text_col: str = "text_cleaned",
) -> dict[str, Any]:
    """Run whole-job invariants after every partition has been recombined."""
    _SparkSession, _Window, F, _T = _spark_modules()
    failures: list[str] = []
    required = expected_feature_columns(config) | {"message_id"}
    missing = sorted(required.difference(silver_df.columns))
    if missing:
        failures.append(f"missing silver columns: {missing}")
    rows = silver_df.count()
    if rows != expected_rows:
        failures.append(f"row count changed from {expected_rows} to {rows}")
    if "message_id" in silver_df.columns:
        invalid_ids = silver_df.where(F.col("message_id").isNull()).limit(1).count()
        duplicates = (
            silver_df.groupBy("message_id").count()
            .where(F.col("count") > 1).limit(1).count()
        )
        if invalid_ids or duplicates:
            failures.append("message_id must be non-null and unique")
    if text_col in silver_df.columns and silver_df.where(F.col(text_col).isNull()).limit(1).count():
        failures.append("cleaned text contains null values")

    nonword_token_rate = None
    nonword_tokens = None
    lexical_tokens = None
    if (
        config.residual_policy in {"reviewed", "api"}
        and text_col in silver_df.columns
    ):
        if (
            config.maximum_nonword_token_rate < 0
            or config.maximum_nonword_token_rate > 1
        ):
            raise ValueError("maximum_nonword_token_rate must be between 0 and 1")
        normalized_text = F.trim(F.regexp_replace(
            F.coalesce(F.col(text_col), F.lit("")),
            r"[^\p{L}\p{N}_'-]+",
            " ",
        ))
        tokens = F.when(
            F.length(normalized_text) == 0,
            F.array().cast("array<string>"),
        ).otherwise(F.split(normalized_text, r"\s+"))
        token_stats = silver_df.select(tokens.alias("tokens")).agg(
            F.sum(F.size("tokens")).alias("lexical_tokens"),
            F.sum(F.size(F.filter(
                F.col("tokens"),
                lambda token: F.upper(token) == F.lit("NONWORD"),
            ))).alias("nonword_tokens"),
        ).first()
        lexical_tokens = int(token_stats["lexical_tokens"] or 0)
        nonword_tokens = int(token_stats["nonword_tokens"] or 0)
        nonword_token_rate = (
            nonword_tokens / lexical_tokens if lexical_tokens else 0.0
        )
        if (
            lexical_tokens >= 100
            and nonword_token_rate > config.maximum_nonword_token_rate
        ):
            failures.append(
                "NONWORD token rate "
                f"{nonword_token_rate:.3%} exceeds configured maximum "
                f"{config.maximum_nonword_token_rate:.3%}"
            )

    for matches, count in (
        ("tech_lexicon_matches", "tech_lexicon_match_count"),
        ("slang_matches", "slang_match_count"),
        ("glued_matches", "glued_match_count"),
    ):
        if {matches, count}.issubset(silver_df.columns):
            mismatch = silver_df.where(
                F.col(matches).isNull() | (F.size(F.col(matches)) != F.col(count))
            ).limit(1).count()
            if mismatch:
                failures.append(f"{count} does not match {matches}")

    available_counts = [column for column in _CORE_COUNT_COLUMNS if column in silver_df.columns]
    if available_counts:
        negative = silver_df.where(
            F.least(*[F.col(column) for column in available_counts]) < 0
        ).limit(1).count()
        if negative:
            failures.append("core count columns contain negative values")
    if {"residual_words", "has_residual"}.issubset(silver_df.columns):
        inconsistent = silver_df.where(
            F.col("has_residual") != (F.size(F.col("residual_words")) > 0)
        ).limit(1).count()
        if inconsistent:
            failures.append("has_residual is inconsistent with residual_words")
    lexicon_inputs = {
        "structural_matches", "tech_lexicon_match_count", "slang_match_count",
        "glued_match_count", "has_lexicon_match",
    }
    if lexicon_inputs.issubset(silver_df.columns) and available_counts:
        anonymized_counts = [
            column for column in (
                "email_count", "domain_count", "phone_count", "ssn_count",
                "ipv4_count", "ipv6_count", "menu_path_count",
                "keyboard_shortcut_count",
            ) if column in silver_df.columns
        ]
        anonymized_total = sum((F.col(column) for column in anonymized_counts), F.lit(0))
        expected_flag = (
            (F.size(F.col("structural_matches")) > 0)
            | (F.col("tech_lexicon_match_count") > 0)
            | (F.col("slang_match_count") > 0)
            | (F.col("glued_match_count") > 0)
            | (anonymized_total > 0)
        )
        inconsistent = silver_df.where(
            F.col("has_lexicon_match") != expected_flag
        ).limit(1).count()
        if inconsistent:
            failures.append("has_lexicon_match is inconsistent with match columns")

    vader_unique_scores = None
    if config.sentiment_mode in {"vader", "both"} and {
        "vader_negative", "vader_neutral", "vader_positive"
    }.issubset(silver_df.columns):
        probability_columns = ["vader_negative", "vader_neutral", "vader_positive"]
        total = sum((F.col(column) for column in probability_columns), F.lit(0.0))
        invalid = silver_df.where(
            total.isNotNull()
            & (
                (F.abs(total - F.lit(1.0)) > F.lit(VADER_PROBABILITY_ATOL))
                | (F.greatest(*[F.col(column) for column in probability_columns]) > 1)
                | (F.least(*[F.col(column) for column in probability_columns]) < 0)
            )
        ).limit(1).count()
        if invalid:
            failures.append("VADER probabilities are invalid or do not sum to 1")
        out_of_range = silver_df.where(
            (F.col("vader_compound") < -1) | (F.col("vader_compound") > 1)
        ).limit(1).count()
        if out_of_range:
            failures.append("VADER compound scores fall outside [-1, 1]")
        vader_stats = silver_df.where(F.col("vader_compound").isNotNull()).agg(
            F.count(F.lit(1)).alias("n"),
            F.countDistinct("vader_compound").alias("unique"),
        ).first()
        vader_unique_scores = int(vader_stats["unique"])
        if vader_stats["n"] >= 100 and vader_unique_scores < 10:
            failures.append("VADER scores have suspiciously low global diversity")

    transformer_unique_scores = None
    if config.sentiment_mode in {"transformer", "both"} and {
        "transformer_negative", "transformer_neutral", "transformer_positive"
    }.issubset(silver_df.columns):
        probability_columns = [
            "transformer_negative", "transformer_neutral", "transformer_positive"
        ]
        total = sum((F.col(column) for column in probability_columns), F.lit(0.0))
        invalid = silver_df.where(
            total.isNotNull()
            & (
                (F.abs(total - F.lit(1.0)) > F.lit(1e-5))
                | (F.greatest(*[F.col(column) for column in probability_columns]) > 1)
                | (F.least(*[F.col(column) for column in probability_columns]) < 0)
            )
        ).limit(1).count()
        if invalid:
            failures.append(
                "transformer probabilities are invalid or do not sum to 1"
            )
        out_of_range = silver_df.where(
            (F.col("transformer_score") < 0) | (F.col("transformer_score") > 1)
            | (F.col("transformer_expected_sentiment") < -1)
            | (F.col("transformer_expected_sentiment") > 1)
        ).limit(1).count()
        if out_of_range:
            failures.append("transformer scores fall outside their valid ranges")
        scored = silver_df.where(F.col("transformer_score").isNotNull())
        score_stats = scored.agg(
            F.count(F.lit(1)).alias("n"),
            F.countDistinct("transformer_score").alias("unique"),
        ).first()
        transformer_unique_scores = int(score_stats["unique"])
        if score_stats["n"] >= 100 and transformer_unique_scores < 10:
            failures.append(
                "transformer scores have suspiciously low global diversity"
            )

    spacy_coverage = None
    if config.advanced_nlp and config.run_spacy and "nlp_token_count" in silver_df.columns:
        spacy_stats = silver_df.agg(
            F.count(F.when(F.length(F.trim(F.col(text_col))) > 0, 1)).alias("scoreable"),
            F.count(F.when(F.col("nlp_token_count") > 0, 1)).alias("tokenized"),
            F.min("nlp_token_count").alias("min_tokens"),
            F.max("nlp_token_count").alias("max_tokens"),
        ).first()
        spacy_coverage = (
            spacy_stats["tokenized"] / spacy_stats["scoreable"]
            if spacy_stats["scoreable"] else 1.0
        )
        if spacy_coverage < 0.99:
            failures.append(f"global spaCy token coverage is only {spacy_coverage:.3f}")
        if (
            spacy_stats["scoreable"] >= 20
            and spacy_stats["min_tokens"] == spacy_stats["max_tokens"]
        ):
            failures.append("spaCy token counts are frozen globally")

    if config.advanced_nlp and config.emotion_mode == "transformer" and {
        "emotion_anger", "emotion_disgust", "emotion_fear", "emotion_joy",
        "emotion_neutral", "emotion_sadness", "emotion_surprise",
    }.issubset(silver_df.columns):
        emotion_columns = [
            "emotion_anger", "emotion_disgust", "emotion_fear", "emotion_joy",
            "emotion_neutral", "emotion_sadness", "emotion_surprise",
        ]
        emotion_total = sum((F.col(column) for column in emotion_columns), F.lit(0.0))
        invalid = silver_df.where(
            emotion_total.isNotNull()
            & (
                (F.abs(emotion_total - F.lit(1.0)) > F.lit(1e-5))
                | (F.greatest(*[F.col(column) for column in emotion_columns]) > 1)
                | (F.least(*[F.col(column) for column in emotion_columns]) < 0)
            )
        ).limit(1).count()
        if invalid:
            failures.append("emotion probabilities are invalid or do not sum to 1")
        emotion_stats = silver_df.where(F.col("emotion_score").isNotNull()).agg(
            F.count(F.lit(1)).alias("n"), F.min("emotion_score").alias("min"),
            F.max("emotion_score").alias("max"),
        ).first()
        if emotion_stats["n"] >= 20 and emotion_stats["min"] == emotion_stats["max"]:
            failures.append("emotion scores are frozen globally")

    topic_coverage = None
    if config.advanced_nlp and config.topic_mode == "nmf" and "topic_vocabulary_terms" in silver_df.columns:
        topic_stats = silver_df.agg(
            F.count(F.when(F.col("topic_vocabulary_terms") > 0, 1)).alias("in_vocabulary"),
            F.count(F.when(F.length(F.trim(F.col(text_col))) > 0, 1)).alias("scoreable"),
            F.countDistinct("topic_id").alias("unique_topics"),
        ).first()
        topic_coverage = (
            topic_stats["in_vocabulary"] / topic_stats["scoreable"]
            if topic_stats["scoreable"] else 1.0
        )
        if topic_coverage < config.topic_minimum_vocabulary_coverage:
            failures.append(f"global topic vocabulary coverage is only {topic_coverage:.3f}")
        if topic_stats["in_vocabulary"] >= 20 and topic_stats["unique_topics"] <= 1:
            failures.append("topic assignments are frozen globally")
        invalid_topics = silver_df.where(
            (F.col("topic_score").isNotNull())
            & ((F.col("topic_score") < 0) | (F.col("topic_score") > 1))
        ).limit(1).count()
        if invalid_topics:
            failures.append("topic scores fall outside [0, 1]")

    metrics = {
        "passed": not failures,
        "failures": failures,
        "rows": rows,
        "columns": len(silver_df.columns),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "lexical_tokens": lexical_tokens,
        "nonword_tokens": nonword_tokens,
        "nonword_token_rate": nonword_token_rate,
        "vader_score_unique": vader_unique_scores,
        "transformer_score_unique": transformer_unique_scores,
        "spacy_nonempty_coverage": spacy_coverage,
        "topic_vocabulary_coverage": topic_coverage,
    }
    if failures:
        raise AssertionError("Spark silver validation failed: " + "; ".join(failures))
    return metrics


def run_bronze_to_silver_spark(
    spark,
    bronze_df,
    *,
    config: PipelineConfig | None = None,
    text_col: str = "text_cleaned",
    source_text_col: str = "text",
    repartition_count: int | None = None,
    api_key: str | None = None,
    maximum_residual_vocabulary: int = 500_000,
    materialization_mode: str = "auto",
    materialization_schema: str = "workspace.default",
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Execute distributed bronze-to-silver and return a materialized Spark frame.

    Classic Spark uses persisted DataFrames. Spark Connect/serverless uses
    managed Delta scratch tables because cache, persist, checkpoint, and
    ``sparkContext`` broadcasts are unsupported there.
    """
    cfg = config or PipelineConfig()
    if cfg.parallel:
        cfg = PipelineConfig(**{**asdict(cfg), "parallel": False, "workers": 1})
    mode = _resolve_materialization_mode(materialization_mode)
    run_id = uuid4().hex
    run_metrics: list[dict[str, Any]] = []
    metrics_context = build_run_metric_context(
        run_id=run_id,
        pipeline_layer="bronze_to_silver",
        config=cfg,
        materialization_mode=mode,
        spark_partitions=repartition_count,
        worker_count=detect_worker_count(spark),
    )
    scratch_tables: list[str] = []
    core = None
    core_persisted = False
    try:
        started = perf_counter()
        prepared = prepare_bronze_spark(bronze_df)
        prepared = sample_bronze_spark(
            prepared, sample_size=cfg.sample_size, seed=cfg.sample_seed
        )
        expected_rows = prepared.count()
        append_stage_metric(
            run_metrics,
            metrics_context,
            stage="bronze_read",
            duration_seconds=perf_counter() - started,
            rows_out=expected_rows,
        )
        if repartition_count is not None:
            if repartition_count < 1:
                raise ValueError("repartition_count must be positive")
            _SparkSession, _Window, F, _T = _spark_modules()
            prepared = prepared.repartition(repartition_count, F.col("message_id"))

        started = perf_counter()
        core_schema = _schema_with_features(prepared.schema, cfg, core_only=True)
        core_columns = [field.name for field in core_schema.fields]
        core_plan = prepared.mapInPandas(
            _core_iterator(text_col, source_text_col, core_columns),
            schema=core_schema,
        )
        if mode == "persist":
            core = core_plan.persist()
            core_persisted = True
            core.count()
        elif mode == "delta":
            core_table = _qualified_scratch_table(
                materialization_schema, "core", run_id
            )
            scratch_tables.append(core_table)
            (
                core_plan.write.format("delta").mode("overwrite")
                .option("overwriteSchema", "true").saveAsTable(core_table)
            )
            core = spark.table(core_table)
        else:
            core = core_plan
            core.count()
        append_stage_metric(
            run_metrics,
            metrics_context,
            stage="core_cleaning",
            duration_seconds=perf_counter() - started,
            rows_in=expected_rows,
            rows_out=expected_rows,
        )

        started = perf_counter()
        resolved_key = (
            api_key or cfg.api_key or os.getenv("OPENAI_API_KEY")
            if cfg.residual_policy == "api"
            else None
        )
        labels, residual_metadata = collect_global_residual_labels(
            core,
            policy=cfg.residual_policy,
            api_key=resolved_key,
            maximum_vocabulary=maximum_residual_vocabulary,
            api_options={
                "model": cfg.residual_api_model,
                "batch_size": cfg.residual_api_batch_size,
                "timeout": cfg.residual_api_timeout_seconds,
                "max_retries": cfg.residual_api_max_retries,
            },
        )
        append_stage_metric(
            run_metrics,
            metrics_context,
            stage="residual_classification",
            duration_seconds=perf_counter() - started,
            rows_in=expected_rows,
            rows_out=expected_rows,
        )
        started = perf_counter()
        topic_model = fit_global_topic_model_spark(
            core, config=cfg, residual_labels=labels, text_col=text_col
        )
        if cfg.topic_mode == "nmf":
            append_stage_metric(
                run_metrics,
                metrics_context,
                stage="topic_model_fit",
                duration_seconds=perf_counter() - started,
                rows_in=expected_rows,
                rows_out=expected_rows,
            )
        if mode == "persist":
            residual_payload = (
                spark.sparkContext.broadcast(labels) if labels else None
            )
            topic_payload = (
                spark.sparkContext.broadcast(topic_model)
                if topic_model is not None else None
            )
        else:
            residual_payload = labels or None
            topic_payload = topic_model

        started = perf_counter()
        output_schema = _schema_with_features(core.schema, cfg, core_only=False)
        output_columns = [field.name for field in output_schema.fields]
        silver_plan = core.mapInPandas(
            _enrichment_iterator(
                cfg, residual_payload, topic_payload, text_col, output_columns
            ),
            schema=output_schema,
        )
        if mode == "persist":
            silver = silver_plan.persist()
            silver.count()
        elif mode == "delta":
            silver_table = _qualified_scratch_table(
                materialization_schema, "silver", run_id
            )
            scratch_tables.append(silver_table)
            (
                silver_plan.write.format("delta").mode("overwrite")
                .option("overwriteSchema", "true").saveAsTable(silver_table)
            )
            silver = spark.table(silver_table)
        else:
            silver = silver_plan
            silver.count()
        append_stage_metric(
            run_metrics,
            metrics_context,
            stage="enrichment",
            duration_seconds=perf_counter() - started,
            rows_in=expected_rows,
            rows_out=expected_rows,
        )

        started = perf_counter()
        validation = validate_silver_spark(
            silver, expected_rows=expected_rows, config=cfg, text_col=text_col
        )
        append_stage_metric(
            run_metrics,
            metrics_context,
            stage="silver_validation",
            duration_seconds=perf_counter() - started,
            rows_in=expected_rows,
            rows_out=int(validation["rows"]),
        )
        if core_persisted:
            core.unpersist()
            core_persisted = False
        if mode == "delta":
            _drop_managed_table(spark, core_table)
            scratch_tables.remove(core_table)
        metadata = {
            "run_id": run_id,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "configuration": {
                key: value for key, value in asdict(cfg).items() if key != "api_key"
            },
            "residuals": residual_metadata,
            "topics": None if topic_model is None else {
                "labels": topic_model.labels,
                "fit_rows": topic_model.fit_rows,
                "random_state": topic_model.random_state,
            },
            "model_loading": "once per Python worker via module-level caches",
            "materialization": {
                "mode": mode,
                "temporary_tables": list(scratch_tables),
            },
            "run_metrics_context": metrics_context,
            "run_metrics": run_metrics,
            "selected_rows": expected_rows,
        }
        return silver, validation, metadata
    except Exception:
        if core_persisted and core is not None:
            core.unpersist()
        for table in reversed(scratch_tables):
            _drop_managed_table(spark, table)
        raise


def write_silver_delta(
    silver_df,
    *,
    output_table: str | None = None,
    output_delta_path: str | None = None,
    mode: str = "errorifexists",
    run_metadata: dict[str, Any] | None = None,
) -> str:
    """Write one validated silver Delta target."""
    if (output_table is None) == (output_delta_path is None):
        raise ValueError("provide exactly one of output_table or output_delta_path")
    started = perf_counter()
    writer = silver_df.write.format("delta").mode(mode).option("mergeSchema", "false")
    if output_table:
        writer.saveAsTable(output_table)
        target = output_table
    else:
        writer.save(output_delta_path)
        target = str(output_delta_path)
    if run_metadata is not None:
        append_stage_metric(
            run_metadata.setdefault("run_metrics", []),
            run_metadata["run_metrics_context"],
            stage="silver_write",
            duration_seconds=perf_counter() - started,
            rows_in=run_metadata.get("selected_rows"),
            rows_out=run_metadata.get("selected_rows"),
        )
    return target


def append_silver_audit_delta(
    spark,
    *,
    audit_table: str,
    output_target: str,
    validation: Mapping[str, Any],
    metadata: Mapping[str, Any],
    metrics_table: str | None = None,
) -> None:
    """Append a secret-free audit row only after validation and Delta write."""
    if not validation.get("passed"):
        raise ValueError("refusing to audit a failed silver output")
    record = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "stage": "silver_message_features",
        "output_target": output_target,
        "rows": int(validation["rows"]),
        "columns": int(validation["columns"]),
        "passed": True,
        "details_json": json.dumps(
            {"validation": validation, "metadata": metadata},
            sort_keys=True, default=str,
        ),
    }
    spark.createDataFrame([record]).write.format("delta").mode("append").saveAsTable(
        audit_table
    )
    if metrics_table:
        append_run_metrics_delta(
            spark,
            metrics_table=metrics_table,
            records=list(metadata.get("run_metrics") or []),
        )


def _build_config(args) -> PipelineConfig:
    return PipelineConfig(
        residual_policy=args.residual_policy,
        maximum_nonword_token_rate=args.maximum_nonword_token_rate,
        residual_api_model=args.residual_api_model,
        residual_api_batch_size=args.residual_api_batch_size,
        residual_api_timeout_seconds=args.residual_api_timeout_seconds,
        residual_api_max_retries=args.residual_api_max_retries,
        sentiment_mode=args.sentiment,
        transformer_model=args.transformer_model,
        transformer_revision=args.transformer_revision,
        transformer_batch_size=args.transformer_batch_size,
        transformer_inference_chunk_size=args.transformer_inference_chunk_size,
        transformer_device=args.transformer_device,
        transformer_dtype=args.transformer_dtype,
        advanced_nlp=args.advanced_nlp,
        run_spacy=not args.no_spacy,
        topic_mode=args.topics,
        topic_fit_sample_size=args.topic_fit_sample_size,
        topic_batch_size=args.topic_batch_size,
        topic_minimum_vocabulary_coverage=args.topic_minimum_vocabulary_coverage,
        emotion_mode=args.emotion,
        spacy_model=args.spacy_model,
        spacy_batch_size=args.spacy_batch_size,
        include_token_details=args.include_token_details,
        emotion_model=args.emotion_model,
        emotion_revision=args.emotion_revision,
        emotion_batch_size=args.emotion_batch_size,
        emotion_inference_chunk_size=args.emotion_inference_chunk_size,
        emotion_device=args.emotion_device,
        emotion_dtype=args.emotion_dtype,
        parallel=False,
        workers=1,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        output_dir="databricks_integration/outputs",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build validated distributed Ubuntu silver message features."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-table")
    source.add_argument("--input-delta-path")
    source.add_argument("--input-csv-path")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-table")
    destination.add_argument("--output-delta-path")
    parser.add_argument("--audit-table", required=True)
    parser.add_argument("--run-metrics-table")
    parser.add_argument("--write-mode", choices=("errorifexists", "overwrite", "append"), default="errorifexists")
    parser.add_argument("--repartition-count", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--residual-policy", choices=("manual_review", "reviewed", "api"), default="manual_review")
    parser.add_argument("--maximum-nonword-token-rate", type=float, default=0.25)
    parser.add_argument(
        "--residual-api-model", default="gpt-4o-mini-2024-07-18"
    )
    parser.add_argument("--residual-api-batch-size", type=int, default=500)
    parser.add_argument("--residual-api-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--residual-api-max-retries", type=int, default=2)
    parser.add_argument("--sentiment", choices=("none", "vader", "transformer", "both"), default="vader")
    parser.add_argument("--transformer-model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    parser.add_argument("--transformer-revision", default="3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7")
    parser.add_argument("--transformer-batch-size", type=int, default=32)
    parser.add_argument("--transformer-inference-chunk-size", type=int, default=4096)
    parser.add_argument("--transformer-device", type=int)
    parser.add_argument("--transformer-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--advanced-nlp", action="store_true")
    parser.add_argument("--no-spacy", action="store_true")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--spacy-batch-size", type=int, default=128)
    parser.add_argument("--include-token-details", action="store_true")
    parser.add_argument("--topics", choices=("none", "nmf"), default="none")
    parser.add_argument("--topic-fit-sample-size", type=int, default=100_000)
    parser.add_argument("--topic-batch-size", type=int, default=50_000)
    parser.add_argument("--topic-minimum-vocabulary-coverage", type=float, default=0.50)
    parser.add_argument("--emotion", choices=("none", "transformer"), default="none")
    parser.add_argument("--emotion-model", default="j-hartmann/emotion-english-distilroberta-base")
    parser.add_argument("--emotion-revision", default="cea2f78f197f0337186a0faa93e00ef93811f6eb")
    parser.add_argument("--emotion-batch-size", type=int, default=32)
    parser.add_argument("--emotion-inference-chunk-size", type=int, default=4096)
    parser.add_argument("--emotion-device", type=int)
    parser.add_argument("--emotion-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--maximum-residual-vocabulary", type=int, default=500_000)
    parser.add_argument(
        "--materialization-mode",
        choices=("auto", "persist", "delta", "none"),
        default="auto",
    )
    parser.add_argument(
        "--materialization-schema",
        default="workspace.default",
        help="managed catalog.schema used for serverless scratch Delta tables",
    )
    args = parser.parse_args()

    SparkSession, _Window, _F, _T = _spark_modules()
    spark = SparkSession.builder.getOrCreate()
    bronze = read_bronze_spark(
        spark,
        input_table=args.input_table,
        input_delta_path=args.input_delta_path,
        input_csv_path=args.input_csv_path,
    )
    config = _build_config(args)
    silver, validation, metadata = run_bronze_to_silver_spark(
        spark,
        bronze,
        config=config,
        repartition_count=args.repartition_count,
        api_key=os.getenv("OPENAI_API_KEY"),
        maximum_residual_vocabulary=args.maximum_residual_vocabulary,
        materialization_mode=args.materialization_mode,
        materialization_schema=args.materialization_schema,
    )
    try:
        target = write_silver_delta(
            silver,
            output_table=args.output_table,
            output_delta_path=args.output_delta_path,
            mode=args.write_mode,
            run_metadata=metadata,
        )
        append_silver_audit_delta(
            spark,
            audit_table=args.audit_table,
            output_target=target,
            validation=validation,
            metadata=metadata,
            metrics_table=args.run_metrics_table,
        )
    finally:
        release_silver_resources(spark, silver, metadata)
    print(json.dumps({"output": target, "validation": validation}, indent=2, default=str))


if __name__ == "__main__":
    main()
