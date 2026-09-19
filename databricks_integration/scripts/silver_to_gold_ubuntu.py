"""Spark/Delta silver-to-gold job for the Ubuntu dialogue corpus.

All PySpark imports are lazy so the repository remains importable and testable
inside the isolated local pandas environment.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from pipeline.aggregation import (
    DATE_GRANULARITIES,
    GOLD_LEVELS,
    RELEASE_AXES,
    _AVERAGE_COLUMNS,
    _LABEL_COLUMNS,
    _SUM_COLUMNS,
)
from pipeline.schema import FEATURE_SCHEMA_VERSION
from pipeline.sentiment_analysis import LANGUAGE_LABEL_TOKENS


def _spark_modules():
    try:
        from pyspark.sql import SparkSession, functions as F, types as T
    except ImportError as exc:
        raise ImportError(
            "The Databricks gold job requires a Databricks/Spark runtime; "
            "do not install PySpark globally for the local pandas project."
        ) from exc
    return SparkSession, F, T


def _require_columns(frame, columns: list[str], context: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{context} requires columns: {missing}")


def _spark_base_aggregate(frame, group_columns: list[str]):
    _SparkSession, F, _T = _spark_modules()
    _require_columns(frame, group_columns, "gold aggregation")
    expressions = [F.count(F.lit(1)).alias("message_count")]
    for source, target in _AVERAGE_COLUMNS.items():
        if source in frame.columns:
            expressions.append(F.avg(F.col(source)).alias(target))
    for source, target in _SUM_COLUMNS.items():
        if source in frame.columns:
            expressions.append(F.sum(F.col(source)).alias(target))
    for source, target in (
        ("conversation_id", "unique_conversation_count"),
        ("from", "unique_sender_count"),
        ("to", "unique_recipient_count"),
    ):
        if source in frame.columns and source not in group_columns:
            expressions.append(F.countDistinct(F.col(source)).alias(target))
    for source, labels in _LABEL_COLUMNS.items():
        if source not in frame.columns:
            continue
        expressions.append(F.expr(f"mode(`{source}`)").alias(f"most_common_{source}"))
        for label in labels:
            expressions.append(
                F.avg(F.when(F.col(source) == label, F.lit(1.0)).otherwise(F.lit(0.0))).alias(
                    f"share_{source}_{label.lower()}"
                )
            )
    if "topic_label" in frame.columns:
        expressions.append(F.expr("mode(`topic_label`)").alias("most_common_topic"))
    return frame.groupBy(*group_columns).agg(*expressions)


def _prefix_spark_metrics(frame, key: str, prefix: str):
    _SparkSession, F, _T = _spark_modules()
    return frame.select(*[
        F.col(column) if column == key else F.col(column).alias(f"{prefix}_{column}")
        for column in frame.columns
    ])


def _spark_conversations(frame):
    _SparkSession, F, _T = _spark_modules()
    _require_columns(frame, ["conversation_id"], "conversation gold")
    result = _spark_base_aggregate(frame, ["conversation_id"])
    if "date" in frame.columns:
        bounds = frame.groupBy("conversation_id").agg(
            F.min(F.to_timestamp("date")).alias("conversation_start"),
            F.max(F.to_timestamp("date")).alias("conversation_end"),
        ).withColumn(
            "conversation_duration_mins",
            (F.col("conversation_end").cast("long") - F.col("conversation_start").cast("long")) / 60.0,
        )
        result = result.join(bounds, "conversation_id", "left")
    participant_frames = []
    for column in ("from", "to"):
        if column in frame.columns:
            participant_frames.append(
                frame.select("conversation_id", F.col(column).alias("participant"))
                .where(F.col("participant").isNotNull())
            )
    if participant_frames:
        participants = participant_frames[0]
        for participant_frame in participant_frames[1:]:
            participants = participants.unionByName(participant_frame)
        counts = participants.groupBy("conversation_id").agg(
            F.countDistinct("participant").alias("participant_count")
        )
        result = result.join(counts, "conversation_id", "left")
    if "from" in frame.columns:
        senders = frame.groupBy("conversation_id").agg(
            F.countDistinct("from").alias("sender_count")
        ).withColumn("conversation_was_answered", F.col("sender_count") > 1)
        result = result.join(senders, "conversation_id", "left")
    else:
        result = result.withColumn("conversation_was_answered", F.lit(False))
    return result


def _spark_users(frame):
    _SparkSession, F, _T = _spark_modules()
    _require_columns(frame, ["from", "to"], "user gold")
    sent = frame.where(F.col("from").isNotNull()).withColumn("user", F.col("from"))
    received = frame.where(F.col("to").isNotNull()).withColumn("user", F.col("to"))
    sent_gold = _prefix_spark_metrics(_spark_base_aggregate(sent, ["user"]), "user", "sent")
    received_gold = _prefix_spark_metrics(
        _spark_base_aggregate(received, ["user"]), "user", "received"
    )
    result = sent_gold.join(received_gold, "user", "full")
    result = result.fillna(0, subset=["sent_message_count", "received_message_count"])
    result = result.withColumn(
        "interaction_message_count",
        F.col("sent_message_count") + F.col("received_message_count"),
    )
    contacts = frame.select(
        F.col("from").alias("user"), F.col("to").alias("contact")
    ).unionByName(frame.select(
        F.col("to").alias("user"), F.col("from").alias("contact")
    )).where(F.col("user").isNotNull() & F.col("contact").isNotNull())
    contact_counts = contacts.groupBy("user").agg(
        F.countDistinct("contact").alias("unique_contact_count")
    )
    result = result.join(contact_counts, "user", "left")
    count_columns = [column for column in result.columns if column.endswith("_count")]
    return result.fillna(0, subset=count_columns)


def _spark_dates(frame, granularity: str):
    _SparkSession, F, _T = _spark_modules()
    _require_columns(frame, ["date"], "date gold")
    if granularity not in DATE_GRANULARITIES:
        raise ValueError(f"date granularity must be one of {sorted(DATE_GRANULARITIES)}")
    dated = frame.withColumn("_parsed_date", F.to_timestamp("date")).where(
        F.col("_parsed_date").isNotNull()
    ).withColumn("date_period", F.date_trunc(granularity, F.col("_parsed_date")))
    return _spark_base_aggregate(dated, ["date_period"])


def _spark_language(frame):
    _SparkSession, F, _T = _spark_modules()
    _require_columns(frame, ["text_cleaned"], "language gold")
    words = F.split(F.coalesce(F.col("text_cleaned"), F.lit("")), r"\s+")
    candidates = F.array(*[
        F.when(F.array_contains(words, label), F.lit(label))
        for label in LANGUAGE_LABEL_TOKENS
    ])
    labels = F.filter(candidates, lambda value: value.isNotNull())
    expanded = frame.withColumn(
        "language_label",
        F.explode(F.when(F.size(labels) > 0, labels).otherwise(F.array(F.lit("UNLABELED"))))
    )
    return _spark_base_aggregate(expanded, ["language_label"])


def _array_element_fields(frame, column: str) -> tuple[str, set[str]]:
    _SparkSession, _F, T = _spark_modules()
    field = frame.schema[column]
    if not isinstance(field.dataType, T.ArrayType):
        raise TypeError(
            f"Spark gold aggregation requires typed array column {column!r}; "
            f"found {field.dataType.simpleString()}"
        )
    element = field.dataType.elementType
    if isinstance(element, T.StructType):
        return "struct", set(element.fieldNames())
    if isinstance(element, T.ArrayType):
        return "array", set()
    if isinstance(element, T.StringType):
        return "string", set()
    raise TypeError(f"unsupported {column} element type: {element.simpleString()}")


def _spark_residual(frame):
    _SparkSession, F, _T = _spark_modules()
    _require_columns(frame, ["residual_words"], "residual gold")
    kind, _fields = _array_element_fields(frame, "residual_words")
    if kind != "string":
        raise TypeError("residual_words must be array<string> in the silver Delta schema")
    expanded = frame.withColumn("residual_word", F.explode_outer("residual_words")).where(
        F.col("residual_word").isNotNull()
    )
    return _spark_base_aggregate(expanded, ["residual_word"])


def _spark_technical(frame):
    _SparkSession, F, _T = _spark_modules()
    _require_columns(frame, ["tech_lexicon_matches"], "technical gold")
    kind, fields = _array_element_fields(frame, "tech_lexicon_matches")
    expanded = frame.withColumn("_technical", F.explode_outer("tech_lexicon_matches"))
    if kind == "struct" and {"term", "category"}.issubset(fields):
        expanded = expanded.withColumn("technical_term", F.col("_technical.term")).withColumn(
            "technical_category", F.col("_technical.category")
        )
    elif kind == "array":
        expanded = expanded.withColumn("technical_term", F.col("_technical")[0]).withColumn(
            "technical_category", F.col("_technical")[1]
        )
    else:
        raise TypeError(
            "tech_lexicon_matches must be array<struct<term,category>> or array<array<string>>"
        )
    expanded = expanded.where(F.col("technical_term").isNotNull())
    return _spark_base_aggregate(expanded, ["technical_term", "technical_category"])


def _spark_entities(frame):
    _SparkSession, F, _T = _spark_modules()
    pieces = []
    if "tech_lexicon_matches" in frame.columns:
        technical = _spark_technical(frame).select(
            F.lit("technical_lexicon").alias("entity_source"),
            F.col("technical_category").alias("entity_label"),
            F.col("technical_term").alias("entity"),
            "message_count",
        )
        pieces.append(technical)
    if "named_entities" in frame.columns:
        field = frame.schema["named_entities"]
        if isinstance(field.dataType, _T.StringType):
            schema = _T.ArrayType(_T.StructType([
                _T.StructField("text", _T.StringType()),
                _T.StructField("label", _T.StringType()),
                _T.StructField("start", _T.IntegerType()),
                _T.StructField("end", _T.IntegerType()),
            ]))
            parsed = F.from_json(F.col("named_entities"), schema)
            malformed = (
                F.length(F.trim(F.col("named_entities"))) > 0
            ) & F.col("named_entities").isNotNull() & parsed.isNull()
            if frame.where(malformed).limit(1).count():
                raise ValueError("malformed JSON in named_entities")
            named_source = frame.withColumn("_named_entities_array", parsed)
            entity_column = "_named_entities_array"
        else:
            kind, fields = _array_element_fields(frame, "named_entities")
            if kind != "struct" or not {"text", "label"}.issubset(fields):
                raise TypeError("named_entities must be JSON or array<struct<text,label,...>>")
            named_source = frame
            entity_column = "named_entities"
        named = named_source.withColumn("_entity", F.explode_outer(entity_column)).where(
            F.col("_entity.text").isNotNull()
        ).groupBy(
            F.lit("spacy_ner").alias("entity_source"),
            F.col("_entity.label").alias("entity_label"),
            F.col("_entity.text").alias("entity"),
        ).agg(F.count(F.lit(1)).alias("message_count"))
        pieces.append(named)
    if not pieces:
        raise KeyError("entity gold requires technical or named-entity columns")
    result = pieces[0]
    for piece in pieces[1:]:
        result = result.unionByName(piece)
    return result


def run_silver_to_gold_spark(
    silver_df,
    *,
    gold_level: str,
    date_granularity: str = "day",
    channel_column: str = "channel",
    release_axis: str = "since",
    sample_size: int | None = None,
    random_state: int = 0,
):
    """Build one globally aggregated Spark gold DataFrame."""
    _SparkSession, F, _T = _spark_modules()
    if gold_level not in GOLD_LEVELS:
        raise ValueError(f"gold_level must be one of {sorted(GOLD_LEVELS)}")
    if gold_level == "conversation":
        return _spark_conversations(silver_df)
    if gold_level == "user":
        return _spark_users(silver_df)
    if gold_level == "date":
        return _spark_dates(silver_df, date_granularity)
    if gold_level == "channel":
        _require_columns(silver_df, [channel_column], "channel gold")
        return _spark_base_aggregate(
            silver_df.where(F.col(channel_column).isNotNull()), [channel_column]
        )
    if gold_level == "release":
        if release_axis not in RELEASE_AXES:
            raise ValueError(f"release axis must be one of {sorted(RELEASE_AXES)}")
        column = f"days_{release_axis}_release_bucket"
        _require_columns(silver_df, [column], "release gold")
        return _spark_base_aggregate(silver_df.where(F.col(column).isNotNull()), [column])
    if gold_level == "language":
        return _spark_language(silver_df)
    if gold_level == "residual":
        return _spark_residual(silver_df)
    if gold_level == "technical":
        return _spark_technical(silver_df)
    if gold_level == "topic":
        _require_columns(silver_df, ["topic_id", "topic_label"], "topic gold")
        return _spark_base_aggregate(
            silver_df.where(F.col("topic_id").isNotNull()), ["topic_id", "topic_label"]
        )
    if gold_level == "entity":
        return _spark_entities(silver_df)
    if sample_size is None or sample_size < 1:
        raise ValueError("positive sample_size is required for sample gold")
    _require_columns(silver_df, ["message_id"], "deterministic sample gold")
    if silver_df.where(F.col("message_id").isNull()).limit(1).count():
        raise ValueError("deterministic sample gold requires non-null message_id values")
    if silver_df.groupBy("message_id").count().where(F.col("count") > 1).limit(1).count():
        raise ValueError("deterministic sample gold requires unique message_id values")
    return (
        silver_df.withColumn(
            "_sample_hash",
            F.xxhash64(F.lit(int(random_state)), F.col("message_id").cast("string")),
        )
        .orderBy("_sample_hash", "message_id")
        .limit(sample_size)
        .drop("_sample_hash")
    )


def validate_gold_spark(
    silver_df,
    gold_df,
    *,
    gold_level: str,
    channel_column: str = "channel",
    release_axis: str = "since",
    sample_size: int | None = None,
    random_state: int = 0,
) -> dict[str, Any]:
    """Run global post-aggregation checks before a Delta write."""
    _SparkSession, F, _T = _spark_modules()
    key_map = {
        "conversation": ["conversation_id"], "user": ["user"],
        "date": ["date_period"], "channel": [channel_column],
        "release": [f"days_{release_axis}_release_bucket"],
        "language": ["language_label"], "residual": ["residual_word"],
        "technical": ["technical_term", "technical_category"],
        "topic": ["topic_id", "topic_label"],
        "entity": ["entity_source", "entity_label", "entity"],
        "sample": ["message_id"],
    }
    failures: list[str] = []
    keys = key_map.get(gold_level, [])
    if keys and gold_df.groupBy(*keys).count().where(F.col("count") > 1).limit(1).count():
        failures.append(f"duplicate gold keys: {keys}")
    for column in (name for name in gold_df.columns if name.endswith("_count")):
        if gold_df.where(F.col(column).isNull() | (F.col(column) < 0)).limit(1).count():
            failures.append(f"invalid count values in {column}")
    bounded_columns = [
        column for column in gold_df.columns
        if column.startswith("avg_vader_")
        or column.startswith("avg_transformer_")
        or column.startswith("share_")
    ]
    for column in bounded_columns:
        lower = -1.0 if column in {
            "avg_vader_compound", "avg_transformer_expected_sentiment"
        } else 0.0
        invalid = (
            F.col(column).isNotNull()
            & (
                F.isnan(F.col(column))
                | (F.col(column) < lower)
                | (F.col(column) > 1.0)
            )
        )
        if gold_df.where(invalid).limit(1).count():
            failures.append(f"out-of-range aggregate values in {column}")
    silver_rows = silver_df.count()
    gold_rows = gold_df.count()
    if gold_level == "conversation" and "message_count" in gold_df.columns:
        total = gold_df.agg(F.sum("message_count")).first()[0] or 0
        if total != silver_rows:
            failures.append("conversation counts do not conserve silver rows")
    elif gold_level == "date" and "message_count" in gold_df.columns:
        expected = silver_df.where(F.to_timestamp("date").isNotNull()).count()
        total = gold_df.agg(F.sum("message_count")).first()[0] or 0
        if total != expected:
            failures.append("date counts do not conserve valid dated rows")
    elif gold_level == "channel" and "message_count" in gold_df.columns:
        expected = silver_df.where(F.col(channel_column).isNotNull()).count()
        total = gold_df.agg(F.sum("message_count")).first()[0] or 0
        if total != expected:
            failures.append("channel counts do not conserve eligible rows")
    elif gold_level == "release" and "message_count" in gold_df.columns:
        release_column = f"days_{release_axis}_release_bucket"
        expected = silver_df.where(F.col(release_column).isNotNull()).count()
        total = gold_df.agg(F.sum("message_count")).first()[0] or 0
        if total != expected:
            failures.append("release counts do not conserve eligible rows")
    elif gold_level == "user":
        sent_expected = silver_df.where(F.col("from").isNotNull()).count()
        received_expected = silver_df.where(F.col("to").isNotNull()).count()
        totals = gold_df.agg(
            F.sum("sent_message_count"), F.sum("received_message_count")
        ).first()
        if (totals[0] or 0) != sent_expected or (totals[1] or 0) != received_expected:
            failures.append("user directional counts do not conserve silver rows")
    elif gold_level == "sample" and gold_rows != min(sample_size or 0, silver_rows):
        failures.append("sample row count is inconsistent")
    metrics = {
        "passed": not failures, "failures": failures,
        "gold_level": gold_level, "silver_rows": silver_rows,
        "gold_rows": gold_rows, "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "sample_size": sample_size, "random_state": random_state,
    }
    if failures:
        raise AssertionError("Spark gold validation failed: " + "; ".join(failures))
    return metrics


def _write_audit_delta(spark, table: str, record: dict[str, Any]) -> None:
    payload = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "stage": f"gold_{record['gold_level']}",
        "gold_level": record["gold_level"],
        "silver_rows": int(record["silver_rows"]),
        "gold_rows": int(record["gold_rows"]),
        "passed": bool(record["passed"]),
        "details_json": json.dumps(record, sort_keys=True),
    }
    spark.createDataFrame([payload]).write.format("delta").mode("append").saveAsTable(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validated Spark/Delta Ubuntu gold tables.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-table")
    source.add_argument("--input-delta-path")
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--audit-table")
    parser.add_argument("--gold-level", choices=sorted(GOLD_LEVELS), required=True)
    parser.add_argument("--date-granularity", choices=sorted(DATE_GRANULARITIES), default="day")
    parser.add_argument("--channel-column", default="channel")
    parser.add_argument("--release-axis", choices=sorted(RELEASE_AXES), default="since")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    SparkSession, _F, _T = _spark_modules()
    spark = SparkSession.builder.getOrCreate()
    silver = (
        spark.table(args.input_table)
        if args.input_table
        else spark.read.format("delta").load(args.input_delta_path)
    )
    gold = run_silver_to_gold_spark(
        silver, gold_level=args.gold_level,
        date_granularity=args.date_granularity,
        channel_column=args.channel_column, release_axis=args.release_axis,
        sample_size=args.sample_size, random_state=args.random_state,
    )
    validation = validate_gold_spark(
        silver, gold, gold_level=args.gold_level,
        channel_column=args.channel_column, release_axis=args.release_axis,
        sample_size=args.sample_size, random_state=args.random_state,
    )
    gold.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(args.output_table)
    _write_audit_delta(
        spark, args.audit_table or f"{args.output_table}_audit", validation
    )
    print(
        f"wrote Delta table {args.output_table} with {validation['gold_rows']:,} rows"
    )


if __name__ == "__main__":
    main()
