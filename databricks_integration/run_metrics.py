"""Portable Databricks run-metrics records for serverless and classic Spark."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Mapping, MutableSequence

from pipeline.schema import FEATURE_SCHEMA_VERSION


RUN_METRIC_COLUMNS = (
    "run_id",
    "stage_sequence",
    "pipeline_layer",
    "stage",
    "status",
    "gold_level",
    "date_granularity",
    "sample_size",
    "sample_seed",
    "rows_in",
    "rows_out",
    "duration_seconds",
    "rows_per_second",
    "duration_per_row_ms",
    "sentiment_mode",
    "residual_policy",
    "advanced_nlp_enabled",
    "topic_mode",
    "emotion_mode",
    "worker_count",
    "python_workers_per_partition",
    "spark_partitions",
    "compute_mode",
    "materialization_mode",
    "cluster_id",
    "runtime_version",
    "feature_schema_version",
    "completed_at_utc",
)


def detect_compute_mode() -> str:
    """Distinguish the Spark APIs without assuming an account tier."""
    try:
        from pyspark.sql.utils import is_remote

        return "spark_connect_or_serverless" if is_remote() else "classic"
    except (ImportError, TypeError):
        return "unknown"


def detect_worker_count(spark) -> int | None:
    """Read an available cluster-size hint without using ``sparkContext``."""
    for key in (
        "spark.databricks.clusterUsageTags.clusterWorkers",
        "spark.executor.instances",
    ):
        try:
            value = spark.conf.get(key)
        except Exception:
            continue
        if value is not None and str(value).isdigit():
            return int(value)
    return None


def build_run_metric_context(
    *,
    run_id: str,
    pipeline_layer: str,
    config: Any | None = None,
    materialization_mode: str | None = None,
    compute_mode: str | None = None,
    spark_partitions: int | None = None,
    worker_count: int | None = None,
    gold_level: str | None = None,
    date_granularity: str | None = None,
    sample_size: int | None = None,
    sample_seed: int | None = None,
) -> dict[str, Any]:
    """Build an explicitly allow-listed, secret-free metrics context."""
    configured_sample_size = getattr(config, "sample_size", None)
    configured_sample_seed = getattr(config, "sample_seed", None)
    return {
        "run_id": str(run_id),
        "pipeline_layer": str(pipeline_layer),
        "gold_level": gold_level,
        "date_granularity": date_granularity,
        "sample_size": (
            configured_sample_size if sample_size is None else sample_size
        ),
        "sample_seed": (
            configured_sample_seed if sample_seed is None else sample_seed
        ),
        "sentiment_mode": getattr(config, "sentiment_mode", None),
        "residual_policy": getattr(config, "residual_policy", None),
        "advanced_nlp_enabled": getattr(config, "advanced_nlp", None),
        "topic_mode": getattr(config, "topic_mode", None),
        "emotion_mode": getattr(config, "emotion_mode", None),
        "worker_count": worker_count,
        "python_workers_per_partition": getattr(config, "workers", None),
        "spark_partitions": spark_partitions,
        "compute_mode": compute_mode or detect_compute_mode(),
        "materialization_mode": materialization_mode,
        "cluster_id": os.getenv("DATABRICKS_CLUSTER_ID"),
        "runtime_version": os.getenv("DATABRICKS_RUNTIME_VERSION"),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }


def append_stage_metric(
    records: MutableSequence[dict[str, Any]],
    context: Mapping[str, Any],
    *,
    stage: str,
    duration_seconds: float,
    rows_in: int | None = None,
    rows_out: int | None = None,
    status: str = "passed",
) -> dict[str, Any]:
    """Append one completed Spark-action timing with derived throughput."""
    duration = max(0.0, float(duration_seconds))
    processed_rows = rows_in if rows_in is not None else rows_out
    throughput = (
        float(processed_rows) / duration
        if processed_rows is not None and duration > 0
        else None
    )
    duration_per_row = (
        duration * 1000.0 / float(processed_rows)
        if processed_rows not in {None, 0}
        else None
    )
    record = {
        **{column: context.get(column) for column in RUN_METRIC_COLUMNS},
        "stage_sequence": len(records) + 1,
        "stage": str(stage),
        "status": str(status),
        "rows_in": None if rows_in is None else int(rows_in),
        "rows_out": None if rows_out is None else int(rows_out),
        "duration_seconds": duration,
        "rows_per_second": throughput,
        "duration_per_row_ms": duration_per_row,
        "completed_at_utc": datetime.now(timezone.utc),
    }
    records.append(record)
    return record


def append_run_metrics_delta(
    spark,
    *,
    metrics_table: str,
    records: list[Mapping[str, Any]],
) -> None:
    """Append metrics with an explicit schema on either Databricks compute API."""
    if not records:
        return
    try:
        from pyspark.sql import types as T
    except ImportError as exc:
        raise ImportError(
            "Delta run metrics require a Databricks/Spark runtime"
        ) from exc

    string_fields = {
        "run_id", "pipeline_layer", "stage", "status", "gold_level",
        "date_granularity", "sentiment_mode", "residual_policy", "topic_mode",
        "emotion_mode", "compute_mode", "materialization_mode", "cluster_id",
        "runtime_version", "feature_schema_version",
    }
    long_fields = {
        "sample_size", "sample_seed", "rows_in", "rows_out",
        "spark_partitions",
    }
    integer_fields = {
        "stage_sequence", "worker_count", "python_workers_per_partition",
    }
    double_fields = {
        "duration_seconds", "rows_per_second", "duration_per_row_ms",
    }
    fields = []
    for column in RUN_METRIC_COLUMNS:
        if column in string_fields:
            data_type = T.StringType()
        elif column in long_fields:
            data_type = T.LongType()
        elif column in integer_fields:
            data_type = T.IntegerType()
        elif column in double_fields:
            data_type = T.DoubleType()
        elif column == "advanced_nlp_enabled":
            data_type = T.BooleanType()
        elif column == "completed_at_utc":
            data_type = T.TimestampType()
        else:  # pragma: no cover - protected by the fixed contract above
            raise KeyError(f"run metric column has no Spark type: {column}")
        fields.append(T.StructField(column, data_type, True))
    schema = T.StructType(fields)
    normalized = [
        tuple(record.get(column) for column in RUN_METRIC_COLUMNS)
        for record in records
    ]
    (
        spark.createDataFrame(normalized, schema=schema)
        .write.format("delta").mode("append").saveAsTable(metrics_table)
    )
