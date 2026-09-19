"""Validated, privacy-bounded exports for the Ubuntu Streamlit dashboard."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from pipeline.schema import FEATURE_SCHEMA_VERSION

FORBIDDEN_DASHBOARD_COLUMNS = frozenset({
    "text", "text_cleaned", "from", "to", "user", "username", "sender", "recipient"
})
REQUIRED_DASHBOARD_TABLES = frozenset({
    "conversation_summary",
    "monthly_summary",
    "correlation_matrix",
    "statistical_summary",
    "hypothesis_tests",
    "model_selection",
    "model_test_metrics",
    "model_feature_importance",
})
_TABLE_REQUIRED_COLUMNS = {
    "correlation_matrix": {"feature"},
    "statistical_summary": {"feature"},
    "hypothesis_tests": {"hypothesis", "test", "p_value"},
    "model_selection": {"model", "average_precision"},
    "model_test_metrics": {"model"},
    "model_feature_importance": {"feature", "importance"},
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _validate_table_name(name: str) -> None:
    if not name or not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe dashboard table name: {name!r}")


def validate_dashboard_tables(
    tables: Mapping[str, pd.DataFrame], *, require_complete: bool = True
) -> dict[str, Any]:
    """Validate the dashboard contract before export or after loading."""
    if require_complete:
        missing = sorted(REQUIRED_DASHBOARD_TABLES.difference(tables))
        if missing:
            raise ValueError(f"missing required dashboard tables: {missing}")

    table_metrics: dict[str, dict[str, Any]] = {}
    for name, frame in tables.items():
        _validate_table_name(name)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"dashboard table {name!r} is not a DataFrame")
        exposed = sorted(FORBIDDEN_DASHBOARD_COLUMNS.intersection(frame.columns))
        if exposed:
            raise ValueError(
                f"dashboard table {name!r} exposes message text or usernames: {exposed}"
            )
        duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
        if duplicate_columns:
            raise ValueError(
                f"dashboard table {name!r} has duplicate columns: {duplicate_columns}"
            )
        missing_columns = sorted(
            _TABLE_REQUIRED_COLUMNS.get(name, set()).difference(frame.columns)
        )
        if missing_columns:
            raise ValueError(
                f"dashboard table {name!r} is missing columns: {missing_columns}"
            )
        for column in (item for item in frame.columns if item.endswith("_count")):
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.isna().any() or (numeric < 0).any():
                raise ValueError(f"invalid count values in {name}.{column}")
        table_metrics[name] = {
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
        }

    if "conversation_summary" in tables:
        conversations = tables["conversation_summary"]
        required = {"conversation_id", "message_count", "conversation_was_answered"}
        missing = sorted(required.difference(conversations.columns))
        if missing:
            raise ValueError(f"conversation_summary is missing columns: {missing}")
        if conversations["conversation_id"].isna().any():
            raise ValueError("conversation_summary contains null conversation IDs")
        if conversations["conversation_id"].duplicated().any():
            raise ValueError("conversation_summary contains duplicate conversation IDs")
        if conversations["conversation_was_answered"].isna().any():
            raise ValueError("conversation_summary contains null response outcomes")

    if "monthly_summary" in tables:
        monthly = tables["monthly_summary"]
        required = {"date_period", "message_count"}
        missing = sorted(required.difference(monthly.columns))
        if missing:
            raise ValueError(f"monthly_summary is missing columns: {missing}")
        parsed_dates = pd.to_datetime(monthly["date_period"], errors="coerce", utc=True)
        if parsed_dates.isna().any():
            raise ValueError("monthly_summary contains invalid date periods")
        if parsed_dates.duplicated().any():
            raise ValueError("monthly_summary contains duplicate date periods")

    return {
        "passed": True,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "tables": table_metrics,
    }


def _relative_reference(path: str | Path | None, base: Path) -> str | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    return Path(os.path.relpath(resolved, start=base.resolve())).as_posix()


def export_dashboard_bundle(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    figures: Iterable[str | Path] = (),
    model_path: str | Path | None = None,
    model_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically export validated CSV tables and a portable JSON manifest."""
    validation = validate_dashboard_tables(tables)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    table_paths: dict[str, str] = {}
    for name, frame in tables.items():
        final_path = destination / f"{name}.csv"
        temporary_path = destination / f".{name}.csv.tmp"
        frame.to_csv(temporary_path, index=False)
        temporary_path.replace(final_path)
        table_paths[name] = final_path.name

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "validation": validation,
        "metadata": _json_safe(metadata or {}),
        "dashboard_tables": table_paths,
        "figures": [
            reference
            for reference in (
                _relative_reference(path, destination) for path in figures
            )
            if reference is not None
        ],
        "model_path": _relative_reference(model_path, destination),
        "model_metadata_path": _relative_reference(model_metadata_path, destination),
    }
    manifest_path = destination / "analysis_manifest.json"
    temporary_manifest = destination / ".analysis_manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def load_dashboard_bundle(
    bundle_dir: str | Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load the manifest and tables, then revalidate the complete contract."""
    source = Path(bundle_dir)
    manifest_path = source / "analysis_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"dashboard manifest not found: {manifest_path}. Run notebook Step 12 first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "dashboard feature schema does not match this checkout: "
            f"{manifest.get('feature_schema_version')!r} != {FEATURE_SCHEMA_VERSION!r}"
        )
    table_paths = manifest.get("dashboard_tables")
    if not isinstance(table_paths, dict):
        raise ValueError("dashboard manifest has no table mapping")

    tables: dict[str, pd.DataFrame] = {}
    for name, relative_path in table_paths.items():
        _validate_table_name(name)
        table_path = (source / relative_path).resolve()
        if source.resolve() not in table_path.parents:
            raise ValueError(f"dashboard table escapes bundle directory: {relative_path!r}")
        if not table_path.exists():
            raise FileNotFoundError(f"dashboard table not found: {table_path}")
        tables[name] = pd.read_csv(table_path)

    validate_dashboard_tables(tables)
    tables["monthly_summary"]["date_period"] = pd.to_datetime(
        tables["monthly_summary"]["date_period"], errors="raise", utc=True
    )
    conversations = tables["conversation_summary"]
    if "conversation_start" in conversations:
        conversations["conversation_start"] = pd.to_datetime(
            conversations["conversation_start"], errors="coerce", utc=True
        )
    if "conversation_end" in conversations:
        conversations["conversation_end"] = pd.to_datetime(
            conversations["conversation_end"], errors="coerce", utc=True
        )
    conversations["conversation_was_answered"] = (
        conversations["conversation_was_answered"].astype(str).str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .astype("boolean")
    )
    if conversations["conversation_was_answered"].isna().any():
        raise ValueError("conversation_summary contains invalid response outcomes")
    return tables, manifest


def filter_conversations(
    conversations: pd.DataFrame,
    *,
    start_date: Any | None = None,
    end_date: Any | None = None,
    answered: bool | None = None,
    topic: str | None = None,
) -> pd.DataFrame:
    """Apply dashboard drilldowns without mutating the source table."""
    result = conversations.copy()
    if start_date is not None and "conversation_start" in result:
        lower = pd.Timestamp(start_date)
        lower = lower.tz_localize("UTC") if lower.tzinfo is None else lower.tz_convert("UTC")
        result = result[result["conversation_start"] >= lower]
    if end_date is not None and "conversation_start" in result:
        upper = pd.Timestamp(end_date)
        upper = upper.tz_localize("UTC") if upper.tzinfo is None else upper.tz_convert("UTC")
        upper += pd.Timedelta(days=1)
        result = result[result["conversation_start"] < upper]
    if answered is not None:
        result = result[result["conversation_was_answered"].eq(answered)]
    if topic and topic != "All" and "most_common_topic" in result:
        result = result[result["most_common_topic"].astype(str).eq(topic)]
    return result


def preferred_sentiment_column(frame: pd.DataFrame) -> str | None:
    """Return the most informative available signed sentiment aggregate."""
    return next((
        column for column in (
            "avg_transformer_expected_sentiment", "avg_vader_compound"
        ) if column in frame.columns
    ), None)


def build_topic_priorities(
    topic_summary: pd.DataFrame, *, minimum_conversations: int = 25
) -> pd.DataFrame:
    """Rank sufficiently supported topics by low answer rate and volume."""
    required = {"topic_label", "conversations", "answer_rate"}
    if not required.issubset(topic_summary.columns):
        return pd.DataFrame(columns=[
            "topic_label", "conversations", "answer_rate", "priority_score"
        ])
    eligible = topic_summary[
        pd.to_numeric(topic_summary["conversations"], errors="coerce")
        .ge(minimum_conversations)
    ].copy()
    if eligible.empty:
        return eligible.assign(priority_score=pd.Series(dtype="float64"))
    volume = np.log1p(pd.to_numeric(eligible["conversations"], errors="coerce"))
    answer_rate = pd.to_numeric(eligible["answer_rate"], errors="coerce").clip(0, 1)
    eligible["priority_score"] = (1 - answer_rate) * volume
    return eligible.sort_values(
        ["priority_score", "conversations"], ascending=[False, False]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an exported Ubuntu dashboard bundle."
    )
    parser.add_argument(
        "bundle_dir", nargs="?", default="outputs/dashboard_data"
    )
    args = parser.parse_args()
    tables, manifest = load_dashboard_bundle(args.bundle_dir)
    print(
        f"validated {len(tables)} dashboard tables for schema "
        f"{manifest['feature_schema_version']}"
    )


if __name__ == "__main__":
    main()
