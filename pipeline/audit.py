"""Append-only, JSON-lines audit records for validated pipeline outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pipeline.schema import FEATURE_SCHEMA_VERSION


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


def append_run_audit(
    log_path: str | Path,
    *,
    df: pd.DataFrame,
    stage: str,
    output_path: str | Path,
    validation: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one record only after validation and output creation succeed."""
    if not validation.get("passed", False):
        raise ValueError("refusing to audit an output that did not pass validation")
    output = Path(output_path)
    if not output.exists():
        raise FileNotFoundError(f"cannot audit missing output: {output}")
    record = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "stage": stage,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "output_path": str(output.resolve()),
        "output_bytes": int(output.stat().st_size),
        "validation": _json_safe(validation),
        "config": _json_safe(config or {}),
        "sentiment_provenance": _json_safe(df.attrs.get("sentiment_provenance", {})),
        "advanced_nlp_provenance": _json_safe(
            df.attrs.get("advanced_nlp_provenance", {})
        ),
        "topic_model_metadata": _json_safe(df.attrs.get("topic_model_metadata", {})),
        "sample_provenance": _json_safe(df.attrs.get("sample_provenance", {})),
        "gold_provenance": _json_safe(df.attrs.get("gold_provenance", {})),
    }
    destination = Path(log_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    return record
