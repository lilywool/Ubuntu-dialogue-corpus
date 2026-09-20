"""Memory-efficient preparation for a pipeline DataFrame."""

from __future__ import annotations

import pandas as pd

_NUMERIC_COLUMNS = {
    "folder",
    "dialogueID",
    "dialogue_id",
    "turn_count",
}
_IDENTIFIER_COLUMNS = {"conversation_id", "message_id"}
_CATEGORICAL_COLUMNS = {"from", "to"}


def prepare_text_column(
    df: pd.DataFrame,
    *,
    text_col: str = "text_cleaned",
    source_text_col: str = "text",
) -> pd.DataFrame:
    """Create and normalize the working text column.

    Existing ``text_cleaned`` values are preserved as the starting point. A
    raw bronze frame may instead provide ``text``; it is copied before the
    reviewed normalization overlays run. Missing values become empty strings
    rather than the literal token ``"nan"``.
    """
    if text_col not in df.columns:
        if source_text_col not in df.columns:
            raise KeyError(
                f"missing text column: expected {text_col!r} or {source_text_col!r}"
            )
        df[text_col] = df[source_text_col]

    from reviewed_overlays.chat_normalization import (
        apply_latin_corrections,
        normalize_chat_shorthand,
    )

    text = df[text_col].fillna("").astype("string")
    df[text_col] = normalize_chat_shorthand(apply_latin_corrections(text))
    return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast known numeric fields and encode repeated participant fields.

    This runs before text matching so the full corpus uses less memory while
    the expensive stages are active. It leaves unknown/object text columns
    untouched rather than guessing their semantic type.
    """
    for column in df.columns:
        if column in _IDENTIFIER_COLUMNS:
            # Spark uses SHA-256 hexadecimal strings for these stable IDs.
            # Never send Arrow-backed identifier columns through to_numeric:
            # aside from changing their meaning, pandas can crash while
            # coercing very large string values inside a mapInPandas worker.
            continue
        if column in _NUMERIC_COLUMNS:
            source = df[column]
            numeric = (
                source
                if pd.api.types.is_numeric_dtype(source.dtype)
                else pd.to_numeric(source.astype("object"), errors="coerce")
            )
            if numeric.notna().all():
                df[column] = pd.to_numeric(numeric, downcast="unsigned")
        elif column in _CATEGORICAL_COLUMNS:
            df[column] = df[column].astype("category")
    return df
