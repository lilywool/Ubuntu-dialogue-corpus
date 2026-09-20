"""Cross-stage invariants for the deterministic text-cleaning pipeline."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


_MATCH_COUNT_PAIRS = (
    ("tech_lexicon_matches", "tech_lexicon_match_count"),
    ("slang_matches", "slang_match_count"),
    ("glued_matches", "glued_match_count"),
)
_PLACEHOLDER_COUNTS = {
    "EMAILADDRESS": "email_count",
    "WEBSITEDOMAIN": "domain_count",
    "PHONENUMBER": "phone_count",
    "SSN": "ssn_count",
    "IPADDRESS": "ipv4_count",
    "IPV6ADDRESS": "ipv6_count",
    "MENUPATH": "menu_path_count",
    "KEYBOARDSHORTCUT": "keyboard_shortcut_count",
}
_LEXICAL_TOKEN_RE = re.compile(r"[^\W]+(?:['-][^\W]+)*", re.UNICODE)
_MIN_TOKENS_FOR_NONWORD_RATE = 100


def _nonword_token_metrics(text: pd.Series) -> tuple[int, int, float]:
    """Count lexical tokens and explicit NONWORD placeholders."""
    token_count = 0
    nonword_count = 0
    for value in text.fillna("").astype(str):
        tokens = _LEXICAL_TOKEN_RE.findall(value)
        token_count += len(tokens)
        nonword_count += sum(token.upper() == "NONWORD" for token in tokens)
    rate = nonword_count / token_count if token_count else 0.0
    return token_count, nonword_count, rate


def validate_core_pipeline(
    df: pd.DataFrame,
    *,
    text_col: str = "text_cleaned",
    expected_rows: int | None = None,
    expected_index: pd.Index | None = None,
    maximum_nonword_token_rate: float | None = None,
) -> dict[str, Any]:
    """Fail on row drift, stale counts, or inconsistent match flags."""
    failures: list[str] = []
    metrics: dict[str, Any] = {"rows": len(df), "columns": len(df.columns)}
    if expected_rows is not None and len(df) != expected_rows:
        failures.append(f"row count changed from {expected_rows} to {len(df)}")
    if expected_index is not None and not df.index.equals(expected_index):
        failures.append("pipeline changed the DataFrame index")
    if df.columns.duplicated().any():
        failures.append("pipeline output contains duplicate column names")
    if text_col not in df:
        failures.append(f"missing cleaned text column: {text_col}")
    elif df[text_col].isna().any():
        failures.append("cleaned text contains null values")

    nonword_token_rate = None
    if maximum_nonword_token_rate is not None:
        if maximum_nonword_token_rate < 0 or maximum_nonword_token_rate > 1:
            raise ValueError("maximum_nonword_token_rate must be between 0 and 1")
        if text_col in df:
            lexical_tokens, nonword_tokens, nonword_token_rate = (
                _nonword_token_metrics(df[text_col])
            )
            metrics["lexical_tokens"] = lexical_tokens
            metrics["nonword_tokens"] = nonword_tokens
            if (
                lexical_tokens >= _MIN_TOKENS_FOR_NONWORD_RATE
                and nonword_token_rate > maximum_nonword_token_rate
            ):
                failures.append(
                    "NONWORD token rate "
                    f"{nonword_token_rate:.3%} exceeds configured maximum "
                    f"{maximum_nonword_token_rate:.3%}"
                )

    for matches_column, count_column in _MATCH_COUNT_PAIRS:
        missing = [column for column in (matches_column, count_column) if column not in df]
        if missing:
            failures.append(f"missing canonical match columns: {missing}")
            continue
        if not df[matches_column].map(lambda value: isinstance(value, list)).all():
            failures.append(f"{matches_column} contains non-list values")
            continue
        expected = df[matches_column].str.len().astype(int)
        if not expected.equals(df[count_column].astype(int)):
            failures.append(f"{count_column} does not match {matches_column}")

    missing_structural = sorted(set(_PLACEHOLDER_COUNTS.values()).difference(df.columns))
    if missing_structural:
        failures.append(f"missing structural count columns: {missing_structural}")
    elif text_col in df:
        structural_total = df[list(_PLACEHOLDER_COUNTS.values())].sum(axis=1)
        if (df[list(_PLACEHOLDER_COUNTS.values())] < 0).any(axis=None):
            failures.append("structural count columns contain negative values")
        for placeholder, count_column in _PLACEHOLDER_COUNTS.items():
            observed = df[text_col].str.count(rf"\b{placeholder}\b").astype(int)
            if not observed.equals(df[count_column].astype(int)):
                failures.append(
                    f"{count_column} does not match {placeholder} placeholders"
                )
    else:
        structural_total = pd.Series(0, index=df.index)

    required_flags = {
        "structural_matches", "has_lexicon_match", "residual_words", "has_residual"
    }
    missing_flags = sorted(required_flags.difference(df.columns))
    if missing_flags:
        failures.append(f"missing core pipeline columns: {missing_flags}")
    else:
        if not df["structural_matches"].map(lambda value: isinstance(value, list)).all():
            failures.append("structural_matches contains non-list values")
        if not df["residual_words"].map(lambda value: isinstance(value, list)).all():
            failures.append("residual_words contains non-list values")
        expected_lexicon = (
            df["tech_lexicon_match_count"].gt(0)
            | df["slang_match_count"].gt(0)
            | df["glued_match_count"].gt(0)
            | df["structural_matches"].str.len().gt(0)
            | structural_total.gt(0)
        )
        if not expected_lexicon.equals(df["has_lexicon_match"].astype(bool)):
            failures.append("has_lexicon_match is inconsistent with match columns")
        expected_residual = df["residual_words"].str.len().gt(0)
        if not expected_residual.equals(df["has_residual"].astype(bool)):
            failures.append("has_residual is inconsistent with residual_words")
        protected = set(_PLACEHOLDER_COUNTS)
        contaminated = df["glued_matches"].map(
            lambda matches: any(str(match.get("run", "")).upper() in protected for match in matches)
        )
        if contaminated.any():
            failures.append("glued matches include structural placeholder tokens")

    metrics["lexicon_matched_rows"] = (
        int(df["has_lexicon_match"].sum()) if "has_lexicon_match" in df else None
    )
    metrics["residual_rows"] = int(df["has_residual"].sum()) if "has_residual" in df else None
    metrics["nonword_token_rate"] = nonword_token_rate
    metrics["passed"] = not failures
    metrics["failures"] = failures
    if failures:
        raise AssertionError("core pipeline validation failed: " + "; ".join(failures))
    return metrics
