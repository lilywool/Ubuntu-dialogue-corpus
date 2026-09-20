"""Corpus-independent residual handling for the active pipeline.

Known lexicon and structural matches are resolved before this stage. Tokens
that are not covered remain explicit residuals. An API labels that vocabulary
when a key is available; otherwise residuals remain available for human review
or are resolved only by explicitly committed reviewed overlays.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

from pipeline.parallel_execution import map_rows

# Unicode-aware and underscore-preserving: keeps accented words and the
# pipeline's underscore-delimited placeholder labels intact.
_TOKEN_RE = re.compile(r"[^\W]+(?:['-][^\W]+)*", re.UNICODE)
_LABELS = frozenset({
    "JARGON", "NONWORD", "UNCERTAIN", "SPANISH", "PORTUGUESE", "FRENCH",
    "GERMAN", "ITALIAN", "DUTCH", "JAPANESE", "HAN_CHINESE_JAPANESE_KANJI",
    "JAPANESE_HIRAGANA", "JAPANESE_KATAKANA", "KOREAN_HANGUL", "CYRILLIC",
    "ARABIC", "HEBREW", "GREEK", "THAI", "DEVANAGARI", "ARMENIAN",
    "GEORGIAN", "ETHIOPIC", "MALAYALAM", "TAMIL", "SINHALA",
})
_STRUCTURAL_PLACEHOLDERS = frozenset({
    "EMAILADDRESS", "WEBSITEDOMAIN", "PHONENUMBER", "SSN", "IPADDRESS",
    "IPV6ADDRESS", "MENUPATH", "KEYBOARDSHORTCUT",
})
_RESERVED_TOKENS_LOWER = frozenset(
    token.lower() for token in (_LABELS | _STRUCTURAL_PLACEHOLDERS)
)
_spellchecker = None

DEFAULT_RESIDUAL_CLASSIFIER_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
RESIDUAL_CLASSIFIER_SCHEMA_VERSION = "1.0.0"
_TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


class ResidualClassifierError(RuntimeError):
    """Raised when an API classification cannot be trusted or completed."""


def residual_api_provenance(**options) -> dict[str, object]:
    """Return the non-secret, reproducibility-relevant API configuration."""
    return {
        "provider": "openai",
        "api": "responses",
        "endpoint": "/responses",
        "model": options.get("model")
        or os.getenv("RESIDUAL_CLASSIFIER_MODEL", DEFAULT_RESIDUAL_CLASSIFIER_MODEL),
        "schema_version": RESIDUAL_CLASSIFIER_SCHEMA_VERSION,
        "response_format": "strict_json_schema",
        "store": False,
        "batch_size": max(1, int(options.get("batch_size", 500))),
        "max_retries": max(0, int(options.get("max_retries", 2))),
    }


def _known_english_words(words: Iterable[str]) -> set[str]:
    """Return exact dictionary hits from the pinned English spellchecker."""
    global _spellchecker
    vocabulary = {str(word).lower() for word in words if str(word)}
    if not vocabulary:
        return set()
    if _spellchecker is None:
        try:
            from spellchecker import SpellChecker
        except ImportError as exc:
            raise ImportError(
                "English residual filtering requires pyspellchecker from "
                "the project requirements"
            ) from exc
        _spellchecker = SpellChecker()
    return {str(word).lower() for word in _spellchecker.known(vocabulary)}


@lru_cache(maxsize=1)
def _reviewed_nonword_terms() -> frozenset[str]:
    """Load only the terms explicitly committed to the NONWORD overlay."""
    path = (
        Path(__file__).parent.parent
        / "reviewed_overlays"
        / "still_unclassified_defaults_to_NONWORD.csv"
    )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "word" not in reader.fieldnames:
            raise ValueError(f"reviewed NONWORD overlay has no 'word' column: {path}")
        return frozenset(
            str(row["word"]).strip().lower()
            for row in reader
            if row.get("word") and str(row["word"]).strip()
        )


def _matched_words(row: pd.Series) -> set[str]:
    """Return the exact tokens already accounted for by deterministic matches."""
    known = set()
    for term, _category in row.get("tech_lexicon_matches", []):
        known.update(token.lower() for token in _TOKEN_RE.findall(str(term)))
    for token, _kind, _meaning in row.get("slang_matches", []):
        known.update(part.lower() for part in _TOKEN_RE.findall(str(token)))
    for match in row.get("glued_matches", []):
        known.add(str(match.get("term", "")).lower())
        known.add(str(match.get("run", "")).lower())
    for match, _category in row.get("structural_matches", []):
        known.update(token.lower() for token in _TOKEN_RE.findall(str(match)))
    return known


def _extract_row_candidates(values: tuple) -> list[str]:
    """Return unmatched candidate tokens for one row.

    This worker stays at module scope so ``ProcessPoolExecutor`` can import it
    on Windows. All multi-process execution is delegated to
    ``pipeline.parallel_execution.map_rows``.
    """
    text, tech_matches, slang_matches, glued_matches, structural_matches = values
    row = {
        "tech_lexicon_matches": tech_matches,
        "slang_matches": slang_matches,
        "glued_matches": glued_matches,
        "structural_matches": structural_matches,
    }
    known = _matched_words(row)
    return [
        token
        for token in _TOKEN_RE.findall(str(text))
        if token.lower() not in known
        and token.lower() not in _RESERVED_TOKENS_LOWER
        and not token.isdigit()
    ]


def extract_residual_vocabulary(
    df: pd.DataFrame,
    text_col: str = "text_cleaned",
    workers: int = 1,
) -> tuple[pd.DataFrame, Counter]:
    """Add explicit residual words and return their corpus counts.

    Structural placeholders are excluded because they are already resolved by
    ``apply_lexicons``. Every other unmatched token is retained for downstream
    API classification or human review.
    """
    match_columns = [
        "tech_lexicon_matches",
        "slang_matches",
        "glued_matches",
        "structural_matches",
    ]
    missing = [column for column in [text_col, *match_columns] if column not in df]
    if missing:
        raise KeyError(f"residual extraction requires columns: {missing}")

    rows = df[[text_col, *match_columns]].itertuples(index=False, name=None)
    candidates = map_rows(_extract_row_candidates, rows, workers=workers)
    known_english = _known_english_words(
        token for row_candidates in candidates for token in row_candidates
    )
    residuals = [
        [
            token for token in row_candidates
            if token.lower() not in known_english
        ]
        for row_candidates in candidates
    ]
    counts = Counter(
        token.lower()
        for row_residuals in residuals
        for token in row_residuals
    )

    df["residual_words"] = residuals
    df["has_residual"] = df["residual_words"].str.len().gt(0)
    return df, counts


def apply_api_labels(
    df: pd.DataFrame,
    labels: dict[str, str],
    text_col: str = "text_cleaned",
) -> pd.DataFrame:
    """Apply only validated API labels; leave unreturned words untouched."""
    normalized = {
        str(word).lower(): str(label).upper()
        for word, label in labels.items()
        if str(label).upper() in _LABELS
    }

    def replace(text: object) -> object:
        if not isinstance(text, str):
            return text
        parts = []
        cursor = 0
        for match in _TOKEN_RE.finditer(text):
            parts.append(text[cursor:match.start()])
            if match.group(0).isdigit():
                parts.append(match.group(0))
                cursor = match.end()
                continue
            label = normalized.get(match.group(0).lower())
            if label in {None, "JARGON", "UNCERTAIN"}:
                parts.append(match.group(0))
            else:
                parts.append(label)
            cursor = match.end()
        parts.append(text[cursor:])
        return "".join(parts)

    df[text_col] = df[text_col].map(replace)
    return df


def classify_residuals_with_api(
    df: pd.DataFrame,
    counts: Counter,
    *,
    api_key: str | None = None,
    text_col: str = "text_cleaned",
    **api_options,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Classify residuals with the API; without a key, leave them for review."""
    words = sorted(counts)
    if not words:
        return df, {}
    if not api_key:
        return df, {}

    provenance: dict[str, object] = {}
    labels = _classify_residual_words_api(
        words,
        counts,
        api_key=api_key,
        provenance=provenance,
        **api_options,
    )
    result = apply_api_labels(df, labels, text_col=text_col)
    result.attrs["residual_api_provenance"] = provenance
    return result, labels


def _reviewed_labels(words: Iterable[str]) -> dict[str, str]:
    """Resolve words with the committed human-reviewed fallback tables."""
    from reviewed_overlays.residual_classification import JARGON_TERMS, LANGUAGE_LABELS

    normalized_words = [str(word).lower() for word in words]
    language_labels = {word.lower(): label for word, label in LANGUAGE_LABELS.items()}
    reviewed_nonwords = _reviewed_nonword_terms()
    known_english = _known_english_words(normalized_words)
    labels = {}
    for normalized in normalized_words:
        if normalized in JARGON_TERMS:
            labels[normalized] = "JARGON"
        elif normalized in language_labels:
            labels[normalized] = language_labels[normalized]
        elif normalized.isdigit():
            labels[normalized] = "UNCERTAIN"
        elif normalized in known_english:
            continue
        elif normalized in reviewed_nonwords:
            labels[normalized] = "NONWORD"
    return labels


def classify_residuals(
    df: pd.DataFrame,
    counts: Counter,
    *,
    policy: str = "manual_review",
    api_key: str | None = None,
    text_col: str = "text_cleaned",
    **api_options,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Apply an explicit residual policy and report each label's source.

    ``manual_review`` leaves text untouched. ``reviewed`` uses the committed
    review tables. ``api`` requires a key and accepts only a complete,
    schema-valid API response; API labels remain traceable as review candidates.
    """
    if policy not in {"manual_review", "reviewed", "api"}:
        raise ValueError("residual policy must be 'manual_review', 'reviewed', or 'api'")

    words = sorted(counts)
    if not words or policy == "manual_review":
        return df, {}, {}

    if policy == "reviewed":
        labels = _reviewed_labels(words)
        return apply_api_labels(df, labels, text_col=text_col), labels, {
            word: "reviewed" for word in labels
        }

    if not api_key:
        raise ValueError("residual policy 'api' requires an API key")

    provenance: dict[str, object] = {}
    api_labels = _classify_residual_words_api(
        words,
        counts,
        api_key=api_key,
        provenance=provenance,
        **api_options,
    )
    result = apply_api_labels(df, api_labels, text_col=text_col)
    result.attrs["residual_api_provenance"] = provenance
    sources = {
        word: "deterministic_numeric"
        if str(word).isdigit()
        else "api_candidate"
        for word in api_labels
    }
    return result, api_labels, sources


def _classify_residual_words_api(
    words: Iterable[str],
    counts: Counter,
    *,
    api_key: str,
    provenance: dict[str, object] | None = None,
    **options,
) -> dict[str, str]:
    """Submit residual tokens in bounded batches and fail on partial output."""
    batch_size = max(1, int(options.pop("batch_size", 500)))
    all_words = list(words)
    deterministic_labels = {
        word: "UNCERTAIN" for word in all_words if str(word).isdigit()
    }
    words = [word for word in all_words if word not in deterministic_labels]
    labels = dict(deterministic_labels)
    request_hashes = []
    response_ids = []
    for start in range(0, len(words), batch_size):
        batch = words[start:start + batch_size]
        batch_labels, batch_metadata = _classify_residual_batch(
            batch,
            counts,
            api_key=api_key,
            **options,
        )
        labels.update(batch_labels)
        request_hashes.append(batch_metadata["request_sha256"])
        if batch_metadata.get("response_id"):
            response_ids.append(batch_metadata["response_id"])
    if provenance is not None:
        provenance.update(residual_api_provenance(batch_size=batch_size, **options))
        provenance.update({
            "vocabulary_size": len(all_words),
            "submitted_vocabulary_size": len(words),
            "deterministic_numeric_passthroughs": len(deterministic_labels),
            "batch_count": len(request_hashes),
            "request_sha256": request_hashes,
            "response_ids": response_ids,
            "classification_status": "candidate_for_human_review",
        })
    return labels


def _classify_residual_batch(
    words: list[str],
    counts: Counter,
    *,
    api_key: str,
    **options,
) -> tuple[dict[str, str], dict[str, object]]:
    """Submit one bounded batch through Responses and validate it completely."""
    base_url = options.get("base_url") or os.getenv(
        "OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL
    )
    model = options.get("model") or os.getenv(
        "RESIDUAL_CLASSIFIER_MODEL", DEFAULT_RESIDUAL_CLASSIFIER_MODEL
    )
    timeout = float(options.get("timeout", 120))
    max_retries = max(0, int(options.get("max_retries", 2)))
    retry_backoff = max(0.0, float(options.get("retry_backoff", 1.0)))
    items = [{"word": word, "count": int(counts[word])} for word in words]
    body = {
        "model": model,
        "instructions": (
            "Classify every supplied residual token exactly once. Use JARGON for "
            "technical terms, the matching language or script label for non-English "
            "tokens, NONWORD only for unusable fragments, and UNCERTAIN when the "
            "token alone is insufficient. Preserve each input token exactly. Do not "
            "add or omit tokens. These classifications are candidates for human review."
        ),
        "input": json.dumps(items, ensure_ascii=False, separators=(",", ":")),
        "temperature": 0,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "residual_token_classifications",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "classifications": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "word": {"type": "string"},
                                    "label": {
                                        "type": "string",
                                        "enum": sorted(_LABELS),
                                    },
                                },
                                "required": ["word", "label"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["classifications"],
                    "additionalProperties": False,
                },
            },
        },
    }
    encoded_body = json.dumps(
        body, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    request_sha256 = hashlib.sha256(encoded_body).hexdigest()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/responses",
        data=encoded_body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    payload = _send_residual_request(
        request,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )
    result = _parse_residual_response(payload, words)
    return result, {
        "request_sha256": request_sha256,
        "response_id": payload.get("id"),
    }


def _send_residual_request(
    request: urllib.request.Request,
    *,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
) -> dict[str, object]:
    """Send one request, retrying only transient failures without leaking secrets."""
    attempts = max_retries + 1
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ResidualClassifierError(
                    "OpenAI residual classification returned a non-object response"
                )
            return payload
        except urllib.error.HTTPError as exc:
            retryable = exc.code in _TRANSIENT_HTTP_STATUS_CODES
            if not retryable or attempt == max_retries:
                raise ResidualClassifierError(
                    "OpenAI residual classification failed with HTTP "
                    f"{exc.code} after {attempt + 1} attempt(s)"
                ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == max_retries:
                raise ResidualClassifierError(
                    "OpenAI residual classification failed after "
                    f"{attempt + 1} transient attempt(s)"
                ) from exc
        except json.JSONDecodeError as exc:
            raise ResidualClassifierError(
                "OpenAI residual classification returned invalid response JSON"
            ) from exc
        if retry_backoff:
            time.sleep(retry_backoff * (2 ** attempt))
    raise AssertionError("retry loop exited unexpectedly")


def _parse_residual_response(
    payload: dict[str, object], words: Iterable[str]
) -> dict[str, str]:
    """Parse Responses output and require exactly one valid label per input word."""
    status = payload.get("status")
    if status not in {None, "completed"}:
        raise ResidualClassifierError(
            f"OpenAI residual classification ended with status {status!r}"
        )
    output_items = payload.get("output", [])
    if not isinstance(output_items, list):
        raise ResidualClassifierError(
            "OpenAI residual classification violated the response schema"
        )
    text_parts = []
    for output in output_items:
        if not isinstance(output, dict) or output.get("type") != "message":
            continue
        content_items = output.get("content", [])
        if not isinstance(content_items, list):
            raise ResidualClassifierError(
                "OpenAI residual classification violated the response schema"
            )
        for content in content_items:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "refusal":
                raise ResidualClassifierError(
                    "OpenAI residual classification was refused"
                )
            if content.get("type") == "output_text" and isinstance(
                content.get("text"), str
            ):
                text_parts.append(content["text"])
    if not text_parts:
        raise ResidualClassifierError(
            "OpenAI residual classification returned no output text"
        )
    try:
        parsed = json.loads("".join(text_parts))
        classifications = parsed["classifications"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ResidualClassifierError(
            "OpenAI residual classification violated the response schema"
        ) from exc
    if not isinstance(classifications, list):
        raise ResidualClassifierError(
            "OpenAI residual classification did not return a classification list"
        )

    expected = {str(word).lower() for word in words}
    labels: dict[str, str] = {}
    for item in classifications:
        if not isinstance(item, dict):
            raise ResidualClassifierError(
                "OpenAI residual classification contained an invalid item"
            )
        word = str(item.get("word", "")).lower()
        label = str(item.get("label", "")).upper()
        if word not in expected or label not in _LABELS or word in labels:
            raise ResidualClassifierError(
                "OpenAI residual classification returned unexpected, duplicate, "
                "or invalid data"
            )
        labels[word] = label
    if set(labels) != expected:
        raise ResidualClassifierError(
            "OpenAI residual classification omitted one or more input tokens"
        )
    return labels
