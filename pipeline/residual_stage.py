"""Corpus-independent residual handling for the active pipeline.

Known lexicon and structural matches are resolved before this stage. Tokens
that are not covered remain explicit residuals. An API labels that vocabulary
when a key is available; otherwise the deterministic fallback is NONWORD while
the original residual list remains available for review or downstream use.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
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

    labels = _classify_residual_words_api(words, counts, api_key=api_key, **api_options)
    return apply_api_labels(df, labels, text_col=text_col), labels


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
    review tables. ``api`` requires a key and falls back to those tables only
    for words omitted from a valid API response.
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

    api_labels = _classify_residual_words_api(words, counts, api_key=api_key, **api_options)
    missing = [word for word in words if word not in api_labels]
    fallback_labels = _reviewed_labels(missing)
    labels = {**fallback_labels, **api_labels}
    sources = {word: "reviewed_fallback" for word in fallback_labels}
    sources.update({word: "api" for word in api_labels})
    return apply_api_labels(df, labels, text_col=text_col), labels, sources


def _classify_residual_words_api(words: Iterable[str], counts: Counter, *, api_key: str, **options) -> dict[str, str]:
    """Submit residual tokens in bounded batches."""
    batch_size = max(1, int(options.pop("batch_size", 500)))
    words = list(words)
    labels = {}
    for start in range(0, len(words), batch_size):
        batch = words[start:start + batch_size]
        labels.update(
            _classify_residual_batch(batch, counts, api_key=api_key, **options)
        )
    return labels


def _classify_residual_batch(words: list[str], counts: Counter, *, api_key: str, **options) -> dict[str, str]:
    """Submit one bounded residual-token batch and validate its response."""
    base_url = options.get("base_url") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = options.get("model") or os.getenv("RESIDUAL_CLASSIFIER_MODEL", "gpt-4o-mini")
    timeout = options.get("timeout", 120)
    labels = ", ".join(sorted(_LABELS))
    items = [{"word": word, "count": int(counts[word])} for word in words]
    seed_material = json.dumps(items, ensure_ascii=False, sort_keys=True).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
    prompt = (
        "Return JSON only as an object mapping each exact residual token to one label. "
        f"Allowed labels: {labels}. Use JARGON for technical terms, a language label "
        "for non-English words, NONWORD for unusable fragments, and UNCERTAIN when "
        "context is insufficient. Do not invent words or labels.\n\n"
        + json.dumps(items, ensure_ascii=False)
    )
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": seed,
            "response_format": {"type": "json_object"},
        }).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    try:
        result = json.loads(payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(result, dict):
        return {}
    expected = set(words)
    return {
        str(word).lower(): str(label).upper()
        for word, label in result.items()
        if str(word).lower() in expected and str(label).upper() in _LABELS
    }
