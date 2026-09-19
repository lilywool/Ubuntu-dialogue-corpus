"""Rerun-safe VADER and three-class transformer sentiment features.

Probabilities remain unrounded until after validation. Transformer inference is
bounded by ``inference_chunk_size`` so millions of messages are never copied
into a second corpus-sized Python list.
"""

from __future__ import annotations

import importlib.metadata
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from pipeline.schema import FEATURE_SCHEMA_VERSION

_LATIN_LANGUAGE_LABELS = [
    "SPANISH", "PORTUGUESE", "FRENCH", "GERMAN", "ITALIAN", "DUTCH", "JAPANESE",
]
_NON_LATIN_LABELS = [
    "HAN_CHINESE_JAPANESE_KANJI", "JAPANESE_HIRAGANA", "JAPANESE_KATAKANA",
    "KOREAN_HANGUL", "CYRILLIC", "ARABIC", "HEBREW", "GREEK", "THAI",
    "DEVANAGARI", "ARMENIAN", "GEORGIAN", "ETHIOPIC", "MALAYALAM", "TAMIL",
    "SINHALA",
]
LANGUAGE_LABEL_TOKENS = tuple(_LATIN_LANGUAGE_LABELS + _NON_LATIN_LABELS)
_STRUCTURAL_LABEL_TOKENS = [
    "EMAILADDRESS", "WEBSITEDOMAIN", "PHONENUMBER", "SSN", "IPADDRESS",
    "IPV6ADDRESS", "MENUPATH", "KEYBOARDSHORTCUT",
]
KNOWN_LABEL_TOKENS = (
    ["NONWORD", "JARGON", "UNCERTAIN"]
    + _LATIN_LANGUAGE_LABELS + _NON_LATIN_LABELS + _STRUCTURAL_LABEL_TOKENS
)
_LABEL_STRIP_PATTERN = re.compile(
    r"\b(?:" + "|".join(
        re.escape(token) for token in sorted(KNOWN_LABEL_TOKENS, key=len, reverse=True)
    ) + r")\b"
)
_ALPHA_RE = re.compile(r"[^\W\d_]")

DEFAULT_TRANSFORMER_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_TRANSFORMER_REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"
SENTIMENT_LABELS = ("NEGATIVE", "NEUTRAL", "POSITIVE")
VADER_COLUMNS = (
    "vader_negative", "vader_neutral", "vader_positive",
    "vader_compound", "vader_label",
)
TRANSFORMER_COLUMNS = (
    "transformer_negative", "transformer_neutral", "transformer_positive",
    "transformer_label", "transformer_score", "transformer_expected_sentiment",
    "transformer_entropy", "transformer_normalized_entropy", "transformer_margin",
)
SENTIMENT_GENERATED_COLUMNS = VADER_COLUMNS + TRANSFORMER_COLUMNS

_TRANSFORMER_LABEL_MAP = {
    "negative": "NEGATIVE", "neutral": "NEUTRAL", "positive": "POSITIVE",
    "label_0": "NEGATIVE", "label_1": "NEUTRAL", "label_2": "POSITIVE",
}
_vader_analyzer = None
_transformer_pipeline = None
_transformer_pipeline_key = None
_transformer_runtime_metadata: dict[str, Any] = {}


def strip_label_tokens(text: object) -> object:
    """Remove upstream all-caps placeholder labels from one value."""
    if not isinstance(text, str) or not text:
        return text
    return _LABEL_STRIP_PATTERN.sub("", text)


def strip_label_tokens_series(series: pd.Series) -> pd.Series:
    return series.str.replace(_LABEL_STRIP_PATTERN, "", regex=True)


def is_effectively_empty(text: object) -> bool:
    return not isinstance(text, str) or _ALPHA_RE.search(text) is None


def clear_sentiment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every stage-owned column before a rerun."""
    existing = [column for column in SENTIMENT_GENERATED_COLUMNS if column in df]
    if existing:
        df.drop(columns=existing, inplace=True)
    df.attrs.pop("sentiment_provenance", None)
    df.attrs.pop("sentiment_validation", None)
    return df


def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def _vader_label(compound: float) -> str:
    if compound >= 0.05:
        return "POSITIVE"
    if compound <= -0.05:
        return "NEGATIVE"
    return "NEUTRAL"


def score_vader(text: object) -> dict[str, Any]:
    """Return VADER's full distribution, compound score, and label."""
    if is_effectively_empty(text):
        return {
            column: (None if column == "vader_label" else math.nan)
            for column in VADER_COLUMNS
        }
    raw = _get_vader().polarity_scores(str(text))
    compound = float(raw["compound"])
    return {
        "vader_negative": float(raw["neg"]),
        "vader_neutral": float(raw["neu"]),
        "vader_positive": float(raw["pos"]),
        "vader_compound": compound,
        "vader_label": _vader_label(compound),
    }


def score_vader_series(series: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(
        series.map(score_vader).tolist(), index=series.index, columns=VADER_COLUMNS
    )
    numeric = [column for column in VADER_COLUMNS if column != "vader_label"]
    frame[numeric] = frame[numeric].astype(np.float32)
    frame["vader_label"] = pd.Categorical(
        frame["vader_label"], categories=SENTIMENT_LABELS
    )
    return frame


def get_transformer_pipeline(
    model_name: str = DEFAULT_TRANSFORMER_MODEL,
    *, revision: str = DEFAULT_TRANSFORMER_REVISION,
    device: int | None = None,
    dtype: str = "float32",
):
    """Load one revision-pinned model with an explicit inference dtype."""
    global _transformer_pipeline, _transformer_pipeline_key, _transformer_runtime_metadata
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except ImportError as exc:
        raise ImportError(
            "Transformer sentiment requires requirements-transformer.txt in the local .venv."
        ) from exc
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    dtype_map = {
        "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        raise ValueError("dtype must be 'float32', 'float16', or 'bfloat16'")
    if device == -1 and dtype == "float16":
        raise ValueError("float16 transformer inference requires a supported GPU")
    cache_key = (model_name, revision, device, dtype)
    if _transformer_pipeline is not None and _transformer_pipeline_key == cache_key:
        return _transformer_pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=revision, torch_dtype=dtype_map[dtype]
    )
    _transformer_pipeline = pipeline(
        task="text-classification", model=model, tokenizer=tokenizer,
        device=device, truncation=True, max_length=512,
    )
    _transformer_pipeline_key = cache_key
    _transformer_runtime_metadata = {
        "resolved_device": str(_transformer_pipeline.device),
        "resolved_model_dtype": str(next(model.parameters()).dtype),
    }
    return _transformer_pipeline


def _canonical_label(raw_label: object) -> str:
    key = str(raw_label).strip().lower()
    try:
        return _TRANSFORMER_LABEL_MAP[key]
    except KeyError as exc:
        raise ValueError(f"unrecognized transformer sentiment label: {raw_label!r}") from exc


def _probability_features(probabilities: dict[str, float]) -> dict[str, Any]:
    missing = set(SENTIMENT_LABELS).difference(probabilities)
    if missing:
        raise ValueError(f"missing sentiment probabilities: {sorted(missing)}")
    values = np.asarray([probabilities[label] for label in SENTIMENT_LABELS], dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError(f"invalid transformer probability vector: {values.tolist()}")
    total = float(values.sum())
    if not np.isclose(total, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError(f"transformer probabilities sum to {total}, not 1")
    values /= total
    best_index = int(values.argmax())
    ordered = np.sort(values)
    entropy = float(-(values[values > 0] * np.log(values[values > 0])).sum())
    return {
        "transformer_negative": float(values[0]),
        "transformer_neutral": float(values[1]),
        "transformer_positive": float(values[2]),
        "transformer_label": SENTIMENT_LABELS[best_index],
        "transformer_score": float(values[best_index]),
        "transformer_expected_sentiment": float(values[2] - values[0]),
        "transformer_entropy": entropy,
        "transformer_normalized_entropy": float(entropy / math.log(3.0)),
        "transformer_margin": float(ordered[-1] - ordered[-2]),
    }


def _features_from_pipeline_result(result: object) -> dict[str, Any]:
    """Normalize one Hugging Face all-label result into stable columns."""
    if isinstance(result, dict):
        result = [result]
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        result = result[0]
    if not isinstance(result, list):
        raise TypeError(f"unexpected transformer result type: {type(result).__name__}")
    probabilities: dict[str, float] = {}
    for item in result:
        if not isinstance(item, dict) or "label" not in item or "score" not in item:
            raise ValueError(f"malformed transformer result: {item!r}")
        probabilities[_canonical_label(item["label"])] = float(item["score"])
    return _probability_features(probabilities)


def score_transformer_series(
    series: pd.Series,
    *, model_name: str = DEFAULT_TRANSFORMER_MODEL,
    revision: str = DEFAULT_TRANSFORMER_REVISION,
    batch_size: int = 32,
    inference_chunk_size: int = 4096,
    device: int | None = None,
    dtype: str = "float32",
) -> pd.DataFrame:
    """Score all three classes in bounded chunks; never round outputs."""
    if batch_size < 1 or inference_chunk_size < 1:
        raise ValueError("transformer batch sizes must be positive")
    classifier = get_transformer_pipeline(
        model_name, revision=revision, device=device, dtype=dtype
    )
    output: dict[str, list[Any]] = {
        column: [None if column == "transformer_label" else math.nan] * len(series)
        for column in TRANSFORMER_COLUMNS
    }
    pending_positions: list[int] = []
    pending_texts: list[str] = []

    def flush() -> None:
        if not pending_texts:
            return
        results = classifier(pending_texts, batch_size=batch_size, top_k=None)
        if isinstance(results, dict):
            results = [results]
        if len(results) != len(pending_texts):
            raise RuntimeError("transformer inference changed the row count")
        for position, result in zip(pending_positions, results):
            for column, value in _features_from_pipeline_result(result).items():
                output[column][position] = value
        pending_positions.clear()
        pending_texts.clear()

    for position, raw_text in enumerate(series):
        text = "" if pd.isna(raw_text) else str(raw_text)
        if is_effectively_empty(text):
            continue
        pending_positions.append(position)
        pending_texts.append(text)
        if len(pending_texts) >= inference_chunk_size:
            flush()
    flush()
    frame = pd.DataFrame(output, index=series.index)
    numeric = [column for column in TRANSFORMER_COLUMNS if column != "transformer_label"]
    frame[numeric] = frame[numeric].astype(np.float32)
    frame["transformer_label"] = pd.Categorical(
        frame["transformer_label"], categories=SENTIMENT_LABELS
    )
    return frame


def _validate_transformer_distribution(
    frame: pd.DataFrame, scoreable: pd.Series,
    failures: list[str], metrics: dict[str, Any],
) -> None:
    missing = sorted(set(TRANSFORMER_COLUMNS).difference(frame.columns))
    if missing:
        failures.append(f"missing transformer sentiment columns: {missing}")
        return
    if frame.loc[~scoreable, list(TRANSFORMER_COLUMNS)].notna().any(axis=None):
        failures.append("unscoreable rows contain transformer sentiment output")
    scored = frame.loc[scoreable]
    probability_columns = [
        "transformer_negative", "transformer_neutral", "transformer_positive"
    ]
    probabilities = scored[probability_columns]
    complete = probabilities.notna().all(axis=1)
    metrics["transformer_rows_scored"] = int(complete.sum())
    if not complete.all():
        failures.append("some scoreable rows have incomplete transformer probabilities")
    scored = scored.loc[complete]
    probabilities = probabilities.loc[complete]
    if scored.empty:
        return
    values = probabilities.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        failures.append("transformer probabilities contain invalid values")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        failures.append("transformer probabilities do not sum to 1")
    labels = np.asarray(SENTIMENT_LABELS, dtype=object)[values.argmax(axis=1)]
    if not np.array_equal(scored["transformer_label"].to_numpy(), labels):
        failures.append("transformer labels do not match probability argmax")
    scores = values.max(axis=1)
    if not np.allclose(scored["transformer_score"], scores, atol=1e-6, rtol=0.0):
        failures.append("transformer scores do not match maximum probability")
    signed = values[:, 2] - values[:, 0]
    if not np.allclose(
        scored["transformer_expected_sentiment"], signed, atol=1e-6, rtol=0.0
    ):
        failures.append("transformer expected sentiment is inconsistent with probabilities")
    entropy = scored["transformer_normalized_entropy"]
    margin = scored["transformer_margin"]
    if ((entropy < 0) | (entropy > 1) | ~np.isfinite(entropy)).any():
        failures.append("transformer normalized entropy falls outside [0, 1]")
    if ((margin < 0) | (margin > 1) | ~np.isfinite(margin)).any():
        failures.append("transformer margin falls outside [0, 1]")
    unique_scores = int(scored["transformer_score"].nunique())
    metrics["transformer_score_unique"] = unique_scores
    if len(scored) >= 100 and unique_scores < 10:
        failures.append("transformer sentiment scores have suspiciously low diversity")


def validate_sentiment(
    df: pd.DataFrame, *, text_col: str = "text_cleaned", mode: str = "both"
) -> dict[str, Any]:
    """Fail closed on incomplete, inconsistent, or collapsed output."""
    if mode not in {"none", "vader", "transformer", "both"}:
        raise ValueError("invalid sentiment mode")
    if text_col not in df:
        raise KeyError(f"missing sentiment text column: {text_col}")
    stripped = strip_label_tokens_series(df[text_col].fillna("").astype(str))
    scoreable = ~stripped.map(is_effectively_empty)
    failures: list[str] = []
    metrics: dict[str, Any] = {
        "sentiment_rows": len(df), "scoreable_rows": int(scoreable.sum())
    }
    if mode in {"vader", "both"}:
        missing = sorted(set(VADER_COLUMNS).difference(df.columns))
        if missing:
            failures.append(f"missing VADER columns: {missing}")
        else:
            if df.loc[~scoreable, list(VADER_COLUMNS)].notna().any(axis=None):
                failures.append("unscoreable rows contain VADER output")
            scored = df.loc[scoreable]
            probability_columns = ["vader_negative", "vader_neutral", "vader_positive"]
            probabilities = scored[probability_columns]
            complete = probabilities.notna().all(axis=1) & scored["vader_compound"].notna()
            metrics["vader_rows_scored"] = int(complete.sum())
            if not complete.all():
                failures.append("some scoreable rows have incomplete VADER output")
            scored = scored.loc[complete]
            probabilities = probabilities.loc[complete]
            if not scored.empty:
                values = probabilities.to_numpy(dtype=float)
                if ((values < 0) | (values > 1) | ~np.isfinite(values)).any():
                    failures.append("VADER probabilities contain invalid values")
                # vaderSentiment exposes its proportions rounded to 3 decimals.
                if not np.allclose(values.sum(axis=1), 1.0, atol=0.002, rtol=0.0):
                    failures.append("VADER probabilities do not sum to 1")
                compounds = scored["vader_compound"].to_numpy(dtype=float)
                if ((compounds < -1) | (compounds > 1) | ~np.isfinite(compounds)).any():
                    failures.append("VADER compound scores fall outside [-1, 1]")
                labels = np.asarray([_vader_label(value) for value in compounds])
                if not np.array_equal(scored["vader_label"].to_numpy(), labels):
                    failures.append("VADER labels do not match compound thresholds")
                unique_scores = int(scored["vader_compound"].nunique())
                metrics["vader_score_unique"] = unique_scores
                if len(scored) >= 100 and unique_scores < 10:
                    failures.append("VADER scores have suspiciously low diversity")
    if mode in {"transformer", "both"}:
        _validate_transformer_distribution(df, scoreable, failures, metrics)
    if mode == "both" and set(VADER_COLUMNS + TRANSFORMER_COLUMNS).issubset(df.columns):
        comparable = scoreable & df["vader_label"].notna() & df["transformer_label"].notna()
        if comparable.any():
            metrics["backend_label_agreement"] = float(
                (df.loc[comparable, "vader_label"] == df.loc[comparable, "transformer_label"]).mean()
            )
            vader_scores = df.loc[comparable, "vader_compound"]
            transformer_scores = df.loc[comparable, "transformer_expected_sentiment"]
            if vader_scores.nunique() > 1 and transformer_scores.nunique() > 1:
                metrics["backend_score_pearson"] = float(
                    vader_scores.corr(transformer_scores)
                )
    metrics["passed"] = not failures
    metrics["failures"] = failures
    if failures:
        raise AssertionError("sentiment validation failed: " + "; ".join(failures))
    return metrics


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def analyze_sentiment(
    df: pd.DataFrame,
    text_column: str = "text_cleaned",
    mode: str = "both",
    *, transformer_model: str = DEFAULT_TRANSFORMER_MODEL,
    transformer_revision: str = DEFAULT_TRANSFORMER_REVISION,
    transformer_batch_size: int = 32,
    transformer_inference_chunk_size: int = 4096,
    transformer_device: int | None = None,
    transformer_dtype: str = "float32",
    validate: bool = True,
) -> pd.DataFrame:
    """Run selected backends, replace stale outputs, and attach provenance."""
    if mode not in {"vader", "transformer", "both"}:
        raise ValueError("mode must be 'vader', 'transformer', or 'both'")
    if text_column not in df:
        raise KeyError(f"missing sentiment text column: {text_column}")
    original_index = df.index.copy()
    clear_sentiment_columns(df)
    stripped = strip_label_tokens_series(df[text_column].fillna("").astype(str))
    if mode in {"vader", "both"}:
        result = score_vader_series(stripped)
        for column in VADER_COLUMNS:
            df[column] = result[column]
    if mode in {"transformer", "both"}:
        result = score_transformer_series(
            stripped, model_name=transformer_model, revision=transformer_revision,
            batch_size=transformer_batch_size,
            inference_chunk_size=transformer_inference_chunk_size,
            device=transformer_device, dtype=transformer_dtype,
        )
        for column in TRANSFORMER_COLUMNS:
            df[column] = result[column]
    provenance: dict[str, Any] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "mode": mode, "text_column": text_column,
        "preprocessing": "strip_known_placeholder_tokens_then_skip_no_alphabetic_content",
        "unrounded": True,
    }
    if mode in {"vader", "both"}:
        provenance["vader"] = {"package_version": _package_version("vaderSentiment")}
    if mode in {"transformer", "both"}:
        provenance["transformer"] = {
            "model": transformer_model, "revision": transformer_revision,
            "dtype": transformer_dtype, "requested_device": transformer_device,
            "batch_size": transformer_batch_size,
            "inference_chunk_size": transformer_inference_chunk_size,
            "transformers_version": _package_version("transformers"),
            "torch_version": _package_version("torch"),
            "label_order": list(SENTIMENT_LABELS),
            **_transformer_runtime_metadata,
        }
    df.attrs["sentiment_provenance"] = provenance
    if validate:
        df.attrs["sentiment_validation"] = validate_sentiment(
            df, text_col=text_column, mode=mode
        )
    if not df.index.equals(original_index):
        raise RuntimeError("sentiment analysis changed the DataFrame index")
    return df


def benchmark_transformer(
    series: pd.Series,
    *, sample_n: int = 2000, batch_size: int = 32,
    inference_chunk_size: int = 4096, device: int | None = None,
    dtype: str = "float32", random_state: int = 0,
) -> tuple[float, float, float]:
    """Measure a deterministic sample before committing to a full run."""
    import time
    sample = series.sample(n=min(sample_n, len(series)), random_state=random_state)
    stripped = strip_label_tokens_series(sample.fillna("").astype(str))
    started = time.perf_counter()
    score_transformer_series(
        stripped, batch_size=batch_size,
        inference_chunk_size=inference_chunk_size, device=device, dtype=dtype,
    )
    elapsed = time.perf_counter() - started
    rate = len(sample) / elapsed if elapsed else math.inf
    projected = len(series) / rate if rate else math.inf
    print(f"{len(sample):,} rows in {elapsed:.1f}s -> {rate:.1f} rows/sec")
    print(f"projected for full corpus ({len(series):,} rows): {projected / 60:.1f} minutes")
    return elapsed, rate, projected
