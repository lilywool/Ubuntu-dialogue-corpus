"""Optional seven-class emotion probabilities with bounded inference."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from pipeline.sentiment_analysis import is_effectively_empty, strip_label_tokens_series

DEFAULT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_EMOTION_REVISION = "cea2f78f197f0337186a0faa93e00ef93811f6eb"
EMOTION_LABELS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
EMOTION_PROBABILITY_COLUMNS = tuple(f"emotion_{label}" for label in EMOTION_LABELS)
EMOTION_COLUMNS = EMOTION_PROBABILITY_COLUMNS + (
    "emotion_label", "emotion_score", "emotion_entropy",
    "emotion_normalized_entropy", "emotion_margin",
)
_EMOTION_PIPELINE = None
_EMOTION_PIPELINE_KEY = None
_EMOTION_RUNTIME_METADATA: dict[str, str] = {}


def get_emotion_pipeline(
    *, model_name: str = DEFAULT_EMOTION_MODEL,
    revision: str = DEFAULT_EMOTION_REVISION,
    device: int | None = None,
    dtype: str = "float32",
):
    """Load one revision-pinned model; GPU detection never implies fp16."""
    global _EMOTION_PIPELINE, _EMOTION_PIPELINE_KEY, _EMOTION_RUNTIME_METADATA
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except ImportError as exc:
        raise ImportError(
            "Emotion classification requires requirements-transformer.txt."
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
        raise ValueError("float16 emotion inference requires a supported GPU")
    key = (model_name, revision, device, dtype)
    if _EMOTION_PIPELINE is not None and _EMOTION_PIPELINE_KEY == key:
        return _EMOTION_PIPELINE
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=revision, torch_dtype=dtype_map[dtype]
    )
    _EMOTION_PIPELINE = pipeline(
        "text-classification", model=model, tokenizer=tokenizer,
        device=device, truncation=True, max_length=512,
    )
    _EMOTION_PIPELINE_KEY = key
    _EMOTION_RUNTIME_METADATA = {
        "resolved_device": str(_EMOTION_PIPELINE.device),
        "resolved_model_dtype": str(next(model.parameters()).dtype),
    }
    return _EMOTION_PIPELINE


def emotion_runtime_metadata() -> dict[str, str]:
    return dict(_EMOTION_RUNTIME_METADATA)


def _emotion_features(result: object) -> dict[str, Any]:
    if isinstance(result, dict):
        result = [result]
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        result = result[0]
    if not isinstance(result, list):
        raise TypeError(f"unexpected emotion result type: {type(result).__name__}")
    probabilities = {str(item["label"]).lower(): float(item["score"]) for item in result}
    missing = set(EMOTION_LABELS).difference(probabilities)
    if missing:
        raise ValueError(f"missing emotion probabilities: {sorted(missing)}")
    values = np.asarray([probabilities[label] for label in EMOTION_LABELS], dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError(f"invalid emotion probability vector: {values.tolist()}")
    total = float(values.sum())
    if not np.isclose(total, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError(f"emotion probabilities sum to {total}, not 1")
    values /= total
    best = int(values.argmax())
    ordered = np.sort(values)
    entropy = float(-(values[values > 0] * np.log(values[values > 0])).sum())
    record = {
        column: float(value)
        for column, value in zip(EMOTION_PROBABILITY_COLUMNS, values)
    }
    record.update({
        "emotion_label": EMOTION_LABELS[best].upper(),
        "emotion_score": float(values[best]),
        "emotion_entropy": entropy,
        "emotion_normalized_entropy": float(entropy / math.log(len(EMOTION_LABELS))),
        "emotion_margin": float(ordered[-1] - ordered[-2]),
    })
    return record


def score_emotion_series(
    series: pd.Series,
    *, model_name: str = DEFAULT_EMOTION_MODEL,
    revision: str = DEFAULT_EMOTION_REVISION,
    batch_size: int = 32,
    inference_chunk_size: int = 4096,
    device: int | None = None,
    dtype: str = "float32",
) -> pd.DataFrame:
    """Score every class in bounded chunks without corpus-sized text copies."""
    if batch_size < 1 or inference_chunk_size < 1:
        raise ValueError("emotion batch sizes must be positive")
    classifier = get_emotion_pipeline(
        model_name=model_name, revision=revision, device=device, dtype=dtype
    )
    stripped = strip_label_tokens_series(series.fillna("").astype(str))
    output: dict[str, list[Any]] = {
        column: [None if column == "emotion_label" else math.nan] * len(series)
        for column in EMOTION_COLUMNS
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
            raise RuntimeError("emotion inference changed the row count")
        for position, result in zip(pending_positions, results):
            for column, value in _emotion_features(result).items():
                output[column][position] = value
        pending_positions.clear()
        pending_texts.clear()

    for position, text in enumerate(stripped):
        if is_effectively_empty(text):
            continue
        pending_positions.append(position)
        pending_texts.append(text)
        if len(pending_texts) >= inference_chunk_size:
            flush()
    flush()
    frame = pd.DataFrame(output, index=series.index)
    numeric = [column for column in EMOTION_COLUMNS if column != "emotion_label"]
    frame[numeric] = frame[numeric].astype(np.float32)
    frame["emotion_label"] = pd.Categorical(
        frame["emotion_label"], categories=[label.upper() for label in EMOTION_LABELS]
    )
    return frame
