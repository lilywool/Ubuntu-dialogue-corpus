"""Gold-label evaluation for sentiment backends without extra dependencies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pipeline.sentiment_analysis import SENTIMENT_LABELS


def evaluate_sentiment_labels(
    df: pd.DataFrame,
    *,
    truth_col: str,
    backend: str,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Measure classification and calibration against human labels.

    ``backend`` is ``vader`` or ``transformer``. Rows with missing predictions
    are excluded from score calculations but reduce the reported coverage.
    """
    if backend not in {"vader", "transformer"}:
        raise ValueError("backend must be 'vader' or 'transformer'")
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least 2")
    required = {
        truth_col,
        f"{backend}_label",
        f"{backend}_negative",
        f"{backend}_neutral",
        f"{backend}_positive",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"missing evaluation columns: {missing}")

    truth = df[truth_col].astype("string").str.upper()
    unknown = sorted(set(truth.dropna()).difference(SENTIMENT_LABELS))
    if unknown:
        raise ValueError(f"unknown gold sentiment labels: {unknown}")
    eligible = truth.notna()
    predicted = df[f"{backend}_label"].astype("string").str.upper()
    unknown_predictions = sorted(set(predicted.dropna()).difference(SENTIMENT_LABELS))
    if unknown_predictions:
        raise ValueError(f"unknown predicted sentiment labels: {unknown_predictions}")
    scored = eligible & predicted.notna()
    if not scored.any():
        raise ValueError("no rows have both a gold label and prediction")

    y_true = truth.loc[scored].to_numpy(dtype=str)
    y_pred = predicted.loc[scored].to_numpy(dtype=str)
    confusion = {
        actual: {
            guess: int(((y_true == actual) & (y_pred == guess)).sum())
            for guess in SENTIMENT_LABELS
        }
        for actual in SENTIMENT_LABELS
    }
    per_class: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    f1_values: list[float] = []
    for label in SENTIMENT_LABELS:
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[actual][label] for actual in SENTIMENT_LABELS if actual != label
        )
        false_negative = sum(
            confusion[label][guess] for guess in SENTIMENT_LABELS if guess != label
        )
        support = sum(confusion[label].values())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }
        recalls.append(recall)
        f1_values.append(f1)

    probabilities = df.loc[scored, [
        f"{backend}_negative", f"{backend}_neutral", f"{backend}_positive",
    ]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("evaluation probabilities contain non-finite values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("evaluation probabilities fall outside [0, 1]")
    probability_tolerance = 0.002 if backend == "vader" else 1e-6
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=probability_tolerance, rtol=0.0
    ):
        raise ValueError("evaluation probabilities do not sum to 1")
    truth_indices = np.asarray([SENTIMENT_LABELS.index(label) for label in y_true])
    one_hot = np.eye(len(SENTIMENT_LABELS), dtype=float)[truth_indices]
    brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
    log_loss = float(-np.log(np.clip(
        probabilities[np.arange(len(probabilities)), truth_indices], 1e-15, 1.0
    )).mean())
    confidence = probabilities.max(axis=1)
    correct = (y_true == y_pred).astype(float)
    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    expected_calibration_error = 0.0
    calibration: list[dict[str, float | int]] = []
    for bin_index in range(calibration_bins):
        lower, upper = edges[bin_index], edges[bin_index + 1]
        in_bin = (
            (confidence >= lower)
            & ((confidence <= upper) if bin_index == calibration_bins - 1 else (confidence < upper))
        )
        count = int(in_bin.sum())
        if not count:
            continue
        bin_accuracy = float(correct[in_bin].mean())
        bin_confidence = float(confidence[in_bin].mean())
        expected_calibration_error += count / len(confidence) * abs(
            bin_accuracy - bin_confidence
        )
        calibration.append({
            "lower": float(lower), "upper": float(upper), "count": count,
            "accuracy": bin_accuracy, "mean_confidence": bin_confidence,
        })

    return {
        "backend": backend,
        "eligible_rows": int(eligible.sum()),
        "scored_rows": int(scored.sum()),
        "coverage": float(scored.sum() / eligible.sum()),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recalls)),
        "multiclass_brier_score": brier,
        "log_loss": log_loss,
        "expected_calibration_error": float(expected_calibration_error),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "calibration_bins": calibration,
    }
