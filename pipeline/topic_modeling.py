"""Deterministic, bounded-memory topic fitting and assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass
class TopicModelBundle:
    """Reusable topic vocabulary/model for local and Databricks parity."""

    vectorizer: Any
    model: Any
    labels: list[str]
    fit_rows: int
    random_state: int


def _clean_texts(values: Iterable[object]) -> pd.Series:
    return pd.Series(values, dtype="object").fillna("").astype(str)


def fit_topic_model(
    texts: Iterable[object],
    *,
    n_topics: int = 12,
    max_features: int = 20_000,
    fit_sample_size: int = 100_000,
    random_state: int = 42,
    top_terms: int = 8,
) -> TopicModelBundle:
    """Fit one deterministic NMF model on a bounded corpus sample.

    Fit this once before a Databricks ``mapInPandas`` run and distribute the
    resulting bundle. Never fit a separate topic model inside each partition.
    """
    try:
        from sklearn.decomposition import NMF
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required. Install requirements-advanced-nlp.txt."
        ) from exc

    if n_topics < 1 or max_features < 1 or fit_sample_size < 1:
        raise ValueError("topic sizes must all be positive")
    series = _clean_texts(texts)
    scoreable = series.loc[series.str.strip().ne("")]
    if scoreable.empty:
        raise ValueError("cannot fit topics without non-empty text")
    if len(scoreable) > fit_sample_size:
        scoreable = scoreable.sample(n=fit_sample_size, random_state=random_state)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        min_df=1 if len(scoreable) < 20 else 2,
        max_df=1.0 if len(scoreable) < 20 else 0.95,
        token_pattern=r"(?u)\b\w[\w'-]+\b",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(scoreable)
    if matrix.shape[1] == 0:
        raise ValueError("topic vocabulary is empty after vectorization")

    topic_count = min(n_topics, matrix.shape[0], matrix.shape[1])
    model = NMF(
        n_components=topic_count,
        init="nndsvda",
        random_state=random_state,
        max_iter=400,
    )
    model.fit(matrix)
    terms = vectorizer.get_feature_names_out()
    labels = [
        " | ".join(terms[component.argsort()[-top_terms:][::-1]])
        for component in model.components_
    ]
    return TopicModelBundle(
        vectorizer=vectorizer,
        model=model,
        labels=labels,
        fit_rows=len(scoreable),
        random_state=random_state,
    )


def annotate_topics(
    df: pd.DataFrame,
    bundle: TopicModelBundle,
    *,
    text_col: str = "text_cleaned",
    batch_size: int = 50_000,
) -> pd.DataFrame:
    """Assign stable topics in bounded batches using one fitted bundle."""
    if text_col not in df:
        raise KeyError(f"missing topic text column: {text_col}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    texts = df[text_col].fillna("").astype(str)
    topic_ids = np.full(len(df), -1, dtype=np.int32)
    weights_out = np.full(len(df), np.nan, dtype=np.float64)
    scores = np.full(len(df), np.nan, dtype=np.float64)
    entropies = np.full(len(df), np.nan, dtype=np.float64)
    margins = np.full(len(df), np.nan, dtype=np.float64)
    vocabulary_terms = np.zeros(len(df), dtype=np.int32)
    labels: list[str | None] = [None] * len(df)

    for start in range(0, len(df), batch_size):
        stop = min(start + batch_size, len(df))
        batch = texts.iloc[start:stop]
        scoreable = batch.str.strip().ne("").to_numpy()
        if not scoreable.any():
            continue
        matrix = bundle.vectorizer.transform(batch.loc[scoreable])
        weights = bundle.model.transform(matrix)
        candidate_positions = np.flatnonzero(scoreable) + start
        vocabulary_terms[candidate_positions] = np.asarray(
            matrix.getnnz(axis=1)
        ).ravel().astype(np.int32)
        totals = weights.sum(axis=1)
        assignable = totals > 0
        if not assignable.any():
            continue
        normalized = weights[assignable] / totals[assignable, None]
        best = normalized.argmax(axis=1)
        positions = candidate_positions[assignable]
        topic_ids[positions] = best
        selected_weights = weights[assignable, :]
        weights_out[positions] = selected_weights[np.arange(len(best)), best]
        scores[positions] = normalized[np.arange(len(best)), best]
        sorted_probabilities = np.sort(normalized, axis=1)
        margins[positions] = (
            sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
            if normalized.shape[1] > 1
            else 1.0
        )
        positive = normalized > 0
        safe_probabilities = np.where(positive, normalized, 1.0)
        raw_entropy = -(
            np.where(positive, normalized * np.log(safe_probabilities), 0.0)
        ).sum(axis=1)
        divisor = np.log(normalized.shape[1]) if normalized.shape[1] > 1 else 1.0
        entropies[positions] = raw_entropy / divisor
        for position, topic_id in zip(positions, best):
            labels[int(position)] = bundle.labels[int(topic_id)]

    df["topic_id"] = pd.array(
        [None if value < 0 else int(value) for value in topic_ids],
        dtype="Int16",
    )
    df["topic_label"] = pd.Categorical(labels, categories=bundle.labels)
    df["topic_weight"] = weights_out.astype(np.float32)
    df["topic_score"] = scores.astype(np.float32)
    df["topic_entropy"] = entropies.astype(np.float32)
    df["topic_margin"] = margins.astype(np.float32)
    df["topic_vocabulary_terms"] = vocabulary_terms.astype(np.int32)
    return df
