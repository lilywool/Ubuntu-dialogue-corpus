"""Batched spaCy token, POS, and general named-entity features."""

from __future__ import annotations

import json
from collections import Counter
from functools import partial

import pandas as pd

from pipeline.parallel_execution import iter_batch_results

DEFAULT_SPACY_MODEL = "en_core_web_sm"
_SPACY_CACHE: dict[str, object] = {}
_NEGATION_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nowhere",
    "cannot", "cant", "can't", "won't", "wont", "isn't", "isnt", "wasn't",
    "wasnt", "aren't", "arent", "weren't", "werent", "doesn't", "doesnt",
    "didn't", "didnt", "don't", "dont", "hasn't", "hasnt", "haven't",
    "havent", "hadn't", "hadnt", "shouldn't", "shouldnt", "wouldn't",
    "wouldnt", "couldn't", "couldnt", "n't",
}


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_spacy(model_name: str):
    cached = _SPACY_CACHE.get(model_name)
    if cached is not None:
        return cached

    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "spaCy is required. Install requirements-advanced-nlp.txt "
            "through the repository-local interpreter."
        ) from exc

    try:
        nlp = spacy.load(model_name, exclude=["parser", "lemmatizer"])
    except OSError as exc:
        raise OSError(
            f"spaCy model {model_name!r} is unavailable. Install the pinned "
            "requirements-advanced-nlp.txt file."
        ) from exc
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)

    _SPACY_CACHE[model_name] = nlp
    return nlp


def _record_from_doc(
    doc,
    *,
    include_token_details: bool,
) -> dict:
    token_objects = [token for token in doc if not token.is_space]
    tokens = [token.text for token in token_objects]
    alpha_tokens = [token for token in token_objects if token.is_alpha]
    pos_counts = Counter(token.pos_ or "UNKNOWN" for token in token_objects)
    named_entities = [
        {
            "text": entity.text,
            "label": entity.label_,
            "start": entity.start_char,
            "end": entity.end_char,
        }
        for entity in doc.ents
    ]
    entity_label_counts = Counter(entity.label_ for entity in doc.ents)
    sentence_lengths = [
        sum(1 for token in sentence if not token.is_space)
        for sentence in doc.sents
        if any(not token.is_space for token in sentence)
    ]
    token_count = len(tokens)
    alpha_count = len(alpha_tokens)
    unique_alpha_count = len({token.lower_ for token in alpha_tokens})

    def ratio(count: int) -> float:
        return count / token_count if token_count else 0.0

    record = {
        "nlp_pos_counts": _json(dict(sorted(pos_counts.items()))),
        "named_entities": _json(named_entities),
        "named_entity_label_counts": _json(dict(sorted(entity_label_counts.items()))),
        "nlp_token_count": token_count,
        "nlp_alpha_token_count": alpha_count,
        "nlp_unique_alpha_token_count": unique_alpha_count,
        "nlp_lexical_diversity": unique_alpha_count / alpha_count if alpha_count else 0.0,
        "nlp_sentence_count": len(sentence_lengths),
        "nlp_avg_sentence_tokens": (
            sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
        ),
        "nlp_stopword_count": sum(token.is_stop for token in token_objects),
        "nlp_negation_count": sum(token.lower_ in _NEGATION_WORDS for token in token_objects),
        "nlp_punctuation_count": sum(token.is_punct for token in token_objects),
        "nlp_exclamation_count": sum(token.text == "!" for token in token_objects),
        "nlp_question_count": sum(token.text == "?" for token in token_objects),
        "nlp_uppercase_token_count": sum(
            token.text.isupper() and len(token.text) > 1 for token in token_objects
        ),
        "noun_count": pos_counts["NOUN"],
        "noun_ratio": ratio(pos_counts["NOUN"]),
        "proper_noun_count": pos_counts["PROPN"],
        "proper_noun_ratio": ratio(pos_counts["PROPN"]),
        "verb_count": pos_counts["VERB"],
        "verb_ratio": ratio(pos_counts["VERB"]),
        "adjective_count": pos_counts["ADJ"],
        "adjective_ratio": ratio(pos_counts["ADJ"]),
        "adverb_count": pos_counts["ADV"],
        "adverb_ratio": ratio(pos_counts["ADV"]),
        "named_entity_count": len(named_entities),
        "person_entity_count": entity_label_counts["PERSON"],
        "location_entity_count": sum(
            entity_label_counts[label] for label in ("GPE", "LOC", "FAC")
        ),
        "organization_entity_count": entity_label_counts["ORG"],
        "product_entity_count": entity_label_counts["PRODUCT"],
    }
    if include_token_details:
        record["nlp_tokens"] = _json(tokens)
        record["nlp_pos_tags"] = _json([
            {"token": token.text, "pos": token.pos_, "tag": token.tag_}
            for token in doc
            if not token.is_space
        ])
    return record


def _annotate_batch(
    texts: list[str],
    *,
    model_name: str,
    include_token_details: bool,
) -> list[dict]:
    """Module-level, Windows-pickleable spaCy batch worker."""
    nlp = _load_spacy(model_name)
    return [
        _record_from_doc(
            doc,
            include_token_details=include_token_details,
        )
        for doc in nlp.pipe(texts, batch_size=len(texts) or 1)
    ]


def annotate_spacy(
    df: pd.DataFrame,
    *,
    text_col: str = "text_cleaned",
    model_name: str = DEFAULT_SPACY_MODEL,
    batch_size: int = 128,
    workers: int = 1,
    include_token_details: bool = False,
) -> pd.DataFrame:
    """Add compact POS/NER features; technical terms stay lexicon-owned."""
    if text_col not in df:
        raise KeyError(f"missing spaCy text column: {text_col}")
    texts = (str(value) if pd.notna(value) else "" for value in df[text_col])
    worker = partial(
        _annotate_batch,
        model_name=model_name,
        include_token_details=include_token_details,
    )
    columns: dict[str, list] = {}
    record_count = 0
    for records in iter_batch_results(
        worker, texts, workers=workers, batch_size=batch_size
    ):
        record_count += len(records)
        for record in records:
            for column, value in record.items():
                columns.setdefault(column, []).append(value)
    if record_count != len(df):
        raise RuntimeError("spaCy annotation changed the row count")
    for column, values in columns.items():
        series = pd.Series(values, index=df.index)
        if column.endswith("_count"):
            series = series.astype("int32")
        elif column.endswith("_ratio") or column in {
            "nlp_lexical_diversity", "nlp_avg_sentence_tokens",
        }:
            series = series.astype("float32")
        df[column] = series
    return df
