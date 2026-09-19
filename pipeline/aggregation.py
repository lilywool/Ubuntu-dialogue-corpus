"""Validated silver-to-gold aggregation for local pandas workflows."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.feature_engineering import engineer_features
from pipeline.schema import FEATURE_SCHEMA_VERSION
from pipeline.sentiment_analysis import LANGUAGE_LABEL_TOKENS

DATE_GRANULARITIES = {"day", "week", "month"}
RELEASE_AXES = {"since", "until"}
GOLD_LEVELS = {
    "conversation", "user", "date", "channel", "release", "language",
    "residual", "technical", "topic", "entity", "sample",
}

_AVERAGE_COLUMNS = {
    "text_length": "avg_text_length",
    "word_count": "avg_word_count",
    "response_gap_mins_between_speakers": "avg_response_gap_mins_between_speakers",
    "vader_compound": "avg_vader_compound",
    "vader_negative": "avg_vader_negative",
    "vader_neutral": "avg_vader_neutral",
    "vader_positive": "avg_vader_positive",
    "transformer_expected_sentiment": "avg_transformer_expected_sentiment",
    "transformer_negative": "avg_transformer_negative",
    "transformer_neutral": "avg_transformer_neutral",
    "transformer_positive": "avg_transformer_positive",
    "transformer_score": "avg_transformer_confidence",
    "transformer_normalized_entropy": "avg_transformer_normalized_entropy",
    "emotion_score": "avg_emotion_confidence",
    "emotion_normalized_entropy": "avg_emotion_normalized_entropy",
    "topic_score": "avg_topic_confidence",
    "topic_entropy": "avg_topic_entropy",
}
_SUM_COLUMNS = {
    "tech_lexicon_match_count": "total_tech_lexicon_matches",
    "slang_match_count": "total_slang_matches",
    "glued_match_count": "total_glued_matches",
    "named_entity_count": "total_named_entities",
}
_LABEL_COLUMNS = {
    "vader_label": ("NEGATIVE", "NEUTRAL", "POSITIVE"),
    "transformer_label": ("NEGATIVE", "NEUTRAL", "POSITIVE"),
    "emotion_label": ("ANGER", "DISGUST", "FEAR", "JOY", "NEUTRAL", "SADNESS", "SURPRISE"),
}


def _parse_collection(value: Any, *, column: str, strict: bool = True) -> list[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            if strict:
                raise ValueError(f"malformed serialized collection in {column}: {value!r}") from exc
            return []
    if not isinstance(parsed, (list, tuple)):
        if strict:
            raise ValueError(f"expected a list-like value in {column}: {value!r}")
        return []
    return list(parsed)


def _technical_records(value: Any, *, strict: bool = True) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for item in _parse_collection(value, column="tech_lexicon_matches", strict=strict):
        if isinstance(item, dict):
            term, category = item.get("term"), item.get("category")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            term, category = item[0], item[1]
        else:
            if strict:
                raise ValueError(f"malformed technical match: {item!r}")
            continue
        if term:
            records.append({"entity": str(term), "entity_label": str(category or "UNKNOWN")})
    return records


def _named_entity_records(value: Any, *, strict: bool = True) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for item in _parse_collection(value, column="named_entities", strict=strict):
        if not isinstance(item, dict) or not item.get("text"):
            if strict:
                raise ValueError(f"malformed named entity: {item!r}")
            continue
        records.append({
            "entity": str(item["text"]),
            "entity_label": str(item.get("label") or "UNKNOWN"),
        })
    return records


def _entity_records(row: pd.Series, *, strict: bool = True) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if "tech_lexicon_matches" in row.index:
        records.extend(
            {**record, "entity_source": "technical_lexicon"}
            for record in _technical_records(row["tech_lexicon_matches"], strict=strict)
        )
    if "named_entities" in row.index:
        records.extend(
            {**record, "entity_source": "spacy_ner"}
            for record in _named_entity_records(row["named_entities"], strict=strict)
        )
    return records


def _mode(series: pd.Series) -> str | None:
    modes = series.dropna().astype(str).mode()
    return str(modes.iloc[0]) if not modes.empty else None


def _base_aggregate(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if not group_columns or not set(group_columns).issubset(frame.columns):
        raise KeyError(f"missing aggregation keys: {group_columns}")
    working = frame.copy()
    working["_row_marker"] = np.int8(1)
    grouped = working.groupby(
        group_columns, sort=False, observed=True, dropna=False
    )
    aggregations: dict[str, tuple[str, str]] = {
        "message_count": ("_row_marker", "count")
    }
    for column, target in _AVERAGE_COLUMNS.items():
        if column in working:
            aggregations[target] = (column, "mean")
    for column, target in _SUM_COLUMNS.items():
        if column in working:
            aggregations[target] = (column, "sum")
    for column, target in (
        ("conversation_id", "unique_conversation_count"),
        ("from", "unique_sender_count"),
        ("to", "unique_recipient_count"),
    ):
        if column in working and column not in group_columns:
            aggregations[target] = (column, "nunique")
    result = grouped.agg(**aggregations).reset_index()

    for source, labels in _LABEL_COLUMNS.items():
        if source not in working:
            continue
        result[f"most_common_{source}"] = grouped[source].agg(_mode).to_numpy()
        for label in labels:
            indicator = f"_share_{source}_{label.lower()}"
            working[indicator] = working[source].astype("string").eq(label).astype("float32")
            shares = working.groupby(
                group_columns, sort=False, observed=True, dropna=False
            )[indicator].mean().reset_index(name=f"share_{source}_{label.lower()}")
            result = result.merge(shares, on=group_columns, how="left", validate="one_to_one")
    if "topic_label" in working:
        result["most_common_topic"] = grouped["topic_label"].agg(_mode).to_numpy()
    return result


def _prefix_metrics(frame: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    return frame.rename(
        columns={column: f"{prefix}_{column}" for column in frame.columns if column != key}
    )


def aggregate_conversations(silver_df: pd.DataFrame, engineer: bool = True) -> pd.DataFrame:
    frame = engineer_features(silver_df) if engineer else silver_df.copy()
    if "conversation_id" not in frame:
        raise KeyError("conversation aggregate requires conversation_id")
    result = _base_aggregate(frame, ["conversation_id"])
    dates = pd.to_datetime(frame.get("date"), errors="coerce", utc=True)
    if dates is not None:
        bounds = frame.assign(_date=dates).groupby(
            "conversation_id", sort=False, observed=True, dropna=False
        )["_date"].agg(conversation_start="min", conversation_end="max").reset_index()
        bounds["conversation_duration_mins"] = (
            bounds["conversation_end"] - bounds["conversation_start"]
        ).dt.total_seconds().div(60)
        result = result.merge(bounds, on="conversation_id", how="left", validate="one_to_one")
    participant_columns = [column for column in ("from", "to") if column in frame]
    if participant_columns:
        participants = frame[["conversation_id", *participant_columns]].melt(
            id_vars="conversation_id", value_name="participant"
        ).dropna(subset=["participant"])
        counts = participants.groupby(
            "conversation_id", observed=True, dropna=False
        )["participant"].nunique().rename("participant_count").reset_index()
        result = result.merge(counts, on="conversation_id", how="left", validate="one_to_one")
        result["participant_count"] = result["participant_count"].fillna(0).astype("uint32")
    if "from" in frame:
        sender_counts = frame.groupby(
            "conversation_id", observed=True, dropna=False
        )["from"].nunique().rename("sender_count").reset_index()
        result = result.merge(sender_counts, on="conversation_id", how="left", validate="one_to_one")
        result["sender_count"] = result["sender_count"].fillna(0).astype("uint32")
        result["conversation_was_answered"] = result["sender_count"].gt(1)
    else:
        result["conversation_was_answered"] = False
    return result


def aggregate_users(silver_df: pd.DataFrame, engineer: bool = True) -> pd.DataFrame:
    frame = engineer_features(silver_df) if engineer else silver_df.copy()
    if not {"from", "to"}.issubset(frame.columns):
        raise KeyError("user aggregate requires from and to")
    sent = frame.dropna(subset=["from"]).rename(columns={"from": "user"})
    received = frame.dropna(subset=["to"]).rename(columns={"to": "user"})
    sent_gold = _prefix_metrics(_base_aggregate(sent, ["user"]), "user", "sent")
    received_gold = _prefix_metrics(
        _base_aggregate(received, ["user"]), "user", "received"
    )
    result = sent_gold.merge(received_gold, on="user", how="outer", validate="one_to_one")
    for column in ("sent_message_count", "received_message_count"):
        if column not in result:
            result[column] = 0
        result[column] = result[column].fillna(0).astype("uint32")
    result["interaction_message_count"] = (
        result["sent_message_count"] + result["received_message_count"]
    )
    contacts = pd.concat([
        frame[["from", "to"]].rename(columns={"from": "user", "to": "contact"}),
        frame[["to", "from"]].rename(columns={"to": "user", "from": "contact"}),
    ], ignore_index=True).dropna(subset=["user", "contact"])
    if not contacts.empty:
        contact_counts = contacts.groupby("user", observed=True)["contact"].nunique().rename(
            "unique_contact_count"
        ).reset_index()
        result = result.merge(contact_counts, on="user", how="left", validate="one_to_one")
    for column in (name for name in result.columns if name.endswith("_count")):
        result[column] = result[column].fillna(0)
    return result


def aggregate_dates(
    silver_df: pd.DataFrame, granularity: str = "day", engineer: bool = True
) -> pd.DataFrame:
    if granularity not in DATE_GRANULARITIES:
        raise ValueError(f"granularity must be one of {sorted(DATE_GRANULARITIES)}")
    frame = engineer_features(silver_df) if engineer else silver_df.copy()
    if "date" not in frame:
        raise KeyError("date aggregate requires date")
    dates = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame = frame.assign(date=dates).dropna(subset=["date"])
    if granularity == "day":
        frame["date_period"] = frame["date"].dt.floor("D")
    else:
        naive = frame["date"].dt.tz_convert(None)
        frequency = "W-MON" if granularity == "week" else "M"
        frame["date_period"] = naive.dt.to_period(frequency).dt.start_time.dt.tz_localize("UTC")
    return _base_aggregate(frame, ["date_period"])


def aggregate_channels(
    silver_df: pd.DataFrame, *, channel_column: str = "channel", engineer: bool = True
) -> pd.DataFrame:
    frame = engineer_features(silver_df) if engineer else silver_df.copy()
    if channel_column not in frame:
        raise KeyError(f"channel aggregate requires {channel_column!r}")
    return _base_aggregate(frame.dropna(subset=[channel_column]), [channel_column])


def aggregate_release_cycles(
    silver_df: pd.DataFrame, *, axis: str = "since", engineer: bool = True
) -> pd.DataFrame:
    if axis not in RELEASE_AXES:
        raise ValueError(f"release axis must be one of {sorted(RELEASE_AXES)}")
    frame = engineer_features(silver_df) if engineer else silver_df.copy()
    column = f"days_{axis}_release_bucket"
    if column not in frame:
        raise KeyError(f"release aggregate requires {column}")
    return _base_aggregate(frame.dropna(subset=[column]), [column])


def _explode_values(
    frame: pd.DataFrame, values: pd.Series, output_column: str
) -> pd.DataFrame:
    expanded = frame.copy()
    expanded[output_column] = values
    return expanded.explode(output_column).dropna(subset=[output_column])


def aggregate_languages(silver_df: pd.DataFrame, engineer: bool = True) -> pd.DataFrame:
    frame = engineer_features(silver_df) if engineer else silver_df.copy()
    if "text_cleaned" not in frame:
        raise KeyError("language aggregate requires text_cleaned")
    language_set = set(LANGUAGE_LABEL_TOKENS)
    values = frame["text_cleaned"].fillna("").astype(str).map(
        lambda text: sorted(language_set.intersection(text.split())) or ["UNLABELED"]
    )
    expanded = _explode_values(frame, values, "language_label")
    return _base_aggregate(expanded, ["language_label"])


def aggregate_residuals(silver_df: pd.DataFrame, engineer: bool = True) -> pd.DataFrame:
    frame = engineer_features(silver_df) if engineer else silver_df.copy()
    if "residual_words" not in frame:
        raise KeyError("residual aggregate requires residual_words")
    values = frame["residual_words"].map(
        lambda value: _parse_collection(value, column="residual_words")
    )
    expanded = _explode_values(frame, values, "residual_word")
    return _base_aggregate(expanded, ["residual_word"])


def aggregate_technical_terms(silver_df: pd.DataFrame, engineer: bool = True) -> pd.DataFrame:
    frame = engineer_features(silver_df) if engineer else silver_df.copy()
    if "tech_lexicon_matches" not in frame:
        raise KeyError("technical aggregate requires tech_lexicon_matches")
    values = frame["tech_lexicon_matches"].map(_technical_records)
    expanded = _explode_values(frame, values, "_technical_record")
    if expanded.empty:
        return pd.DataFrame(columns=["technical_term", "technical_category", "message_count"])
    expanded["technical_term"] = expanded["_technical_record"].map(lambda value: value["entity"])
    expanded["technical_category"] = expanded["_technical_record"].map(
        lambda value: value["entity_label"]
    )
    return _base_aggregate(expanded, ["technical_term", "technical_category"])


def aggregate_topics(silver_df: pd.DataFrame, engineer: bool = True) -> pd.DataFrame:
    frame = engineer_features(silver_df) if engineer else silver_df.copy()
    required = {"topic_id", "topic_label"}
    if not required.issubset(frame.columns):
        raise KeyError(f"topic aggregate requires {sorted(required)}")
    return _base_aggregate(frame.dropna(subset=list(required)), ["topic_id", "topic_label"])


def aggregate_entities(silver_df: pd.DataFrame, engineer: bool = True) -> pd.DataFrame:
    frame = engineer_features(silver_df) if engineer else silver_df.copy()
    if not {"tech_lexicon_matches", "named_entities"}.intersection(frame.columns):
        raise KeyError("entity aggregate requires technical or named-entity columns")
    records = frame.apply(_entity_records, axis=1)
    expanded = _explode_values(frame, records, "_entity_record")
    if expanded.empty:
        return pd.DataFrame(
            columns=["entity_source", "entity_label", "entity", "message_count"]
        )
    for column in ("entity_source", "entity_label", "entity"):
        expanded[column] = expanded["_entity_record"].map(lambda value: value[column])
    return _base_aggregate(expanded, ["entity_source", "entity_label", "entity"])


def sample_messages(
    silver_df: pd.DataFrame, sample_size: int, random_state: int = 0
) -> pd.DataFrame:
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    if sample_size >= len(silver_df):
        return silver_df.copy()
    rng = np.random.default_rng(random_state)
    positions = np.sort(rng.choice(len(silver_df), size=sample_size, replace=False))
    return silver_df.iloc[positions].copy()


def _gold_keys(gold_level: str, *, channel_column: str, release_axis: str) -> list[str]:
    return {
        "conversation": ["conversation_id"], "user": ["user"],
        "date": ["date_period"], "channel": [channel_column],
        "release": [f"days_{release_axis}_release_bucket"],
        "language": ["language_label"], "residual": ["residual_word"],
        "technical": ["technical_term", "technical_category"],
        "topic": ["topic_id", "topic_label"],
        "entity": ["entity_source", "entity_label", "entity"],
    }.get(gold_level, [])


def validate_gold(
    silver_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    *, gold_level: str,
    channel_column: str = "channel",
    release_axis: str = "since",
    sample_size: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    keys = _gold_keys(
        gold_level, channel_column=channel_column, release_axis=release_axis
    )
    if gold_level == "sample" and "message_id" in gold_df:
        keys = ["message_id"]
    if keys and set(keys).issubset(gold_df.columns) and gold_df.duplicated(keys).any():
        failures.append(f"duplicate gold keys: {keys}")
    count_columns = [column for column in gold_df if column.endswith("_count")]
    for column in count_columns:
        numeric = pd.to_numeric(gold_df[column], errors="coerce")
        if numeric.isna().any() or (numeric < 0).any():
            failures.append(f"invalid count values in {column}")
    bounded_columns = [
        column for column in gold_df
        if column.startswith("avg_vader_")
        or column.startswith("avg_transformer_")
        or column.startswith("share_")
    ]
    for column in bounded_columns:
        values = pd.to_numeric(gold_df[column], errors="coerce").dropna()
        lower = -1 if column in {"avg_vader_compound", "avg_transformer_expected_sentiment"} else 0
        if ((values < lower) | (values > 1) | ~np.isfinite(values)).any():
            failures.append(f"out-of-range aggregate values in {column}")
    if gold_level == "conversation" and "message_count" in gold_df:
        if int(gold_df["message_count"].sum()) != len(silver_df):
            failures.append("conversation message counts do not conserve silver rows")
    elif gold_level == "date" and "message_count" in gold_df:
        expected = int(pd.to_datetime(silver_df["date"], errors="coerce", utc=True).notna().sum())
        if int(gold_df["message_count"].sum()) != expected:
            failures.append("date message counts do not conserve valid dated rows")
    elif gold_level == "channel" and "message_count" in gold_df:
        expected = int(silver_df[channel_column].notna().sum())
        if int(gold_df["message_count"].sum()) != expected:
            failures.append("channel message counts do not conserve eligible rows")
    elif gold_level == "release" and "message_count" in gold_df:
        release_column = f"days_{release_axis}_release_bucket"
        expected = int(silver_df[release_column].notna().sum())
        if int(gold_df["message_count"].sum()) != expected:
            failures.append("release message counts do not conserve eligible rows")
    elif gold_level == "topic" and "message_count" in gold_df:
        expected = int(silver_df[["topic_id", "topic_label"]].notna().all(axis=1).sum())
        if int(gold_df["message_count"].sum()) != expected:
            failures.append("topic message counts do not conserve assigned rows")
    elif gold_level == "language" and "message_count" in gold_df:
        if int(gold_df["message_count"].sum()) < len(silver_df):
            failures.append("language message counts omit one or more silver rows")
    elif gold_level == "user":
        if int(gold_df.get("sent_message_count", pd.Series(dtype=int)).sum()) != int(silver_df["from"].notna().sum()):
            failures.append("sent user counts do not conserve sender rows")
        if int(gold_df.get("received_message_count", pd.Series(dtype=int)).sum()) != int(silver_df["to"].notna().sum()):
            failures.append("received user counts do not conserve recipient rows")
    elif gold_level == "sample":
        expected = min(sample_size or 0, len(silver_df))
        if len(gold_df) != expected:
            failures.append("sample gold row count is inconsistent")
    metrics = {
        "passed": not failures,
        "failures": failures,
        "gold_level": gold_level,
        "silver_rows": len(silver_df),
        "gold_rows": len(gold_df),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
    if failures:
        raise AssertionError("gold validation failed: " + "; ".join(failures))
    return metrics


def run_silver_to_gold(
    silver_df: pd.DataFrame,
    *, gold_level: str,
    date_granularity: str = "day",
    channel_column: str = "channel",
    release_axis: str = "since",
    sample_size: int | None = None,
    random_state: int = 0,
    engineer: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    if gold_level not in GOLD_LEVELS:
        raise ValueError(f"gold_level must be one of {sorted(GOLD_LEVELS)}")
    routes = {
        "conversation": lambda: aggregate_conversations(silver_df, engineer),
        "user": lambda: aggregate_users(silver_df, engineer),
        "date": lambda: aggregate_dates(silver_df, date_granularity, engineer),
        "channel": lambda: aggregate_channels(
            silver_df, channel_column=channel_column, engineer=engineer
        ),
        "release": lambda: aggregate_release_cycles(
            silver_df, axis=release_axis, engineer=engineer
        ),
        "language": lambda: aggregate_languages(silver_df, engineer),
        "residual": lambda: aggregate_residuals(silver_df, engineer),
        "technical": lambda: aggregate_technical_terms(silver_df, engineer),
        "topic": lambda: aggregate_topics(silver_df, engineer),
        "entity": lambda: aggregate_entities(silver_df, engineer),
    }
    if gold_level == "sample":
        if sample_size is None:
            raise ValueError("sample_size is required when gold_level='sample'")
        gold = sample_messages(silver_df, sample_size, random_state)
    else:
        gold = routes[gold_level]()
    validation = validate_gold(
        silver_df, gold, gold_level=gold_level, channel_column=channel_column,
        release_axis=release_axis, sample_size=sample_size,
    ) if validate else {"passed": True, "validation_skipped": True}
    gold.attrs["gold_validation"] = validation
    gold.attrs["gold_provenance"] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "gold_level": gold_level,
        "date_granularity": date_granularity,
        "channel_column": channel_column,
        "release_axis": release_axis,
        "sample_size": sample_size,
        "random_state": random_state,
        "engineer_features": engineer,
    }
    return gold


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validated Ubuntu gold outputs locally.")
    parser.add_argument("input_csv")
    parser.add_argument("--gold-level", choices=sorted(GOLD_LEVELS), required=True)
    parser.add_argument("--date-granularity", choices=sorted(DATE_GRANULARITIES), default="day")
    parser.add_argument("--channel-column", default="channel")
    parser.add_argument("--release-axis", choices=sorted(RELEASE_AXES), default="since")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--no-feature-engineering", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    silver = pd.read_csv(args.input_csv)
    gold = run_silver_to_gold(
        silver, gold_level=args.gold_level,
        date_granularity=args.date_granularity,
        channel_column=args.channel_column, release_axis=args.release_axis,
        sample_size=args.sample_size, random_state=args.random_state,
        engineer=not args.no_feature_engineering,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.gold_level == "date":
        suffix = args.date_granularity
    elif args.gold_level == "release":
        suffix = f"release_{args.release_axis}"
    else:
        suffix = args.gold_level
    output_path = output_dir / f"gold_{suffix}.csv"
    gold.to_csv(output_path, index=False)
    from pipeline.audit import append_run_audit
    append_run_audit(
        output_dir / "pipeline_run_log.jsonl", df=gold,
        stage=f"gold_{args.gold_level}", output_path=output_path,
        validation=gold.attrs["gold_validation"], config=vars(args),
    )
    print(f"wrote {output_path} with {len(gold):,} rows")


if __name__ == "__main__":
    main()
