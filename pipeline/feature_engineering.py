"""Reusable, order-safe message and conversation feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _restore_source_order(frame: pd.DataFrame, source_index: pd.Index) -> pd.DataFrame:
    result = frame.sort_values("_source_position", kind="stable").drop(
        columns=["_source_position"]
    )
    result.index = source_index
    return result


def add_message_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic temporal, response, and activity features.

    Sequence-dependent features are calculated chronologically, independent of
    input row order, and the caller's original row order/index is restored.
    """
    source_index = df.index.copy()
    result = df.copy()
    result["_source_position"] = np.arange(len(result), dtype=np.int64)

    if "date" in result:
        result["date"] = pd.to_datetime(result["date"], errors="coerce", utc=True)
        result["year"] = result["date"].dt.year.astype("Int16")
        result["month"] = result["date"].dt.month.astype("Int8")
        result["day"] = result["date"].dt.day.astype("Int8")
        result["day_of_week"] = result["date"].dt.day_name().astype("category")
        result["hour"] = result["date"].dt.hour.astype("Int8")

    if "conversation_id" in result:
        ordering = ["conversation_id"]
        if "date" in result:
            ordering.append("date")
        ordering.append("_source_position")
        chronological = result.sort_values(
            ordering, kind="stable", na_position="last"
        ).copy()
        conversation_group = chronological.groupby(
            "conversation_id", sort=False, observed=True, dropna=False
        )
        chronological["turn_count"] = conversation_group.cumcount().astype("uint32")
        if "date" in chronological and "from" in chronological:
            previous_speaker = conversation_group["from"].shift()
            speaker_changed = chronological["from"].ne(previous_speaker)
            response_gap = conversation_group["date"].diff().dt.total_seconds().div(60)
            chronological["response_gap_mins_between_speakers"] = response_gap.where(
                speaker_changed
            )

        sequential_columns = ["turn_count"]
        if "response_gap_mins_between_speakers" in chronological:
            sequential_columns.append("response_gap_mins_between_speakers")
        by_position = chronological.set_index("_source_position")
        for column in sequential_columns:
            result[column] = result["_source_position"].map(by_position[column])

    if "from" in result:
        user_order = ["date", "_source_position"] if "date" in result else ["_source_position"]
        user_chronology = result.sort_values(
            user_order, kind="stable", na_position="last"
        ).copy()
        user_chronology["user_message_count"] = (
            user_chronology.groupby("from", sort=False, observed=True, dropna=False)
            .cumcount()
            .add(1)
            .astype("uint32")
        )
        result["user_message_count"] = result["_source_position"].map(
            user_chronology.set_index("_source_position")["user_message_count"]
        )

        if {"year", "month"}.issubset(result.columns):
            result["user_messages_this_month"] = (
                result.groupby(
                    ["from", "year", "month"],
                    observed=True,
                    dropna=False,
                )["_source_position"]
                .transform("size")
                .astype("uint32")
            )
            result["user_tier"] = pd.cut(
                result["user_messages_this_month"],
                bins=[0, 10, 50, 150, 500, float("inf")],
                labels=["occasional", "regular", "active", "frequent", "power_user"],
            )

    if "text_length" in result:
        result["text_length_bucket"] = pd.cut(
            result["text_length"],
            bins=[-1, 20, 50, 100, 200, float("inf")],
            labels=["very_short", "short", "medium", "long", "very_long"],
        )
    if "word_count" in result:
        result["word_count_bucket"] = pd.cut(
            result["word_count"],
            bins=[-1, 5, 10, 20, 40, float("inf")],
            labels=["very_short", "short", "medium", "long", "very_long"],
        )
    if "response_gap_mins_between_speakers" in result:
        result["response_gap_bucket"] = pd.cut(
            result["response_gap_mins_between_speakers"],
            bins=[-1, 5, 30, 120, 1440, float("inf")],
            labels=["<5min", "5-30min", "30min-2hr", "2hr-1day", "1day+"],
        )
    return _restore_source_order(result, source_index)


def add_conversation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add conversation bounds and response status without row-changing joins."""
    if not {"conversation_id", "date"}.issubset(df.columns):
        return df
    result = df.copy()
    dates = pd.to_datetime(result["date"], errors="coerce", utc=True)
    group_key = result["conversation_id"]
    result["conversation_start"] = dates.groupby(group_key, observed=True).transform("min")
    result["conversation_end"] = dates.groupby(group_key, observed=True).transform("max")
    result["conversation_duration_mins"] = (
        result["conversation_end"] - result["conversation_start"]
    ).dt.total_seconds().div(60)
    result["conversation_message_count"] = (
        group_key.groupby(group_key, observed=True).transform("size").astype("uint32")
    )
    participant_columns = [column for column in ("from", "to") if column in result]
    if participant_columns:
        participants = result[["conversation_id", *participant_columns]].melt(
            id_vars="conversation_id", value_name="participant"
        ).dropna(subset=["participant"])
        participant_counts = participants.groupby(
            "conversation_id", observed=True, dropna=False
        )["participant"].nunique()
        result["conversation_participant_count"] = (
            result["conversation_id"].map(participant_counts).fillna(0).astype("uint32")
        )
        if "from" in result:
            sender_counts = result.groupby(
                "conversation_id", observed=True, dropna=False
            )["from"].transform("nunique")
            result["conversation_was_answered"] = sender_counts.gt(1)
        else:
            result["conversation_was_answered"] = False
    else:
        result["conversation_was_answered"] = False
    return result


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic features while preserving row order and index."""
    expected_index = df.index.copy()
    result = add_conversation_features(add_message_features(df))
    if len(result) != len(df):
        raise RuntimeError("feature engineering changed the message row count")
    if not result.index.equals(expected_index):
        raise RuntimeError("feature engineering changed the message index")
    return result
