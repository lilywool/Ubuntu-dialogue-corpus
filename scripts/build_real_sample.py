"""
One-time, read-only fixture extraction from the Ubuntu corpus checkout.

NOT part of the ongoing transformer pipeline -- this is a standalone,
manually-run script that produces a small deterministic CSV fixture inside
THIS repo (data/real_sample.csv). It never writes back to the source
checkout and never copies the full source file.

Per project isolation rules, this script must not run until explicitly
authorized, and the source path below is intentionally left unset so it
can't be run by accident.

Runs in the EXTRACTION environment (.venv-extract, pandas 3.0.1 +
pyarrow 19.0.0 -- matching the pickle's writer), NOT the transformer
runtime. Emits CSV so the transformer env can read the fixture with
plain pandas 2.2.3 and needs neither pyarrow nor pandas 3.x.

Usage (only under the step-2 extraction authorization):
    .\.venv-extract\python.exe scripts\build_real_sample.py
"""
import sys
import numpy as np
import pandas as pd

# --- CONFIG -- column names CONFIRMED against the real schema on 2026-09-04
# via scripts/peek_pickle_schema.py (see docs_schema_df_with_sentiment.md).
SOURCE_PATH = None  # set only under the step-2 extraction authorization
OUTPUT_PATH = "data/real_sample.csv"
N_SAMPLES = 2000
SEED = 42
N_LENGTH_BANDS = 5

# Stable row identity. Deliberately EXCLUDES `from`/`to`: participant
# identity is unnecessary for transformer sentiment parity and is avoidable
# data exposure. message_id is the stable per-row id.
ID_COLS = ["message_id", "conversation_id"]

# Both text variants are carried. text_cleaned is the PRIMARY arm: the
# committed entry point is analyze_sentiment(df, text_column='text_cleaned')
# (sentiment_analysis.py:214), so that is the path the historical run took.
# Raw text is the COMPARISON arm -- if the two diverge sharply, that is a
# preprocessing finding rather than a dtype one.
PRIMARY_TEXT_COL = "text_cleaned"
COMPARISON_TEXT_COL = "text"
TEXT_COLS = [PRIMARY_TEXT_COL, COMPARISON_TEXT_COL]

# Prefer the corpus's own pre-computed band column; fall back to quantile
# banding on the raw length if it is absent.
BAND_COL = "word_count_bucket"
LENGTH_COL = "word_count"

# Existing scores carried for parity comparison. Note there are NO
# transformer columns in the source -- the historical collapsed output was
# never persisted -- so VADER is the only prior signal available.
CARRY_COLS = ["vader_compound", "vader_label", "text_length"]
# -------------------------------------------------------------------------


def main():
    if not SOURCE_PATH:
        sys.exit(
            "SOURCE_PATH is not set. This script stays inert until the "
            "step-2 extraction is explicitly authorized -- set SOURCE_PATH, "
            "then re-run. Requires pyarrow (the source's string columns are "
            "pyarrow-backed) and enough free RAM for a ~2.9GB pickle."
        )

    df = pd.read_pickle(SOURCE_PATH)
    print(f"Loaded source: {df.shape[0]:,} rows x {df.shape[1]} cols")

    id_cols = [c for c in ID_COLS if c in df.columns]
    text_cols = [c for c in TEXT_COLS if c in df.columns]
    carry_cols = [c for c in CARRY_COLS if c in df.columns]
    if not text_cols:
        sys.exit(f"None of {TEXT_COLS} present; columns are {list(df.columns)}")

    keep = id_cols + text_cols + [LENGTH_COL] + carry_cols
    keep = [c for c in dict.fromkeys(keep) if c in df.columns]
    work = df[keep].copy()
    del df  # release the 2.9GB source as early as possible

    primary_text = PRIMARY_TEXT_COL if PRIMARY_TEXT_COL in text_cols else text_cols[0]
    work = work.dropna(subset=[primary_text])

    # Stratify on the corpus's own band column when present, else derive
    # quantile bands -- either way the sample spans short..long text rather
    # than being a first-N slice.
    if BAND_COL in work.columns:
        band = work[BAND_COL]
        band_source = BAND_COL
    else:
        band = pd.qcut(work[LENGTH_COL], q=N_LENGTH_BANDS, duplicates="drop")
        band_source = f"qcut({LENGTH_COL}, {N_LENGTH_BANDS})"
    print(f"Stratifying on: {band_source}")

    rng = np.random.default_rng(SEED)
    groups = [g for _, g in work.groupby(band, observed=True)]
    per_band = N_SAMPLES // max(len(groups), 1)

    parts = []
    for group in groups:
        n = min(per_band, len(group))
        picked = rng.choice(group.index.to_numpy(), size=n, replace=False)
        parts.append(work.loc[picked])
    sample = pd.concat(parts)

    shortfall = N_SAMPLES - len(sample)
    if shortfall > 0:
        remaining = work.drop(index=sample.index)
        n = min(shortfall, len(remaining))
        picked = rng.choice(remaining.index.to_numpy(), size=n, replace=False)
        sample = pd.concat([sample, remaining.loc[picked]])

    # Deterministic ordering by stable id, not by position in the source.
    sort_key = id_cols[0] if id_cols else primary_text
    sample = sample.sort_values(sort_key).reset_index(drop=True)

    # CSV, not parquet: keeps the transformer runtime free of pyarrow.
    sample.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(sample):,} rows to {OUTPUT_PATH}")
    print(f"Columns: {list(sample.columns)}")
    print("No from/to usernames included by design.")


if __name__ == "__main__":
    main()
