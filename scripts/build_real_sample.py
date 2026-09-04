"""
One-time, read-only fixture extraction from the Ubuntu corpus checkout.

NOT part of the ongoing transformer pipeline -- this is a standalone,
manually-run script that produces a small deterministic CSV fixture inside
THIS repo (data/real_sample.csv). It never writes back to the source
checkout and never copies the full source file.

Per project isolation rules, this script must not run until explicitly
authorized, and the source path below is intentionally left unset so it
can't be run by accident.

Usage (after authorization + confirming column names, see CONFIG below):
    ..\.venv\python.exe scripts\build_real_sample.py
"""
import sys
import numpy as np
import pandas as pd

# --- CONFIG -- confirm/adjust these against the actual file before running ---
SOURCE_PATH = None  # e.g. r"...\1. Ubuntu-dialogue-corpus\df_with_sentiment.pkl"
OUTPUT_PATH = "data/real_sample.csv"
N_SAMPLES = 2000
SEED = 42
N_LENGTH_BANDS = 5

# Column names as understood from prior notebook work -- CONFIRM these match
# the actual dataframe's dtypes/columns before running, since this script
# has not yet been run against the real file.
LENGTH_COL = "word_count"          # falls back to "text_length" if absent
TEXT_COL = "text"
ID_COLS = ["conversation_id", "folder", "from", "to", "date"]
CARRY_COLS = ["vader_compound", "vader_label"]  # existing scores for parity
# -------------------------------------------------------------------------


def main():
    if not SOURCE_PATH:
        sys.exit(
            "SOURCE_PATH is not set. This script is intentionally inert "
            "until explicitly authorized and pointed at a real file -- "
            "set SOURCE_PATH above, confirm LENGTH_COL/ID_COLS/CARRY_COLS "
            "against the actual columns, then re-run."
        )

    df = pd.read_pickle(SOURCE_PATH)

    length_col = LENGTH_COL if LENGTH_COL in df.columns else "text_length"
    if length_col not in df.columns:
        sys.exit(f"Neither '{LENGTH_COL}' nor 'text_length' found in columns: {list(df.columns)}")

    missing_id = [c for c in ID_COLS if c not in df.columns]
    missing_carry = [c for c in CARRY_COLS if c not in df.columns]
    if missing_id:
        print(f"NOTE: missing expected id columns, dropping: {missing_id}")
    if missing_carry:
        print(f"NOTE: missing expected carry columns, dropping: {missing_carry}")
    id_cols = [c for c in ID_COLS if c in df.columns]
    carry_cols = [c for c in CARRY_COLS if c in df.columns]

    keep_cols = id_cols + [TEXT_COL, length_col] + carry_cols
    work = df[keep_cols].copy()
    work = work.dropna(subset=[TEXT_COL])

    # Deterministic stratified sample across length bands (quantile-based,
    # not simply first-N), so short/medium/long/very-long text are all
    # represented proportionally to their share of the corpus.
    work["_band"] = pd.qcut(work[length_col], q=N_LENGTH_BANDS, duplicates="drop")

    rng = np.random.default_rng(SEED)
    parts = []
    band_groups = list(work.groupby("_band", observed=True))
    per_band = N_SAMPLES // len(band_groups)
    for _, group in band_groups:
        n = min(per_band, len(group))
        idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
        parts.append(work.loc[idx])

    sample = pd.concat(parts).drop(columns="_band")

    # Top up to N_SAMPLES deterministically if bands didn't divide evenly.
    shortfall = N_SAMPLES - len(sample)
    if shortfall > 0:
        remaining = work.drop(index=sample.index).drop(columns="_band")
        idx = rng.choice(remaining.index.to_numpy(), size=min(shortfall, len(remaining)), replace=False)
        sample = pd.concat([sample, remaining.loc[idx]])

    sample = sample.sort_index()  # deterministic, reproducible ordering
    sample.to_csv(OUTPUT_PATH, index=True, index_label="source_row_index")
    print(f"Wrote {len(sample)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
