"""
Preprocessing diagnostic -- NOT a reproduction of the historical collapse.

What this can establish (CPU, float32):
  * how placeholder stripping changes the number of distinct effective
    inputs actually sent to the model
  * which rows are excluded as effectively empty and never scored at all
  * whether label/score diversity differs between raw text, cleaned text,
    and the real pipeline path (cleaned + stripped)
  * whether batch size changes results (padding/attention effects)

What it CANNOT establish: anything about the historical collapse. That run
was fp16-on-GPU. get_transformer_pipeline() sets
torch_dtype = float16 if device != -1 else float32, so on CPU (device=-1)
this is unavoidably the float32 arm. Treat every number here as a
preprocessing measurement and a float32 baseline, nothing more.

Note on the empty-row path: score_transformer_series() filters with
is_effectively_empty BEFORE inference (sentiment_analysis.py:169), so empty
rows get transformer_label=None / transformer_score=NaN without reaching
the model. Stripping can therefore reduce input *diversity*, but repeated
empty inputs cannot by themselves produce repeated model scores.

Arms:
  raw_unstripped     -- df['text'], no stripping
  cleaned_unstripped -- df['text_cleaned'], no stripping
  cleaned_stripped   -- strip_label_tokens_series(df['text_cleaned']),
                        i.e. the actual path analyze_sentiment() takes

Run in the transformer runtime (.venv), from the repo root:
    .\.venv\python.exe scripts\diagnose_preprocessing.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from sentiment_analysis import (
    is_effectively_empty,
    score_transformer_series,
    score_vader_series,
    strip_label_tokens_series,
)

FIXTURE = "data/diagnostic_fixture.csv"
PER_ROW_OUT = "data/diagnostic_results_per_row.csv"
SUMMARY_OUT = "data/diagnostic_results_summary.csv"
BATCH_SIZES = [1, 32]


def report_runtime():
    import torch
    import transformers
    device = 0 if torch.cuda.is_available() else -1
    dtype = "float16" if device != -1 else "float32"
    print("=== runtime ===")
    print(f"  torch            {torch.__version__}")
    print(f"  transformers     {transformers.__version__}")
    print(f"  pandas           {pd.__version__}")
    print(f"  cuda_available   {torch.cuda.is_available()}")
    print(f"  resolved device  {device}")
    print(f"  resolved dtype   {dtype}  "
          f"({'fp16 arm -- NOT the CPU baseline this script claims' if dtype == 'float16' else 'float32 baseline'})")
    if dtype == "float16":
        print("  WARNING: a GPU was detected, so get_transformer_pipeline will")
        print("  force fp16. This script is written as a float32 CPU diagnostic;")
        print("  results below are NOT the float32 baseline.")
    print()
    return dtype


def build_arms(df):
    return {
        "raw_unstripped": df["text"].fillna("").astype(str),
        "cleaned_unstripped": df["text_cleaned"].fillna("").astype(str),
        "cleaned_stripped": strip_label_tokens_series(
            df["text_cleaned"].fillna("").astype(str)
        ),
    }


def n_unique_exact(values):
    """Unique count over full-precision floats. No rounding anywhere.
    NaN is excluded rather than counted as a value."""
    kept = [v for v in values if pd.notna(v)]
    return len(set(kept))


def main():
    df = pd.read_csv(FIXTURE, keep_default_na=False, na_values=[])
    print(f"Loaded {len(df)} fixture rows from {FIXTURE}\n")

    dtype = report_runtime()
    arms = build_arms(df)

    per_row, summary = [], []

    for arm_name, effective in arms.items():
        empties = effective.map(is_effectively_empty)
        vader = score_vader_series(effective)

        for batch_size in BATCH_SIZES:
            print(f"--- arm={arm_name} batch_size={batch_size} ---")
            tf = score_transformer_series(effective, batch_size=batch_size)

            for i in range(len(df)):
                per_row.append({
                    "row_id": df["row_id"].iloc[i],
                    "category": df["category"].iloc[i],
                    "arm": arm_name,
                    "batch_size": batch_size,
                    "original_input": df["text"].iloc[i],
                    "effective_input": effective.iloc[i],
                    "is_effectively_empty": bool(empties.iloc[i]),
                    "transformer_label": tf["transformer_label"].iloc[i],
                    "transformer_score": tf["transformer_score"].iloc[i],
                    "vader_compound": vader["vader_compound"].iloc[i],
                    "vader_label": vader["vader_label"].iloc[i],
                })

            scored_mask = ~empties.to_numpy()
            scored_inputs = effective[scored_mask]
            summary.append({
                "arm": arm_name,
                "batch_size": batch_size,
                "dtype": dtype,
                "n_rows": len(df),
                "n_effectively_empty": int(empties.sum()),
                "n_sent_to_model": int(scored_mask.sum()),
                "n_unique_effective_inputs_all": effective.nunique(),
                "n_unique_effective_inputs_scored": scored_inputs.nunique(),
                "n_unique_transformer_labels": tf["transformer_label"].dropna().nunique(),
                "n_unique_transformer_scores": n_unique_exact(tf["transformer_score"].tolist()),
                "n_unique_vader_compound": n_unique_exact(vader["vader_compound"].tolist()),
            })

    per_row_df = pd.DataFrame(per_row)
    summary_df = pd.DataFrame(summary)
    per_row_df.to_csv(PER_ROW_OUT, index=False)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    print("\n=== SUMMARY (no rounding applied to any score) ===")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary_df.to_string(index=False))

    # Batch-size sensitivity: exact float comparison, per arm.
    print("\n=== batch-size sensitivity (exact float equality) ===")
    for arm_name in arms:
        a = per_row_df[(per_row_df.arm == arm_name) & (per_row_df.batch_size == BATCH_SIZES[0])]
        b = per_row_df[(per_row_df.arm == arm_name) & (per_row_df.batch_size == BATCH_SIZES[1])]
        a = a.set_index("row_id")["transformer_score"]
        b = b.set_index("row_id")["transformer_score"]
        both_nan = a.isna() & b.isna()
        identical = ((a == b) | both_nan).all()
        n_diff = int((~((a == b) | both_nan)).sum())
        print(f"  {arm_name:<20} identical={identical}  differing_rows={n_diff}")
        if n_diff:
            diff = pd.DataFrame({"bs%d" % BATCH_SIZES[0]: a, "bs%d" % BATCH_SIZES[1]: b})
            print(diff[~((a == b) | both_nan)].to_string())

    print(f"\nWrote {PER_ROW_OUT} and {SUMMARY_OUT}")
    print("\nReminder: float32 CPU preprocessing diagnostic. It does not, and")
    print("cannot, reproduce the fp16-on-GPU collapse.")


if __name__ == "__main__":
    main()
