# Schema: df_with_sentiment.pkl (source corpus artifact) -- PARTIAL

**Status: partial, not a completed schema inspection.** What follows was
recovered from the raw byte/opcode stream, which yields column *names* and
structural class names and nothing more. Dtypes, exact row count, index
metadata, null behavior, and actual row contents remain **unverified** and
stay that way until a successful deserialization. Do not treat this as a
substitute for that step; treat it as enough to configure the sampler
correctly in advance.

Established 2026-09-04 **without deserializing the file**, via
`scripts/peek_pickle_schema.py` (opcode/byte-stream inspection, read-only,
bounded memory). A full `pd.read_pickle()` was not possible: the native
Windows machine had only ~0.54 GiB of 15.3 GiB physically available, and
Claude's Linux-VM bridge has ~3.5 GiB and no swap -- neither is safe for a
2.9 GB pickle. Nothing was written to the source checkout.

- File: `1. Ubuntu-dialogue-corpus/df_with_sentiment.pkl`
- Size: 3,018,774,810 bytes
- Pickle protocol: 5, uncompressed
- Structure: `pandas.DataFrame` -> `BlockManager`, blocks written FIRST,
  axes near the end (columns index at byte ~2,950,060,300, followed by a
  ~68.7 MB int64 row index -- consistent with ~8.59M rows -- then the
  final `_typ`/`_metadata`/`attrs`/`_flags` frame).
- String columns are **pyarrow-backed** (`pandas.arrays.ArrowStringArray`,
  `pyarrow` `large_string`), including the columns index itself.

## Columns (41, in order)

    message_id                          conversation_id
    date                                from
    to                                  text
    turn_count                          is_op
    response_gap_mins                   user_message_count
    text_length                         word_count
    days_since_release                  days_until_release
    response_gap_mins_between_speakers  year
    month                               days_since_release_bucket
    days_until_release_bucket           text_length_bucket
    word_count_bucket                   user_messages_this_month
    response_gap_bucket                 user_tier
    text_cleaned                        structural_matches
    email_count                         domain_count
    phone_count                         ssn_count
    ipv4_count                          ipv6_count
    menu_path_count                     keyboard_shortcut_count
    tech_lexicon_matches                slang_matches
    tech_lexicon_match_count            slang_match_count
    has_lexicon_match                   vader_compound
    vader_label

## Consequences for the plan

1. **`message_id` exists** -- a stable per-row identifier. The sample
   fixture uses `message_id` (+ `conversation_id`) as its `source_row_id`
   and carries **no `from`/`to` usernames**, per the data-exposure note.
   Participant identity is not needed for transformer sentiment parity.

2. **No transformer output columns exist** (nothing matching
   transformer/roberta/sentiment_score). The historical collapsed scores
   were never persisted into this artifact. So this file cannot serve as a
   record of the collapse -- it can only provide inputs plus the VADER
   baseline. Reproducing the collapse means re-running the model, which on
   CPU establishes a float32 baseline only; the GPU fp16 comparison stays
   pending regardless.

3. **`text` and `text_cleaned` both exist.** Which one was fed to the
   transformer materially changes the diagnosis, and that is not recorded
   anywhere. The sampler carries both so the reproduction can be run each
   way.

4. **pyarrow is a hard dependency for loading this file**, and neither
   pyarrow nor a writer-compatible pandas was present anywhere --
   `pd.read_pickle` would have failed with `ModuleNotFoundError: pyarrow`
   even with unlimited RAM. The memory wall was masking a second blocker.

   Resolved by splitting environments rather than by widening the model
   runtime's pins. The writer's environment was confirmed read-only as
   **pandas 3.0.1 / pyarrow 19.0.0**, and pandas pickle compatibility
   across major versions is not a reliable archival contract, so:
   - `requirements-extract.txt` (-> `.venv-extract`) matches the writer and
     is used only to load the pickle once and emit the CSV fixture.
   - `requirements.txt` (-> `.venv`) stays a pure model runtime on
     pandas 2.2.3 with no pyarrow, since it only ever reads a 2,000-row CSV.
   The earlier guessed `pyarrow==18.1.0` pin has been dropped entirely.

5. Pre-computed banding columns already exist (`word_count_bucket`,
   `text_length_bucket`), so the sampler stratifies on the corpus's own
   bands when available instead of re-deriving quantiles.

## Still unknown without a full load -- schema inspection NOT complete

Unverified until a successful deserialization:

- per-column dtypes (the pyarrow backing is inferred from class names in
  the stream, not from a dtype listing)
- exact row count (~8.59M is inferred from a ~68.7 MB int64 index buffer,
  not read from the frame)
- index metadata: type, name, monotonicity, whether it aligns with
  `message_id`
- null counts and null behavior per column
- any actual row contents, and therefore all value distributions

These need the real load (step 2), with adequate RAM and the extraction
environment.

## Collapse hypotheses -- fp16 remains a prime suspect

An earlier note here argued that fp16 could not explain **8** unique scores
across 8.59M rows, on the grounds that fp16 has thousands of representable
values across [-1, 1]. **That argument was incomplete and is withdrawn.**
It addressed only rounding granularity. fp16's more likely failure mode
here is numerical instability, not rounding: overflow in attention
producing `inf`/`NaN` logits, and a softmax that saturates to ~1.0. That
mode does yield a handful of repeated confidence values, which fits the
reported figure well. fp16 therefore stays a leading hypothesis, and the
CPU work cannot speak to it either way -- `get_transformer_pipeline` sets
`torch_dtype = float16 if device != -1 else float32`, so CPU is always the
float32 arm.

Two further facts shape what is testable:

1. `transformer_score` is the model's **confidence in its predicted
   label**, not a signed compound like VADER's. Confidences cluster high
   by nature, so a low unique-count is less surprising for this column
   than it would be for a compound score -- worth keeping in mind before
   treating any particular count as anomalous.

2. `score_transformer_series` filters with `is_effectively_empty` **before
   inference** (sentiment_analysis.py:169). Empty rows get
   `transformer_label=None` / `transformer_score=NaN` without reaching the
   model. So placeholder stripping can reduce the *diversity of inputs*,
   but repeated empty inputs cannot themselves produce repeated model
   scores. Stripping is a contributing factor to investigate, not a
   competing explanation for the score count.

The CPU diagnostic (`scripts/diagnose_preprocessing.py`) is therefore
scoped as a **preprocessing diagnostic and float32 baseline**, explicitly
not a reproduction of the historical collapse.
