# Schema: df_with_sentiment.pkl (source corpus artifact)

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

4. **pyarrow is a hard dependency for loading this file** and was missing
   from `requirements.txt`/`requirements.lock.txt` -- `pd.read_pickle`
   would have failed with `ModuleNotFoundError: pyarrow` even with enough
   RAM. Now pinned. If unpickling raises an Arrow-specific error rather
   than a memory error, the pin likely needs to match the pyarrow version
   that *wrote* the file (check the base Anaconda env with
   `conda list pyarrow` -- a read-only query, no changes).

5. Pre-computed banding columns already exist (`word_count_bucket`,
   `text_length_bucket`), so the sampler stratifies on the corpus's own
   bands when available instead of re-deriving quantiles.

## Still unknown without a full load

Per-column dtypes, exact row count, null counts, and the actual value
distributions. Those need the real load, i.e. step 2, with adequate RAM.
