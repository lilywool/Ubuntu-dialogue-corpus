# transformer branch -- isolated environment notes

Genuine clone of the Ubuntu Dialogue Corpus repo (full history, `origin` =
https://github.com/lilywool/Ubuntu-dialogue-corpus.git), checked out on its
own `transformer` branch, per project isolation rules: separate clone,
separate `.venv`, own pinned Python/PyTorch/Transformers/CUDA versions,
never shared with the Ubuntu `master` clone or the Yelp repo.

## Status as of 2026-09-04

- [x] Genuine clone with full history
- [x] `transformer` branch created
- [x] `requirements.txt` pinned: PyPI primary index, PyTorch CPU repo as an
      *extra* index only, `torch==2.5.1+cpu` pinned to the explicit local
      build tag
- [x] `.venv` created as a **prefix-based Conda environment** via the
      existing Anaconda `conda.exe` (`C:\Users\lilli_37fwt34\anaconda3\Scripts\conda.exe`,
      Conda 25.5.1) -- `conda create --prefix ".\.venv" python=3.11 -y`.
      No global install, no PATH/profile changes, no base-env changes.
- [x] Exact interpreter version confirmed and pinned:
      **`.python-version` = `3.11.16`** (not just the bare `3.11` minor).
      Future re-creation of this env should use `python=3.11.16` for exact
      reproducibility, not the bare minor version.
- [ ] **Blocked:** `pip install -r requirements.txt` failed partway with a
      Windows `MAX_PATH` (260-char) error while unpacking a deeply nested
      `transformers` file:
      ```
      OSError: [Errno 2] No such file or directory: 'C:\...\5. Transformer
      Sentiment\transformer-sentiment-clone\.venv\Lib\site-packages\
      transformers\models\deprecated\trajectory_transformer\
      convert_trajectory_transformer_original_pytorch_checkpoint_to_pytorch.py'
      ```
      Caused by the combined length of the descriptive parent folder names
      (`1. Github Root Repos\5. Transformer Sentiment\...`) plus a long
      nested package path, not anything about the pins themselves. The
      environment is **partially installed** -- do not trust or generate
      `requirements.lock.txt` (a `pip freeze`) until install completes
      cleanly.
- [~] **Step 1 (schema inspection) PARTIAL -- not complete.** Column
      names and structure were recovered without loading the pickle, but
      dtypes, exact shape, index metadata, null behavior and row contents
      remain unverified until a successful deserialization. Do not mark
      this phase done. Both machines lacked the RAM (native Windows: 0.54 GiB of
      15.3 GiB available; Claude's bridge VM: ~3.5 GiB, no swap). Schema
      was instead recovered from the pickle byte/opcode stream without
      deserializing anything -- see `docs_schema_df_with_sentiment.md` for
      the full 41-column list and consequences, and
      `scripts/peek_pickle_schema.py` for the tool (head/tail/find/window
      modes, read-only, bounded memory).
- [x] `scripts/build_real_sample.py` config corrected against the real
      schema: `message_id` as the stable id, **no `from`/`to` usernames**,
      both `text` and `text_cleaned` carried, stratification on the
      corpus's own `word_count_bucket`.
- [x] **Two-environment split** instead of widening the runtime's pins.
      Writer env confirmed read-only as pandas 3.0.1 / pyarrow 19.0.0;
      cross-major-version pandas pickle compat is not a reliable archival
      contract, so:
        * `requirements-extract.txt` -> `.venv-extract` (pandas==3.0.1,
          pyarrow==19.0.0) loads the pickle once and emits the CSV fixture.
        * `requirements.txt` -> `.venv` stays a pure model runtime
          (pandas 2.2.3, no pyarrow) -- it only reads a 2,000-row CSV.
      The guessed `pyarrow==18.1.0` pin is dropped. `requirements.txt` is
      unchanged in substance for the runtime, so `requirements.lock.txt`
      remains valid for `.venv`; `.venv-extract` gets its own lock when
      it is created.
- [x] Sampler arms set: **`text_cleaned` primary** (matches the committed
      `analyze_sentiment(..., text_column='text_cleaned')` default,
      verified at sentiment_analysis.py:214), raw `text` as comparison.
- [!] **Competing hypothesis for the collapse, testable on CPU now:**
      `strip_label_tokens_series` removes NONWORD/language/script
      placeholders before scoring. fp16 has thousands of representable
      values in [-1,1], so precision loss alone does not plausibly yield
      **8** unique scores across 8.59M rows -- degenerate post-strip inputs
      or an assignment bug fit that number far better. See
      docs_schema_df_with_sentiment.md.
- [ ] **Step 2 (extraction) NOT authorized yet** and additionally blocked
      on RAM. Needs ~8-10 GiB free.
- [ ] GPU presence still unconfirmed (not "no GPU").
- [ ] No dtype/CUDA-related code changes made yet to `sentiment_analysis.py`
      in this branch -- unmodified copy, carried over via the clone.

## Fix for the long-path install failure (chosen approach)

Use a temporary `subst` virtual drive mapping to shorten the effective path
during install -- this is session-scoped, per-repo, and reversible; it does
**not** touch the registry, `LongPathsEnabled`, global `PATH`, or any other
project. (The alternative -- enabling Windows long-path support globally --
is a system-wide change and is explicitly not being made without separate
approval.)

PowerShell, run as the current user (no elevation needed for `subst`):

    subst X: "C:\Users\lilli_37fwt34\Home\Projects\1. Github Root Repos\5. Transformer Sentiment\transformer-sentiment-clone"
    cd X:\
    .\.venv\python.exe -m pip install -r requirements.txt
    .\.venv\python.exe -m pip freeze > requirements.lock.txt
    cd C:\
    subst X: /D

The last two lines return to a real path and remove the virtual drive so
nothing lingers past this install -- `subst` mappings don't survive a
reboot by default, but tearing it down explicitly keeps this fully
temporary rather than relying on that.

## Next steps

Once `pip install` completes cleanly via the `subst` path and
`requirements.lock.txt` is generated and committed, move on to step 2 of
the agreed plan: reproduce the transformer behavior on a small
deterministic sample using `.\.venv\python.exe`, before touching any
dtype/validation code.
