"""Multi-core tokenizer for the row-level "all words known" pass (see the
2026-08-19 lexicon review): tokenizes every row TWO ways -- digit-inclusive
and digit-exclusive -- and strips+counts the non-Latin-script placeholder
labels (CYRILLIC, HAN_CHINESE_JAPANESE_KANJI, ...) written by the cells
62-66 language-labeling step BEFORE tokenizing, so those labels don't get
shredded at their underscores into fragments ("han", "chinese", "kanji")
that would otherwise pollute the known/unknown-word buckets.

Mirrors parallel_match.py's approach (ProcessPoolExecutor + a module-level
worker function for Windows/Jupyter picklability -- see that module's
docstring for why the worker can't be a lambda or notebook-local closure).

Usage:

    from parallel_vocab_check import parallel_tokenize
    with_tokens, no_tokens, label_skip_counts = parallel_tokenize(
        no_match_df['text_cleaned'], n_workers=8
    )
    print(f"Non-Roman Language Labels Skipped: {sum(label_skip_counts):,}")
"""

import re
from concurrent.futures import ProcessPoolExecutor

try:
    from tqdm.auto import tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False

# Must match cells 62-66's NON_LATIN_LABELS list exactly.
NON_LATIN_LABELS = ['han_chinese_japanese_kanji', 'japanese_hiragana', 'japanese_katakana',
                     'korean_hangul', 'cyrillic', 'arabic', 'hebrew', 'greek', 'thai',
                     'devanagari', 'armenian', 'georgian', 'ethiopic', 'malayalam',
                     'tamil', 'sinhala']

# Exact substring match on the uppercase tokens cells 62-66 wrote in. Replaced
# with a space (not '') so a label sitting between two real words -- or glued
# directly onto one, e.g. "HAN_CHINESE_JAPANESE_KANJIaMuleHAN_CHINESE_JAPANESE_KANJI"
# -- doesn't fuse its neighbors together once it's removed.
_LABEL_PATTERN = re.compile('|'.join(re.escape(label.upper()) for label in NON_LATIN_LABELS))

# Unicode-aware: keeps accented Latin letters (café, über, señor) as part of
# a word instead of truncating at the first non a-zA-Z char. Apostrophes
# stay glued to their word ("don't").
WORD_RE_WITH_DIGITS = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
WORD_RE_NO_DIGITS   = re.compile(r"[^\W_\d]+(?:'[^\W_\d]+)*")


def _tokenize_chunk(texts):
    """Worker: strip+count non-Latin labels, then tokenize both ways, for
    one chunk (a plain list of strings). Must stay a top-level function --
    see module docstring."""
    with_all, no_all, label_counts = [], [], []
    for t in texts:
        label_counts.append(len(_LABEL_PATTERN.findall(t)))
        stripped = _LABEL_PATTERN.sub(' ', t)
        with_all.append([w.lower() for w in WORD_RE_WITH_DIGITS.findall(stripped)])
        no_all.append([w.lower() for w in WORD_RE_NO_DIGITS.findall(stripped)])
    return with_all, no_all, label_counts


def parallel_tokenize(text_series, n_workers=8, chunks_per_worker=4):
    """Tokenize `text_series` two ways across a process pool, after
    stripping non-Latin-script placeholder labels out of each row first.
    Returns (with_tokens, no_tokens, label_skip_counts) in original row
    order -- with_tokens/no_tokens are lists-of-token-lists safe to feed
    straight into Counter() or zip with the source index; label_skip_counts
    is a per-row count of labels stripped (sum it for a total, or assign it
    to a column to see which rows had them)."""
    texts = text_series.tolist()
    n = len(texts)
    chunk_count = max(1, n_workers * chunks_per_worker)
    size = -(-n // chunk_count)  # ceil division
    chunks = [texts[i:i + size] for i in range(0, n, size)]

    with_tokens, no_tokens, label_counts = [], [], []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = ex.map(_tokenize_chunk, chunks)
        if _HAVE_TQDM:
            results = tqdm(results, total=len(chunks), desc="parallel tokenize (with/without digits)")
        for with_chunk, no_chunk, label_chunk in results:
            with_tokens.extend(with_chunk)
            no_tokens.extend(no_chunk)
            label_counts.extend(label_chunk)

    assert len(with_tokens) == n and len(no_tokens) == n and len(label_counts) == n, \
        f"row count mismatch after reassembly: got {len(with_tokens)}/{len(no_tokens)}/{len(label_counts)}, expected {n}"
    return with_tokens, no_tokens, label_counts