from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from residual_classification import classify_text, prepare_classification_sets

try:
    from tqdm.auto import tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False

# Populated by _init_worker in EACH worker process (via initializer/initargs
# below), not by the parent. Stays None in the parent process itself -- that's
# fine, the parent never calls _classify_chunk.
_LANG_DICT = None
_NONWORD_SET = None


def _init_worker(lang_dict, nonword_set):
    """Runs once per worker process at pool startup (both fork and spawn).
    initargs below hands it the real dict/set explicitly, so this doesn't
    depend on the worker having inherited anything from the parent."""
    global _LANG_DICT, _NONWORD_SET
    _LANG_DICT = lang_dict
    _NONWORD_SET = nonword_set


def _classify_chunk(texts):
    """Worker: classify one chunk (a plain list of strings). Must stay a
    top-level function -- see module docstring."""
    return [classify_text(t, _LANG_DICT, _NONWORD_SET) for t in texts]


def parallel_classify_remaining_words(text_series, unresolved_words, n_workers=8, chunks_per_worker=4):
    """Drop-in replacement for residual_classification.classify_remaining_words,
    parallelized and without the giant-regex bottleneck. Same three outcomes
    (jargon left alone, language words labeled, everything else -> NONWORD),
    same \\b-equivalent tokenization. Returns a Series aligned to
    text_series's original index -- safe to assign straight back to
    df['text_cleaned']."""
    lang_dict, nonword_set = prepare_classification_sets(unresolved_words)

    texts = text_series.tolist()
    n = len(texts)
    chunk_count = max(1, n_workers * chunks_per_worker)
    size = -(-n // chunk_count)  # ceil division
    chunks = [texts[i:i + size] for i in range(0, n, size)]

    out = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(lang_dict, nonword_set)) as ex:
        results = ex.map(_classify_chunk, chunks)
        if _HAVE_TQDM:
            results = tqdm(results, total=len(chunks), desc="parallel jargon/NONWORD classification")
        for chunk_result in results:
            out.extend(chunk_result)

    assert len(out) == n, f"row count mismatch after reassembly: got {len(out)}, expected {n}"
    return pd.Series(out, index=text_series.index)