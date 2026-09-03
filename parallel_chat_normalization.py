from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from chat_normalization import apply_latin_corrections, normalize_chat_shorthand

try:
    from tqdm.auto import tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False


def _normalize_chunk(texts):
    """Worker: apply both chat_normalization passes to one chunk (a plain
    list of strings). Rebuilds a throwaway Series so the .str accessor
    works, same as the module-level functions expect. Must stay a top-level
    function -- see parallel_match.py's docstring for why (Windows/Jupyter
    picklability -- a lambda or notebook-local closure can't be pickled to
    send to a worker process)."""
    s = apply_latin_corrections(pd.Series(texts))
    s = normalize_chat_shorthand(s)
    return s.tolist()


def parallel_normalize(text_series, n_workers=8, chunks_per_worker=4):
    """Run apply_latin_corrections + normalize_chat_shorthand across a
    process pool. Returns a Series aligned to text_series's original index --
    safe to assign straight back to frame['text_cleaned']."""
    texts = text_series.tolist()
    n = len(texts)
    chunk_count = max(1, n_workers * chunks_per_worker)
    size = -(-n // chunk_count)  # ceil division
    chunks = [texts[i:i + size] for i in range(0, n, size)]

    out = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = ex.map(_normalize_chunk, chunks)
        if _HAVE_TQDM:
            results = tqdm(results, total=len(chunks), desc="parallel chat-normalization")
        for chunk_result in results:
            out.extend(chunk_result)

    assert len(out) == n, f"row count mismatch after reassembly: got {len(out)}, expected {n}"
    return pd.Series(out, index=text_series.index)