from concurrent.futures import ProcessPoolExecutor

from Ubuntu_Developer_lexicon_terms import find_tech_matches
from sms_slang_lexicon import find_slang_matches
from apply_lexicons import _find_remaining_structural_matches

try:
    from tqdm.auto import tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False


def _match_chunk(texts):
    """Worker: run both lexicon matchers over one chunk (a plain list of
    strings). Must stay a top-level function -- see module docstring."""
    return (
        [find_tech_matches(t) for t in texts],
        [find_slang_matches(t) for t in texts],
    )


def _structural_chunk(texts):
    """Worker: run apply_lexicons.py's phase-1 unix_absolute_path scan over
    one chunk. Must stay a top-level function -- see module docstring."""
    return [_find_remaining_structural_matches(t) for t in texts]


def _chunk_series(text_series, n_workers, chunks_per_worker):
    texts = text_series.tolist()
    n = len(texts)
    chunk_count = max(1, n_workers * chunks_per_worker)
    size = -(-n // chunk_count)  # ceil division
    return texts, n, [texts[i:i + size] for i in range(0, n, size)]


def parallel_match(text_series, n_workers=8, chunks_per_worker=4):
    """Run find_tech_matches/find_slang_matches (phases 3-4) across a
    process pool. Returns (tech_matches_list, slang_matches_list) in
    original row order -- safe to assign directly to
    df['tech_lexicon_matches'] / df['slang_matches']. Must run on the
    ALREADY-ANONYMIZED text_cleaned column -- see module docstring.

    n_workers defaults to 8 (physical-core count on the AMD Ryzen 7 8840HS --
    an 8P/16T chip; use the physical count, not the logical thread count,
    since this is pure CPU-bound work where hyperthreading doesn't add real
    throughput). chunks_per_worker=4 means each worker processes ~4 chunks
    rather than 1, so the progress bar updates more often and one slow chunk
    (e.g. an unusually long pasted-log row) doesn't stall a whole worker for
    the entire run -- the work gets spread more evenly."""
    texts, n, chunks = _chunk_series(text_series, n_workers, chunks_per_worker)

    tech_all, slang_all = [], []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = ex.map(_match_chunk, chunks)
        if _HAVE_TQDM:
            results = tqdm(results, total=len(chunks), desc="parallel tech+slang matching")
        for tech_chunk, slang_chunk in results:
            tech_all.extend(tech_chunk)
            slang_all.extend(slang_chunk)

    assert len(tech_all) == n and len(slang_all) == n, \
        f"row count mismatch after reassembly: got {len(tech_all)}/{len(slang_all)}, expected {n}"
    return tech_all, slang_all


def parallel_structural_scan(text_series, n_workers=8, chunks_per_worker=4):
    """Run apply_lexicons.py's phase-1 unix_absolute_path scan across a
    process pool. Returns a structural_matches_list in original row order --
    safe to assign directly to df['structural_matches']. Must run on the RAW
    (pre-anonymization) text_cleaned column -- see module docstring."""
    texts, n, chunks = _chunk_series(text_series, n_workers, chunks_per_worker)

    structural_all = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = ex.map(_structural_chunk, chunks)
        if _HAVE_TQDM:
            results = tqdm(results, total=len(chunks), desc="parallel structural (unix_absolute_path) scan")
        for chunk in results:
            structural_all.extend(chunk)

    assert len(structural_all) == n, \
        f"row count mismatch after reassembly: got {len(structural_all)}, expected {n}"
    return structural_all