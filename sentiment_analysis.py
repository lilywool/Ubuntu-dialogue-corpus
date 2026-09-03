import re

import pandas as pd

# --- known placeholder tokens from the cleanup pipeline -------------------

# LANGUAGE_LABELS values (Latin-script non-English words), from
# residual_classification.py.
_LATIN_LANGUAGE_LABELS = [
    'SPANISH', 'PORTUGUESE', 'FRENCH', 'GERMAN', 'ITALIAN', 'DUTCH', 'JAPANESE',
]

# NON_LATIN_LABELS.upper(), from the non-Latin-script character-category step
# (cells 62-66, upstream of the Claude-review stage).
_NON_LATIN_LABELS = [
    'HAN_CHINESE_JAPANESE_KANJI', 'JAPANESE_HIRAGANA', 'JAPANESE_KATAKANA',
    'KOREAN_HANGUL', 'CYRILLIC', 'ARABIC', 'HEBREW', 'GREEK', 'THAI',
    'DEVANAGARI', 'ARMENIAN', 'GEORGIAN', 'ETHIOPIC', 'MALAYALAM', 'TAMIL',
    'SINHALA',
]

KNOWN_LABEL_TOKENS = ['NONWORD'] + _LATIN_LANGUAGE_LABELS + _NON_LATIN_LABELS

_LABEL_STRIP_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(t) for t in sorted(KNOWN_LABEL_TOKENS, key=len, reverse=True)) + r')\b'
)

# after stripping labels, a row with nothing but punctuation/whitespace left
# has no content to score
_ALNUM_RE = re.compile(r'[^\W\d_]')  # at least one alphabetic char


def strip_label_tokens(text):
    """Remove every known placeholder token from one string. Case-sensitive
    on purpose -- these tokens are always inserted upstream in all-caps, so
    matching case-sensitively avoids accidentally stripping a genuine
    lowercase/mixed-case word that happens to share spelling (there isn't
    one in this token list, but it's the correct default)."""
    if not isinstance(text, str) or not text:
        return text
    return _LABEL_STRIP_PATTERN.sub('', text)


def strip_label_tokens_series(series):
    """Vectorized version of strip_label_tokens. KNOWN_LABEL_TOKENS is only
    24 entries, nowhere near the scale that made the earlier regex passes
    slow -- this is cheap regardless of corpus size."""
    return series.str.replace(_LABEL_STRIP_PATTERN, '', regex=True)


def is_effectively_empty(text):
    """True if there's no alphabetic content left to score (empty string,
    all whitespace, or only punctuation/digits/leftover NONWORD debris)."""
    if not isinstance(text, str):
        return True
    return _ALNUM_RE.search(text) is None


# --- VADER (lexicon-based baseline) ----------------------------------------

_vader_analyzer = None


def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def score_vader(text):
    """Score one (already label-stripped) string with VADER. Returns
    (compound, label) where label is POSITIVE/NEUTRAL/NEGATIVE using
    VADER's own recommended compound thresholds (>=0.05 / <=-0.05).
    Returns (float('nan'), None) for empty/unscoreable text."""
    if is_effectively_empty(text):
        return float('nan'), None
    scores = _get_vader().polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        label = 'POSITIVE'
    elif compound <= -0.05:
        label = 'NEGATIVE'
    else:
        label = 'NEUTRAL'
    return compound, label


def score_vader_series(series):
    """Runs score_vader over a Series of ALREADY label-stripped text.
    Returns a DataFrame with vader_compound / vader_label columns, aligned
    to the input Series's index. Pure Python + dict lookups -- fast enough
    single-threaded regardless of corpus size (no parallel wrapper needed,
    same reasoning apply_lexicons.py's anonymize_structural uses for cheap
    per-row work)."""
    results = series.map(score_vader)
    compound = results.map(lambda r: r[0])
    label = results.map(lambda r: r[1])
    return pd.DataFrame({'vader_compound': compound, 'vader_label': label}, index=series.index)


# --- Transformer (cardiffnlp/twitter-roberta-base-sentiment-latest) --------

DEFAULT_TRANSFORMER_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
_TRANSFORMER_LABEL_MAP = {
    'negative': 'NEGATIVE', 'neutral': 'NEUTRAL', 'positive': 'POSITIVE',
    'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL', 'LABEL_2': 'POSITIVE',
}

_transformer_pipeline = None
_transformer_model_name = None


def get_transformer_pipeline(model_name=DEFAULT_TRANSFORMER_MODEL, device=None):
    global _transformer_pipeline, _transformer_model_name
    if _transformer_pipeline is not None and _transformer_model_name == model_name:
        return _transformer_pipeline

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    # Determine the correct torch dtype for the GPU
    torch_dtype = torch.float16 if device != -1 else torch.float32

    # Load tokenizer and model explicitly with dtype casting
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        torch_dtype=torch_dtype
    )

    # Pass the instantiated model and tokenizer to the pipeline
    _transformer_pipeline = pipeline(
        task='sentiment-analysis', 
        model=model, 
        tokenizer=tokenizer,
        device=device, 
        truncation=True, 
        max_length=512,
    )
    
    _transformer_model_name = model_name
    return _transformer_pipeline


def score_transformer_series(series, model_name=DEFAULT_TRANSFORMER_MODEL, batch_size=32, device=None):
    """Runs the transformer over a Series of ALREADY label-stripped text.
    Returns a DataFrame with transformer_label / transformer_score columns
    (transformer_score is the model's confidence in transformer_label, not
    a signed compound like VADER's). Empty/unscoreable rows get NaN/None
    without being sent to the model at all.

    Batched within ONE process via the HF pipeline's own batch_size, not
    parallelized across a process pool the way the earlier regex passes
    were -- spinning up 8 worker processes would mean loading 8 separate
    copies of a ~500MB model, which costs more in load time and RAM than
    it saves on CPU-bound inference. Benchmark on a few thousand rows
    first (see benchmark_transformer below) before committing to a full-df
    run -- CPU-only transformer inference over a multi-million-row corpus
    can still take a while even batched; that's a real decision, not
    something to guess past."""
    pipe = get_transformer_pipeline(model_name=model_name, device=device)

    texts = series.tolist()
    scoreable_idx = [i for i, t in enumerate(texts) if not is_effectively_empty(t)]
    scoreable_texts = [texts[i] for i in scoreable_idx]

    labels = [None] * len(texts)
    scores = [float('nan')] * len(texts)

    if scoreable_texts:
        try:
            from tqdm.auto import tqdm
            results = list(tqdm(
                pipe(scoreable_texts, batch_size=batch_size),
                total=len(scoreable_texts), desc="transformer sentiment",
            ))
        except ImportError:
            results = pipe(scoreable_texts, batch_size=batch_size)
        for idx, r in zip(scoreable_idx, results):
            labels[idx] = _TRANSFORMER_LABEL_MAP.get(r['label'], r['label'].upper())
            scores[idx] = r['score']

    return pd.DataFrame({'transformer_label': labels, 'transformer_score': scores}, index=series.index)


def benchmark_transformer(series, sample_n=2000, batch_size=32, device=None, random_state=0):
    """Times the transformer on a random sample and reports rows/sec, so
    you can extrapolate to your actual corpus size BEFORE committing to a
    full run. Returns (elapsed_seconds, rows_per_sec, projected_full_corpus_seconds)
    where the projection is scaled to len(series)."""
    import time

    sample = series.sample(n=min(sample_n, len(series)), random_state=random_state)
    stripped = strip_label_tokens_series(sample)

    t0 = time.time()
    score_transformer_series(stripped, batch_size=batch_size, device=device)
    elapsed = time.time() - t0

    rate = len(sample) / elapsed if elapsed > 0 else float('inf')
    projected = len(series) / rate if rate > 0 else float('inf')
    print(f"{len(sample):,} rows in {elapsed:.1f}s -> {rate:.1f} rows/sec")
    print(f"projected for full corpus ({len(series):,} rows): {projected/60:.1f} minutes")
    return elapsed, rate, projected


# --- combined entry point ---------------------------------------------------

def analyze_sentiment(df, text_column='text_cleaned', transformer_batch_size=32, transformer_device=None):
    """Adds vader_compound, vader_label, transformer_label, transformer_score
    columns to df, scoring both models on df[text_column] with all known
    placeholder tokens stripped first. Returns df (modified in place, same
    as the rest of this pipeline's convention)."""
    stripped = strip_label_tokens_series(df[text_column])

    vader_df = score_vader_series(stripped)
    df['vader_compound'] = vader_df['vader_compound']
    df['vader_label'] = vader_df['vader_label']

    transformer_df = score_transformer_series(
        stripped, batch_size=transformer_batch_size, device=transformer_device,
    )
    df['transformer_label'] = transformer_df['transformer_label']
    df['transformer_score'] = transformer_df['transformer_score']

    return df