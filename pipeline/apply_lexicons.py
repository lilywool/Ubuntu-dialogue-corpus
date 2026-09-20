import re
import time
from collections import Counter

import pandas as pd

try:
    from tqdm.auto import tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False

from lexicons_and_templates.Ubuntu_Developer_lexicon_terms import find_tech_matches
from lexicons_and_templates.sms_slang_lexicon import find_slang_matches
from lexicons_and_templates.structural_patterns import (
    EMAIL_ADDRESS_RE,
    PHONE_NUMBER_RE,
    URL_OR_DOMAIN_RE,
    IPV4_ADDRESS_RE,
    IPV6_ADDRESS_RE,
    MENU_PATH_RE,
    KEYBOARD_SHORTCUT_RE,
    SSN_RE,
    STRUCTURAL_PATTERNS,
)
from pipeline.parallel_execution import map_rows
from pipeline.glued_terms import find_glued_matches

# category -> (compiled pattern, placeholder token, count-column name).
# Order matters for the replacement pass below: a complete URL is a broader
# structural container and must be consumed before an email-like fragment in
# its path (for example, a mailing-list archive URL containing
# ``lug@linux.or.ug``). The bare-domain branch already excludes domains
# immediately preceded by ``@``, so ordinary email addresses remain available
# for the following email pass.
_ANONYMIZATION_SPEC = [
    ("url_or_domain", URL_OR_DOMAIN_RE, "WEBSITEDOMAIN", "domain_count"),
    ("email_address", EMAIL_ADDRESS_RE, "EMAILADDRESS", "email_count"),
    ("phone_number", PHONE_NUMBER_RE, "PHONENUMBER", "phone_count"),
    ("ssn", SSN_RE, "SSN", "ssn_count"),
    ("ipv4_address", IPV4_ADDRESS_RE, "IPADDRESS", "ipv4_count"),
    ("ipv6_address", IPV6_ADDRESS_RE, "IPV6ADDRESS", "ipv6_count"),
    ("menu_navigation_path", MENU_PATH_RE, "MENUPATH", "menu_path_count"),
    ("keyboard_shortcut", KEYBOARD_SHORTCUT_RE, "KEYBOARDSHORTCUT", "keyboard_shortcut_count"),
]
_ANONYMIZED_CATEGORIES = {category for category, _pat, _tok, _col in _ANONYMIZATION_SPEC}
_COUNT_COLUMNS = [col for _cat, _pat, _tok, col in _ANONYMIZATION_SPEC]

# Structural categories NOT part of the anonymization pass -- currently just
# unix_absolute_path. Kept verbatim in its own match column since it isn't PII.
_REMAINING_STRUCTURAL_PATTERNS = {
    k: v for k, v in STRUCTURAL_PATTERNS.items() if k not in _ANONYMIZED_CATEGORIES
}


def _count_matches(series, pattern):
    """Per-row match count for a compiled pattern. NOT series.str.count(pattern)
    -- on this pandas version (3.0.2, Arrow-backed string dtype), .str.count()
    silently drops flags baked into a compiled re.Pattern (it only honors an
    explicit flags= argument, and takes a fast path through PyArrow's RE2
    engine using just pattern.pattern as a bare string). Every regex in
    structural_patterns.py that needs case-insensitivity (EMAIL_ADDRESS_RE,
    URL_OR_DOMAIN_RE, KEYBOARD_SHORTCUT_RE) would silently undercount mixed-
    or upper-case matches through .str.count() -- confirmed directly: 'CTRL+ALT+T'
    counts as 0 via .str.count() but 1 via this method. .str.findall() and
    .str.replace() both go through Python's actual re engine and honor the
    compiled flags correctly, so counting via findall's length sidesteps the
    bug entirely."""
    return series.str.findall(pattern).str.len()


def anonymize_structural(series, verbose=False):
    """Replace all _ANONYMIZATION_SPEC categories with their placeholder
    tokens in a text Series, in order. Returns (scrubbed_series, counts_df)
    where counts_df has one integer column per category ('email_count',
    'domain_count', 'phone_count', 'ipv4_count', 'ipv6_count',
    'menu_path_count', 'keyboard_shortcut_count') -- counts only, never the
    matched strings themselves (see the module docstring's privacy note).
    Counts describe placeholders in the final scrubbed text. Deriving them
    after every replacement prevents a later, broader structural replacement
    from leaving a stale count for a placeholder it consumed.
    Each of the 7 replacement passes is vectorized (not per-row), so there's
    no per-row progress bar to show here -- verbose=True instead prints a
    one-line timestamp per category as it completes, so you can at least see
    it moving through the 7 steps rather than staring at a silent cell."""
    out = series
    for category, pattern, placeholder, count_col in _ANONYMIZATION_SPEC:
        t0 = time.time()
        out = out.str.replace(pattern, placeholder, regex=True)
        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] anonymized {category} ({time.time()-t0:.1f}s)")
    counts = {
        count_col: _count_matches(
            out,
            re.compile(rf"\b{re.escape(placeholder)}\b"),
        )
        for _category, _pattern, placeholder, count_col in _ANONYMIZATION_SPEC
    }
    return out, pd.DataFrame(counts)


def _find_remaining_structural_matches(text):
    """Like structural_patterns.find_structural_matches, but only the
    categories not covered by anonymize_structural (currently just
    unix_absolute_path), and returns a flat list of (matched_text, category)
    tuples to match the shape of find_tech_matches / find_slang_matches."""
    results = []
    for category, pattern in _REMAINING_STRUCTURAL_PATTERNS.items():
        for match in pattern.findall(text):
            results.append((match, category))
    return results


def apply_lexicons(df, text_col='text_cleaned', anonymize=True, show_progress=True, workers=1):
    """Run all three lexicon components against df[text_col].

    Returns (df, matched_subset, tally):
      - df: the input dataframe (mutated in place) with new columns:
          text_col (anonymized in place, if anonymize=True: emails ->
                    EMAILADDRESS, domains/URLs -> WEBSITEDOMAIN, phone
                    numbers -> PHONENUMBER, IPv4 addresses -> IPADDRESS,
                    IPv6 addresses -> IPV6ADDRESS, menu paths -> MENUPATH,
                    keyboard shortcuts -> KEYBOARDSHORTCUT)
          'email_count' / 'domain_count' / 'phone_count' / 'ipv4_count' /
          'ipv6_count' / 'menu_path_count' / 'keyboard_shortcut_count'
                                        -- int, how many of each were found
                                           and scrubbed per row (counts only
                                           -- see the module docstring's
                                           privacy note for why the matched
                                           strings themselves are never kept)
          'tech_lexicon_matches'       -- list[(term, category)]
          'tech_lexicon_match_count'   -- number of technical matches
          'slang_matches'              -- list[(token, type, meaning)]
          'slang_match_count'          -- number of slang matches
          'structural_matches'         -- list[(matched_text, category)] for
                                           unix_absolute_path only -- the one
                                           structural category left as raw
                                           text (see module docstring)
          'glued_matches'             -- list[dict] for known technical terms
                                           embedded in larger tokens
          'has_lexicon_match'          -- bool, True if any of the above
                                           (including the seven placeholder
                                           counts) are non-empty/non-zero.
                                           As of the 2026-08-19 lexicon
                                           review there is no more separate
                                           low-confidence tier held out of
                                           this flag -- every tech_lexicon
                                           category (including the former
                                           'low_confidence' terms, now under
                                           'general_computing') counts
                                           equally.
      - matched_subset: df[df['has_lexicon_match']], a copy
      - tally: dict of Counters -- {'tech', 'slang', 'structural',
        'anonymized'} -- each counting how many times each
        term/token/category fired across the whole corpus (not just the
        subset), for sanity-checking what's actually matching before you
        trust it. 'anonymized' counts totals for the six placeholder-ized
        categories only (again, never the matched strings).

    show_progress=True (default) prints a timestamped line as each of the
    4 phases (structural scan, anonymization, tech lexicon, slang lexicon)
    starts and finishes, and -- if the `tqdm` package is installed -- shows a
    live per-row progress bar with a rolling ETA on each of the 3 .apply()
    passes, instead of a silent cell with no feedback until it returns. Set
    False to fully suppress this (e.g. for a smaller/faster call where the
    overhead of the progress bar itself isn't worth it).
    """
    def _log(msg):
        if show_progress:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _apply(series, func, desc):
        if workers > 1:
            return pd.Series(map_rows(func, series, workers), index=series.index)
        if show_progress and _HAVE_TQDM:
            tqdm.pandas(desc=desc)
            return series.progress_apply(func)
        return series.apply(func)

    n = len(df)
    _log(f"starting apply_lexicons on {n:,} rows"
         + ("" if _HAVE_TQDM else "  (tip: `pip install tqdm` for a live per-row progress bar)"))

    text = df[text_col].astype(str)

    # The one structural category not anonymized -- safe to keep verbatim.
    _log("phase 1/4: scanning unix_absolute_path...")
    t0 = time.time()
    df['structural_matches'] = _apply(text, _find_remaining_structural_matches, "structural_matches")
    _log(f"phase 1/4 done ({time.time()-t0:.1f}s)")

    if anonymize:
        _log("phase 2/4: anonymizing email/domain/phone/ipv4/ipv6/menu/shortcut...")
        t0 = time.time()
        scrubbed, counts_df = anonymize_structural(text, verbose=show_progress)
        df[text_col] = scrubbed
        for col in _COUNT_COLUMNS:
            df[col] = counts_df[col].values
        text = df[text_col].astype(str)  # re-read scrubbed text for tech/slang matching below
        _log(f"phase 2/4 done ({time.time()-t0:.1f}s)")
    else:
        for _category, pattern, _placeholder, count_col in _ANONYMIZATION_SPEC:
            df[count_col] = _count_matches(text, pattern)

    _log("phase 3/4: tech lexicon matching...")
    t0 = time.time()
    df['tech_lexicon_matches'] = _apply(text, find_tech_matches, "tech_lexicon_matches")
    df['tech_lexicon_match_count'] = df['tech_lexicon_matches'].str.len()
    _log(f"phase 3/4 done ({time.time()-t0:.1f}s)")

    _log("phase 4/4: slang/emoticon lexicon matching...")
    t0 = time.time()
    df['slang_matches'] = _apply(text, find_slang_matches, "slang_matches")
    df['slang_match_count'] = df['slang_matches'].str.len()
    _log(f"phase 4/4 done ({time.time()-t0:.1f}s)")

    _log("glued-term matching...")
    t0 = time.time()
    df['glued_matches'] = _apply(text, find_glued_matches, "glued_matches")
    df['glued_match_count'] = df['glued_matches'].str.len()
    _log(f"glued-term matching done ({time.time()-t0:.1f}s)")

    anonymized_total_count = df[_COUNT_COLUMNS].sum(axis=1)
    df['has_lexicon_match'] = (
        df['tech_lexicon_matches'].str.len().gt(0)
        | df['slang_matches'].str.len().gt(0)
        | df['glued_matches'].str.len().gt(0)
        | df['structural_matches'].str.len().gt(0)
        | anonymized_total_count.gt(0)
    )

    matched_subset = df[df['has_lexicon_match']].copy()

    tally = {
        'tech': Counter(term.lower() for matches in df['tech_lexicon_matches'] for term, _cat in matches),
        'slang': Counter(tok for matches in df['slang_matches'] for tok, _typ, _meaning in matches),
        'glued': Counter(match['term'] for matches in df['glued_matches'] for match in matches),
        'structural': Counter(cat for matches in df['structural_matches'] for _match, cat in matches),
        'anonymized': Counter({
            category: int(df[count_col].sum())
            for category, _pat, _tok, count_col in _ANONYMIZATION_SPEC
        }),
    }

    _log(f"done. {len(matched_subset):,} / {n:,} rows have at least one lexicon match")

    return df, matched_subset, tally
