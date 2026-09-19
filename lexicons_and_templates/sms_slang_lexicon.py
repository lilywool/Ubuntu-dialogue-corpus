import csv
import re
from pathlib import Path

_DEFAULT_LEXICON_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = _DEFAULT_LEXICON_DIR / "sms_slang_emoticons_dictionary_filled.csv"

# Types held out of automatic slang/sentiment tagging -- the CSV author
# flagged these as needing a human look, not blanket auto-labeling.
REVIEW_REQUIRED_TYPES = {"Flagged/Ambiguous", "Other/Non-Slang"}

# Types whose tokens are made of ordinary word characters (letters/digits),
# and therefore need \b word-boundary matching to avoid substring false
# positives. Everything else (Emoticon, Kaomoji, most Symbol rows) is
# matched as a literal, escaped, boundary-free sequence instead.
_WORD_LIKE_TYPES = {"Textese", "Slang", "Technical/IRC", "Expression"}

def load_entries(csv_path=DEFAULT_CSV_PATH):
    """Read the CSV into a list of {token, type, meaning} dicts, preserving
    file order and every row (including review-required ones)."""
    entries = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entries.append({
                "token": row["Token"],
                "type": row["Type"],
                "meaning": row["Meaning"],
            })
    return entries


ALL_ENTRIES = load_entries()
AUTO_ENTRIES = [e for e in ALL_ENTRIES if e["type"] not in REVIEW_REQUIRED_TYPES]
REVIEW_ENTRIES = [e for e in ALL_ENTRIES if e["type"] in REVIEW_REQUIRED_TYPES]

# Exact-case lookup -- covers symbol tokens (which have no case) and serves
# as the first lookup attempt for word-like tokens matched in their
# canonical case.
SLANG_LOOKUP = {e["token"]: (e["type"], e["meaning"]) for e in AUTO_ENTRIES}
REVIEW_LOOKUP = {e["token"]: (e["type"], e["meaning"]) for e in REVIEW_ENTRIES}


def _is_word_like(entry):
    return entry["type"] in _WORD_LIKE_TYPES and re.fullmatch(r"[\w'/-]+", entry["token"])


# Case-folded lookup for word-like tokens ('thx' or 'THX' both need to find
# the 'thx' entry; 'u'/'U' both need to find the 'U' entry) -- built from
# lower-cased keys so a match's casing doesn't matter at lookup time.
_WORD_LOOKUP_CI = {
    e["token"].lower(): (e["type"], e["meaning"])
    for e in AUTO_ENTRIES
    if _is_word_like(e)
}


def _build_matcher(entries):
    """Compile one regex that finds any token in `entries` as a standalone
    unit. Returns None if `entries` is empty."""
    word_tokens, symbol_tokens = [], []
    for e in entries:
        tok = e["token"]
        (word_tokens if _is_word_like(e) else symbol_tokens).append(tok)

    branches = []
    if word_tokens:
        # Case-insensitive by design -- see the module docstring's hazard #1
        # for the recall/precision tradeoff this implies for 'U'/'R'.
        alt = "|".join(re.escape(t) for t in sorted(word_tokens, key=len, reverse=True))
        branches.append(rf"\b(?i:{alt})\b")
    if symbol_tokens:
        # Longest-first so e.g. a 3-char kaomoji matches whole rather than a
        # 1-char alternative elsewhere in the list matching a piece of it first.
        alt = "|".join(re.escape(t) for t in sorted(symbol_tokens, key=len, reverse=True))
        branches.append(f"(?:{alt})")

    if not branches:
        return None
    return re.compile("|".join(branches))


# Compiled once at import time. AUTO_MATCHER excludes review-required tokens
# by default; REVIEW_MATCHER is exposed separately for callers that
# explicitly want to also surface flagged/ambiguous hits (e.g. to route them
# to a human-review queue rather than auto-tag them).
AUTO_MATCHER = _build_matcher(AUTO_ENTRIES)
REVIEW_MATCHER = _build_matcher(REVIEW_ENTRIES)


def find_slang_matches(text, include_review=False):
    """Scan `text` and return a list of (token, type, meaning) tuples for
    every match found, in order of appearance. Review-required tokens
    (Flagged/Ambiguous, Other/Non-Slang) are excluded unless
    include_review=True -- and even then, callers should treat those hits as
    needing a human look, not as settled classifications."""
    results = []
    if AUTO_MATCHER is not None:
        for m in AUTO_MATCHER.finditer(text):
            tok = m.group(0)
            # Exact-case lookup first (covers symbols and any word-like token
            # typed in its canonical case); fall back to the case-folded map
            # for a word-like token matched in a different case ('thx'
            # matched -> look up via 'thx'.lower() against the 'thx'/'THX'
            # entry's folded key; 'u' matched -> look up 'u' against 'U').
            if tok in SLANG_LOOKUP:
                typ, meaning = SLANG_LOOKUP[tok]
            else:
                typ, meaning = _WORD_LOOKUP_CI.get(tok.lower(), (None, None))
            results.append((tok, typ, meaning))
    if include_review and REVIEW_MATCHER is not None:
        for m in REVIEW_MATCHER.finditer(text):
            tok = m.group(0)
            typ, meaning = REVIEW_LOOKUP.get(tok, (None, None))
            results.append((tok, typ, meaning))
    return results