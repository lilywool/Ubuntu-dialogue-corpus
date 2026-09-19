import re

from lexicons_and_templates.Ubuntu_Developer_lexicon_terms import TECH_LEXICON, TERM_CATEGORY

# Shorter terms ('sd', 'k', 'ram', 'o') produce too much noise as substrings
# of ordinary words to be worth checking here -- 'ram' alone would flag
# 'program', 'grammar', 'gram', etc. on every row. Multi-word phrases
# ('video card') are excluded too since they can't be "glued" the same way
# a single token can.
MIN_TERM_LEN = 4
_WORD_LIKE_RE = re.compile(r'^\w+$')
_PROTECTED_PLACEHOLDER_RUNS = {
    'EMAILADDRESS', 'WEBSITEDOMAIN', 'PHONENUMBER', 'SSN', 'IPADDRESS',
    'IPV6ADDRESS', 'MENUPATH', 'KEYBOARDSHORTCUT',
}


def _candidate_terms(lexicon=TECH_LEXICON, min_len=MIN_TERM_LEN):
    seen = set()
    terms = []
    for words in lexicon.values():
        for w in words:
            wl = w.lower()
            if wl in seen or len(wl) < min_len or not _WORD_LIKE_RE.match(wl):
                continue
            seen.add(wl)
            terms.append(wl)
    return terms


_CANDIDATE_TERMS = _candidate_terms()

# Deliberately NOT \b-bounded -- boundary checks are done manually per-match
# in find_glued_matches so prefix/suffix/infix can be told apart instead of
# just rejecting anything without a clean boundary on both sides.
_UNBOUNDED_MATCHER = re.compile(
    '|'.join(re.escape(t) for t in sorted(_CANDIDATE_TERMS, key=len, reverse=True)),
    re.IGNORECASE,
) if _CANDIDATE_TERMS else None


def _is_word_char(ch):
    return ch is not None and (ch.isalnum() or ch == '_')


def find_glued_matches(text):
    """Scan `text` for known tech-lexicon terms glued to adjacent word
    characters with no separator. Returns a list of dicts:
        {'term': ..., 'category': ..., 'glue': 'prefix'|'suffix'|'infix',
         'run': <the full contiguous \\w+ run the match sits inside>}
    Empty list if nothing found. Intended for rows that already have zero
    matches from find_tech_matches/find_slang_matches/structural matching --
    see module docstring for why this is a separate review pass rather than
    folded into the primary matcher."""
    if _UNBOUNDED_MATCHER is None:
        return []
    results = []
    for m in _UNBOUNDED_MATCHER.finditer(text):
        start, end = m.span()
        left_boundary = not _is_word_char(text[start - 1] if start > 0 else None)
        right_boundary = not _is_word_char(text[end] if end < len(text) else None)

        if left_boundary and right_boundary:
            continue  # clean bounded match -- find_tech_matches already has it

        glue = 'prefix' if left_boundary else ('suffix' if right_boundary else 'infix')

        run_start = start
        while run_start > 0 and _is_word_char(text[run_start - 1]):
            run_start -= 1
        run_end = end
        while run_end < len(text) and _is_word_char(text[run_end]):
            run_end += 1

        run = text[run_start:run_end]
        if run.upper() in _PROTECTED_PLACEHOLDER_RUNS:
            continue

        term = m.group(0).lower()
        results.append({
            'term': term,
            'category': TERM_CATEGORY.get(term),
            'glue': glue,
            'run': run,
        })
    return results
