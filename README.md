# Ubuntu Dialogue Corpus — text cleaning and sentiment pipeline

Cleaning, normalization and sentiment analysis over the Ubuntu Dialogue
Corpus: roughly 8.6 million messages from the Ubuntu IRC technical support
channels.

Support chat is messy in ways that general-purpose NLP tools handle badly.
A single message can contain a shell command, a pasted stack trace, a URL, an
emoticon, a typo, and a word of Portuguese. This repository is the tooling
built to make that text tractable: scrub structural artifacts, normalize chat
shorthand, identify technical jargon separately from genuine noise, and score
sentiment on what remains.

## What's here

**Anonymization and structural scrubbing**

- `structural_patterns.py` — regex patterns for emails, URLs and bare domains,
  phone numbers, IPv4/IPv6 addresses, SSNs, menu paths and keyboard shortcuts
- `apply_lexicons.py` — applies the scrubbing and lexicon matching in order,
  returning cleaned text plus per-category match counts

**Vocabulary and normalization**

- `Ubuntu_Developer_lexicon_terms.py` — categorized technical vocabulary
  (packages, commands, hardware, desktop environments, and so on)
- `sms_slang_lexicon.py` — chat slang and emoticon matching, driven by
  `sms_slang_emoticons_dictionary_filled.csv`
- `chat_normalization.py` — typo and shorthand correction, plus regex-family
  canonicalization for things like laughter and filler words
- `glued_terms.py` — detects run-together terms (`configureatiojn`,
  `driverloader`) that neither the lexicon nor a spellchecker catches
- `residual_classification.py` — sorts leftover tokens into technical jargon,
  non-English language labels, or `NONWORD`

**Sentiment**

- `sentiment_analysis.py` — VADER scoring, plus a transformer path using
  `cardiffnlp/twitter-roberta-base-sentiment-latest`

**Parallel wrappers**

- `parallel_match.py`, `parallel_vocab_check.py`,
  `parallel_chat_normalization.py`, `parallel_classify.py` — process-pool
  wrappers for the expensive passes, with module-level worker functions so
  they pickle correctly on Windows

**Reference data**

The CSVs are working artifacts from the classification passes: the slang and
emoticon dictionary, reviewed glued-term matches, and residual word lists with
corpus frequencies.

## Status

Cleaning and normalization are complete and validated at full corpus scale,
as is VADER sentiment. The transformer sentiment pass is paused: a full run
collapsed to a handful of distinct scores, and the cause is still under
investigation on the `transformer` branch.

The `notebooks/` directory holds the exploratory work the modules were
extracted from. It is kept for reference, not as the way to run anything.
There is no single entry-point script yet; the modules are imported and run
per stage.

## Data

The corpus is not included here — it is far too large for git, and it isn't
mine to redistribute. The pipeline expects the V1 `dialogueText` CSV; paths in
the notebooks assume a `_corpus_data/` folder alongside this repository.

## A note on method

The residual-word classification step — sorting roughly 177,000 leftover
tokens into jargon, language labels, or noise — was done with LLM assistance
rather than by hand. Ubuntu support chat is dense with vocabulary that needs
subject-matter knowledge to separate real jargon from garbage, and labeling
that volume manually wasn't feasible. The review CSVs in this repository are
the record of that process. Anyone evaluating the resulting labels should know
a model produced them.

## Attribution

Built on the Ubuntu Dialogue Corpus:

> Ryan Lowe, Nissan Pow, Iulian Serban, and Joelle Pineau. "The Ubuntu
> Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn
> Dialogue Systems." *SIGDIAL*, 2015. arXiv:1506.08909

- Paper: https://arxiv.org/abs/1506.08909
- Dataset generation scripts: https://github.com/rkadlec/ubuntu-ranking-dataset-creator
  (Apache-2.0)

The corpus derives from publicly archived Ubuntu IRC channel logs. Its own
redistribution terms are not clearly stated by the original release — the
Hugging Face dataset card lists the license as unknown — so treat the original
sources above as authoritative rather than anything here. The derived word
lists and review CSVs in this repository are frequency counts and token
classifications, not message content.

## License

The code in this repository is MIT licensed — see `LICENSE`. That covers the
code only, not the underlying corpus, whose terms are as described above.
