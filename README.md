# Ubuntu Dialogue Corpus — text cleaning and sentiment pipeline

This repository prepares the Ubuntu Dialogue Corpus for sentiment analysis and
later conversation-level analytics. The corpus contains roughly 8.6 million
technical-support messages whose text can mix commands, stack traces, URLs,
slang, emoticons, typos, and multiple languages.

## Current workflow

The text-processing sequence is complete:

```text
human normalization
    -> structural anonymization
    -> lexicon/slang matching
    -> glued-term matching
    -> residual vocabulary analysis
    -> human-validated residual overlay
    -> NONWORD/language labels
```

The immediate next stage is sentiment:

```text
cleaned text
    -> VADER, transformer, or both
    -> sentiment-enriched DataFrame
    -> outputs/df_with_sentiment.pkl
```

The same Step 8 artifact is written only after compact spaCy/POS and named-
entity features, stable NMF topics, and the advanced-NLP validation gate have
run. Transformer emotion classification is an explicit optional arm.

After sentiment, the next architectural layer is silver-to-gold aggregation:

- conversation-level summaries
- user-level summaries
- sentiment by conversation, user, channel, and time
- Ubuntu release-cycle trends
- language and residual distributions
- topic and technical-term trends

The implemented optional advanced-NLP stage covers spaCy token/POS summaries,
general named entities, topic modeling, and transformer emotion
classification. Technical entities remain the responsibility of the reviewed
Ubuntu lexicon stage. Later work can build phrase- and conversation-level
models on the validated message features and gold summaries.

## Repository layout

- `pipeline/` — canonical normalization, matching, residual-classification,
  sentiment, and orchestration modules
- `reviewed_overlays/` — corpus-specific human-reviewed corrections and
  classifications
- `lexicons_and_templates/` — technical vocabulary, slang, emoticons, and
  structural patterns
- `notebooks/` — the exploratory steps 1–6 and the continuing steps 7–12
- `databricks_integration/` — migration notes plus the Spark-native,
  validated silver-to-gold Delta job; bronze-to-silver remains staged work
- `dashboard/` — the final Streamlit presentation layer for validated notebook
  outputs
- `tests/` — regression and integration tests
- `tools/` — repository maintenance utilities
- `outputs/` — generated artifacts; ignored by Git

## Local environment

The project is isolated in a repository-local `.venv` and pins Python
3.11.16 in `.python-version`.

```powershell
& "C:\Users\lilli_37fwt34\anaconda3\Scripts\conda.exe" create --prefix ".\.venv" python=3.11.16 -y
.\.venv\python.exe -m pip install --upgrade pip
.\.venv\python.exe -m pip install -r requirements.lock.txt
```

Optional dependencies are separated:

- requirements-dashboard.txt — Streamlit and Plotly for the final dashboard
- `requirements-transformer.txt` — CPU PyTorch and Transformers
- `requirements-advanced-nlp.txt` — spaCy, its pinned English model, and
  scikit-learn topic modeling
- `requirements-notebooks.txt` — the local Jupyter kernel plus plotting and
  statistical notebook packages

Install the notebook environment through the same local interpreter, then
select `.venv\python.exe` as the notebook kernel:

```powershell
.\.venv\python.exe -m pip install -r notebooks\requirements.txt
```

Run the tests through the local interpreter:

```powershell
.\.venv\python.exe -m unittest discover -s tests -v
```

For a bounded local transformer run, sample before cleaning/NLP extraction:

```powershell
.\.venv\python.exe -m pipeline.pipeline data\dialogueText_196.csv `
  --sample-size 10000 --sample-seed 42 --sentiment both --advanced-nlp
```

The same seed selects the same source rows. Selected rows are restored to
their original source order before sequence-dependent work; requesting at
least the full input size does not shuffle the dataset. Source row count,
selected row count, seed, and whether sampling occurred are retained in
provenance and the output audit record.

No project step requires modifying the system Python, global `PATH`, Conda
base environment, Java, CUDA, or GPU drivers.

## Data and paths

The raw CSV is local-only at:

```text
data/dialogueText_196.csv
```

It is intentionally ignored by Git. The notebooks resolve the repository root
whether Jupyter starts from the root or from `notebooks/`, and read the CSV
from the repository-local `data/` directory.

The steps 7–8.5 notebook runs steps 1–6 in the same kernel. It does not read or
write intermediate pickle checkpoints. The sole notebook pickle export is the
completed sentiment artifact:

```text
outputs/df_with_sentiment.pkl
```

Generated review CSVs and final artifacts also stay under `outputs/`, which is
ignored by Git.

## Sentiment behavior

`pipeline.sentiment_analysis.analyze_sentiment` supports `vader`,
`transformer`, and `both`. Transformer dtype is explicit and defaults to
`float32`; GPU availability never silently forces fp16. Model repository
revisions are pinned as well as Python package versions.

VADER retains its negative, neutral, and positive proportions in addition to
compound score and label. The three-class RoBERTa path retains every class
probability plus predicted label, confidence, expected sentiment
(`positive - negative`), entropy, normalized entropy, and top-two margin.
Values are not rounded before validation. Inference uses a configurable
bounded chunk size, so the 8.6-million-row corpus is never duplicated as one
giant Python text list.

Every rerun first removes all columns owned by the sentiment stage, preventing
outputs from an older mode from surviving unnoticed. Validation checks
coverage, numeric ranges, probability sums, label/argmax agreement,
confidence/max-probability agreement, signed-score consistency, uncertainty,
and suspiciously collapsed score diversity. Model, revision, dtype, package
versions, label order, preprocessing, and batching settings are retained in
DataFrame provenance metadata.

Upstream model cards: [CardiffNLP three-class sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
and [Hartmann seven-class emotion](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base).

The final gate also validates the deterministic cleaning layer: row/index
preservation, duplicate columns, non-null cleaned text, match-list/count
agreement, placeholder/count agreement, residual flags, and lexicon flags.
Structural placeholder tokens are protected from glued-term scanning (for
example, `address` inside `EMAILADDRESS` is not a technical-term match).

When a human-labeled evaluation set is available,
`pipeline.evaluation.evaluate_sentiment_labels` measures coverage, accuracy,
macro-F1, balanced accuracy, per-class precision/recall/F1, confusion matrix,
multiclass Brier score, log loss, and expected calibration error. When both
backends run, validation also records VADER/transformer label agreement and
score correlation. These are diagnostics, not hard-coded claims of model
accuracy; the project does not substitute Yelp star ratings for Ubuntu gold
labels.

The CPU preprocessing diagnostic did not reproduce the historical transformer
score collapse. A representative real-data parity run and a controlled GPU
dtype comparison remain separate validation work before a full transformer
run.

## Advanced NLP behavior

`pipeline/advanced_nlp.py` coordinates three independently callable stages:

- `pipeline/spacy_features.py` — batched tokenization, compact POS summaries,
  normalized POS ratios, sentence/lexical/negation/punctuation features, and
  general named entities with label-level counts
- `pipeline/topic_modeling.py` — one deterministic bounded-sample NMF fit,
  followed by bounded-batch assignment using the same model, explicit
  vocabulary coverage, normalized topic confidence, entropy, and margin
- `pipeline/emotion_analysis.py` — optional Hugging Face emotion scoring with
  explicit device and dtype

Technical entities are not re-counted by spaCy. The earlier
`tech_lexicon_matches` and `tech_lexicon_match_count` columns remain the
canonical technical-entity features. Full token/POS JSON is opt-in because it
is unsuitable as the default representation for 8.6 million pandas rows.

The advanced stage validates coverage, JSON/count consistency, topic scores,
transformer score ranges, and frozen outputs before the notebook writes its
single Step 8 artifact.

The remainder of `notebooks/Ubuntu_step7-12.ipynb` completes the analytical
workflow in the same notebook style:

- Step 9 selects a deterministic sample of complete conversations, builds
  validated gold summaries, calculates robust Spearman correlations, and runs
  effect-size-aware hypothesis tests with Holm correction;
- Step 10 produces bounded distribution/relationship plots and aggregated time,
  release-cycle, and topic figures at publication resolution;
- Step 11 compares a dummy baseline, logistic regression, and random forest,
  tunes the strongest real model, and evaluates it on an untouched test set;
- Step 12 evaluates held-out model performance and baseline lift, quantifies a
  bounded triage-impact scenario, records limitations and next steps, and writes
  reproducibility documentation. A separate downstream handoff exports the
  privacy-bounded tables and manifest used by the Streamlit dashboard.

The supervised target is whether another sender replies to a conversation's
initial message. Only features available at posting time are eligible;
conversation duration, message count, response gaps, later messages, and raw
user identities are explicitly excluded as leakage.

### Design lineage and improvements

The advanced-NLP workflow was modeled on the architectural patterns proven in
the separate
[Yelp Review Intelligence](https://github.com/lilywool/yelp-review-intelligence)
repository: import-safe model initialization, optional transformer features,
validation before output, and a clear separation between source text and
generated features. The implementation here is independent and adapted to the
Ubuntu corpus rather than sharing code, data, or environments with Yelp.

For this repository, those patterns were extended in several ways:

- advanced NLP is separated into dedicated spaCy, topic-modeling, emotion,
  sentiment, and orchestration modules rather than one feature-engineering
  script;
- expensive local stages use the shared, Windows-safe execution backend in
  `pipeline/parallel_execution.py`, while transformer inference remains
  batched in one process to avoid loading multiple copies of a large model;
- Ubuntu technical entities remain governed by the existing reviewed lexicon
  and residual pipeline, avoiding a second spaCy-derived count for the same
  concept;
- spaCy uses batched processing and compact POS/entity summaries by default,
  adds length-normalized POS ratios and lexical/sentence/negation features,
  and keeps full token-level JSON as an explicit opt-in;
- topic modeling fits one deterministic model on a bounded representative
  sample and reuses it for every row, so topic IDs retain one corpus-wide
  meaning and can later be shared across Databricks partitions; out-of-
  vocabulary messages stay explicitly unassigned instead of being falsely
  labeled topic 0;
- transformer sentiment and emotion both use explicit device and dtype
  settings and pinned model revisions, so GPU availability never silently
  enables fp16 and a moving model repository cannot silently change results;
- sentiment retains full VADER and three-class transformer distributions,
  principled signed transformer expectation, entropy, and decision margin;
- optional emotion output retains all seven class probabilities plus entropy
  and decision margin instead of only the winning label/confidence;
- transformer stages stream bounded inference chunks rather than building a
  corpus-sized Python list, while still using internal model batching;
- stage-owned column registries remove stale outputs before every rerun;
- numeric features use compact, explicit dtypes (`float32`, `int32`, and
  nullable `Int16` topic IDs) to keep the 8.6-million-row artifact tractable;
- validation covers probability conservation and internal agreement as well
  as score ranges, coverage, JSON/count consistency, topic diversity, and the
  historical transformer-collapse failure mode;
- successful outputs append a JSON-lines audit record with validation results,
  configuration, provenance, output size, and UTC completion time; and
- a dependency-light gold-label evaluator covers both discrimination and
  probability calibration rather than relying only on a rating correlation;
- the modules preserve one message per row and expose partition-callable
  functions for the planned Spark `mapInPandas` implementation.

These are concrete extensions beyond Yelp's sentiment baseline, which stores
the VADER distribution but reduces its optional binary transformer to a
winning label/confidence and a derived signed confidence. Ubuntu's neutral-
aware three-class distribution and expected score preserve information needed
for later conversation-level aggregation without treating neutral messages as
implicitly positive or negative.

## Streamlit dashboard

The final pipeline application is the Ubuntu Dialogue Intelligence dashboard
under dashboard/. After notebook Step 12 completes its evaluation, the separate
dashboard-handoff cells call pipeline.dashboard.export_dashboard_bundle to
validate and atomically write the dashboard's input tables and portable
manifest. The dashboard revalidates the bundle on load and refuses
schema-mismatched, incomplete, or privacy-unsafe inputs.

Install its optional dependencies into this repository's local environment:

~~~powershell
.\.venv\python.exe -m pip install -r requirements-dashboard.txt
~~~

After running notebook Steps 7-12, launch it from the repository root:

~~~powershell
.\.venv\python.exe -m streamlit run dashboard\app.py
~~~

The four views mirror the Yelp project's information architecture while using
Ubuntu-specific measures:

- **Descriptive** — response outcomes, conversation size, time trends, topics
- **Diagnostic** — correlations, corrected hypothesis tests, response gaps
- **Predictive** — held-out model metrics, model comparison, feature importance
- **Prescriptive** — supported low-response topic priorities and release-cycle
  monitoring, explicitly presented as operational leads rather than causal facts

Sidebar filters drill into conversation date, response outcome, and dominant
topic. Dashboard exports contain no raw message text or usernames. Set
UBUNTU_DASHBOARD_DATA only in the current process or an uncommitted local
configuration if the bundle lives somewhere other than
outputs/dashboard_data.

## Databricks status

The local pandas pipeline remains the feature-definition source of truth. A
Spark-native silver-to-gold job now reads a silver Delta table/path, performs
global Spark aggregations, validates conservation and key uniqueness, writes
the selected Delta gold table, and appends a Delta audit record. It supports
conversation, directional user, date, channel, release-cycle, language,
residual, technical-term, topic, entity, and seeded-sample outputs.

Spark-native bronze ingestion, the partition-safe `mapInPandas` NLP stage,
Asset Bundles, and complete local-to-distributed parity tests remain future
work. The pandas bronze-to-silver adapter is not represented as a distributed
Databricks implementation.

## Attribution and license

Built on the Ubuntu Dialogue Corpus:

> Ryan Lowe, Nissan Pow, Iulian Serban, and Joelle Pineau. “The Ubuntu
> Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn
> Dialogue Systems.” *SIGDIAL*, 2015. arXiv:1506.08909.

- Paper: https://arxiv.org/abs/1506.08909
- Dataset-generation scripts:
  https://github.com/rkadlec/ubuntu-ranking-dataset-creator (Apache-2.0)

Repository code is MIT licensed. That license does not cover the underlying
corpus. The raw corpus is not intended for redistribution from this repository.
