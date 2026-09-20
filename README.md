# Ubuntu Dialogue Intelligence

I began this project as a notebook-based investigation of roughly 8.6 million
Ubuntu technical-support messages. Because the corpus mixes conversation with
commands, URLs, stack traces, slang, misspellings, glued technical terms, and
multiple languages, the work grew beyond generic text cleaning.

The notebooks established the analysis through Step 8.5. I then converted that
work into a reusable local and Databricks pipeline with optional parallelism,
sampling, transformers, human-reviewed residual labels, and AI-assisted
classification. After the architecture was complete, I finished the notebooks
through Step 12 to demonstrate the full data-science process without hiding it
behind one pipeline call.

```text
Kaggle CSV
  -> bronze messages
  -> silver text, sentiment, and NLP features
  -> gold analytical summaries
  -> notebook analysis + Streamlit dashboard
```

The local pipeline and Spark/Delta bronze-to-silver-to-gold implementation are
built. A live Databricks workspace run, deployment configuration, and the
PowerPoint presentation remain forthcoming.

## Quick start

Download the [Ubuntu Dialogue Corpus from Kaggle](https://www.kaggle.com/datasets/rtatman/ubuntu-dialogue-corpus)
and place `dialogueText_196.csv` at:

```text
data/dialogueText_196.csv
```

Create the repository-local Python 3.11.16 environment:

```powershell
conda create --prefix ".\.venv" python=3.11.16 -y
.\.venv\python.exe -m pip install --upgrade pip
.\.venv\python.exe -m pip install -r requirements.lock.txt
```

Optional dependencies are separated into
`requirements-transformer.txt`, `requirements-advanced-nlp.txt`,
`requirements-dashboard.txt`, and `notebooks/requirements.txt`. Install
them only for the features you intend to run. Nothing requires modifying
system Python, Conda base, Java, CUDA, drivers, global `PATH`, or another
repository's environment.

Run a CPU-friendly VADER pipeline:

```powershell
.\.venv\python.exe -m pipeline.pipeline data\dialogueText_196.csv `
  --parallel off --sentiment vader --residual-policy manual_review
```

Or run expensive features on a reproducible sample:

```powershell
.\.venv\python.exe -m pipeline.pipeline data\dialogueText_196.csv `
  --sample-size 10000 --sample-seed 42 --parallel auto `
  --sentiment both --residual-policy reviewed `
  --advanced-nlp --topics nmf --emotion transformer
```

| Choice | Supported options |
|---|---|
| Local execution | Automatic, forced, or disabled multiprocessing; optional worker count |
| Input | Full corpus or seeded sample |
| Residual handling | Manual review, committed reviewed overlay, or opt-in API |
| Sentiment | None, VADER, RoBERTa, or both |
| Advanced NLP | spaCy features, global NMF topics, transformer emotion |
| Precision | Explicit float32, float16, or bfloat16; GPU detection never silently selects fp16 |

Generated silver data, review files, model artifacts, dashboard data, and
append-only audit logs stay under the ignored `outputs/` directory.

## Pipeline design

The deterministic layer combines:

- 873 Ubuntu/Linux and computing terms across 15 categories;
- 351 reviewed slang, emoticon, IRC, and symbolic entries;
- structural anonymization for contact/network identifiers, URLs, paths,
  shortcuts, dumps, quantities, and related patterns;
- more than 1,300 reviewed chat, typo, and shorthand normalizations;
- glued-term detection; and
- thousands of human-reviewed jargon and language decisions.

Residual behavior is explicit. A pinned English dictionary first removes
ordinary words from the unresolved vocabulary. `manual_review` leaves the
remaining terms unchanged, `reviewed` applies only explicit committed human
classifications, and `api` sends only that bounded residual vocabulary—not
full messages—to an OpenAI-compatible endpoint. Unreviewed terms are never
silently converted to `NONWORD`; reviewed/API runs also fail validation if the
resulting `NONWORD` token rate exceeds the configured ceiling. API mode
requires a key and never stores it in pipeline output.

Sentiment retains the complete VADER distribution and, when selected, all
three RoBERTa probabilities plus confidence, expected sentiment, entropy, and
margin. Optional NLP adds compact spaCy POS/entity summaries, one globally
fitted NMF topic model, and seven-class transformer emotion probabilities.
Technical terms remain governed by the reviewed Ubuntu lexicon rather than
being counted again through spaCy NER.

Model revisions, device, dtype, batching, and inference chunks are explicit.
Validation checks row conservation, feature consistency, NLP coverage,
probability distributions, topic diversity, and collapsed model output before
writes are allowed.

The architecture was modeled on my
[Yelp Review Intelligence](https://github.com/lilywool/yelp-review-intelligence)
project and extended for neutral-aware sentiment, Ubuntu technical language,
global topic identity, bounded inference, richer uncertainty features,
rerun-safe outputs, and stricter validation.

## Notebooks and dashboard

- `Ubuntu_Project_Steps1-6.ipynb` covers ingestion, memory, structure, and
  conversation, user, and temporal features.
- `Ubuntu_step7-12.ipynb` continues through cleaning, sentiment, advanced NLP,
  statistics, visualization, machine learning, and evaluation.
- Step 12 reports held-out performance, baseline lift, business impact,
  limitations, next steps, and reproducibility documentation.

The completed notebook artifact is `outputs/df_with_sentiment.pkl`. The
notebooks expose each analytical step and do not call the top-level pipeline.

The Streamlit dashboard provides descriptive, diagnostic, predictive, and
prescriptive views over privacy-bounded outputs. It excludes raw messages and
user identities.

```powershell
.\.venv\python.exe -m pip install -r requirements-dashboard.txt
.\.venv\python.exe -m streamlit run dashboard\app.py
```

## Databricks

The bronze-to-silver job reads a Unity Catalog table, Delta path, or CSV with
Spark. Its first `mapInPandas` pass performs deterministic text processing.
Spark then aggregates the residual vocabulary and fits one bounded global NMF
model when requested. A second pass applies broadcast labels/topics and runs
sentiment and optional NLP. Models are cached per Python worker.

The same job adapts to its Databricks compute. In `auto` mode, classic Spark
reuses intermediates with persisted DataFrames and broadcast variables;
serverless/Spark Connect instead uses short-lived managed Delta tables and
serialized immutable artifacts. Databricks still selects the CPU/GPU runtime,
worker shape, dependencies, model device, and dtype through the job setup. A
future Declarative Automation Bundle can record separate serverless and classic
deployment targets without duplicating the processing code.

Whole-job validation—including the residual `NONWORD` rate—precedes the Delta
silver write and audit record. A shared run-metrics Delta contract records
materialized stage latency and throughput on both serverless and classic
compute. The silver-to-gold job then produces any of these grains:

`conversation`, `user`, `date`, `channel`, `release`, `language`,
`residual`, `technical`, `topic`, `entity`, or seeded `sample`.

Optional residual API classification is shared by local and Databricks runs.
It sends only unique residual tokens and counts—not dialogue rows—to OpenAI's
Responses API, using the pinned `gpt-4o-mini-2024-07-18` snapshot, a strict JSON
schema, `store: false`, bounded retries, and fail-closed response validation.
Labels are identified as API-generated review candidates; non-secret request
hashes, response IDs, model, and schema version are retained for provenance.
Local runs read `OPENAI_API_KEY` from the environment. Databricks should inject
the same variable from a secret scope; the key is never sent to Spark workers
or written to audit metadata.

```text
python -m databricks_integration.scripts.bronze_to_silver_ubuntu \
  --input-table catalog.schema.bronze_ubuntu_dialogue \
  --output-table catalog.schema.silver_ubuntu_dialogue \
  --audit-table catalog.schema.pipeline_audit \
  --run-metrics-table catalog.schema.ubuntu_pipeline_run_metrics \
  --residual-policy reviewed --sentiment vader
```

Partition transformations have deterministic local parity coverage. Live
Bronze ingestion and a reviewed/VADER Silver smoke test have been verified in
Databricks Free Edition; the complete Silver/Gold production run remains to be
validated on its selected runtime and job configuration. See
[databricks_integration/README.md](databricks_integration/README.md) for the
full execution contract.

## Repository map

- `pipeline/` — reusable processing, NLP, validation, aggregation, and audit
- `lexicons_and_templates/` — reviewed terms and structural patterns
- `reviewed_overlays/` — human normalization and residual decisions
- `notebooks/` — complete twelve-step analysis
- `databricks_integration/` — distributed Spark/Delta jobs
- `dashboard/` — Streamlit application
- `tests/` — regression, parity, notebook, aggregation, and dashboard tests

Run all tests with:

```powershell
.\.venv\python.exe -m unittest discover -s tests -v
```

## Next steps

- validate a small real-data slice on the selected Databricks runtime;
- add an Asset Bundle after the cloud and cluster shape are known;
- run the controlled GPU dtype and full-corpus transformer jobs; and
- publish the PowerPoint presentation.

## Attribution and license

Built on Ryan Lowe, Nissan Pow, Iulian Serban, and Joelle Pineau's
*The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured
Multi-Turn Dialogue Systems* ([paper](https://arxiv.org/abs/1506.08909);
[dataset-generation code](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)).

Repository code is MIT licensed. The underlying corpus is not covered by that
license and is not redistributed here.
