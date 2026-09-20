# Databricks integration layer

For an exhaustive comparison of local PowerShell CLI arguments and Databricks
Python-task Parameters JSON—including sampling semantics and JSON-array
formatting—see
the designed [Local and Databricks Parameter Guide](../docs/local_and_databricks_parameters.pdf).

This subfolder contains the distributed Databricks implementation of the
Ubuntu pipeline: Spark-native bronze ingestion and feature preparation, two
partition-safe `mapInPandas` NLP passes, validated Delta silver output, and
Spark-native silver-to-gold aggregation. Free Edition/serverless verification
is complete; larger classic/GPU validation and an optional Asset Bundle remain
deployment work.

The intended architecture is:

- Bronze: raw message ingestion and storage
- Silver: cleaned text, normalized tokens, residual classification, and review tracking
- Gold: aggregated analytics and reporting

## Folder layout

- `scripts/` — executable Databricks job code and orchestration steps
- `outputs/` — generated intermediate/final tables and export artifacts
- `notebooks/` — optional notebook-driven exploration and validation

## Distributed workflow

1. Read one Unity Catalog table, Delta path, or CSV with Spark.
2. Normalize the raw Ubuntu fields and derive stable SHA-256 conversation and
   message identifiers plus temporal, user, release-cycle, and conversation
   features.
3. Run deterministic normalization, anonymization, lexicon matching,
   glued-term matching, and residual extraction in the first `mapInPandas`
   pass. A pinned English dictionary removes ordinary words before residual
   classification. Each Python partition uses one local worker because Spark
   supplies the outer parallelism.
4. Aggregate the unresolved vocabulary globally with Spark. Leave it untouched
   for `manual_review`, apply the committed overlay for `reviewed`, or classify
   only that bounded vocabulary on the driver for `api`.
5. When NMF is enabled, collect one seeded, bounded fitting sample and fit one
   model on the driver. Broadcast the residual lookup and fitted NMF bundle so
   every partition uses the same labels and topic IDs.
6. Run sentiment, spaCy, global-topic assignment, and optional transformer
   emotion scoring in the second `mapInPandas` pass. Transformer and spaCy
   module caches load a model once per Python worker and reuse it across Arrow
   batches.
7. Validate the recombined Spark DataFrame globally, write one Delta silver
   target, and append a secret-free audit row.
8. Aggregate the validated silver table into the selected gold grain with the
   separate Spark/Delta gold job.

The only intentional driver collections are the bounded NMF fit sample and the
unique residual vocabulary; the latter has a configurable maximum-size guard.
API credentials remain driver-only and are removed from the configuration
serialized into `mapInPandas` task payloads.
The full message table is never converted to pandas. Complete probability and
uncertainty columns are retained rather than reducing transformer output to a
winning label.

Intermediate reuse adapts to the compute API. Classic Spark uses persisted
DataFrames and broadcast variables. Serverless/Spark Connect uses short-lived
managed Delta tables plus serialized immutable artifacts because serverless
does not support DataFrame cache, persist, checkpoint, or `sparkContext`.
`--materialization-mode auto` selects the appropriate route;
`--materialization-schema` controls the scratch-table location. Successful and
failed runs remove their scratch tables, and final-output cleanup is explicit
for notebook callers through `release_silver_resources` and
`release_gold_resources`.

Partition-level deterministic checks run inside the pandas stages. Full-job
validation runs again after Spark recombines the partitions so missing rows,
duplicate identifiers, excessive `NONWORD` replacement, collapsed global
distributions, invalid probabilities, and inadequate NLP coverage cannot hide
behind per-partition success. Both
silver and gold jobs append to a Delta audit table and record
`pipeline.schema.FEATURE_SCHEMA_VERSION`.
`pipeline.schema.FEATURE_SCHEMA_VERSION` is the explicit contract version to
store with every Delta write; schema changes should update that version rather
than silently altering downstream tables.

## API policy

The residual API call should only happen after the lexical and glued-match passes. The model should not be asked to classify the entire corpus; it should only classify the residual set.

If no API key is configured, do not silently run the API and do not convert
the residual vocabulary to `NONWORD`. The adapter defaults to
`manual_review`; API mode without a key fails explicitly. The `reviewed`
fallback converts only terms explicitly present in the committed NONWORD
overlay; unseen terms remain unchanged.

## Verification boundary

The partition transformation has a deterministic small-slice parity test
against the canonical local VADER pipeline. Both Databricks entry points are
imported and syntax-checked in the local test suite. A 1,000-row VADER smoke
test has also been validated on Databricks Free Edition/serverless compute;
larger classic/GPU runtime choices still need to be selected and recorded.
Asset Bundle configuration remains optional deployment work.

## Suggested Delta tables

- `bronze_messages`
- `silver_cleaned_messages`
- `silver_residual_words`
- `silver_manual_review`
- `silver_final_classified_tokens`
- `gold_conversation_summary`
- `gold_user_summary`
- `gold_date_summary`
- `gold_channel_summary`
- `gold_release_summary`
- `gold_language_summary`
- `gold_residual_summary`
- `gold_technical_summary`
- `gold_topic_summary`
- `gold_entity_summary`
- `pipeline_audit`
- `ubuntu_pipeline_run_metrics`

## Operational run metrics

Both compute paths record the same stage contract in a Delta table. Timers
surround Spark actions and materialized writes rather than lazy DataFrame-plan
construction. Bronze-to-silver records `bronze_read`, `core_cleaning`,
`residual_classification`, optional `topic_model_fit`, `enrichment`,
`silver_validation`, and `silver_write`. The single `enrichment` stage reflects
the actual second `mapInPandas` pass, where sentiment and optional advanced NLP
run together. Silver-to-gold records `silver_read`, `gold_aggregation`,
`gold_validation`, and `gold_write`.

Each row includes the run/stage identity, input and output rows, duration,
throughput, per-row latency, sample settings, NLP configuration, compute and
materialization modes, runtime/cluster identifiers when available, and the
feature schema version. Configuration fields are allow-listed; API keys are
never included.

```sql
SELECT
  stage,
  materialization_mode,
  AVG(duration_seconds) AS avg_duration_seconds,
  AVG(rows_per_second) AS avg_rows_per_second
FROM workspace.default.ubuntu_pipeline_run_metrics
GROUP BY stage, materialization_mode
ORDER BY avg_duration_seconds DESC;
```

Use `materialization_mode = 'delta'` for the serverless/Spark Connect route and
`materialization_mode = 'persist'` for the classic-cluster route selected by
`auto`. The same Delta table can therefore drive one comparison dashboard.

## Local silver-to-gold CLI

The message-level silver output can be aggregated without discarding the
message-level table:

```powershell
.\.venv\python.exe -m pipeline.aggregation silver.csv --gold-level conversation
.\.venv\python.exe -m pipeline.aggregation silver.csv --gold-level user
.\.venv\python.exe -m pipeline.aggregation silver.csv --gold-level date --date-granularity month
.\.venv\python.exe -m pipeline.aggregation silver.csv --gold-level release --release-axis since
.\.venv\python.exe -m pipeline.aggregation silver.csv --gold-level language
.\.venv\python.exe -m pipeline.aggregation silver.csv --gold-level technical
.\.venv\python.exe -m pipeline.aggregation silver.csv --gold-level topic
.\.venv\python.exe -m pipeline.aggregation silver.csv --gold-level entity
.\.venv\python.exe -m pipeline.aggregation silver.csv --gold-level sample --sample-size 10000 --random-state 42
```

Every local gold write is validated and appended to the project-local JSONL
audit log.

## Databricks Spark/Delta bronze-to-silver job

The bronze job accepts exactly one source and one destination. A typical
CPU/VADER run is:

```text
python -m databricks_integration.scripts.bronze_to_silver_ubuntu \
  --input-table catalog.schema.bronze_ubuntu_dialogue \
  --output-table catalog.schema.silver_ubuntu_dialogue \
  --audit-table catalog.schema.pipeline_audit \
  --run-metrics-table catalog.schema.ubuntu_pipeline_run_metrics \
  --write-mode overwrite \
  --repartition-count 32 \
  --residual-policy reviewed \
  --sentiment vader \
  --advanced-nlp --topics nmf
```

Sources may instead use `--input-delta-path` or `--input-csv-path`; destinations
may use `--output-delta-path`. Local-style bounded execution is available with
`--sample-size` and `--sample-seed`. Transformer sentiment and emotion retain
explicit model revision, batch size, inference chunk, device, and dtype
arguments. GPU availability never changes dtype automatically.

For API residual classification, retrieve the key from a Databricks secret
scope in a notebook or job wrapper and pass it only to
`run_bronze_to_silver_spark(..., api_key=secret)`. The key is used on the driver
and is neither broadcast nor written to output metadata. The shared adapter
sends only the global residual token/count vocabulary through the Responses
API with the pinned `gpt-4o-mini-2024-07-18` snapshot, strict Structured
Outputs, `store: false`, bounded retries, and complete-response validation.
Returned labels are recorded as review candidates with non-secret provenance;
API mode without a key or with partial/invalid output fails closed.

Use a Databricks Runtime or ML Runtime whose built-in Python, Arrow, Spark, and
PyTorch versions match the selected workload. Install this repository's
ordinary and optional Python dependencies as cluster libraries without
replacing Spark's own PySpark installation. CPU-only PyTorch wheels from
`requirements-transformer.txt` are for the repository-local Windows
environment; do not install that CPU wheel on a GPU Databricks cluster.

For a serverless Python-script task, add the repository root (the directory
containing `pyproject.toml`) as a Jobs environment dependency, for example:

```text
/Workspace/Users/<user>/Ubuntu-dialogue-corpus
```

This installs the canonical `pipeline`, `lexicons_and_templates`,
`reviewed_overlays`, and `databricks_integration` packages on both the driver
and the isolated `mapInPandas` workers. Pointing the task only at the entry
script is insufficient because Workspace source files are not automatically
added to each Python worker's import path. The project package includes the
two CSV assets required at runtime; raw corpus files and generated outputs are
not packaged. Increment the project version in `pyproject.toml` whenever this
worker package changes so a serverless Jobs environment does not reuse a
cached build.

## Databricks Spark/Delta silver-to-gold job

Run the Databricks entry point as a job against either a Unity Catalog table or
a Delta path. For example:

```text
python -m databricks_integration.scripts.silver_to_gold_ubuntu \
  --input-table catalog.schema.silver_ubuntu_dialogue \
  --output-table catalog.schema.gold_conversation_summary \
  --audit-table catalog.schema.pipeline_audit \
  --run-metrics-table catalog.schema.ubuntu_pipeline_run_metrics \
  --gold-level conversation
```

The Databricks job never converts the full table to pandas. Group keys are
aggregated globally with Spark, validated before the write, stored as Delta,
and recorded in the audit table. Residual and technical aggregations require
typed array/struct columns in the silver Delta schema. Named entities may be a
typed struct array or the pipeline's JSON representation. Malformed values
fail closed rather than disappearing silently.

The sample route requires a non-null, unique `message_id` and ranks rows by a
seeded stable hash. This makes the selected message IDs reproducible even when
Spark repartitions the source table; it does not depend on partition-local
random-number order.

Conversation and directional user aggregates include full sentiment
probabilities, expected transformer sentiment, label shares, uncertainty,
topic/emotion confidence, and message features when those silver columns
exist. User-sent and user-received metrics remain separate. Gold aggregation
must be global Spark `groupBy` work—not independently fitted or finalized per
partition.

## Notes

This integration remains inside the same repository because it imports the
canonical `pipeline/`, `lexicons_and_templates/`, and `reviewed_overlays/`
modules. It does not duplicate or fork those feature definitions.
