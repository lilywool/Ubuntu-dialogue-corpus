# Databricks integration layer

This subfolder contains the staged Databricks adapter for bronze-to-silver work
and a Spark-native, validated silver-to-gold Delta job. It is not yet a complete
production deployment because distributed ingestion/NLP orchestration and Asset
Bundle configuration remain outstanding.

The intended architecture is:

- Bronze: raw message ingestion and storage
- Silver: cleaned text, normalized tokens, residual classification, and review tracking
- Gold: aggregated analytics and reporting

## Folder layout

- `scripts/` — executable Databricks job code and orchestration steps
- `lexicons/` — curated lexicon inputs and review references
- `outputs/` — generated intermediate/final tables and export artifacts
- `notebooks/` — optional notebook-driven exploration and validation

## Intended workflow

1. Load raw text into a bronze table.
2. Apply the lexicon pass and normalization rules.
3. Build the unresolved residual vocabulary.
4. Separate glued matches from remaining residuals.
5. If an API key is configured, call the residual classifier only on the unresolved set.
6. Otherwise, leave residuals unchanged for manual review. Use the explicit
   `reviewed` policy when the committed deterministic fallback is desired.
7. Write the resolved silver table and then aggregate into gold-level metrics.

Advanced NLP functions are importable from `pipeline/` for a future
`mapInPandas` stage. spaCy should run with one local worker inside each Spark
partition, because Spark supplies the outer parallelism. Topic modeling must
be fitted once on a representative sample with
`pipeline.topic_modeling.fit_topic_model`; broadcast and reuse that model for
every partition so topic IDs have one corpus-wide meaning. Never fit NMF
inside individual partitions. Transformer sentiment and emotion inference
likewise remain batched within one process per GPU/partition, use bounded
inference chunks, and require explicit model revision, device, and dtype.
Partition results must preserve the complete probability/uncertainty columns;
do not reduce them to only the winning label before the Delta write.

The local validators are partition-callable, but full-job validation must run
again after unioning partitions so collapsed global distributions, probability
inconsistencies, and missing partitions cannot hide behind per-partition
success. The local JSON-lines audit schema is mirrored by the implemented
append-only Delta audit table for the Spark gold job.
`pipeline.schema.FEATURE_SCHEMA_VERSION` is the explicit contract version to
store with every Delta write; schema changes should update that version rather
than silently altering downstream tables.

## API policy

The residual API call should only happen after the lexical and glued-match passes. The model should not be asked to classify the entire corpus; it should only classify the residual set.

If no API key is configured, do not silently run the API and do not convert
the residual vocabulary to `NONWORD`. The adapter defaults to
`manual_review`; API mode without a key fails explicitly.

## Not implemented yet

- Spark-native bronze ingestion and a partition-safe `mapInPandas` NLP stage
- Asset Bundle job/cluster configuration
- small-slice parity tests against the local pipeline

Spark/Delta silver-to-gold aggregation, global validation, Delta writes, and
append-only Delta audit records are implemented in
`scripts/silver_to_gold_ubuntu.py`.

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

## Databricks Spark/Delta gold job

Run the Databricks entry point as a job against either a Unity Catalog table or
a Delta path. For example:

```text
python -m databricks_integration.scripts.silver_to_gold_ubuntu \
  --input-table catalog.schema.silver_ubuntu_dialogue \
  --output-table catalog.schema.gold_conversation_summary \
  --audit-table catalog.schema.pipeline_audit \
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

This is intentionally isolated from the main repo root to avoid breaking module imports while still creating a clean Databricks-ready structure.
