# Local and Databricks Parameters

## The pipeline has two executable stages

“The pipeline” refers to the complete Bronze/Silver/Gold workflow, but local
and Databricks execution are both divided into separate Silver and Gold entry
points:

| Environment | Silver stage | Gold stage |
|---|---|---|
| Local | `python -m pipeline.pipeline` reads a raw CSV and writes a Silver CSV. | `python -m pipeline.aggregation` reads that Silver CSV and writes one selected Gold CSV. |
| Databricks | `bronze_to_silver_ubuntu.py` reads Bronze and writes a Delta Silver table. | `silver_to_gold_ubuntu.py` reads Silver and writes one selected Delta Gold table. |

The local workflow is therefore not one command that automatically creates
both layers. Run the Silver command first, then run the Gold command once for
each Gold output you want. A Databricks job follows the same sequence with two
dependent Python-script tasks.

## Where parameters are entered

### Local PowerShell CLI

Open PowerShell in the repository root. Parameters follow the module name as
ordinary command-line arguments. A backtick continues the same command.

```powershell
# Raw CSV -> Silver CSV
& .\.venv\python.exe -m pipeline.pipeline `
    ".\data\dialogueText_196.csv" `
    --sample-size 10000 `
    --sample-seed 42 `
    --residual-policy reviewed `
    --sentiment vader `
    --output-dir ".\outputs\silver_10k"

# Silver CSV -> conversation Gold CSV
& .\.venv\python.exe -m pipeline.aggregation `
    ".\outputs\silver_10k\silver_ubuntu_dialogue.csv" `
    --gold-level conversation `
    --output-dir ".\outputs\gold_10k"
```

If `--residual-policy api` is selected, the pipeline does **not** open its own
key prompt. Set the key securely in the same PowerShell session immediately
before the Silver command, then remove it when the run ends:

```powershell
$openAiSecureKey = Read-Host "Paste your OpenAI API key" -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new(
    "",
    $openAiSecureKey
).Password
Remove-Variable openAiSecureKey

try {
    # Run the Silver command here with --residual-policy api.
}
finally {
    Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
}
```

The pasted key is hidden, is available only to that PowerShell process and its
child pipeline process, and is not placed in the CLI command or audit output.

### Databricks Jobs Parameters JSON

1. Upload the source data to a Unity Catalog volume and create Bronze, or use a
   supported CSV/Delta path.
2. Add this repository as a Databricks Git folder under **Workspace**.
3. Under **Jobs & Pipelines**, create a job with a **Python script** task.
4. Select the Bronze-to-Silver script from the Git folder.
5. Paste a JSON array of strings into that task's **Parameters** textbox.
6. Add a dependent second task for the Silver-to-Gold script and give it its
   own Parameters JSON.

Every flag and value is a separate quoted item. Numbers are quoted because the
textbox requires strings. Standalone flags have no value after them. The basic
shape is `["--flag", "value", "--standalone-flag"]`.

If `"--residual-policy", "api"` is selected, Databricks does **not** prompt for
the key during the job. Do not put it in the Parameters JSON. Store it in a
Databricks secret scope and expose it to the Python task as the environment
variable `OPENAI_API_KEY`. On classic job compute, configure the compute's
secret-backed environment variable as:

```text
OPENAI_API_KEY={{secrets/<scope-name>/<secret-name>}}
```

On serverless compute, use the job-level environment-variable configuration
(currently Beta) to expose the same variable name from a secret. If that UI is
not enabled for the workspace, use a small secret-reading job wrapper that
passes the retrieved secret to `run_bronze_to_silver_spark(..., api_key=...)`.
The key is not a Silver-to-Gold parameter and is never written to pipeline
output or audit metadata.

Databricks requires this JSON-array format for Python-script tasks. Notebook
tasks use a different key-value interface. See the official
[task-parameter documentation](https://docs.databricks.com/aws/en/jobs/task-parameters).

## The two sampling controls

| When it runs | Parameter | Meaning |
|---|---|---|
| Before Silver processing | `--sample-size N --sample-seed S` | Selects `N` individual input messages. Only those messages are cleaned, scored, and written to Silver. |
| When Gold level is `sample` | `--gold-level sample --sample-size N --random-state S` | Creates a Gold output containing `N` individual messages selected from Silver. It does not limit any other Gold output. |

## Complete parameter reference

The workflow above is enough for normal use. Expand the relevant section below
when you need the complete flag-by-flag reference.

<details>
<summary><strong>Silver parameters: local and Databricks</strong></summary>

## Silver parameter reference

`—` means that entry point does not expose the parameter.

### Input, output, execution, and sampling

| Purpose | Local CLI | Databricks Parameters JSON | Default or requirement |
|---|---|---|---|
| Help | `-h` / `--help` | `"--help"` | Optional |
| Input CSV | Positional `input_csv` | `"--input-csv-path", "PATH"` | Local: required. Databricks: choose one input. |
| Input table | — | `"--input-table", "CATALOG.SCHEMA.TABLE"` | Choose one input. |
| Input Delta path | — | `"--input-delta-path", "PATH"` | Choose one input. |
| Processed-text column | `--text-column NAME` | — | `text_cleaned` |
| Source-text column | `--source-text-column NAME` | — | `text` |
| Local output folder | `--output-dir PATH` | — | `outputs` |
| Output table | — | `"--output-table", "CATALOG.SCHEMA.TABLE"` | Choose one output. |
| Output Delta path | — | `"--output-delta-path", "PATH"` | Choose one output. |
| Audit table | — | `"--audit-table", "TABLE"` | Required in Databricks; automatic locally. |
| Metrics table | — | `"--run-metrics-table", "TABLE"` | Optional |
| Write mode | — | `"--write-mode", "errorifexists\|overwrite\|append"` | `errorifexists` |
| Local parallel mode | `--parallel auto\|on\|off` | — | `auto` |
| Local process count | `--workers N` | — | Automatic, conservatively capped |
| Spark partitions | — | `"--repartition-count", "N"` | Optional positive integer |
| Input-message sample | `--sample-size N` | `"--sample-size", "N"` | All messages |
| Input sample seed | `--sample-seed S` | `"--sample-seed", "S"` | `0` |
| Intermediate strategy | — | `"--materialization-mode", "auto\|persist\|delta\|none"` | `auto` |
| Scratch-table schema | — | `"--materialization-schema", "CATALOG.SCHEMA"` | `workspace.default` |

Databricks requires exactly one input (`--input-table`, `--input-delta-path`,
or `--input-csv-path`) and one output (`--output-table` or
`--output-delta-path`).

### Residual classification

| Purpose | Local CLI | Databricks Parameters JSON | Default |
|---|---|---|---|
| Policy | `--residual-policy manual_review\|reviewed\|api` | `"--residual-policy", "VALUE"` | `manual_review` |
| Maximum NONWORD rate | `--maximum-nonword-token-rate X` | `"--maximum-nonword-token-rate", "X"` | `0.25` |
| API model | `--residual-api-model NAME` | `"--residual-api-model", "NAME"` | `gpt-4o-mini-2024-07-18` |
| API batch size | `--residual-api-batch-size N` | `"--residual-api-batch-size", "N"` | `500` unique terms |
| API timeout | `--residual-api-timeout-seconds X` | `"--residual-api-timeout-seconds", "X"` | `120.0` seconds |
| API retries | `--residual-api-max-retries N` | `"--residual-api-max-retries", "N"` | `2` |
| Residual vocabulary ceiling | — | `"--maximum-residual-vocabulary", "N"` | `500000` |
| API key | Environment variable `OPENAI_API_KEY` | Secret-backed `OPENAI_API_KEY` | Never a CLI/JSON parameter |

### Sentiment transformer

| Purpose | Local CLI | Databricks Parameters JSON | Default |
|---|---|---|---|
| Sentiment mode | `--sentiment none\|vader\|transformer\|both` | `"--sentiment", "VALUE"` | `vader` |
| Model | `--transformer-model NAME` | `"--transformer-model", "NAME"` | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Revision | `--transformer-revision REV` | `"--transformer-revision", "REV"` | Pinned commit `3216a57f...57c20c49fb4bf9c7` |
| Batch size | `--transformer-batch-size N` | `"--transformer-batch-size", "N"` | `32` |
| Inference chunk | `--transformer-inference-chunk-size N` | `"--transformer-inference-chunk-size", "N"` | `4096` |
| Device | `--transformer-device N` | `"--transformer-device", "N"` | Automatic; `-1` CPU, `0+` GPU index |
| Dtype | `--transformer-dtype float32\|float16\|bfloat16` | `"--transformer-dtype", "VALUE"` | `float32` |

### Advanced NLP, topics, and emotion

| Purpose | Local CLI | Databricks Parameters JSON | Default |
|---|---|---|---|
| Enable advanced NLP | `--advanced-nlp` | `"--advanced-nlp"` | Off |
| Disable spaCy | — | `"--no-spacy"` | Off; local CLI runs spaCy with advanced NLP. |
| spaCy model | `--spacy-model NAME` | `"--spacy-model", "NAME"` | `en_core_web_sm` |
| spaCy batch size | — | `"--spacy-batch-size", "N"` | `128` |
| Include token details | — | `"--include-token-details"` | Off |
| Topic mode | `--topics none\|nmf` | `"--topics", "VALUE"` | `none` |
| Topic-fit sample | — | `"--topic-fit-sample-size", "N"` | `100000`; affects NMF fitting, not Silver row count. |
| Topic batch size | — | `"--topic-batch-size", "N"` | `50000` |
| Minimum topic coverage | `--topic-minimum-vocabulary-coverage X` | `"--topic-minimum-vocabulary-coverage", "X"` | `0.50` |
| Emotion mode | `--emotion none\|transformer` | `"--emotion", "VALUE"` | `none` |
| Emotion model | `--emotion-model NAME` | `"--emotion-model", "NAME"` | `j-hartmann/emotion-english-distilroberta-base` |
| Emotion revision | `--emotion-revision REV` | `"--emotion-revision", "REV"` | Pinned commit `cea2f78...e00ef93811f6eb` |
| Emotion batch size | `--emotion-batch-size N` | `"--emotion-batch-size", "N"` | `32` |
| Emotion inference chunk | `--emotion-inference-chunk-size N` | `"--emotion-inference-chunk-size", "N"` | `4096` |
| Emotion device | `--emotion-device N` | `"--emotion-device", "N"` | Automatic; `-1` CPU, `0+` GPU index |
| Emotion dtype | `--emotion-dtype float32\|float16\|bfloat16` | `"--emotion-dtype", "VALUE"` | `float32` |

Local `PipelineConfig` also supports `spacy_batch_size`,
`include_token_details`, `topic_fit_sample_size`, `topic_batch_size`, and
`validate_advanced_nlp` programmatically. The local CLI does not expose them.

</details>

<details>
<summary><strong>Gold parameters: local and Databricks</strong></summary>

## Gold parameter reference

| Purpose | Local Gold CLI | Databricks Parameters JSON | Default or requirement |
|---|---|---|---|
| Help | `-h` / `--help` | `"--help"` | Optional |
| Silver CSV | Positional `input_csv` | — | Required locally |
| Silver table | — | `"--input-table", "TABLE"` | Choose one Databricks input. |
| Silver Delta path | — | `"--input-delta-path", "PATH"` | Choose one Databricks input. |
| Local output folder | `--output-dir PATH` | — | `outputs` |
| Gold output table | — | `"--output-table", "TABLE"` | Required |
| Audit table | — | `"--audit-table", "TABLE"` | Databricks default: `<output_table>_audit`; automatic locally. |
| Metrics table | — | `"--run-metrics-table", "TABLE"` | Optional |
| Gold level | `--gold-level LEVEL` | `"--gold-level", "LEVEL"` | Required |
| Date grain | `--date-granularity day\|week\|month` | `"--date-granularity", "VALUE"` | `day`; date Gold only |
| Channel column | `--channel-column NAME` | `"--channel-column", "NAME"` | `channel`; channel Gold only |
| Release axis | `--release-axis since\|until` | `"--release-axis", "VALUE"` | `since`; release Gold only |
| Sample-Gold size | `--sample-size N` | `"--sample-size", "N"` | Required only when Gold level is `sample`. |
| Sample-Gold seed | `--random-state S` | `"--random-state", "S"` | `0` |
| Skip local feature engineering | `--no-feature-engineering` | — | Feature engineering runs locally by default. |
| Intermediate strategy | — | `"--materialization-mode", "auto\|persist\|delta\|none"` | `auto` |
| Scratch-table schema | — | `"--materialization-schema", "CATALOG.SCHEMA"` | `workspace.default` |
| Write mode | — | — | Both replace their named Gold output; no user-facing flag. |

Gold-level choices are:

| Level | One output row per… |
|---|---|
| `conversation` | distinct conversation represented in Silver |
| `user` | user represented in Silver |
| `date` | selected day, week, or month |
| `channel` | value in the selected channel column |
| `release` | Ubuntu release-cycle bucket |
| `language` | language/script label |
| `residual` | residual term/classification |
| `technical` | technical term/category |
| `topic` | fitted topic |
| `entity` | named entity |
| `sample` | selected Silver message |

</details>

## Formatting rules to remember

- Local: enter one PowerShell command; do not add JSON brackets or quotes around
  every argument.
- Databricks: enter a JSON array; every flag and value is a separate string.
- Databricks numbers are quoted strings: `"--sample-size", "10000"`.
- Databricks standalone flags are single items: `"--advanced-nlp"`.
- `--sample-seed` controls pre-Silver sampling; `--random-state` controls only
  `--gold-level sample`.
- Never place an OpenAI API key in a command, Parameters JSON, Git, or audit
  output.

## Verify the installation

From the repository root, run the complete tracked test suite with:

```powershell
.\.venv\python.exe -m unittest discover -s tests -v
```

This validates the core pipeline, residual/API handling, local-Databricks
parity, packaging, notebook structure, aggregation, run metrics, and dashboard.
