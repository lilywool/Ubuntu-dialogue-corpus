# CPU preprocessing diagnostic -- run 2026-09-04

Result files: `data/diagnostic_results_per_row.csv`,
`data/diagnostic_results_summary.csv`.

## Runtime (float32 arm, as intended)

    torch            2.5.1+cpu
    transformers     4.46.3
    pandas           2.2.3
    cuda_available   False
    resolved device  -1
    resolved dtype   float32

## Results

| Arm                | Sent to model | Effectively empty | Unique scores |
|--------------------|--------------:|------------------:|--------------:|
| raw, unstripped    |            22 |                 3 |            22 |
| cleaned, unstripped|            23 |                 2 |            23 |
| cleaned, stripped  |            18 |                 7 |            18 |

Every input that reached the model produced a **distinct** confidence
(unique scores == rows sent, in all three arms), with all three labels
represented. Stripping excluded five further inputs as effectively empty
but produced no confidence collapse among the rows that were scored.

Batch sizes 1 vs 32:
- zero label changes
- max |confidence delta| ~5.36e-7
- mean |delta| ~1.5e-7 to 1.8e-7

## What this establishes, and what it does not

**Ruled out, on this fixture:** placeholder stripping as a collapse
mechanism, and batch size as a collapse mechanism. Neither reduced score
diversity at all.

**Not established:**
- anything about fp16. `get_transformer_pipeline` sets
  `torch_dtype = float16 if device != -1 else float32`, so a CPU run is
  structurally incapable of exercising the fp16 path. fp16 remains the
  standing hypothesis and stays blocked on CUDA-capable execution.
- anything about the real corpus. The fixture's `text_cleaned` column is
  synthetic (see `data/README_diagnostic_fixture.md`). Real placeholder
  densities, tokenization and anonymization conventions may differ.

The negative result is still informative: it means the two mechanisms that
*could* have been checked cheaply, on CPU, today, are not the explanation.
That narrows the field rather than closing it.

## Correction adopted from this run

The script's original batch-parity check used **exact float equality**,
which is the wrong test -- batching changes accumulation order, so
bit-identical results are not expected even on a healthy run, and the
observed 5.36e-7 would have been reported as a failure. Replaced with:
scores compared within `SCORE_TOLERANCE = 1e-6`, labels compared exactly
(a label flip is behavioral, not numerical). This is the standard to carry
into any future batch-parity validation, including on Databricks.
