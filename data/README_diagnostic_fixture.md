# diagnostic_fixture.csv -- synthetic, hand-built

25 hand-picked rows for the preprocessing diagnostic. Columns:

- `text` -- raw-style input
- `text_cleaned` -- **SYNTHETIC**. A hand-written approximation of what the
  upstream pipeline might emit, constructed specifically to exercise
  `strip_label_tokens_series`: it contains `NONWORD`, `SPANISH`, `FRENCH`
  and `CYRILLIC` placeholders in varying densities (none / partial / total).
- `note` -- what each row is meant to expose

**This is not real pipeline output.** No row here came from the corpus. The
`text_cleaned` values are invented to test how the stripping path behaves
across placeholder densities, and they may not match the real cleaner's
tokenization, casing, or anonymization conventions. Anything this fixture
shows about placeholder stripping is indicative, not evidence about the
actual corpus -- that requires the real `text_cleaned` column, i.e. the
step-2 extraction.

Placeholder densities covered:
- none (most rows)
- partial, content survives stripping (diag_005, diag_015, diag_018,
  diag_023, diag_024)
- total, nothing survives stripping (diag_002, diag_003, diag_014,
  diag_016, diag_022)
- no alphabetic content at all, so never sent to the model (diag_001,
  diag_025)
