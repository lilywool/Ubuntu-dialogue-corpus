These notebooks are legacy: they were exploratory work done pre-pipeline, before the processing steps were consolidated into standalone .py modules and chained into a single run_pipeline.py script. They're kept here for reference and reproducibility of that exploratory process, not as the current way to run the corpus processing.

## Where the corpus data lives

The large corpus artifacts are **not** in this repository and are not
tracked by git. As of 2026-09-11 they live in a sibling folder:

    2. Ubuntu Dialogue Corpus/
        _corpus_data/                 <- dialogueText_196.csv (951MB)
        |                                df_with_sentiment.pkl (2.9GB)
        1. Ubuntu-dialogue-corpus/    <- this repo

They were moved out of the working tree deliberately. Multi-GB untracked
blobs inside a checkout slow `git status`, risk an accidental forced add,
make the repo awkward to clone or back up, and put bulk data within
relative-path reach of any environment created inside the repo.

Paths in the legacy notebooks were updated to `../_corpus_data/...`, which
assumes the working directory is this repo root.
