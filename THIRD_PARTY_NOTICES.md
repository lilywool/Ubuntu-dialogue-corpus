# Third-party notices and data boundaries

This file records the material that is used by, or represented in, this
repository but is not covered by the repository's MIT code license.

## Ubuntu Dialogue Corpus

The pipeline was developed against version 1 of the Ubuntu Dialogue Corpus,
described by Lowe, Pow, Serban, and Pineau in the SIGDIAL 2015 paper:
<https://arxiv.org/abs/1506.08909>. The dataset generation scripts are at
<https://github.com/rkadlec/ubuntu-ranking-dataset-creator>.

The original corpus is intentionally not distributed in this repository. Its
redistribution terms are not established here, so users must obtain it from an
authorized source and satisfy the source's terms. The same caution applies to
raw-message excerpts embedded in historical notebook outputs.

## Derived files

The CSV files in this repository contain working artifacts such as slang
entries, corpus-derived frequencies, token classifications, and reviewed
matches. They should not be treated as a license to redistribute the original
messages. Some entries may also reflect third-party lexicons or model-assisted
review. Their provenance is documented, but no blanket third-party license is
asserted for them until each source has been verified.

## Software and models

- The code authored for this repository is released under the MIT License in
  `LICENSE`.
- VADER is loaded from the `vaderSentiment` package; consult that package's
  license when redistributing it.
- The optional transformer path uses Hugging Face `transformers`, PyTorch, and
  `cardiffnlp/twitter-roberta-base-sentiment-latest`. Their code, weights, and
  licenses remain governed by their respective upstream terms.
- Optional emotion classification uses
  `j-hartmann/emotion-english-distilroberta-base`; its weights, model card, and
  license remain governed by the upstream repository.
- Advanced linguistic/topic features use spaCy, `en_core_web_sm`, and
  scikit-learn. Their software and model licenses remain upstream.
- The exploratory notebooks may import additional packages. Those packages
  are dependencies of the notebooks, not components relicensed by this
  repository.
