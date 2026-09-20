"""Canonical orchestration for the deterministic Ubuntu dialogue pipeline."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class PipelineConfig:
    residual_policy: str = "manual_review"
    maximum_nonword_token_rate: float = 0.25
    api_key: Optional[str] = None
    sentiment_mode: str = "vader"
    transformer_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    transformer_revision: str = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"
    transformer_batch_size: int = 32
    transformer_inference_chunk_size: int = 4096
    transformer_device: Optional[int] = None
    transformer_dtype: str = "float32"
    advanced_nlp: bool = False
    run_spacy: bool = True
    topic_mode: str = "none"
    emotion_mode: str = "none"
    spacy_model: str = "en_core_web_sm"
    spacy_batch_size: int = 128
    include_token_details: bool = False
    topic_fit_sample_size: int = 100_000
    topic_batch_size: int = 50_000
    topic_minimum_vocabulary_coverage: float = 0.50
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    emotion_revision: str = "cea2f78f197f0337186a0faa93e00ef93811f6eb"
    emotion_batch_size: int = 32
    emotion_inference_chunk_size: int = 4096
    emotion_device: Optional[int] = None
    emotion_dtype: str = "float32"
    validate_advanced_nlp: bool = True
    parallel: bool = True
    workers: Optional[int] = None
    sample_size: Optional[int] = None
    sample_seed: int = 0
    output_dir: str = "outputs"


def _is_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ or "DBFS" in os.environ


def _resolve_api_key(api_key: Optional[str] = None) -> Optional[str]:
    if api_key:
        return api_key
    return os.getenv("OPENAI_API_KEY")


def run_pipeline(
    df=None,
    config: Optional[PipelineConfig] = None,
    text_col: str = "text_cleaned",
    source_text_col: str = "text",
    **overrides,
):
    """Run the active stages in order and return the processed DataFrame.

    Known lexicon and structural matches are handled deterministically. The
    Residual behavior is explicit: the default ``manual_review`` policy leaves
    unresolved tokens untouched, ``reviewed`` applies committed review tables,
    and ``api`` requires a configured API key.
    """
    if df is None:
        raise ValueError("run_pipeline requires a DataFrame containing the cleaned text")

    cfg = PipelineConfig()
    if config is not None:
        cfg = config

    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise TypeError(f"unknown pipeline option: {key}")
        setattr(cfg, key, value)

    source_row_count = len(df)
    sample_provenance: dict[str, Any] = {
        "requested_size": cfg.sample_size,
        "seed": cfg.sample_seed,
        "source_rows": source_row_count,
        "selected_rows": source_row_count,
        "sampled": False,
    }
    if cfg.sample_size is not None:
        if cfg.sample_size < 1:
            raise ValueError("sample_size must be at least 1")
        if cfg.sample_size < source_row_count:
            rng = np.random.default_rng(cfg.sample_seed)
            selected_positions = np.sort(
                rng.choice(source_row_count, size=cfg.sample_size, replace=False)
            )
            df = df.iloc[selected_positions].copy()
            sample_provenance["sampled"] = True
        else:
            df = df.copy()
        sample_provenance["selected_rows"] = len(df)

    expected_rows = len(df)
    expected_index = df.index.copy()

    from pipeline.apply_lexicons import apply_lexicons
    from pipeline.data_preparation import optimize_dtypes, prepare_text_column
    from pipeline.parallel_execution import recommended_workers
    from pipeline.residual_stage import classify_residuals, extract_residual_vocabulary

    workers = cfg.workers if cfg.parallel and cfg.workers else (
        recommended_workers(len(df)) if cfg.parallel else 1
    )
    df = optimize_dtypes(df)
    df = prepare_text_column(df, text_col=text_col, source_text_col=source_text_col)
    df, _matched, tally = apply_lexicons(df, text_col=text_col, workers=workers)
    df, residual_counts = extract_residual_vocabulary(
        df,
        text_col=text_col,
        workers=workers,
    )

    api_key = _resolve_api_key(cfg.api_key) if cfg.residual_policy == "api" else None
    df, residual_labels, residual_sources = classify_residuals(
        df,
        residual_counts,
        policy=cfg.residual_policy,
        api_key=api_key,
        text_col=text_col,
    )

    residual_output = Path(cfg.output_dir) / "residual_words_for_review.csv"
    residual_output.parent.mkdir(parents=True, exist_ok=True)
    residual_frame = pd.DataFrame({
        "word": list(residual_counts),
        "count": [residual_counts[word] for word in residual_counts],
        "classification_label": [residual_labels.get(word, "") for word in residual_counts],
        "classification_source": [residual_sources.get(word, "manual_review") for word in residual_counts],
    })
    residual_frame.to_csv(residual_output, index=False)

    if cfg.sentiment_mode != "none":
        from pipeline.sentiment_analysis import analyze_sentiment
        df = analyze_sentiment(
            df,
            text_column=text_col,
            mode=cfg.sentiment_mode,
            transformer_model=cfg.transformer_model,
            transformer_revision=cfg.transformer_revision,
            transformer_batch_size=cfg.transformer_batch_size,
            transformer_inference_chunk_size=cfg.transformer_inference_chunk_size,
            transformer_device=cfg.transformer_device,
            transformer_dtype=cfg.transformer_dtype,
        )

    if cfg.advanced_nlp:
        from pipeline.advanced_nlp import annotate_messages, validate_advanced_nlp
        df = annotate_messages(
            df,
            text_col=text_col,
            run_spacy=cfg.run_spacy,
            spacy_model=cfg.spacy_model,
            spacy_batch_size=cfg.spacy_batch_size,
            workers=workers,
            include_token_details=cfg.include_token_details,
            topic_mode=cfg.topic_mode,
            topic_fit_sample_size=cfg.topic_fit_sample_size,
            topic_batch_size=cfg.topic_batch_size,
            emotion_mode=cfg.emotion_mode,
            emotion_model=cfg.emotion_model,
            emotion_revision=cfg.emotion_revision,
            emotion_batch_size=cfg.emotion_batch_size,
            emotion_inference_chunk_size=cfg.emotion_inference_chunk_size,
            emotion_device=cfg.emotion_device,
            emotion_dtype=cfg.emotion_dtype,
        )
        if cfg.validate_advanced_nlp:
            df.attrs["advanced_nlp_validation"] = validate_advanced_nlp(
                df,
                text_col=text_col,
                sentiment_mode=cfg.sentiment_mode,
                run_spacy=cfg.run_spacy,
                topic_mode=cfg.topic_mode,
                emotion_mode=cfg.emotion_mode,
                require_technical_lexicon=True,
                minimum_topic_vocabulary_coverage=cfg.topic_minimum_vocabulary_coverage,
            )

    from pipeline.validation import validate_core_pipeline
    core_validation = validate_core_pipeline(
        df,
        text_col=text_col,
        expected_rows=expected_rows,
        expected_index=expected_index,
        maximum_nonword_token_rate=(
            cfg.maximum_nonword_token_rate
            if cfg.residual_policy in {"reviewed", "api"}
            else None
        ),
    )
    validation_stages: dict[str, Any] = {"core": core_validation}
    if "sentiment_validation" in df.attrs:
        validation_stages["sentiment"] = df.attrs["sentiment_validation"]
    if "advanced_nlp_validation" in df.attrs:
        validation_stages["advanced_nlp"] = df.attrs["advanced_nlp_validation"]
    df.attrs["pipeline_validation"] = {
        "passed": all(stage["passed"] for stage in validation_stages.values()),
        "rows": len(df),
        "stages": validation_stages,
    }

    df.attrs["pipeline_tally"] = tally
    df.attrs["residual_counts"] = residual_counts
    df.attrs["residual_labels"] = residual_labels
    df.attrs["residual_sources"] = residual_sources
    df.attrs["residual_output"] = str(residual_output)
    df.attrs["parallel_workers"] = workers
    df.attrs["sample_provenance"] = sample_provenance
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Ubuntu dialogue pipeline.")
    parser.add_argument("input_csv", help="CSV containing text_cleaned or raw text")
    parser.add_argument("--text-column", default="text_cleaned")
    parser.add_argument("--source-text-column", default="text")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--parallel", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--workers", type=int, default=None, help="Override automatic worker selection")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample this many input messages before cleaning and NLP extraction.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Random seed for the pre-extraction input sample.",
    )
    parser.add_argument(
        "--sentiment",
        choices=("none", "vader", "transformer", "both"),
        default="vader",
        help="Sentiment backend: VADER is cheap; transformer is expensive; both runs both.",
    )
    parser.add_argument(
        "--residual-policy",
        choices=("manual_review", "reviewed", "api"),
        default="manual_review",
    )
    parser.add_argument(
        "--maximum-nonword-token-rate",
        type=float,
        default=0.25,
        help="Fail reviewed/API runs when NONWORD exceeds this token share.",
    )
    parser.add_argument(
        "--transformer-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Explicit transformer dtype; float16 is never selected implicitly.",
    )
    parser.add_argument(
        "--transformer-model",
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )
    parser.add_argument(
        "--transformer-revision",
        default="3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7",
    )
    parser.add_argument("--transformer-batch-size", type=int, default=32)
    parser.add_argument("--transformer-inference-chunk-size", type=int, default=4096)
    parser.add_argument("--transformer-device", type=int, default=None)
    parser.add_argument(
        "--advanced-nlp",
        action="store_true",
        help="Run optional spaCy message-level annotations.",
    )
    parser.add_argument(
        "--topics",
        choices=("none", "nmf"),
        default="none",
        help="Optional topic modeling after spaCy annotations.",
    )
    parser.add_argument(
        "--topic-minimum-vocabulary-coverage",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--emotion",
        choices=("none", "transformer"),
        default="none",
        help="Optional Hugging Face emotion classification.",
    )
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument(
        "--emotion-model",
        default="j-hartmann/emotion-english-distilroberta-base",
    )
    parser.add_argument(
        "--emotion-revision",
        default="cea2f78f197f0337186a0faa93e00ef93811f6eb",
    )
    parser.add_argument("--emotion-batch-size", type=int, default=32)
    parser.add_argument("--emotion-inference-chunk-size", type=int, default=4096)
    parser.add_argument("--emotion-device", type=int, default=None)
    parser.add_argument(
        "--emotion-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Explicit emotion-model dtype; GPU availability never implies fp16.",
    )
    args = parser.parse_args()

    import pandas as pd

    frame = pd.read_csv(args.input_csv)
    config = PipelineConfig(
        residual_policy=args.residual_policy,
        maximum_nonword_token_rate=args.maximum_nonword_token_rate,
        sentiment_mode=args.sentiment,
        transformer_model=args.transformer_model,
        transformer_revision=args.transformer_revision,
        transformer_batch_size=args.transformer_batch_size,
        transformer_inference_chunk_size=args.transformer_inference_chunk_size,
        transformer_device=args.transformer_device,
        transformer_dtype=args.transformer_dtype,
        advanced_nlp=args.advanced_nlp,
        topic_mode=args.topics,
        topic_minimum_vocabulary_coverage=args.topic_minimum_vocabulary_coverage,
        emotion_mode=args.emotion,
        spacy_model=args.spacy_model,
        emotion_model=args.emotion_model,
        emotion_revision=args.emotion_revision,
        emotion_batch_size=args.emotion_batch_size,
        emotion_inference_chunk_size=args.emotion_inference_chunk_size,
        emotion_device=args.emotion_device,
        emotion_dtype=args.emotion_dtype,
        parallel=args.parallel != "off",
        workers=args.workers,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        output_dir=args.output_dir,
    )
    result = run_pipeline(
        frame,
        config=config,
        text_col=args.text_column,
        source_text_col=args.source_text_column,
    )
    output_path = Path(args.output_dir) / "silver_ubuntu_dialogue.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    from pipeline.audit import append_run_audit
    append_run_audit(
        Path(args.output_dir) / "pipeline_run_log.jsonl",
        df=result,
        stage="silver_message_features",
        output_path=output_path,
        validation=result.attrs["pipeline_validation"],
        config=vars(args),
    )
    print(f"wrote {output_path} using {result.attrs['parallel_workers']} worker(s)")


if __name__ == "__main__":
    main()
