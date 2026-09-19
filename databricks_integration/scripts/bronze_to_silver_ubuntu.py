"""bronze_to_silver_ubuntu

Databricks-oriented orchestration for the Ubuntu bronze-to-silver workflow.

This is the intended bronze -> silver -> gold progression:
1. bronze: raw chat records
2. silver: cleaned text + residual classification
3. gold: aggregated metrics and reporting

This adapter delegates the complete bronze-to-silver message-feature pipeline
to the shared ``pipeline/`` modules. Databricks ingestion, partitioning, and
Delta writes belong around this transformation boundary.
"""

import os
import pandas as pd

from pipeline.pipeline import PipelineConfig, run_pipeline


def residual_stage_policy(use_api: bool = False, api_key: str | None = None) -> str:
    """Return the residual classification policy for a Databricks job.

    Use API classification only when the env/secret is present. Otherwise,
    leave residuals unchanged for manual review.
    """
    if not use_api:
        return "manual_review"
    return "api" if api_key or os.getenv("OPENAI_API_KEY") else "manual_review"


def build_residual_plan(use_api: bool = False, api_key: str | None = None) -> dict:
    """Return the step-by-step plan for the residual stage."""
    policy = residual_stage_policy(use_api=use_api, api_key=api_key)
    if policy == "manual_review":
        return {
            "status": "manual_review",
            "message": "Residual words are unchanged and ready for manual review.",
            "api_enabled": False,
        }

    return {
        "status": "api",
        "message": "Residual words will be classified via the API only after residual extraction.",
        "api_enabled": True,
        "api_key_present": bool(api_key or os.getenv("OPENAI_API_KEY")),
    }


def run_bronze_to_silver(
    bronze_df: pd.DataFrame,
    *,
    text_col: str = "text_cleaned",
    source_text_col: str = "text",
    residual_policy: str = "manual_review",
    api_key: str | None = None,
    sentiment_mode: str = "vader",
    transformer_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    transformer_revision: str = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7",
    transformer_batch_size: int = 32,
    transformer_inference_chunk_size: int = 4096,
    transformer_device: int | None = None,
    transformer_dtype: str = "float32",
    advanced_nlp: bool = False,
    run_spacy: bool = True,
    spacy_batch_size: int = 128,
    include_token_details: bool = False,
    topic_mode: str = "none",
    topic_fit_sample_size: int = 100_000,
    topic_batch_size: int = 50_000,
    topic_minimum_vocabulary_coverage: float = 0.50,
    emotion_mode: str = "none",
    spacy_model: str = "en_core_web_sm",
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
    emotion_revision: str = "cea2f78f197f0337186a0faa93e00ef93811f6eb",
    emotion_batch_size: int = 32,
    emotion_inference_chunk_size: int = 4096,
    emotion_device: int | None = None,
    emotion_dtype: str = "float32",
    validate_advanced_nlp: bool = True,
    workers: int | None = None,
    sample_size: int | None = None,
    sample_seed: int = 0,
) -> pd.DataFrame:
    """Delegate bronze-to-silver transformation to the shared pipeline.

    Databricks ingestion and table writes belong around this adapter. The
    transformation rules remain centralized in ``pipeline/`` so local and
    Databricks runs cannot drift.
    """
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    config = PipelineConfig(
        residual_policy=residual_policy,
        api_key=resolved_api_key,
        sentiment_mode=sentiment_mode,
        transformer_model=transformer_model,
        transformer_revision=transformer_revision,
        transformer_batch_size=transformer_batch_size,
        transformer_inference_chunk_size=transformer_inference_chunk_size,
        transformer_device=transformer_device,
        transformer_dtype=transformer_dtype,
        advanced_nlp=advanced_nlp,
        run_spacy=run_spacy,
        spacy_batch_size=spacy_batch_size,
        include_token_details=include_token_details,
        topic_mode=topic_mode,
        topic_fit_sample_size=topic_fit_sample_size,
        topic_batch_size=topic_batch_size,
        topic_minimum_vocabulary_coverage=topic_minimum_vocabulary_coverage,
        emotion_mode=emotion_mode,
        spacy_model=spacy_model,
        emotion_model=emotion_model,
        emotion_revision=emotion_revision,
        emotion_batch_size=emotion_batch_size,
        emotion_inference_chunk_size=emotion_inference_chunk_size,
        emotion_device=emotion_device,
        emotion_dtype=emotion_dtype,
        validate_advanced_nlp=validate_advanced_nlp,
        parallel=False,
        workers=workers,
        sample_size=sample_size,
        sample_seed=sample_seed,
        output_dir="databricks_integration/outputs",
    )
    return run_pipeline(
        bronze_df,
        config=config,
        text_col=text_col,
        source_text_col=source_text_col,
    )


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    plan = build_residual_plan(use_api=bool(api_key), api_key=api_key)
    print(plan)


if __name__ == "__main__":
    main()
