r"""Ubuntu Dialogue Intelligence Streamlit dashboard.

Run from the repository root with:
    .venv\python.exe -m streamlit run dashboard\app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dashboard import (  # noqa: E402
    build_topic_priorities,
    filter_conversations,
    load_dashboard_bundle,
    preferred_sentiment_column,
)

st.set_page_config(
    page_title="Ubuntu Dialogue Intelligence",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

UBUNTU_ORANGE = "#E95420"
AUBERGINE = "#772953"
LIGHT_AUBERGINE = "#9B6A8D"
WARM_GRAY = "#AEA79F"
PAPER = "#F7F5F2"
DARK_INK = "#2C2C2C"

st.markdown(
    f"""
    <style>
    :root {{
        --ubuntu-orange: {UBUNTU_ORANGE};
        --aubergine: {AUBERGINE};
        --paper: {PAPER};
        --ink: {DARK_INK};
        --line: #DDD8D2;
    }}
    .stApp {{ background: var(--paper); color: var(--ink); }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #2C001E 0%, var(--aubergine) 100%);
        border-right: 4px solid var(--ubuntu-orange);
    }}
    [data-testid="stSidebar"] * {{ color: #FFFFFF; }}
    h1 {{
        border-left: 10px solid var(--ubuntu-orange);
        padding-left: 18px;
        color: var(--aubergine);
        letter-spacing: .02em;
    }}
    h2, h3 {{ color: var(--aubergine); }}
    [data-testid="stMetric"] {{
        background: #FFFFFF;
        border-top: 4px solid var(--ubuntu-orange);
        border-bottom: 1px solid var(--line);
        border-radius: 6px;
        padding: 14px 16px;
        box-shadow: 0 2px 8px rgba(44, 0, 30, .06);
    }}
    [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {{
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 6px;
    }}
    [data-baseweb="tab-list"] {{
        background: rgba(255, 255, 255, .75);
        border-radius: 6px;
        padding: 4px;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--ubuntu-orange);
        font-weight: 800;
    }}
    .scope-note {{
        background: #FFFFFF;
        border-left: 5px solid var(--ubuntu-orange);
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 14px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _bundle_directory() -> Path:
    configured = os.getenv("UBUNTU_DASHBOARD_DATA")
    return Path(configured).resolve() if configured else (
        PROJECT_ROOT / "outputs" / "dashboard_data"
    )


@st.cache_data(show_spinner=False)
def _load_bundle(path: str):
    return load_dashboard_bundle(path)


def _weighted_monthly(
    frame: pd.DataFrame, sentiment_column: str | None
) -> pd.DataFrame:
    if "conversation_start" not in frame or frame["conversation_start"].notna().sum() == 0:
        return pd.DataFrame()
    working = frame.dropna(subset=["conversation_start"]).copy()
    working["date_period"] = (
        working["conversation_start"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    )
    working["message_count"] = pd.to_numeric(
        working["message_count"], errors="coerce"
    ).fillna(0)
    grouped = working.groupby("date_period", observed=True, sort=True)
    monthly = grouped.agg(
        message_count=("message_count", "sum"),
        conversation_count=("conversation_id", "nunique"),
        answer_rate=("conversation_was_answered", "mean"),
    ).reset_index()
    if sentiment_column:
        working["_weighted_sentiment"] = (
            pd.to_numeric(working[sentiment_column], errors="coerce")
            * working["message_count"]
        )
        sentiment = working.groupby(
            "date_period", observed=True, sort=True
        )["_weighted_sentiment"].sum(min_count=1)
        weights = working.loc[
            working[sentiment_column].notna()
        ].groupby("date_period", observed=True)["message_count"].sum()
        monthly = monthly.merge(
            sentiment.div(weights).rename("average_sentiment").reset_index(),
            on="date_period",
            how="left",
            validate="one_to_one",
        )
    return monthly


def _safe_relative_file(bundle_dir: Path, reference: str | None) -> Path | None:
    if not reference:
        return None
    return (bundle_dir / reference).resolve()


bundle_dir = _bundle_directory()
try:
    tables, manifest = _load_bundle(str(bundle_dir))
except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError) as exc:
    st.title("Ubuntu Dialogue Intelligence")
    st.error(str(exc))
    st.code(
        r".\.venv\python.exe -m streamlit run dashboard\app.py",
        language="powershell",
    )
    st.info(
        "Run Steps 7-12 first so outputs/dashboard_data contains the validated "
        "dashboard bundle and analysis_manifest.json."
    )
    st.stop()

conversations = tables["conversation_summary"]
sentiment_column = preferred_sentiment_column(conversations)

# ---------------------------------------------------------------------------
# Sidebar drilldowns
# ---------------------------------------------------------------------------

st.sidebar.title("Filters")
metadata = manifest.get("metadata", {})
st.sidebar.caption(
    f"Schema {manifest['feature_schema_version']} · "
    f"seed {metadata.get('analysis_seed', 'not recorded')}"
)

start_date = end_date = None
if "conversation_start" in conversations and conversations["conversation_start"].notna().any():
    minimum_date = conversations["conversation_start"].min().date()
    maximum_date = conversations["conversation_start"].max().date()
    selected_dates = st.sidebar.date_input(
        "Conversation dates",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
    )
    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates

outcome_choice = st.sidebar.selectbox(
    "Response outcome", ["All", "Answered", "Unanswered"]
)
answered_filter = {
    "All": None, "Answered": True, "Unanswered": False
}[outcome_choice]

topic_choice = "All"
if "most_common_topic" in conversations:
    topics = sorted(
        conversations["most_common_topic"].dropna().astype(str).unique().tolist()
    )
    topic_choice = st.sidebar.selectbox("Dominant topic", ["All", *topics])

filtered = filter_conversations(
    conversations,
    start_date=start_date,
    end_date=end_date,
    answered=answered_filter,
    topic=topic_choice,
)
st.sidebar.markdown(f"**{len(filtered):,} conversations** in current selection")
st.sidebar.download_button(
    "Download filtered summary",
    filtered.to_csv(index=False).encode("utf-8"),
    file_name="ubuntu_conversation_summary_filtered.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Header and KPIs
# ---------------------------------------------------------------------------

st.title("Ubuntu Dialogue Intelligence")
scope_parts = [outcome_choice]
if topic_choice != "All":
    scope_parts.append(topic_choice)
if start_date and end_date:
    scope_parts.append(f"{start_date} to {end_date}")
st.markdown(
    f'<div class="scope-note"><strong>Scope:</strong> {" · ".join(scope_parts)}</div>',
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("No conversations match the current filters.")
    st.stop()

message_total = int(pd.to_numeric(filtered["message_count"], errors="coerce").sum())
answer_rate = float(filtered["conversation_was_answered"].mean())
average_sentiment = (
    float(pd.to_numeric(filtered[sentiment_column], errors="coerce").mean())
    if sentiment_column else np.nan
)
median_response = (
    float(pd.to_numeric(
        filtered["avg_response_gap_mins_between_speakers"], errors="coerce"
    ).median())
    if "avg_response_gap_mins_between_speakers" in filtered else np.nan
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Conversations", f"{len(filtered):,}")
k2.metric("Messages", f"{message_total:,}")
k3.metric("Answer rate", f"{answer_rate:.1%}")
k4.metric(
    "Average sentiment",
    f"{average_sentiment:+.3f}" if np.isfinite(average_sentiment) else "Unavailable",
)
k5.metric(
    "Median response gap",
    f"{median_response:.1f} min" if np.isfinite(median_response) else "Unavailable",
)

tab_desc, tab_diag, tab_pred, tab_presc = st.tabs(
    ["Descriptive", "Diagnostic", "Predictive", "Prescriptive"]
)

# ---------------------------------------------------------------------------
# Descriptive
# ---------------------------------------------------------------------------

with tab_desc:
    left, right = st.columns(2)
    with left:
        st.subheader("Response outcomes")
        outcome_counts = (
            filtered["conversation_was_answered"]
            .map({True: "Answered", False: "Unanswered"})
            .value_counts()
            .rename_axis("outcome")
            .reset_index(name="conversations")
        )
        figure = px.pie(
            outcome_counts,
            names="outcome",
            values="conversations",
            hole=0.58,
            color="outcome",
            color_discrete_map={"Answered": UBUNTU_ORANGE, "Unanswered": WARM_GRAY},
        )
        figure.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    with right:
        st.subheader("Conversation size")
        figure = px.histogram(
            filtered,
            x="message_count",
            nbins=40,
            color_discrete_sequence=[AUBERGINE],
            labels={"message_count": "Messages per conversation"},
        )
        figure.update_layout(showlegend=False, height=350, margin=dict(t=20, b=20))
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    st.subheader("Conversation activity over time")
    dynamic_monthly = _weighted_monthly(filtered, sentiment_column)
    if dynamic_monthly.empty:
        st.info("No dated conversations are available for the selected scope.")
    else:
        figure = go.Figure()
        figure.add_trace(go.Bar(
            x=dynamic_monthly["date_period"],
            y=dynamic_monthly["conversation_count"],
            name="Conversations",
            marker_color=UBUNTU_ORANGE,
            opacity=0.35,
            yaxis="y2",
        ))
        if "average_sentiment" in dynamic_monthly:
            figure.add_trace(go.Scatter(
                x=dynamic_monthly["date_period"],
                y=dynamic_monthly["average_sentiment"],
                name="Average sentiment",
                line=dict(color=AUBERGINE, width=3),
            ))
        figure.update_layout(
            yaxis=dict(title="Average sentiment"),
            yaxis2=dict(title="Conversations", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.10),
            margin=dict(t=50, b=30),
            height=430,
        )
        st.plotly_chart(figure, width="stretch")

    if "most_common_topic" in filtered:
        st.subheader("Most common conversation topics")
        topic_counts = (
            filtered["most_common_topic"].dropna().astype(str).value_counts()
            .head(15).sort_values().rename_axis("topic").reset_index(name="conversations")
        )
        if not topic_counts.empty:
            figure = px.bar(
                topic_counts,
                x="conversations",
                y="topic",
                orientation="h",
                color_discrete_sequence=[UBUNTU_ORANGE],
            )
            figure.update_layout(showlegend=False, height=430, margin=dict(t=20, b=20))
            st.plotly_chart(figure, width="stretch")

# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

with tab_diag:
    st.subheader("Initial-message feature correlations")
    correlation = tables["correlation_matrix"].copy()
    if "feature" in correlation:
        correlation = correlation.set_index("feature")
    correlation = correlation.apply(pd.to_numeric, errors="coerce")
    figure = px.imshow(
        correlation,
        color_continuous_scale=[AUBERGINE, "#FFFFFF", UBUNTU_ORANGE],
        color_continuous_midpoint=0,
        zmin=-1,
        zmax=1,
        aspect="auto",
        labels={"color": "Spearman r"},
    )
    figure.update_layout(height=max(480, 24 * len(correlation)), margin=dict(t=20, b=20))
    st.plotly_chart(figure, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Exploratory hypothesis tests")
        tests = tables["hypothesis_tests"]
        if tests.empty:
            st.info("No hypothesis test met its minimum sample-size requirement.")
        else:
            shown = [
                column for column in (
                    "hypothesis", "test", "effect_size", "p_value_holm",
                    "significant_at_0_05"
                ) if column in tests
            ]
            st.dataframe(tests[shown], width="stretch", hide_index=True)
    with right:
        st.subheader("Response-gap distribution")
        gap_column = "avg_response_gap_mins_between_speakers"
        if gap_column in filtered and filtered[gap_column].notna().any():
            gaps = pd.to_numeric(filtered[gap_column], errors="coerce").dropna()
            upper = gaps.quantile(0.99)
            figure = px.histogram(
                gaps[gaps <= upper].to_frame("minutes"),
                x="minutes",
                nbins=40,
                color_discrete_sequence=[LIGHT_AUBERGINE],
            )
            figure.update_layout(showlegend=False, height=360)
            st.plotly_chart(figure, width="stretch")
            st.caption("Displayed through the 99th percentile so extreme waits do not flatten the chart.")
        else:
            st.info("No between-speaker response gaps are available in this scope.")

    if sentiment_column:
        st.subheader("Sentiment and conversation length")
        relationship = filtered[[
            "message_count", sentiment_column, "conversation_was_answered"
        ]].dropna()
        if len(relationship) > 5_000:
            relationship = relationship.sample(5_000, random_state=42)
        relationship["outcome"] = relationship["conversation_was_answered"].map({
            True: "Answered", False: "Unanswered"
        })
        figure = px.scatter(
            relationship,
            x="message_count",
            y=sentiment_column,
            color="outcome",
            opacity=0.45,
            color_discrete_map={"Answered": UBUNTU_ORANGE, "Unanswered": WARM_GRAY},
            labels={"message_count": "Messages", sentiment_column: "Average sentiment"},
        )
        figure.update_layout(height=430)
        st.plotly_chart(figure, width="stretch")

# ---------------------------------------------------------------------------
# Predictive
# ---------------------------------------------------------------------------

with tab_pred:
    st.subheader("Initial-message response model")
    st.caption(
        "Predicts whether another sender replies. The model excludes conversation "
        "duration, later messages, response gaps, and user identities."
    )
    test_metrics = tables["model_test_metrics"]
    if test_metrics.empty:
        st.info("No held-out model metrics are available.")
    else:
        metric = test_metrics.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC AUC", f"{metric.get('roc_auc', np.nan):.3f}")
        m2.metric("Average precision", f"{metric.get('average_precision', np.nan):.3f}")
        m3.metric("Balanced accuracy", f"{metric.get('balanced_accuracy', np.nan):.3f}")
        m4.metric("F1", f"{metric.get('f1', np.nan):.3f}")

    st.subheader("Model selection")
    selection = tables["model_selection"].copy()
    if not selection.empty and {"model", "average_precision"}.issubset(selection):
        figure = px.bar(
            selection.sort_values("average_precision"),
            x="average_precision",
            y="model",
            orientation="h",
            color="average_precision",
            color_continuous_scale=[WARM_GRAY, UBUNTU_ORANGE],
            range_x=[0, 1],
        )
        figure.update_layout(coloraxis_showscale=False, height=320)
        st.plotly_chart(figure, width="stretch")

    st.subheader("Strongest model factors")
    importance = tables["model_feature_importance"].copy()
    if not importance.empty and {"feature", "importance"}.issubset(importance):
        importance["absolute_importance"] = pd.to_numeric(
            importance.get("absolute_importance", importance["importance"].abs()),
            errors="coerce",
        )
        top_importance = importance.nlargest(20, "absolute_importance").sort_values("importance")
        figure = px.bar(
            top_importance,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=[AUBERGINE, "#FFFFFF", UBUNTU_ORANGE],
            color_continuous_midpoint=0,
        )
        figure.update_layout(coloraxis_showscale=False, height=560)
        st.plotly_chart(figure, width="stretch")

    model_metadata_path = _safe_relative_file(
        bundle_dir, manifest.get("model_metadata_path")
    )
    if model_metadata_path and model_metadata_path.exists():
        with st.expander("Model card and leakage exclusions"):
            model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
            st.json(model_metadata)

# ---------------------------------------------------------------------------
# Prescriptive
# ---------------------------------------------------------------------------

with tab_presc:
    st.subheader("Where should support triage focus first?")
    st.caption(
        "Priorities combine low observed answer rate with enough conversation "
        "volume to avoid overreacting to tiny groups. They are operational leads, not causal claims."
    )

    if "most_common_topic" in filtered:
        dynamic_topics = (
            filtered.dropna(subset=["most_common_topic"])
            .groupby("most_common_topic", observed=True)
            .agg(
                conversations=("conversation_id", "size"),
                answer_rate=("conversation_was_answered", "mean"),
            )
            .reset_index()
            .rename(columns={"most_common_topic": "topic_label"})
        )
        topic_priorities = build_topic_priorities(dynamic_topics)
    elif "topic_response_summary" in tables:
        topic_priorities = build_topic_priorities(tables["topic_response_summary"])
    else:
        topic_priorities = pd.DataFrame()

    if topic_priorities.empty:
        st.info("No topic group meets the minimum support threshold.")
    else:
        top_priority = topic_priorities.iloc[0]
        st.warning(
            f"Highest-volume low-response topic: {top_priority['topic_label']} · "
            f"{top_priority['answer_rate']:.1%} answered across "
            f"{int(top_priority['conversations']):,} conversations."
        )
        st.dataframe(
            topic_priorities.head(15)[[
                "topic_label", "conversations", "answer_rate", "priority_score"
            ]].style.format({
                "answer_rate": "{:.1%}", "priority_score": "{:.3f}"
            }),
            width="stretch",
            hide_index=True,
        )

    if "release_summary" in tables:
        release = tables["release_summary"].copy()
        release_key = next((
            column for column in release.columns if column.startswith("days_")
            and column.endswith("_release_bucket")
        ), None)
        if release_key:
            st.subheader("Release-cycle monitoring")
            shown = [
                column for column in (
                    release_key, "message_count", "avg_vader_compound",
                    "avg_transformer_expected_sentiment",
                    "avg_response_gap_mins_between_speakers",
                ) if column in release
            ]
            st.dataframe(release[shown], width="stretch", hide_index=True)

    st.subheader("Recommended operational use")
    st.markdown(
        """
        1. Route high-volume, low-answer topics to queue monitoring and documentation review.
        2. Compare release-cycle spikes with response gaps before scheduling extra coverage.
        3. Use model factors to prioritize investigation, never to suppress or auto-reject messages.
        4. Rebuild the bundle and model after material pipeline, lexicon, or corpus changes.
        """
    )

with st.expander("Data provenance and limitations"):
    st.json({
        "bundle_directory": str(bundle_dir),
        "created_at_utc": manifest.get("created_at_utc"),
        "feature_schema_version": manifest.get("feature_schema_version"),
        "analysis_metadata": metadata,
        "limitations": [
            "Dashboard tables exclude raw message text and usernames.",
            "Statistical tests are exploratory associations, not causal estimates.",
            "The response model is evaluated on a held-out split but not yet on a later time period.",
            "Results inherit the configured sentiment, topic, and sampling choices.",
        ],
    })
