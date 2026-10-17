"""Streamlit entry point for the AI Data Analyst Agent."""

import streamlit as st

from analyst.charts import build_chart
from analyst.config import ConfigError, load_settings
from analyst.data_loader import DataLoadError, load_csv
from analyst.graph import AnalysisOutcome, run_analysis
from analyst.llm import GroqClient, LLMError

st.set_page_config(page_title="AI Data Analyst Agent", page_icon="📊", layout="wide")

st.title("📊 AI Data Analyst Agent")
st.caption("Upload a CSV, ask a question, get a data-backed answer.")

try:
    settings = load_settings()
except ConfigError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()

with st.sidebar:
    st.header("Setup status")
    st.write(f"Max upload size: **{settings.max_upload_mb} MB**")
    st.write(f"LLM model: `{settings.groq_model}`")
    if settings.has_api_key:
        st.success("Groq API key: configured")
    else:
        st.warning("Groq API key: not set (add it to your .env file)")
    if settings.reviewer_enabled:
        st.write(f"Reviewer agent: on (`{settings.reviewer_model}`)")
    else:
        st.write("Reviewer agent: off")


def render_outcome(outcome: AnalysisOutcome) -> None:
    if outcome.status == "success":
        st.subheader("Answer")
        st.markdown(outcome.message)
        st.subheader("Result table")
        st.dataframe(outcome.result.table)
        chart = build_chart(outcome.plan, outcome.result)
        if chart.figure is not None:
            st.subheader("Chart")
            st.plotly_chart(chart.figure)
        if chart.note:
            st.caption(chart.note)
    elif outcome.status == "unsupported":
        st.info(outcome.message)
    else:
        st.error(outcome.message)

    if outcome.repair_attempts:
        st.caption("The first plan was rejected, so one automatic repair attempt was made.")

    review = outcome.review
    if review is not None and review.verdict == "mismatch":
        st.warning(f"Reviewer (AI opinion): this may not answer your question. {review.reason}")
    elif review is not None and review.verdict == "partial":
        st.warning(f"Reviewer (AI opinion): this may only partly answer your question. {review.reason}")
    elif review is not None:
        st.caption(f"Reviewer (AI opinion): looks like it answers the question. {review.reason}")
    elif outcome.review_error:
        st.caption(f"Reviewer unavailable: {outcome.review_error}")
    if outcome.plan is not None:
        with st.expander("How this was computed (analysis plan)"):
            st.json(outcome.plan.model_dump(mode="json"))
    if outcome.critic is not None:
        with st.expander("Consistency checks"):
            st.write(f"Checks run: {outcome.critic.checks_run}")
            for issue in outcome.critic.issues:
                st.error(issue)
            for warning in outcome.critic.warnings:
                st.warning(warning)
            st.caption(
                "These checks look for internal inconsistencies. "
                "They do not guarantee the answer is correct."
            )


uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

raw_bytes = uploaded.getvalue()
try:
    df = load_csv(raw_bytes, uploaded.name, settings.max_upload_bytes)
except DataLoadError as exc:
    st.error(str(exc))
    st.stop()

# A new file means any saved answer belongs to the old data, so forget it.
signature = (uploaded.name, len(raw_bytes))
if st.session_state.get("file_signature") != signature:
    st.session_state["file_signature"] = signature
    st.session_state.pop("saved", None)

st.success(f"Loaded **{uploaded.name}**: {len(df):,} rows × {df.shape[1]} columns")

st.subheader("Preview (first 20 rows)")
st.dataframe(df.head(20))

st.subheader("Columns")
st.dataframe({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})

st.divider()
st.subheader("Ask a question")
question = st.text_input(
    "Your question", placeholder="e.g. Bar chart of average salary by city"
)

if st.button("Analyze", type="primary"):
    if not settings.has_api_key:
        st.error("Groq API key is not set. Add GROQ_API_KEY to your .env file.")
        st.stop()
    try:
        client = GroqClient(settings.groq_api_key, settings.groq_model)
        reviewer_client = (
            GroqClient(settings.groq_api_key, settings.reviewer_model)
            if settings.reviewer_enabled
            else None
        )
    except LLMError as exc:
        st.error(str(exc))
        st.stop()

    with st.spinner("Planning, running and checking..."):
        outcome = run_analysis(question, df, client, reviewer_client)
    st.session_state["saved"] = {"question": question, "outcome": outcome}

# Rendered from session_state, so the answer survives reruns (e.g. typing in a widget).
saved = st.session_state.get("saved")
if saved is not None:
    st.text(f"Question: {saved['question']}")
    render_outcome(saved["outcome"])