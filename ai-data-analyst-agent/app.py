"""Streamlit entry point for the AI Data Analyst Agent."""

import streamlit as st

from analyst.charts import build_chart
from analyst.clarifier import combine_question
from analyst.config import ConfigError, load_settings
from analyst.data_loader import DataLoadError, load_csv
from analyst.graph import AnalysisOutcome, run_analysis
from analyst.llm import GroqClient, LLMError
from analyst.reporter import _escape as escape_markdown

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
    st.write(
        "Clarifier agent: " + (f"on (`{settings.clarifier_model}`)" if settings.clarifier_enabled else "off")
    )
    st.write(
        "Reviewer agent: " + (f"on (`{settings.reviewer_model}`)" if settings.reviewer_enabled else "off")
    )
    st.write(
        "Narrator agent: " + (f"on (`{settings.narrator_model}`)" if settings.narrator_enabled else "off")
    )


# ---- callbacks -----------------------------------------------------------
# A callback runs BEFORE the script reruns. It only records what the user asked
# for in session_state["pending"]; the actual analysis runs once, further down.


def run_from_input() -> None:
    # Read the text box through its key: values captured earlier would be stale.
    question = st.session_state.get("question_input", "")
    st.session_state["pending"] = {"question": question, "use_clarifier": True}


def run_with_option(original: str, option: str) -> None:
    st.session_state["pending"] = {
        "question": combine_question(original, option),
        "use_clarifier": False,  # one clarification round at most
    }


def run_without_clarification(original: str) -> None:
    st.session_state["pending"] = {"question": original, "use_clarifier": False}


def build_clients(use_clarifier: bool):
    main = GroqClient(settings.groq_api_key, settings.groq_model)
    reviewer = (
        GroqClient(settings.groq_api_key, settings.reviewer_model)
        if settings.reviewer_enabled
        else None
    )
    narrator = (
        GroqClient(settings.groq_api_key, settings.narrator_model)
        if settings.narrator_enabled
        else None
    )
    clarifier = (
        GroqClient(settings.groq_api_key, settings.clarifier_model)
        if (settings.clarifier_enabled and use_clarifier)
        else None
    )
    return main, reviewer, narrator, clarifier


def render_outcome(outcome: AnalysisOutcome, question: str) -> None:
    if outcome.status == "clarify":
        # Option text is LLM text: escape it before it is shown anywhere.
        st.info(escape_markdown(outcome.clarification.question, 300))
        for i, option in enumerate(outcome.clarification.options):
            st.button(
                escape_markdown(option, 80),
                key=f"clarify_{i}",
                on_click=run_with_option,
                args=(question, option),
            )
        st.button(
            "Just answer my original question",
            key="clarify_skip",
            on_click=run_without_clarification,
            args=(question,),
        )
        return

    if outcome.status == "success":
        st.subheader("Answer")
        if outcome.narrative:
            st.markdown(outcome.narrative)
            st.caption(
                "AI-written summary. The numbers in it were checked against the "
                "result table; the wording was not verified."
            )
            with st.expander("Computed details"):
                st.markdown(outcome.message)
        else:
            st.markdown(outcome.message)
            if outcome.narrative_error:
                st.caption(f"AI summary unavailable: {outcome.narrative_error}")
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

    if outcome.clarify_error:
        st.caption(f"Clarifier unavailable (your question was used as written): {outcome.clarify_error}")

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
    st.session_state.pop("pending", None)

st.success(f"Loaded **{uploaded.name}**: {len(df):,} rows × {df.shape[1]} columns")

st.subheader("Preview (first 20 rows)")
st.dataframe(df.head(20))

st.subheader("Columns")
st.dataframe({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})

st.divider()
st.subheader("Ask a question")
st.text_input(
    "Your question",
    key="question_input",
    placeholder="e.g. Bar chart of average salary by city",
)
st.button("Analyze", type="primary", on_click=run_from_input)

# The analysis runs here, exactly once per request recorded by a callback.
pending = st.session_state.pop("pending", None)
if pending is not None:
    if not settings.has_api_key:
        st.error("Groq API key is not set. Add GROQ_API_KEY to your .env file.")
        st.stop()
    try:
        clients = build_clients(pending["use_clarifier"])
    except LLMError as exc:
        st.error(str(exc))
        st.stop()
    with st.spinner("Planning, running and checking..."):
        outcome = run_analysis(pending["question"], df, *clients)
    st.session_state["saved"] = {"question": pending["question"], "outcome": outcome}

# Rendered from session_state, so the answer survives reruns (e.g. typing in a widget).
saved = st.session_state.get("saved")
if saved is not None:
    st.text(f"Question: {saved['question']}")
    render_outcome(saved["outcome"], saved["question"])