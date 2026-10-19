"""Streamlit entry point for the AI Data Analyst Agent."""

import json
import os

import streamlit as st

from analyst.charts import build_chart
from analyst.clarifier import combine_question
from analyst.config import ConfigError, load_settings
from analyst.data_loader import DataLoadError, load_csv
from analyst.graph import AnalysisOutcome, run_analysis
from analyst.llm import LLMError
from analyst.providers import LABELS, make_client, missing_key_message, provider_for
from analyst.planner import build_user_prompt
from analyst.privacy import HiddenColumnError, PrivacyPolicy
from analyst.reviewer import build_review_prompt
from analyst.reporter import _escape as escape_markdown
from evals.dataset import csv_bytes  # built-in sample data for the demo button

# Demo protection: max analysis runs per browser session (0 or unset = no limit).
try:
    MAX_QUERIES = max(0, int(os.environ.get("MAX_QUERIES_PER_SESSION", "0") or 0))
except ValueError:
    MAX_QUERIES = 0

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
    planner_provider = provider_for(settings, "planner")
    st.write(
        f"LLM model: `{settings.model_for('planner')}`"
        + ("" if planner_provider == "groq" else f" ({LABELS[planner_provider]})")
    )
    if missing_key_message(settings, ["planner"]) is None:
        st.success(f"{LABELS[planner_provider]} API key: configured")
    else:
        st.warning(f"{LABELS[planner_provider]} API key: not set (add it to your .env file)")
    st.write(
        "Clarifier agent: " + (f"on (`{settings.clarifier_model}`)" if settings.clarifier_enabled else "off")
    )
    st.write(
        "Reviewer agent: " + (f"on (`{settings.reviewer_model}`)" if settings.reviewer_enabled else "off")
    )
    privacy_on = st.toggle(
        "Privacy mode",
        value=settings.privacy_mode == "strict",
        help="The AI sees only column names, types and counts. No sample values, "
        "no result rows. Your real data and results stay on this computer.",
    )
    if privacy_on:
        st.write("Narrator agent: off (privacy mode)")
    else:
        st.write(
            "Narrator agent: "
            + (f"on (`{settings.narrator_model}`)" if settings.narrator_enabled else "off")
        )
    if MAX_QUERIES:
        used = min(st.session_state.get("queries_used", 0), MAX_QUERIES)
        st.caption(f"Demo limit: {used} / {MAX_QUERIES} analyses used in this session.")


# ---- callbacks -----------------------------------------------------------
# A callback runs BEFORE the script reruns. It only records what the user asked
# for in session_state["pending"]; the actual analysis runs once, further down.


def queue_run(question: str, use_clarifier: bool) -> None:
    st.session_state["pending"] = {"question": question, "use_clarifier": use_clarifier}
    st.session_state["queries_used"] = st.session_state.get("queries_used", 0) + 1


def run_from_input() -> None:
    # Read the text box through its key: values captured earlier would be stale.
    queue_run(st.session_state.get("question_input", ""), True)


def run_with_option(original: str, option: str) -> None:
    # use_clarifier=False: one clarification round at most
    queue_run(combine_question(original, option), False)


def run_without_clarification(original: str) -> None:
    queue_run(original, False)


def use_sample_data() -> None:
    st.session_state["use_sample"] = True


def active_roles(use_clarifier: bool) -> list[str]:
    """The agents that will actually run for this request."""
    roles = ["planner"]
    if settings.reviewer_enabled:
        roles.append("reviewer")
    if settings.narrator_enabled and not privacy_on:
        roles.append("narrator")
    if settings.clarifier_enabled and use_clarifier:
        roles.append("clarifier")
    return roles


def build_clients(use_clarifier: bool):
    """One client per active agent. Each agent may use a different provider."""
    clients = {role: make_client(settings, role) for role in active_roles(use_clarifier)}
    return clients["planner"], clients.get("reviewer"), clients.get("narrator"), clients.get("clarifier")


def render_privacy_panel() -> None:
    """Shows the real outbound payloads. They are built by the same functions the agents use."""
    with st.expander("What is sent to the AI?"):
        st.markdown(
            "**Your question is sent to the AI exactly as you typed it.** Privacy mode does "
            "not mask or edit it, so do not type sensitive details into the question box."
        )
        if policy is None:
            st.warning("Privacy mode is off: sample values and the first 10 result rows are also sent.")
        else:
            hidden = len(policy.hidden_columns)
            st.caption(
                "Privacy mode: only column names, types and counts are sent. No sample values, "
                "no result rows."
                + (f" {hidden} column(s) are hidden from the AI." if hidden else "")
            )
        question = st.session_state.get("question_input", "").strip()
        try:
            payload = build_user_prompt(question or "(your question appears here)", df, None, policy)
        except HiddenColumnError as exc:
            st.warning(
                "Your question names column(s) you hid from the AI: "
                + escape_markdown(", ".join(exc.columns), 200)
                + ". It will not be sent."
            )
        else:
            st.caption(
                "Planner and Clarifier receive exactly this. If a clarification is needed, "
                "the option you pick is added to your question."
            )
            st.json(json.loads(payload))
        saved_run = st.session_state.get("saved")
        if saved_run is not None:
            outcome = saved_run["outcome"]
            reviewer_ran = outcome.review is not None or outcome.review_error
            if reviewer_ran and outcome.plan is not None and outcome.result is not None:
                try:
                    review_payload = build_review_prompt(
                        saved_run["question"], outcome.plan, outcome.result, saved_run["policy"]
                    )
                except HiddenColumnError:
                    pass
                else:
                    st.caption("Reviewer received exactly this (for the last answer):")
                    st.json(json.loads(review_payload))
        if policy is not None:
            st.caption("The Narrator needs result rows, so it is switched off in privacy mode.")


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

if uploaded is not None:
    st.session_state.pop("use_sample", None)  # a real upload replaces the sample
    raw_bytes, file_name = uploaded.getvalue(), uploaded.name
elif st.session_state.get("use_sample"):
    raw_bytes, file_name = csv_bytes(), "sample_sales.csv"
else:
    st.info("Upload a CSV file to get started, or try the built-in sample data.")
    st.button("Try sample sales data", on_click=use_sample_data)
    st.stop()

try:
    df = load_csv(raw_bytes, file_name, settings.max_upload_bytes)
except DataLoadError as exc:
    st.error(str(exc))
    st.stop()

# A new file means any saved answer belongs to the old data, so forget it.
signature = (file_name, len(raw_bytes))
if st.session_state.get("file_signature") != signature:
    st.session_state["file_signature"] = signature
    st.session_state.pop("saved", None)
    st.session_state.pop("pending", None)
    st.session_state.pop("hidden_columns", None)  # hidden columns belong to the old file

st.success(f"Loaded **{file_name}**: {len(df):,} rows × {df.shape[1]} columns")

st.subheader("Preview (first 20 rows)")
st.dataframe(df.head(20))

st.subheader("Columns")
st.dataframe({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})

st.subheader("Privacy")
st.multiselect(
    "Hide these columns from the AI (optional)",
    options=list(df.columns),
    key="hidden_columns",
    disabled=not privacy_on,
    help="The AI will not see these column names. Questions that need them cannot be answered.",
)
policy = PrivacyPolicy.strict(st.session_state.get("hidden_columns", [])) if privacy_on else None
st.divider()
st.subheader("Ask a question")
st.text_input(
    "Your question",
    key="question_input",
    placeholder="e.g. Bar chart of average salary by city",
)
st.button("Analyze", type="primary", on_click=run_from_input)
render_privacy_panel()

# The analysis runs here, exactly once per request recorded by a callback.
pending = st.session_state.pop("pending", None)
if pending is not None:
    if MAX_QUERIES and st.session_state.get("queries_used", 0) > MAX_QUERIES:
        st.error(
            f"Demo limit reached ({MAX_QUERIES} analyses per session). "
            "Run the app locally to use it without a limit."
        )
        st.stop()
    key_problem = missing_key_message(settings, active_roles(pending["use_clarifier"]))
    if key_problem:
        st.error(key_problem)
        st.stop()
    try:
        clients = build_clients(pending["use_clarifier"])
    except LLMError as exc:
        st.error(str(exc))
        st.stop()
    with st.spinner("Planning, running and checking..."):
        outcome = run_analysis(pending["question"], df, *clients, policy=policy)
    for role, client in zip(("planner", "reviewer", "narrator", "clarifier"), clients):
        if getattr(client, "used_fallback", False):
            st.toast(f"{role}: main provider was busy, answered by {client.fallback_label}")
    st.session_state["saved"] = {
        "question": pending["question"],
        "outcome": outcome,
        "policy": policy,
    }

# Rendered from session_state, so the answer survives reruns (e.g. typing in a widget).
saved = st.session_state.get("saved")
if saved is not None:
    st.text(f"Question: {saved['question']}")
    render_outcome(saved["outcome"], saved["question"])