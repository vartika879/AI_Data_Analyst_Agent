"""LangGraph workflow:
[Clarifier] -> Planner -> Executor -> Critic -> [Reviewer] -> Reporter -> [Narrator].

Flow:
    START -> (clarifier) -> planner -> executor -> critic -> (reviewer)
          -> reporter -> (narrator) -> END
The Clarifier may stop the run early with status "clarify": the UI then asks the
user a question and starts a NEW run (without the Clarifier) with the answer.
A recoverable error (bad plan, plan does not fit the data) goes to `repair`,
which sends the problem back to the planner. The Reviewer can also request a
repair when it judges the result is a "mismatch". All repairs share ONE budget:
MAX_REPAIR_ATTEMPTS. Fatal errors go straight to `fail`.
The Narrator only adds an optional summary; it never changes `message`.
"""

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from analyst.clarifier import ClarificationError, ClarifierDecision, decide_clarification  # NEW (Phase 12)
from analyst.critic import CriticReport, review_execution
from analyst.executor import ExecutionError, ExecutionResult, execute_plan
from analyst.llm import LLMClient, LLMError
from analyst.narrator import MAX_ANSWER_CHARS, NarrationError, narrate_result
from analyst.plan_validation import PlanValidationError
from analyst.privacy import (
    HiddenColumnError,
    PrivacyPolicy,
    find_hidden_mentions,
    hidden_columns_used,
    tracing_disabled,
)
from analyst.planner import MAX_QUESTION_CHARS, PlannerError, generate_plan
from analyst.reporter import _escape as escape_markdown
from analyst.reporter import build_report
from analyst.reviewer import ReviewError, ReviewVerdict, review_result
from analyst.schemas import AnalysisPlan

MAX_REPAIR_ATTEMPTS = 1
RECURSION_LIMIT = 25  # safety net only; the routing already bounds the loop
MAX_MESSAGE_CHARS = 400

Status = Literal["success", "unsupported", "failed", "clarify"]  # NEW (Phase 12): "clarify"


class AnalysisState(TypedDict, total=False):
    question: str
    df: pd.DataFrame
    clarification: ClarifierDecision | None  # NEW (Phase 12)
    clarify_error: str | None  # NEW (Phase 12): why the clarifier was skipped
    plan: AnalysisPlan | None
    result: ExecutionResult | None
    critic: CriticReport | None
    review: ReviewVerdict | None
    review_error: str | None  # why the review was skipped
    narrative: str | None
    narrative_error: str | None  # why there is no AI summary
    error: str | None  # user-safe message
    error_detail: str | None  # text sent back to the LLM during repair
    error_kind: Literal["recoverable", "fatal"] | None
    repair_attempts: int
    repair_feedback: str | None
    status: Status
    message: str


@dataclass(frozen=True, eq=False)
class AnalysisOutcome:
    status: Status
    message: str  # the template report (success) or a user-safe explanation
    plan: AnalysisPlan | None = None
    result: ExecutionResult | None = None
    critic: CriticReport | None = None
    repair_attempts: int = 0
    review: ReviewVerdict | None = None
    review_error: str | None = None
    narrative: str | None = None  # optional AI summary, numbers checked
    narrative_error: str | None = None
    # NEW (Phase 12): raw (unescaped) so the UI can use the option text. The UI
    # must escape it before showing it.
    clarification: ClarifierDecision | None = None
    clarify_error: str | None = None


# ---------------------------------------------------------------- routing
# Routers are plain functions of the state, so they are easy to unit-test.


def _next_after_error_check(state: AnalysisState, ok_node: str) -> str:
    kind = state.get("error_kind")
    if kind is None:
        return ok_node
    if kind == "recoverable" and state.get("repair_attempts", 0) < MAX_REPAIR_ATTEMPTS:
        return "repair"
    return "fail"


def route_after_clarifier(state: AnalysisState) -> str:  # NEW (Phase 12)
    if state.get("status") == "clarify":
        return END
    return "planner"


def route_after_planner(state: AnalysisState) -> str:
    if state.get("status") == "unsupported":
        return END
    return _next_after_error_check(state, "executor")


def route_after_executor(state: AnalysisState) -> str:
    return _next_after_error_check(state, "critic")


def route_after_critic(state: AnalysisState) -> str:
    return _next_after_error_check(state, "reporter")


def route_after_reviewer(state: AnalysisState) -> str:
    review = state.get("review")
    budget_left = state.get("repair_attempts", 0) < MAX_REPAIR_ATTEMPTS
    if review is not None and review.verdict == "mismatch" and budget_left:
        return "repair"
    return "reporter"  # includes: reviewer skipped, "answers", "partial", or no budget


# ------------------------------------------------------------------ nodes


def prepare_repair(state: AnalysisState) -> dict:
    """Clear the failed attempt and carry the problem description forward."""
    return {
        "repair_attempts": state.get("repair_attempts", 0) + 1,
        "repair_feedback": state.get("error_detail"),
        "error": None,
        "error_detail": None,
        "error_kind": None,
        "plan": None,
        "result": None,
        "critic": None,
        "review": None,
        "review_error": None,
    }


def _hidden_question_message(columns: list[str]) -> str:
    """Shown on the user's own screen only. It is never sent to the LLM."""
    return (
        "Your question names column(s) you hid from the AI: "
        + ", ".join(columns)
        + ". The question was not sent. Unhide the column(s), or ask without naming them."
    )


def fail_node(state: AnalysisState) -> dict:
    text = state.get("error") or "The analysis failed."
    if state.get("repair_attempts", 0) > 0:
        text += " An automatic repair attempt was also made."
    # Error text can contain column names or LLM text, so make it markdown-safe.
    return {"status": "failed", "message": escape_markdown(text, MAX_MESSAGE_CHARS)}


def build_graph(
    client: LLMClient,
    reviewer_client: LLMClient | None = None,
    narrator_client: LLMClient | None = None,
    clarifier_client: LLMClient | None = None,  # NEW (Phase 12)
    policy: PrivacyPolicy | None = None,  # NEW (privacy mode)
):
    """Build and compile the workflow. LLM clients are injected (testable).

    A client of None means that agent's node is not part of the graph at all.
    With a strict privacy policy the Narrator is dropped: it needs result rows.
    """
    if policy is not None and not policy.rows_allowed:
        narrator_client = None

    def guard_node(state: AnalysisState) -> dict:
        """First node when columns are hidden: reject a question that names one.

        Runs locally, before any LLM call, so nothing is sent.
        """
        found = find_hidden_mentions(state["question"], policy.hidden_columns)
        if not found:
            return {}
        return {
            "status": "failed",
            "message": escape_markdown(_hidden_question_message(found), MAX_MESSAGE_CHARS),
        }

    def route_after_guard(state: AnalysisState) -> str:
        return END if state.get("status") == "failed" else entry

    def clarifier_node(state: AnalysisState) -> dict:  # NEW (Phase 12)
        question = state["question"].strip()
        if not question or len(question) > MAX_QUESTION_CHARS:
            # The planner reports these problems with a clearer message.
            return {"clarify_error": None}
        try:
            decision = decide_clarification(
                question, state["df"], clarifier_client, policy
            )
        except (LLMError, ClarificationError) as exc:
            # Fail open: a broken Clarifier must never block the question.
            return {"clarify_error": str(exc)}
        if decision.action != "ask":
            return {"clarify_error": None}
        return {
            "clarification": decision,
            "clarify_error": None,
            "status": "clarify",
            "message": escape_markdown(decision.question or "", MAX_MESSAGE_CHARS),
        }

    def planner_node(state: AnalysisState) -> dict:
        try:
            plan = generate_plan(
                state["question"],
                state["df"],
                client,
                feedback=state.get("repair_feedback"),
                policy=policy,
            )
        except PlannerError as exc:
            # `detail` is only set when the LLM output was malformed. Question
            # problems (empty / too long) have no detail and cannot be repaired.
            kind = "recoverable" if exc.detail else "fatal"
            return {
                "error": str(exc),
                "error_detail": exc.detail or str(exc),
                "error_kind": kind,
            }
        except LLMError as exc:
            return {"error": str(exc), "error_kind": "fatal"}

        hidden = hidden_columns_used(plan, policy) if policy is not None else []
        if hidden:
            # The AI guessed a hidden column from the question. Do not run it.
            return {
                "error": "This question needs columns you hid from the AI: "
                + ", ".join(hidden)
                + ". Unhide them or ask about other columns.",
                "error_detail": "The plan used a column that is not available. "
                "Use only the columns in the dataset, or return unsupported.",
                "error_kind": "recoverable",
            }
        if not plan.is_supported:
            reason = escape_markdown(plan.unsupported_reason or "", MAX_MESSAGE_CHARS)
            return {
                "plan": plan,
                "status": "unsupported",
                "message": f"This question cannot be answered with the supported operations: {reason}",
            }
        return {"plan": plan}

    def executor_node(state: AnalysisState) -> dict:
        try:
            result = execute_plan(state["plan"], state["df"])
        except (PlanValidationError, ExecutionError) as exc:
            # The detail goes back to the LLM, so hidden column names are removed.
            detail = policy.redact(str(exc)) if policy is not None else str(exc)
            return {
                "error": str(exc),
                "error_detail": detail,
                "error_kind": "recoverable",
            }
        return {"result": result}

    def critic_node(state: AnalysisState) -> dict:
        report = review_execution(state["plan"], state["result"], state["df"])
        if not report.passed:
            return {
                "critic": report,
                "error": "The result failed internal consistency checks, so it is not shown.",
                "error_kind": "fatal",
            }
        return {"critic": report}

    def reviewer_node(state: AnalysisState) -> dict:
        try:
            verdict = review_result(
                state["question"], state["plan"], state["result"], reviewer_client, policy
            )
        except (LLMError, ReviewError) as exc:
            # Advisory agent: its failure never blocks the answer.
            return {"review": None, "review_error": str(exc)}
        update: dict = {"review": verdict, "review_error": None}
        if verdict.verdict == "mismatch":
            # Used only if route_after_reviewer decides a repair is still allowed.
            update["error_detail"] = (
                "A reviewer judged that the result does not answer the question: "
                f"{verdict.reason}"
            )
        return update

    def reporter_node(state: AnalysisState) -> dict:
        text = build_report(state["plan"], state["result"], state["critic"])
        return {"status": "success", "message": text}

    def narrator_node(state: AnalysisState) -> dict:
        review = state.get("review")
        if review is not None and review.verdict == "mismatch":
            return {
                "narrative": None,
                "narrative_error": "AI summary skipped because the reviewer flagged a possible mismatch.",
            }
        try:
            text = narrate_result(
                state["question"], state["plan"], state["result"], narrator_client
            )
        except (LLMError, NarrationError) as exc:
            # The template report in `message` is still there, so nothing is lost.
            return {"narrative": None, "narrative_error": str(exc)}
        return {"narrative": text, "narrative_error": None}

    graph = StateGraph(AnalysisState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critic", critic_node)
    graph.add_node("reporter", reporter_node)
    graph.add_node("repair", prepare_repair)
    graph.add_node("fail", fail_node)

    entry = "planner" if clarifier_client is None else "clarifier"
    if clarifier_client is not None:  # NEW (Phase 12)
        graph.add_node("clarifier", clarifier_node)
        graph.add_conditional_edges("clarifier", route_after_clarifier, ["planner", END])
    if policy is not None and policy.hidden_columns:
        graph.add_node("guard", guard_node)
        graph.add_edge(START, "guard")
        graph.add_conditional_edges("guard", route_after_guard, [entry, END])
    else:
        graph.add_edge(START, entry)

    graph.add_conditional_edges("planner", route_after_planner, ["executor", "repair", "fail", END])
    graph.add_conditional_edges("executor", route_after_executor, ["critic", "repair", "fail"])

    if reviewer_client is None:
        graph.add_conditional_edges("critic", route_after_critic, ["reporter", "fail"])
    else:
        graph.add_node("reviewer", reviewer_node)
        # path map: the router still returns "reporter"; we send it to the reviewer first.
        graph.add_conditional_edges(
            "critic", route_after_critic, {"reporter": "reviewer", "fail": "fail"}
        )
        graph.add_conditional_edges("reviewer", route_after_reviewer, ["reporter", "repair"])

    graph.add_edge("repair", "planner")  # repairs never go back through the Clarifier
    if narrator_client is None:
        graph.add_edge("reporter", END)
    else:
        graph.add_node("narrator", narrator_node)
        graph.add_edge("reporter", "narrator")
        graph.add_edge("narrator", END)
    graph.add_edge("fail", END)
    return graph.compile()


def run_analysis(
    question: str,
    df: pd.DataFrame,
    client: LLMClient,
    reviewer_client: LLMClient | None = None,
    narrator_client: LLMClient | None = None,
    clarifier_client: LLMClient | None = None,  # NEW (Phase 12)
    policy: PrivacyPolicy | None = None,  # NEW (privacy mode)
) -> AnalysisOutcome:
    """Run the whole workflow. Never raises: failures become status='failed'."""
    initial: AnalysisState = {"question": question, "df": df, "repair_attempts": 0}
    try:
        with tracing_disabled(policy):  # LangSmith could otherwise receive the DataFrame
            final = build_graph(
                client, reviewer_client, narrator_client, clarifier_client, policy
            ).invoke(initial, config={"recursion_limit": RECURSION_LIMIT})
    except HiddenColumnError as exc:  # safety net: a lower-level guard fired
        return AnalysisOutcome(
            status="failed",
            message=escape_markdown(_hidden_question_message(list(exc.columns)), MAX_MESSAGE_CHARS),
        )
    except Exception as exc:  # last-resort boundary; only the type is shown
        return AnalysisOutcome(
            status="failed",
            message=f"Unexpected internal error ({type(exc).__name__}).",
        )

    review = final.get("review")
    if review is not None:  # reviewer text is LLM text, so escape it
        review = review.model_copy(
            update={"reason": escape_markdown(review.reason, MAX_MESSAGE_CHARS)}
        )
    narrative = final.get("narrative")  # LLM text, so escape it too
    if narrative is not None:
        narrative = escape_markdown(narrative, MAX_ANSWER_CHARS)
    narrative_error = final.get("narrative_error")
    if narrative_error is not None:
        narrative_error = escape_markdown(narrative_error, MAX_MESSAGE_CHARS)
    return AnalysisOutcome(
        status=final.get("status", "failed"),
        message=final.get("message", "The analysis failed."),
        plan=final.get("plan"),
        result=final.get("result"),
        critic=final.get("critic"),
        repair_attempts=final.get("repair_attempts", 0),
        review=review,
        review_error=final.get("review_error"),
        narrative=narrative,
        narrative_error=narrative_error,
        clarification=final.get("clarification"),
        clarify_error=final.get("clarify_error"),
    )