"""LangGraph workflow: Planner -> Executor -> Critic -> [Reviewer] -> Reporter.

Flow:
    START -> planner -> executor -> critic -> (reviewer) -> reporter -> END
A recoverable error (bad plan, plan does not fit the data) goes to `repair`,
which sends the problem back to the planner. The Reviewer can also request a
repair when it judges the result is a "mismatch". All repairs share ONE budget:
MAX_REPAIR_ATTEMPTS. Fatal errors go straight to `fail`.
"""

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from analyst.critic import CriticReport, review_execution
from analyst.executor import ExecutionError, ExecutionResult, execute_plan
from analyst.llm import LLMClient, LLMError
from analyst.plan_validation import PlanValidationError
from analyst.planner import PlannerError, generate_plan
from analyst.reporter import _escape as escape_markdown
from analyst.reporter import build_report
from analyst.reviewer import ReviewError, ReviewVerdict, review_result  # NEW (Phase 10)
from analyst.schemas import AnalysisPlan

MAX_REPAIR_ATTEMPTS = 1
RECURSION_LIMIT = 25  # safety net only; the routing already bounds the loop
MAX_MESSAGE_CHARS = 400

Status = Literal["success", "unsupported", "failed"]


class AnalysisState(TypedDict, total=False):
    question: str
    df: pd.DataFrame
    plan: AnalysisPlan | None
    result: ExecutionResult | None
    critic: CriticReport | None
    review: ReviewVerdict | None  # NEW (Phase 10)
    review_error: str | None  # NEW (Phase 10): why the review was skipped
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
    message: str  # the answer (success) or a user-safe explanation
    plan: AnalysisPlan | None = None
    result: ExecutionResult | None = None
    critic: CriticReport | None = None
    repair_attempts: int = 0
    review: ReviewVerdict | None = None  # NEW (Phase 10)
    review_error: str | None = None  # NEW (Phase 10)


# ---------------------------------------------------------------- routing
# Routers are plain functions of the state, so they are easy to unit-test.


def _next_after_error_check(state: AnalysisState, ok_node: str) -> str:
    kind = state.get("error_kind")
    if kind is None:
        return ok_node
    if kind == "recoverable" and state.get("repair_attempts", 0) < MAX_REPAIR_ATTEMPTS:
        return "repair"
    return "fail"


def route_after_planner(state: AnalysisState) -> str:
    if state.get("status") == "unsupported":
        return END
    return _next_after_error_check(state, "executor")


def route_after_executor(state: AnalysisState) -> str:
    return _next_after_error_check(state, "critic")


def route_after_critic(state: AnalysisState) -> str:
    return _next_after_error_check(state, "reporter")


def route_after_reviewer(state: AnalysisState) -> str:  # NEW (Phase 10)
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
        "critic": None,  # NEW (Phase 10): stale after a repair
        "review": None,  # NEW (Phase 10)
        "review_error": None,  # NEW (Phase 10)
    }


def fail_node(state: AnalysisState) -> dict:
    text = state.get("error") or "The analysis failed."
    if state.get("repair_attempts", 0) > 0:
        text += " An automatic repair attempt was also made."
    # Error text can contain column names or LLM text, so make it markdown-safe.
    return {"status": "failed", "message": escape_markdown(text, MAX_MESSAGE_CHARS)}


def build_graph(client: LLMClient, reviewer_client: LLMClient | None = None):
    """Build and compile the workflow. LLM clients are injected (testable).

    reviewer_client=None means: no Reviewer node at all.
    """

    def planner_node(state: AnalysisState) -> dict:
        try:
            plan = generate_plan(
                state["question"],
                state["df"],
                client,
                feedback=state.get("repair_feedback"),
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
            return {
                "error": str(exc),
                "error_detail": str(exc),
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

    def reviewer_node(state: AnalysisState) -> dict:  # NEW (Phase 10)
        try:
            verdict = review_result(
                state["question"], state["plan"], state["result"], reviewer_client
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

    graph = StateGraph(AnalysisState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critic", critic_node)
    graph.add_node("reporter", reporter_node)
    graph.add_node("repair", prepare_repair)
    graph.add_node("fail", fail_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_after_planner, ["executor", "repair", "fail", END])
    graph.add_conditional_edges("executor", route_after_executor, ["critic", "repair", "fail"])

    if reviewer_client is None:
        graph.add_conditional_edges("critic", route_after_critic, ["reporter", "fail"])
    else:  # NEW (Phase 10)
        graph.add_node("reviewer", reviewer_node)
        # path map: the router still returns "reporter"; we send it to the reviewer first.
        graph.add_conditional_edges(
            "critic", route_after_critic, {"reporter": "reviewer", "fail": "fail"}
        )
        graph.add_conditional_edges("reviewer", route_after_reviewer, ["reporter", "repair"])

    graph.add_edge("repair", "planner")
    graph.add_edge("reporter", END)
    graph.add_edge("fail", END)
    return graph.compile()


def run_analysis(
    question: str,
    df: pd.DataFrame,
    client: LLMClient,
    reviewer_client: LLMClient | None = None,
) -> AnalysisOutcome:
    """Run the whole workflow. Never raises: failures become status='failed'."""
    initial: AnalysisState = {"question": question, "df": df, "repair_attempts": 0}
    try:
        final = build_graph(client, reviewer_client).invoke(
            initial, config={"recursion_limit": RECURSION_LIMIT}
        )
    except Exception as exc:  # last-resort boundary; only the type is shown
        return AnalysisOutcome(
            status="failed",
            message=f"Unexpected internal error ({type(exc).__name__}).",
        )

    review = final.get("review")
    if review is not None:  # NEW (Phase 10): reviewer text is LLM text, so escape it
        review = review.model_copy(
            update={"reason": escape_markdown(review.reason, MAX_MESSAGE_CHARS)}
        )
    return AnalysisOutcome(
        status=final.get("status", "failed"),
        message=final.get("message", "The analysis failed."),
        plan=final.get("plan"),
        result=final.get("result"),
        critic=final.get("critic"),
        repair_attempts=final.get("repair_attempts", 0),
        review=review,
        review_error=final.get("review_error"),
    )