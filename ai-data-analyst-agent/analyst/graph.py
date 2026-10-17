"""LangGraph workflow: Planner -> Executor -> Critic -> Reporter, with one repair.

Flow:
    START -> planner -> executor -> critic -> reporter -> END
A recoverable error (bad plan, plan does not fit the data) goes to `repair`,
which sends the problem back to the planner. Repair happens at most
MAX_REPAIR_ATTEMPTS times. Fatal errors go straight to `fail`.
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
    }


def fail_node(state: AnalysisState) -> dict:
    text = state.get("error") or "The analysis failed."
    if state.get("repair_attempts", 0) > 0:
        text += " An automatic repair attempt was also made."
    # Error text can contain column names or LLM text, so make it markdown-safe.
    return {"status": "failed", "message": escape_markdown(text, MAX_MESSAGE_CHARS)}


def build_graph(client: LLMClient):
    """Build and compile the workflow. The LLM client is injected (testable)."""

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
    graph.add_conditional_edges("critic", route_after_critic, ["reporter", "fail"])
    graph.add_edge("repair", "planner")
    graph.add_edge("reporter", END)
    graph.add_edge("fail", END)
    return graph.compile()


def run_analysis(question: str, df: pd.DataFrame, client: LLMClient) -> AnalysisOutcome:
    """Run the whole workflow. Never raises: failures become status='failed'."""
    initial: AnalysisState = {"question": question, "df": df, "repair_attempts": 0}
    try:
        final = build_graph(client).invoke(
            initial, config={"recursion_limit": RECURSION_LIMIT}
        )
    except Exception as exc:  # last-resort boundary; only the type is shown
        return AnalysisOutcome(
            status="failed",
            message=f"Unexpected internal error ({type(exc).__name__}).",
        )
    return AnalysisOutcome(
        status=final.get("status", "failed"),
        message=final.get("message", "The analysis failed."),
        plan=final.get("plan"),
        result=final.get("result"),
        critic=final.get("critic"),
        repair_attempts=final.get("repair_attempts", 0),
    )