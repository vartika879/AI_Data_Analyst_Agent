import json

import pandas as pd
import pytest
from langgraph.graph import END

from analyst.critic import CriticReport
from analyst.graph import prepare_repair, route_after_critic, route_after_executor, route_after_planner, run_analysis
from analyst.llm import LLMError


class ScriptedClient:
    """Fake LLM that returns pre-written replies, one per call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete_json(self, system_prompt, user_prompt):
        self.calls.append(user_prompt)
        if not self.replies:
            raise AssertionError("unexpected extra LLM call")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Asha", "Ravi", "Charu", "Dev"],
            "age": [30, 25, 40, 35],
            "city": ["X", "Y", "X", None],
            "salary": [10.0, 20.0, 30.0, None],
        }
    )


GOOD_AGG = json.dumps(
    {
        "operation": "aggregate",
        "group_by": ["city"],
        "aggregations": [{"function": "mean", "column": "salary"}],
        "sort_by": "salary_mean",
        "ascending": False,
    }
)
BAD_COLUMN = json.dumps({"operation": "sort", "sort_by": "height"})
NOT_JSON = "I think the answer is 42."


# ------------------------------------------------------------ happy paths


def test_successful_analysis(df):
    client = ScriptedClient(GOOD_AGG)
    outcome = run_analysis("average salary by city", df, client)
    assert outcome.status == "success"
    assert "salary_mean" in outcome.message
    assert outcome.result.total_result_rows == 3
    assert outcome.critic.passed is True
    assert outcome.repair_attempts == 0
    assert len(client.calls) == 1


def test_unsupported_question_stops_without_executing(df):
    reply = json.dumps({"operation": "unsupported", "unsupported_reason": "Forecasting."})
    client = ScriptedClient(reply)
    outcome = run_analysis("predict next year", df, client)
    assert outcome.status == "unsupported"
    assert "Forecasting" in outcome.message
    assert outcome.result is None
    assert len(client.calls) == 1


# ------------------------------------------------------------ repair logic


def test_unknown_column_is_repaired_once(df):
    client = ScriptedClient(BAD_COLUMN, GOOD_AGG)
    outcome = run_analysis("q", df, client)
    assert outcome.status == "success"
    assert outcome.repair_attempts == 1
    assert len(client.calls) == 2
    second_prompt = json.loads(client.calls[1])
    assert "height" in second_prompt["previous_attempt_problem"]


def test_repair_is_limited_to_one_attempt(df):
    client = ScriptedClient(BAD_COLUMN, BAD_COLUMN)
    outcome = run_analysis("q", df, client)
    assert outcome.status == "failed"
    assert "Unknown column" in outcome.message
    assert "repair attempt" in outcome.message
    assert outcome.repair_attempts == 1
    assert len(client.calls) == 2  # a third call would raise AssertionError


def test_malformed_llm_output_is_repaired_once(df):
    client = ScriptedClient(NOT_JSON, GOOD_AGG)
    outcome = run_analysis("q", df, client)
    assert outcome.status == "success"
    assert outcome.repair_attempts == 1


def test_malformed_llm_output_twice_fails(df):
    client = ScriptedClient(NOT_JSON, NOT_JSON)
    outcome = run_analysis("q", df, client)
    assert outcome.status == "failed"
    assert len(client.calls) == 2


def test_execution_error_is_repaired_once(df):
    bad_value = json.dumps(
        {
            "operation": "filter",
            "filters": [{"column": "age", "operator": "==", "value": "abc"}],
        }
    )
    client = ScriptedClient(bad_value, GOOD_AGG)
    outcome = run_analysis("q", df, client)
    assert outcome.status == "success"
    assert "not a number" in json.loads(client.calls[1])["previous_attempt_problem"]


# ------------------------------------------------------------ fatal paths


def test_llm_failure_is_not_retried(df):
    client = ScriptedClient(LLMError("Groq rate limit reached."))
    outcome = run_analysis("q", df, client)
    assert outcome.status == "failed"
    assert "rate limit" in outcome.message
    assert outcome.repair_attempts == 0
    assert len(client.calls) == 1


def test_empty_question_fails_without_calling_llm(df):
    client = ScriptedClient(GOOD_AGG)
    outcome = run_analysis("   ", df, client)
    assert outcome.status == "failed"
    assert "enter a question" in outcome.message
    assert client.calls == []


def test_critic_failure_blocks_the_answer(df, monkeypatch):
    def failing_review(plan, result, data):
        return CriticReport(passed=False, issues=("x",), checks_run=1)

    monkeypatch.setattr("analyst.graph.review_execution", failing_review)
    client = ScriptedClient(GOOD_AGG)
    outcome = run_analysis("q", df, client)
    assert outcome.status == "failed"
    assert "consistency" in outcome.message
    assert outcome.critic.issues == ("x",)
    assert outcome.repair_attempts == 0
    assert len(client.calls) == 1


def test_unexpected_exception_becomes_failed_outcome(df, monkeypatch):
    def boom(plan, data):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr("analyst.graph.execute_plan", boom)
    outcome = run_analysis("q", df, ScriptedClient(GOOD_AGG))
    assert outcome.status == "failed"
    assert "RuntimeError" in outcome.message
    assert "secret internal detail" not in outcome.message


# ------------------------------------------------------------ safe display


EVIL = "![x](http://evil.test/a.png)"


def test_markdown_in_failure_message_is_neutralised(df):
    reply = json.dumps({"operation": "sort", "sort_by": EVIL})
    outcome = run_analysis("q", df, ScriptedClient(reply, reply))
    assert outcome.status == "failed"
    assert "![" not in outcome.message
    assert "](" not in outcome.message


def test_markdown_in_unsupported_reason_is_neutralised(df):
    reply = json.dumps({"operation": "unsupported", "unsupported_reason": EVIL})
    outcome = run_analysis("q", df, ScriptedClient(reply))
    assert outcome.status == "unsupported"
    assert "![" not in outcome.message
    assert "](" not in outcome.message


# ------------------------------------------------------------ routers


def test_planner_router_unsupported_ends():
    assert route_after_planner({"status": "unsupported"}) == END


def test_planner_router_recoverable_goes_to_repair_first():
    state = {"error_kind": "recoverable", "repair_attempts": 0}
    assert route_after_planner(state) == "repair"


def test_planner_router_recoverable_after_repair_fails():
    state = {"error_kind": "recoverable", "repair_attempts": 1}
    assert route_after_planner(state) == "fail"


def test_planner_router_fatal_fails():
    assert route_after_planner({"error_kind": "fatal", "repair_attempts": 0}) == "fail"


def test_planner_router_no_error_goes_to_executor():
    assert route_after_planner({"repair_attempts": 0}) == "executor"


def test_executor_router():
    assert route_after_executor({}) == "critic"
    assert route_after_executor({"error_kind": "recoverable", "repair_attempts": 0}) == "repair"


def test_critic_router():
    assert route_after_critic({}) == "reporter"
    assert route_after_critic({"error_kind": "fatal"}) == "fail"


# ------------------------------------------------------------ repair node


def test_prepare_repair_resets_state_and_keeps_feedback():
    update = prepare_repair(
        {"repair_attempts": 0, "error": "e", "error_detail": "Unknown column x", "error_kind": "recoverable"}
    )
    assert update["repair_attempts"] == 1
    assert update["repair_feedback"] == "Unknown column x"
    assert update["error"] is None
    assert update["error_kind"] is None
    assert update["plan"] is None