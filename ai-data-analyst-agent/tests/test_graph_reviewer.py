import json

import pandas as pd
import pytest

from analyst.critic import CriticReport
from analyst.graph import route_after_reviewer, run_analysis
from analyst.llm import LLMError
from analyst.reviewer import ReviewVerdict


class ScriptedClient:
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
    }
)
WRONG_SORT = json.dumps({"operation": "sort", "sort_by": "salary"})
BAD_COLUMN = json.dumps({"operation": "sort", "sort_by": "height"})


def review(verdict, reason="because"):
    return json.dumps({"verdict": verdict, "reason": reason})


def test_answers_verdict_passes_through(df):
    planner, reviewer = ScriptedClient(GOOD_AGG), ScriptedClient(review("answers"))
    outcome = run_analysis("q", df, planner, reviewer)
    assert outcome.status == "success"
    assert outcome.review.verdict == "answers"
    assert outcome.repair_attempts == 0
    assert len(planner.calls) == 1 and len(reviewer.calls) == 1


def test_partial_verdict_still_shows_the_answer(df):
    outcome = run_analysis("q", df, ScriptedClient(GOOD_AGG), ScriptedClient(review("partial")))
    assert outcome.status == "success"
    assert outcome.review.verdict == "partial"


def test_mismatch_triggers_repair_with_reviewer_reason(df):
    planner = ScriptedClient(WRONG_SORT, GOOD_AGG)
    reviewer = ScriptedClient(review("mismatch", "The plan sorts instead of averaging."), review("answers"))
    outcome = run_analysis("average salary by city", df, planner, reviewer)
    assert outcome.status == "success"
    assert outcome.repair_attempts == 1
    assert outcome.review.verdict == "answers"  # the review of the NEW result
    assert outcome.plan.operation == "aggregate"
    feedback = json.loads(planner.calls[1])["previous_attempt_problem"]
    assert "averaging" in feedback
    assert len(reviewer.calls) == 2


def test_repeated_mismatch_is_shown_with_warning_not_hidden(df):
    planner = ScriptedClient(WRONG_SORT, WRONG_SORT)
    reviewer = ScriptedClient(review("mismatch"), review("mismatch"))
    outcome = run_analysis("q", df, planner, reviewer)
    assert outcome.status == "success"
    assert outcome.review.verdict == "mismatch"
    assert outcome.repair_attempts == 1
    assert len(planner.calls) == 2  # no second repair


def test_repair_budget_is_shared_with_planner_errors(df):
    planner = ScriptedClient(BAD_COLUMN, GOOD_AGG)  # one repair already used here
    reviewer = ScriptedClient(review("mismatch"))
    outcome = run_analysis("q", df, planner, reviewer)
    assert outcome.status == "success"
    assert outcome.review.verdict == "mismatch"
    assert outcome.repair_attempts == 1
    assert len(planner.calls) == 2  # the reviewer could not trigger a second repair
    assert len(reviewer.calls) == 1


def test_reviewer_api_failure_does_not_block_the_answer(df):
    reviewer = ScriptedClient(LLMError("Groq rate limit reached."))
    outcome = run_analysis("q", df, ScriptedClient(GOOD_AGG), reviewer)
    assert outcome.status == "success"
    assert outcome.review is None
    assert "rate limit" in outcome.review_error
    assert outcome.repair_attempts == 0


def test_malformed_reviewer_output_does_not_block_the_answer(df):
    outcome = run_analysis("q", df, ScriptedClient(GOOD_AGG), ScriptedClient("not json"))
    assert outcome.status == "success"
    assert outcome.review is None
    assert "usable verdict" in outcome.review_error


def test_without_reviewer_client_nothing_changes(df):
    outcome = run_analysis("q", df, ScriptedClient(GOOD_AGG))
    assert outcome.status == "success"
    assert outcome.review is None
    assert outcome.review_error is None


def test_critic_failure_skips_the_reviewer(df, monkeypatch):
    monkeypatch.setattr(
        "analyst.graph.review_execution",
        lambda plan, result, data: CriticReport(passed=False, issues=("x",), checks_run=1),
    )
    reviewer = ScriptedClient(review("answers"))
    outcome = run_analysis("q", df, ScriptedClient(GOOD_AGG), reviewer)
    assert outcome.status == "failed"
    assert reviewer.calls == []


def test_unsupported_question_skips_the_reviewer(df):
    reply = json.dumps({"operation": "unsupported", "unsupported_reason": "Forecasting."})
    reviewer = ScriptedClient(review("answers"))
    outcome = run_analysis("predict", df, ScriptedClient(reply), reviewer)
    assert outcome.status == "unsupported"
    assert reviewer.calls == []


def test_reviewer_text_is_markdown_safe(df):
    evil = "![x](http://evil.test/a.png)"
    outcome = run_analysis("q", df, ScriptedClient(GOOD_AGG), ScriptedClient(review("answers", evil)))
    assert "![" not in outcome.review.reason
    assert "](" not in outcome.review.reason


# ------------------------------------------------------------ router


def verdict(name):
    return ReviewVerdict(verdict=name, reason="r")


def test_router_mismatch_with_budget_repairs():
    state = {"review": verdict("mismatch"), "repair_attempts": 0}
    assert route_after_reviewer(state) == "repair"


def test_router_mismatch_without_budget_reports():
    state = {"review": verdict("mismatch"), "repair_attempts": 1}
    assert route_after_reviewer(state) == "reporter"


def test_router_answers_and_partial_report():
    assert route_after_reviewer({"review": verdict("answers"), "repair_attempts": 0}) == "reporter"
    assert route_after_reviewer({"review": verdict("partial"), "repair_attempts": 0}) == "reporter"


def test_router_skipped_review_reports():
    assert route_after_reviewer({"review": None, "repair_attempts": 0}) == "reporter"