import json

import pandas as pd
import pytest

from analyst.critic import CriticReport
from analyst.graph import run_analysis
from analyst.llm import LLMError


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
GROUNDED = "Groups X and Y both have an average salary of 20."


def narration(answer=GROUNDED):
    return json.dumps({"answer": answer})


def review(verdict, reason="because"):
    return json.dumps({"verdict": verdict, "reason": reason})


def test_summary_is_added_and_template_message_is_kept(df):
    narrator = ScriptedClient(narration())
    outcome = run_analysis("q", df, ScriptedClient(GOOD_AGG), narrator_client=narrator)
    assert outcome.status == "success"
    assert outcome.narrative == GROUNDED
    assert outcome.narrative_error is None
    assert "Analysed 4 of 4 rows" in outcome.message  # the template report is untouched


def test_ungrounded_summary_is_dropped_but_answer_still_shown(df):
    narrator = ScriptedClient(narration("Group X averages 25."))
    outcome = run_analysis("q", df, ScriptedClient(GOOD_AGG), narrator_client=narrator)
    assert outcome.status == "success"
    assert outcome.narrative is None
    assert "not in the result" in outcome.narrative_error
    assert "Analysed 4 of 4 rows" in outcome.message


def test_narrator_api_failure_does_not_block_the_answer(df):
    narrator = ScriptedClient(LLMError("Groq rate limit reached."))
    outcome = run_analysis("q", df, ScriptedClient(GOOD_AGG), narrator_client=narrator)
    assert outcome.status == "success"
    assert outcome.narrative is None
    assert "rate limit" in outcome.narrative_error


def test_narrator_is_skipped_when_reviewer_flags_a_mismatch(df):
    planner = ScriptedClient(WRONG_SORT, WRONG_SORT)
    reviewer = ScriptedClient(review("mismatch"), review("mismatch"))
    narrator = ScriptedClient()  # any call would raise AssertionError
    outcome = run_analysis("q", df, planner, reviewer, narrator)
    assert outcome.status == "success"
    assert outcome.narrative is None
    assert "skipped" in outcome.narrative_error
    assert narrator.calls == []


def test_unsupported_question_skips_the_narrator(df):
    reply = json.dumps({"operation": "unsupported", "unsupported_reason": "Forecasting."})
    narrator = ScriptedClient(narration())
    outcome = run_analysis("predict", df, ScriptedClient(reply), narrator_client=narrator)
    assert outcome.status == "unsupported"
    assert narrator.calls == []


def test_critic_failure_skips_the_narrator(df, monkeypatch):
    monkeypatch.setattr(
        "analyst.graph.review_execution",
        lambda plan, result, data: CriticReport(passed=False, issues=("x",), checks_run=1),
    )
    narrator = ScriptedClient(narration())
    outcome = run_analysis("q", df, ScriptedClient(GOOD_AGG), narrator_client=narrator)
    assert outcome.status == "failed"
    assert narrator.calls == []


def test_without_narrator_client_nothing_changes(df):
    outcome = run_analysis("q", df, ScriptedClient(GOOD_AGG))
    assert outcome.status == "success"
    assert outcome.narrative is None
    assert outcome.narrative_error is None


def test_narrator_runs_once_after_a_repair(df):
    planner = ScriptedClient(BAD_COLUMN, GOOD_AGG)
    narrator = ScriptedClient(narration())
    outcome = run_analysis("q", df, planner, narrator_client=narrator)
    assert outcome.status == "success"
    assert outcome.repair_attempts == 1
    assert outcome.narrative == GROUNDED
    assert len(narrator.calls) == 1


def test_narrative_is_markdown_safe(df):
    evil = "Result ready ![x](http://evil.test/a.png)"
    narrator = ScriptedClient(narration(evil))
    outcome = run_analysis("q", df, ScriptedClient(GOOD_AGG), narrator_client=narrator)
    assert outcome.narrative is not None
    assert "![" not in outcome.narrative
    assert "](" not in outcome.narrative


def test_all_three_agents_together(df):
    planner = ScriptedClient(GOOD_AGG)
    reviewer = ScriptedClient(review("answers"))
    narrator = ScriptedClient(narration())
    outcome = run_analysis("q", df, planner, reviewer, narrator)
    assert outcome.status == "success"
    assert outcome.review.verdict == "answers"
    assert outcome.narrative == GROUNDED
    assert len(planner.calls) == len(reviewer.calls) == len(narrator.calls) == 1