import json

import pandas as pd
import pytest

from analyst.clarifier import combine_question
from analyst.graph import route_after_clarifier, run_analysis
from analyst.llm import LLMError
from langgraph.graph import END


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
BAD_COLUMN = json.dumps({"operation": "sort", "sort_by": "height"})
PROCEED = json.dumps({"action": "proceed"})
ASK = json.dumps(
    {"action": "ask", "question": "Which column do you mean?", "options": ["salary", "age"]}
)


def test_ask_stops_the_run_before_the_planner(df):
    clarifier, planner = ScriptedClient(ASK), ScriptedClient()
    outcome = run_analysis("average pay", df, planner, clarifier_client=clarifier)
    assert outcome.status == "clarify"
    assert outcome.clarification.options == ["salary", "age"]
    assert outcome.clarification.question == "Which column do you mean?"
    assert outcome.result is None
    assert planner.calls == []


def test_proceed_continues_to_the_planner(df):
    clarifier, planner = ScriptedClient(PROCEED), ScriptedClient(GOOD_AGG)
    outcome = run_analysis("q", df, planner, clarifier_client=clarifier)
    assert outcome.status == "success"
    assert outcome.clarification is None
    assert len(clarifier.calls) == 1 and len(planner.calls) == 1


def test_clarifier_api_failure_does_not_block_the_question(df):
    clarifier = ScriptedClient(LLMError("Groq rate limit reached."))
    outcome = run_analysis("q", df, ScriptedClient(GOOD_AGG), clarifier_client=clarifier)
    assert outcome.status == "success"
    assert "rate limit" in outcome.clarify_error


def test_malformed_clarifier_output_does_not_block_the_question(df):
    outcome = run_analysis(
        "q", df, ScriptedClient(GOOD_AGG), clarifier_client=ScriptedClient("not json")
    )
    assert outcome.status == "success"
    assert "usable decision" in outcome.clarify_error


def test_without_clarifier_client_nothing_changes(df):
    outcome = run_analysis("q", df, ScriptedClient(GOOD_AGG))
    assert outcome.status == "success"
    assert outcome.clarification is None
    assert outcome.clarify_error is None


def test_empty_question_never_reaches_the_clarifier(df):
    clarifier = ScriptedClient(ASK)
    outcome = run_analysis("   ", df, ScriptedClient(GOOD_AGG), clarifier_client=clarifier)
    assert outcome.status == "failed"
    assert "enter a question" in outcome.message
    assert clarifier.calls == []


def test_repair_does_not_call_the_clarifier_again(df):
    clarifier = ScriptedClient(PROCEED)  # a second call would raise AssertionError
    planner = ScriptedClient(BAD_COLUMN, GOOD_AGG)
    outcome = run_analysis("q", df, planner, clarifier_client=clarifier)
    assert outcome.status == "success"
    assert outcome.repair_attempts == 1
    assert len(clarifier.calls) == 1


def test_two_step_flow_with_the_users_choice(df):
    first = run_analysis("average pay", df, ScriptedClient(), clarifier_client=ScriptedClient(ASK))
    assert first.status == "clarify"

    combined = combine_question("average pay", first.clarification.options[0])
    planner = ScriptedClient(GOOD_AGG)
    second = run_analysis(combined, df, planner)  # no clarifier: one round at most
    assert second.status == "success"
    assert json.loads(planner.calls[0])["question"] == "average pay\nClarification: salary"


def test_ask_skips_reviewer_and_narrator(df):
    reviewer, narrator = ScriptedClient(), ScriptedClient()
    outcome = run_analysis(
        "q", df, ScriptedClient(), reviewer, narrator, clarifier_client=ScriptedClient(ASK)
    )
    assert outcome.status == "clarify"
    assert reviewer.calls == [] and narrator.calls == []


def test_all_four_agents_together(df):
    clarifier = ScriptedClient(PROCEED)
    planner = ScriptedClient(GOOD_AGG)
    reviewer = ScriptedClient(json.dumps({"verdict": "answers", "reason": "ok"}))
    narrator = ScriptedClient(json.dumps({"answer": "Groups X and Y both average 20."}))
    outcome = run_analysis("q", df, planner, reviewer, narrator, clarifier)
    assert outcome.status == "success"
    assert outcome.review.verdict == "answers"
    assert outcome.narrative == "Groups X and Y both average 20."
    assert all(len(c.calls) == 1 for c in (clarifier, planner, reviewer, narrator))


def test_clarify_message_is_markdown_safe(df):
    evil = "Which? ![x](http://evil.test/a.png)"
    reply = json.dumps({"action": "ask", "question": evil, "options": ["a", "b"]})
    outcome = run_analysis("q", df, ScriptedClient(), clarifier_client=ScriptedClient(reply))
    assert outcome.status == "clarify"
    assert "![" not in outcome.message
    assert "](" not in outcome.message


def test_router_ask_ends_the_run():
    assert route_after_clarifier({"status": "clarify"}) == END


def test_router_otherwise_goes_to_the_planner():
    assert route_after_clarifier({}) == "planner"
    assert route_after_clarifier({"clarify_error": "x"}) == "planner"