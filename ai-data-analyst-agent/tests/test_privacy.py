"""Privacy mode: what the LLM is (not) allowed to see. No real API calls."""

import json

import pandas as pd
import pytest

from analyst.clarifier import build_clarifier_prompt
from analyst.config import load_settings
from analyst.graph import run_analysis
from analyst.planner import build_user_prompt, describe_dataframe
from analyst.privacy import PrivacyPolicy, hidden_columns_used
from analyst.schemas import AnalysisPlan

SECRET_CITY = "Zanzibarville"
SECRET_NAME = "Asha Verma"


class ScriptedClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete_json(self, system_prompt, user_prompt):
        self.calls.append(user_prompt)
        if not self.replies:
            raise AssertionError("unexpected extra LLM call")
        return self.replies.pop(0)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [SECRET_NAME, "Ravi Rao", "Charu Jain"],
            "city": [SECRET_CITY, "Pune", SECRET_CITY],
            "salary": [10.0, 20.0, 30.0],
        }
    )


AGG = json.dumps(
    {
        "operation": "aggregate",
        "group_by": ["city"],
        "aggregations": [{"function": "mean", "column": "salary"}],
    }
)
VERDICT = json.dumps({"verdict": "answers", "reason": "ok"})


# ------------------------------------------------------------- policy basics
def test_default_policy_changes_nothing():
    p = PrivacyPolicy()
    assert p.share_samples and p.share_result_rows and p.rows_allowed
    assert p.visible("anything")


def test_strict_policy_blocks_samples_and_rows():
    p = PrivacyPolicy.strict()
    assert not p.share_samples and not p.rows_allowed


def test_hidden_columns_block_rows_even_if_sharing_is_on():
    p = PrivacyPolicy(hidden_columns=frozenset({"name"}))
    assert not p.rows_allowed


def test_redact_removes_hidden_names_longest_first():
    p = PrivacyPolicy.strict(["salary", "salary_total"])
    assert p.redact("bad: salary_total and salary") == "bad: [hidden] and [hidden]"


def test_hidden_columns_used_finds_every_place():
    p = PrivacyPolicy.strict(["name", "salary"])
    plan = AnalysisPlan.model_validate(
        {
            "operation": "aggregate",
            "group_by": ["city"],
            "filters": [{"column": "name", "operator": "==", "value": "x"}],
            "aggregations": [{"function": "mean", "column": "salary"}],
        }
    )
    assert hidden_columns_used(plan, p) == ["name", "salary"]
    assert hidden_columns_used(plan, PrivacyPolicy()) == []


# --------------------------------------------------------------- prompts
def test_describe_has_no_samples_in_strict_mode(df):
    info = describe_dataframe(df, PrivacyPolicy.strict())
    assert all("sample_values" not in c for c in info["columns"])
    assert SECRET_CITY not in json.dumps(info)


def test_describe_without_policy_still_has_samples(df):
    info = describe_dataframe(df)
    assert SECRET_CITY in json.dumps(info)


def test_hidden_column_is_not_in_planner_or_clarifier_prompt(df):
    p = PrivacyPolicy.strict(["name"])
    for prompt in (build_user_prompt("q", df, None, p), build_clarifier_prompt("q", df, p)):
        assert '"name": "name"' not in prompt
        assert SECRET_NAME not in prompt
        assert "city" in prompt


def test_repair_feedback_is_redacted(df):
    p = PrivacyPolicy.strict(["name"])
    prompt = build_user_prompt("q", df, "Unknown column name. Available: name, city", p)
    assert "name" not in json.loads(prompt)["previous_attempt_problem"]


# ----------------------------------------------------------- whole pipeline
def test_strict_run_never_sends_data_values_to_any_llm(df):
    planner, reviewer, narrator = ScriptedClient(AGG), ScriptedClient(VERDICT), ScriptedClient()
    outcome = run_analysis(
        "average salary by city", df, planner, reviewer, narrator, policy=PrivacyPolicy.strict()
    )
    assert outcome.status == "success"
    sent = " ".join(planner.calls + reviewer.calls)
    for secret in (SECRET_CITY, SECRET_NAME, "Pune"):
        assert secret not in sent
    assert narrator.calls == []  # Narrator is skipped in privacy mode
    assert outcome.narrative is None
    # The user still gets the real table.
    assert SECRET_CITY in outcome.result.table.to_string()


def test_reviewer_payload_says_rows_are_hidden(df):
    planner, reviewer = ScriptedClient(AGG), ScriptedClient(VERDICT)
    run_analysis("q", df, planner, reviewer, policy=PrivacyPolicy.strict())
    result = json.loads(reviewer.calls[0])["result"]
    assert result["rows"] == [] and result["rows_hidden"] is True


def test_old_behaviour_without_policy_sends_rows(df):
    planner, reviewer = ScriptedClient(AGG), ScriptedClient(VERDICT)
    run_analysis("q", df, planner, reviewer)
    assert SECRET_CITY in reviewer.calls[0]


def test_plan_using_a_hidden_column_is_not_run(df):
    uses_hidden = json.dumps({"operation": "sort", "sort_by": "name"})
    planner = ScriptedClient(uses_hidden, uses_hidden)
    outcome = run_analysis("q", df, planner, policy=PrivacyPolicy.strict(["name"]))
    assert outcome.status == "failed"
    assert outcome.result is None
    assert outcome.repair_attempts == 1
    assert "name" not in planner.calls[1].split("previous_attempt_problem")[1]


def test_executor_error_does_not_leak_hidden_names_on_repair(df):
    bad = json.dumps({"operation": "sort", "sort_by": "height"})
    planner = ScriptedClient(bad, AGG)
    outcome = run_analysis("q", df, planner, policy=PrivacyPolicy.strict(["name"]))
    assert outcome.status == "success"
    feedback = json.loads(planner.calls[1])["previous_attempt_problem"]
    assert "name" not in feedback.replace("[hidden]", "")


# ------------------------------------------------------------------ config
def test_privacy_mode_defaults_to_strict_and_can_be_standard():
    assert load_settings({}).privacy_mode == "strict"
    assert load_settings({"PRIVACY_MODE": "false"}).privacy_mode == "standard"
