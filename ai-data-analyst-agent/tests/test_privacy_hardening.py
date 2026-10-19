"""Privacy hardening: hidden-column questions, redact(), and every outbound LLM request.

All data is synthetic. No real API is used: the Groq SDK is replaced by a recorder.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from analyst.clarifier import build_clarifier_prompt, combine_question, decide_clarification
from analyst.graph import build_graph, run_analysis
from analyst.llm import GroqClient
from analyst.planner import build_user_prompt, generate_plan
from analyst.privacy import (
    HiddenColumnError,
    PrivacyPolicy,
    ensure_no_hidden_mentions,
    find_hidden_mentions,
    tracing_disabled,
)
from analyst.reviewer import build_review_prompt, review_result

HIDDEN_NAME = "employee name"  # hidden column (contains a space on purpose)
SECRET_VALUES = ["Asha Verma", "Zanzibarville"]


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            HIDDEN_NAME: ["Asha Verma", "Ravi Rao", "Charu Jain"],
            "city": ["Zanzibarville", "Pune", "Zanzibarville"],
            "salary": [10.0, 20.0, 30.0],
        }
    )


class ScriptedClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if not self.replies:
            raise AssertionError("unexpected extra LLM call")
        return self.replies.pop(0)


class RecordingSDK:
    """Stands in for the Groq SDK client: records every request that would leave the machine."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


AGG = json.dumps(
    {
        "operation": "aggregate",
        "group_by": ["city"],
        "aggregations": [{"function": "mean", "column": "salary"}],
    }
)
BAD_SORT = json.dumps({"operation": "sort", "sort_by": "height"})
PROCEED = json.dumps({"action": "proceed"})
ANSWERS = json.dumps({"verdict": "answers", "reason": "ok"})
MISMATCH = json.dumps({"verdict": "mismatch", "reason": "wrong measure"})


# ============================================================ 1. matching rules
@pytest.mark.parametrize(
    "question",
    [
        "Show salary by city",
        "SHOW SALARY BY CITY",
        "average Salary per city",
        "what is the salary?",
        "salary, city",
        "(salary)",
        "the salary's total",
    ],
)
def test_case_and_punctuation_around_a_hidden_name(question):
    assert find_hidden_mentions(question, ["salary"]) == ["salary"]


@pytest.mark.parametrize(
    "question",
    ["employee name", "Employee Name", "employee_name", "Employee-Name", "EMPLOYEE   NAME", "employee names"],
)
def test_names_with_spaces_match_space_underscore_and_hyphen(question):
    assert find_hidden_mentions(f"list {question} by city", [HIDDEN_NAME]) == [HIDDEN_NAME]


@pytest.mark.parametrize(
    "hidden, question",
    [
        ("age", "average salary by city"),
        ("age", "manage the page message"),
        ("id", "paid valid idea"),
        ("salary", "salary_band by city"),  # a different (visible) column name
        ("salary", "salaryx"),
        ("name", "rename the surname"),
        ("tax", "syntax"),
    ],
)
def test_no_partial_word_false_positives(hidden, question):
    assert find_hidden_mentions(question, [hidden]) == []


def test_overlapping_hidden_names_are_both_reported():
    found = find_hidden_mentions("salary total per city", ["salary", "salary total"])
    assert found == ["salary", "salary total"]
    assert find_hidden_mentions("salary_total per city", ["salary", "salary total"]) == ["salary total"]


def test_multiple_hidden_columns():
    found = find_hidden_mentions("Compare Salary and EMPLOYEE_NAME", ["salary", HIDDEN_NAME, "city"])
    assert found == [HIDDEN_NAME, "salary"]


def test_names_with_punctuation_and_brackets():
    assert find_hidden_mentions("show Salary (USD) by city", ["salary (USD)"]) == ["salary (USD)"]
    assert find_hidden_mentions("show salary by city", ["salary (USD)"]) == ["salary (USD)"]
    assert find_hidden_mentions("revenue in q3.revenue", ["q3.revenue"]) == ["q3.revenue"]


def test_unicode_tricks_are_normalised():
    assert find_hidden_mentions("show \uff33\uff21\uff2c\uff21\uff32\uff39", ["salary"]) == ["salary"]  # full-width
    assert find_hidden_mentions("show sal\u200bary", ["salary"]) == ["salary"]  # zero-width space


@pytest.mark.parametrize("hidden", [[""], ["   "], ["%"], ["###"], ["", "  ", "%"]])
def test_empty_or_symbol_only_names_never_match_and_never_crash(hidden):
    assert find_hidden_mentions("average salary by city 100% #1", hidden) == []
    policy = PrivacyPolicy.strict(hidden)
    assert policy.redact("average salary by city 100% #1") == "average salary by city 100% #1"
    ensure_no_hidden_mentions("average salary", policy)  # does not raise


def test_empty_name_does_not_disable_other_hidden_names():
    assert find_hidden_mentions("show salary", ["", "salary"]) == ["salary"]


def test_regex_metacharacters_in_names_are_literal():
    assert find_hidden_mentions("use a+b here", ["a+b"]) == ["a+b"]
    assert find_hidden_mentions("use aab here", ["a+b"]) == []
    assert find_hidden_mentions("the (x) value", ["(x)"]) == ["(x)"]


def test_numeric_column_names_are_handled_as_text():
    policy = PrivacyPolicy.strict([2023])
    assert not policy.visible(2023)
    assert find_hidden_mentions("revenue in 2023", [2023]) == ["2023"]
    assert find_hidden_mentions("revenue in 12023", [2023]) == []


def test_known_limits_synonyms_and_irregular_plurals_are_not_detected():
    # Documented limit: only the column NAME is matched, not what it means.
    assert find_hidden_mentions("show pay by city", ["salary"]) == []
    assert find_hidden_mentions("show salaries by city", ["salary"]) == []


# ================================================================== 2. redact()
def test_redact_is_case_insensitive():
    p = PrivacyPolicy.strict(["Salary"])
    assert p.redact("bad SALARY and salary and Salary") == "bad [hidden] and [hidden] and [hidden]"


def test_redact_names_with_spaces():
    p = PrivacyPolicy.strict([HIDDEN_NAME])
    assert p.redact("no Employee_Name; EMPLOYEE NAME; employee-name") == "no [hidden]; [hidden]; [hidden]"


def test_redact_overlapping_names_longest_wins():
    p = PrivacyPolicy.strict(["salary", "salary total"])
    assert p.redact("salary total, salary") == "[hidden], [hidden]"


def test_redact_has_no_partial_word_false_positives():
    p = PrivacyPolicy.strict(["age"])
    assert p.redact("average age, manage, page") == "average [hidden], manage, page"


def test_redact_multiple_hidden_columns():
    p = PrivacyPolicy.strict(["salary", HIDDEN_NAME])
    text = "Unknown column. Available columns: employee name, city, salary"
    assert p.redact(text) == "Unknown column. Available columns: [hidden], city, [hidden]"


def test_redact_does_not_rescan_its_own_placeholder():
    p = PrivacyPolicy.strict(["hidden"])
    assert p.redact("a hidden column") == "a [hidden] column"


def test_redact_without_hidden_columns_returns_text_unchanged():
    assert PrivacyPolicy().redact("Salary stays") == "Salary stays"
    assert PrivacyPolicy.strict().redact("Salary stays") == "Salary stays"


# ============================ 3. a question that names a hidden column never reaches an LLM
def _clients():
    return ScriptedClient(), ScriptedClient(), ScriptedClient(), ScriptedClient()


@pytest.mark.parametrize(
    "question",
    ["Show employee name by city", "list EMPLOYEE_NAME", "Employee-Name per city", "Show salary by city"],
)
def test_hidden_question_is_rejected_and_no_llm_is_called(df, question):
    planner, reviewer, narrator, clarifier = _clients()
    policy = PrivacyPolicy.strict([HIDDEN_NAME, "salary"])
    outcome = run_analysis(question, df, planner, reviewer, narrator, clarifier, policy=policy)
    assert outcome.status == "failed"
    assert "not sent" in outcome.message
    assert outcome.plan is None and outcome.result is None
    for client in (planner, reviewer, narrator, clarifier):
        assert client.calls == []


def test_rejection_message_is_clear_and_names_the_column_for_the_user(df):
    planner = ScriptedClient()
    outcome = run_analysis("show salary", df, planner, policy=PrivacyPolicy.strict(["salary"]))
    assert "salary" in outcome.message and "hid" in outcome.message
    assert planner.calls == []


def test_question_without_the_name_is_still_processed(df):
    planner = ScriptedClient(AGG)
    outcome = run_analysis("average salary by city", df, planner, policy=PrivacyPolicy.strict([HIDDEN_NAME]))
    assert outcome.status == "success" and len(planner.calls) == 1


def test_clarification_option_cannot_smuggle_a_hidden_name(df):
    # Alternate path: the UI joins the original question and an LLM-written option.
    question = combine_question("Show totals by city", f"Use {HIDDEN_NAME}")
    planner, reviewer, narrator, clarifier = _clients()
    outcome = run_analysis(
        question, df, planner, reviewer, narrator, clarifier, policy=PrivacyPolicy.strict([HIDDEN_NAME])
    )
    assert outcome.status == "failed"
    for client in (planner, reviewer, narrator, clarifier):
        assert client.calls == []


def test_build_graph_used_directly_is_protected_too(df):
    planner, clarifier = ScriptedClient(), ScriptedClient()
    graph = build_graph(planner, clarifier_client=clarifier, policy=PrivacyPolicy.strict(["salary"]))
    final = graph.invoke({"question": "salary by city", "df": df, "repair_attempts": 0})
    assert final["status"] == "failed"
    assert planner.calls == [] and clarifier.calls == []


def test_prompt_builders_refuse_a_hidden_question_without_leaking_the_name(df):
    policy = PrivacyPolicy.strict(["salary"])
    for build in (
        lambda: build_user_prompt("salary by city", df, None, policy),
        lambda: build_clarifier_prompt("salary by city", df, policy),
        lambda: build_review_prompt("salary by city", None, None, policy),
    ):
        with pytest.raises(HiddenColumnError) as info:
            build()
        assert "salary" not in str(info.value).lower()
        assert info.value.columns == ("salary",)


def test_generate_plan_decide_clarification_and_review_do_not_call_the_client(df):
    policy = PrivacyPolicy.strict(["salary"])
    client = ScriptedClient()
    with pytest.raises(HiddenColumnError):
        generate_plan("salary by city", df, client, policy=policy)
    with pytest.raises(HiddenColumnError):
        decide_clarification("salary by city", df, client, policy)
    with pytest.raises(HiddenColumnError):
        review_result("salary by city", None, None, client, policy)
    assert client.calls == []


def test_safety_net_turns_a_low_level_guard_into_a_clean_failure(df, monkeypatch):
    def boom(*args, **kwargs):
        raise HiddenColumnError(["salary"])

    monkeypatch.setattr("analyst.graph.generate_plan", boom)
    outcome = run_analysis("q", df, ScriptedClient(), policy=PrivacyPolicy.strict(["salary"]))
    assert outcome.status == "failed" and "not sent" in outcome.message


def test_plan_that_guesses_a_hidden_column_is_not_run_and_repair_does_not_leak(df):
    uses_hidden = json.dumps({"operation": "sort", "sort_by": HIDDEN_NAME})
    planner = ScriptedClient(uses_hidden, uses_hidden)
    outcome = run_analysis("sort the rows", df, planner, policy=PrivacyPolicy.strict([HIDDEN_NAME]))
    assert outcome.status == "failed" and outcome.result is None
    repair_prompt = planner.calls[1][1].lower()
    assert HIDDEN_NAME not in repair_prompt and "employee_name" not in repair_prompt


# ===================== 4. every real outbound request (Groq SDK replaced by a recorder)
def _run_with_real_clients(df, question, planner_replies, policy, *, reviewer_replies=(), narrator_replies=(),
                           clarifier_replies=(PROCEED,)):
    sdks = {
        "planner": RecordingSDK(*planner_replies),
        "reviewer": RecordingSDK(*reviewer_replies),
        "narrator": RecordingSDK(*narrator_replies),
        "clarifier": RecordingSDK(*clarifier_replies),
    }
    clients = {role: GroqClient(None, f"model-{role}", sdk_client=sdk) for role, sdk in sdks.items()}
    outcome = run_analysis(
        question,
        df,
        clients["planner"],
        reviewer_client=clients["reviewer"],
        narrator_client=clients["narrator"],
        clarifier_client=clients["clarifier"],
        policy=policy,
    )
    return outcome, sdks


def _all_outbound_text(sdks) -> str:
    return json.dumps([r for sdk in sdks.values() for r in sdk.requests], ensure_ascii=False).lower()


def _assert_no_secret_in_outbound(sdks):
    text = _all_outbound_text(sdks)
    for secret in [HIDDEN_NAME, "employee_name", "employee-name", "ravi rao", "charu jain", "pune", *map(str.lower, SECRET_VALUES)]:
        assert secret not in text, f"leaked: {secret!r}"


def test_outbound_requests_success_path(df):
    policy = PrivacyPolicy.strict([HIDDEN_NAME])
    outcome, sdks = _run_with_real_clients(
        df, "average salary by city", [AGG], policy, reviewer_replies=[ANSWERS]
    )
    assert outcome.status == "success"
    _assert_no_secret_in_outbound(sdks)
    assert sdks["narrator"].requests == []  # the Narrator needs rows, so it never runs
    assert len(sdks["planner"].requests) == 1 and len(sdks["clarifier"].requests) == 1
    assert len(sdks["reviewer"].requests) == 1
    assert "Zanzibarville" in outcome.result.table.to_string()  # the user still sees real data


def test_outbound_requests_executor_error_repair_path(df):
    policy = PrivacyPolicy.strict([HIDDEN_NAME])
    outcome, sdks = _run_with_real_clients(
        df, "sort the rows by height", [BAD_SORT, AGG], policy, reviewer_replies=[ANSWERS]
    )
    assert outcome.status == "success" and outcome.repair_attempts == 1
    _assert_no_secret_in_outbound(sdks)
    repair = json.loads(sdks["planner"].requests[1]["messages"][1]["content"])
    assert "[hidden]" in repair["previous_attempt_problem"]  # "Available columns" was redacted


def test_outbound_requests_reviewer_mismatch_repair_path(df):
    policy = PrivacyPolicy.strict([HIDDEN_NAME])
    outcome, sdks = _run_with_real_clients(
        df, "average salary by city", [AGG, AGG], policy, reviewer_replies=[MISMATCH, ANSWERS]
    )
    assert outcome.status == "success" and outcome.repair_attempts == 1
    _assert_no_secret_in_outbound(sdks)


def test_outbound_requests_when_the_question_names_a_hidden_column(df):
    policy = PrivacyPolicy.strict([HIDDEN_NAME])
    outcome, sdks = _run_with_real_clients(df, "list Employee Name by city", [], policy)
    assert outcome.status == "failed"
    assert all(sdk.requests == [] for sdk in sdks.values())


def test_outbound_request_carries_only_the_expected_fields(df):
    outcome, sdks = _run_with_real_clients(
        df, "average salary by city", [AGG], PrivacyPolicy.strict(), reviewer_replies=[ANSWERS]
    )
    for sdk in sdks.values():
        for request in sdk.requests:
            assert set(request) == {"model", "messages", "temperature", "max_completion_tokens", "response_format"}
            assert [m["role"] for m in request["messages"]] == ["system", "user"]


def test_what_the_ui_previews_is_what_is_actually_sent(df):
    policy = PrivacyPolicy.strict([HIDDEN_NAME])
    question = "average salary by city"
    outcome, sdks = _run_with_real_clients(df, question, [AGG], policy, reviewer_replies=[ANSWERS])

    sent_planner = sdks["planner"].requests[0]["messages"][1]["content"]
    assert sent_planner == build_user_prompt(question, df, None, policy)
    sent_clarifier = sdks["clarifier"].requests[0]["messages"][1]["content"]
    assert sent_clarifier == build_clarifier_prompt(question, df, policy)
    assert json.loads(sent_clarifier) == json.loads(sent_planner)
    sent_reviewer = sdks["reviewer"].requests[0]["messages"][1]["content"]
    assert sent_reviewer == build_review_prompt(question, outcome.plan, outcome.result, policy)


def test_preview_matches_request_when_privacy_is_off(df):
    question = "average salary by city"
    outcome, sdks = _run_with_real_clients(df, question, [AGG], None, reviewer_replies=[ANSWERS])
    sent = sdks["planner"].requests[0]["messages"][1]["content"]
    assert sent == build_user_prompt(question, df, None, None)
    assert "Zanzibarville" in sent  # off means samples ARE sent; the UI warns about this


def test_question_text_is_sent_as_typed_in_privacy_mode(df):
    # Documented limit: the question itself is not masked.
    question = "average salary by city for Zanzibarville"
    outcome, sdks = _run_with_real_clients(df, question, [AGG], PrivacyPolicy.strict(), reviewer_replies=[ANSWERS])
    assert json.loads(sdks["planner"].requests[0]["messages"][1]["content"])["question"] == question
    assert "zanzibarville" in _all_outbound_text(sdks)


# ======================================================== 5. evaluation call path
def test_eval_private_config_sends_no_data_values_and_baseline_does(monkeypatch):
    from evals import runner
    from evals.cases import CASES
    from evals.dataset import csv_bytes

    df = pd.read_csv(pd.io.common.BytesIO(csv_bytes()))
    seen: dict[str, list[str]] = {}

    def factory_for(label):
        def factory(role):
            client = ScriptedClient()
            client.complete_json = lambda s, u, _l=label: (
                seen.setdefault(_l, []).append(u) or json.dumps({"operation": "column_summary"})
            )
            return client

        return factory

    for config in ("baseline", "private"):
        runner.run_case(CASES[0], df, runner.CONFIGS[config], factory_for(config), config=config)
    assert any("North" in prompt for prompt in seen["baseline"])  # synthetic samples are sent
    assert seen["private"], "private config made no LLM call"
    assert not any("North" in prompt or "Pen" in prompt for prompt in seen["private"])


# ===================================================== 6. tracing (LangSmith) switch
def test_tracing_is_switched_off_under_a_limiting_policy(df, monkeypatch):
    ls_utils = pytest.importorskip("langsmith.utils")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    for key in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):  # so nothing can ever be uploaded from a test
        monkeypatch.delenv(key, raising=False)
    ls_utils.get_env_var.cache_clear()
    observed = []

    class Probe:
        def complete_json(self, system_prompt, user_prompt):
            observed.append(ls_utils.tracing_is_enabled())
            return AGG

    try:
        assert ls_utils.tracing_is_enabled() is True  # control: the env flag really turns tracing on
        with tracing_disabled(PrivacyPolicy()):  # default policy limits nothing
            assert ls_utils.tracing_is_enabled() is True
        outcome = run_analysis("average salary by city", df, Probe(), policy=PrivacyPolicy.strict())
    finally:
        ls_utils.get_env_var.cache_clear()
    assert outcome.status == "success"
    assert observed == [False]  # tracing was off while the pipeline ran


def test_default_policy_does_not_touch_tracing():
    assert not PrivacyPolicy().limits_sharing
    assert PrivacyPolicy.strict().limits_sharing
    assert PrivacyPolicy(share_samples=False).limits_sharing


# ============================================================ 7. config hygiene
def test_env_example_has_a_single_privacy_mode_assignment():
    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    assert len(re.findall(r"^PRIVACY_MODE=", text, flags=re.MULTILINE)) == 1


def test_tracing_import_failure_fails_closed_without_llm_call(df, monkeypatch):
    import builtins

    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "langsmith" or name.startswith("langsmith."):
            raise ImportError("simulated missing LangSmith")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    client = ScriptedClient(AGG)
    outcome = run_analysis("average salary by city", df, client, policy=PrivacyPolicy.strict())

    assert outcome.status == "failed"
    assert "RuntimeError" in outcome.message
    assert client.calls == []


def test_tracing_context_type_error_fails_closed_without_llm_call(df, monkeypatch):
    import langsmith

    def unsupported_context(*args, **kwargs):
        raise TypeError("simulated unsupported tracing API")

    monkeypatch.setattr(langsmith, "tracing_context", unsupported_context, raising=False)
    client = ScriptedClient(AGG)
    outcome = run_analysis("average salary by city", df, client, policy=PrivacyPolicy.strict())

    assert outcome.status == "failed"
    assert "RuntimeError" in outcome.message
    assert client.calls == []
