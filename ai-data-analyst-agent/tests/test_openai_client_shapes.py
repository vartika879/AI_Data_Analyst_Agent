"""Providers answer in slightly different shapes. None may crash the app. No network."""

import json
from types import SimpleNamespace as NS

import pytest

from analyst.graph import run_analysis
from analyst.llm import LLMError, OpenAICompatClient
from evals import check_providers
from evals.cases import CASES
from evals.dataset import csv_bytes
from evals.runner import CONFIGS, run_case
from analyst.data_loader import load_csv

PROCEED = '{"action": "proceed"}'
PLAN = json.dumps({"operation": "aggregate", "aggregations": [{"function": "sum", "column": "gross_revenue"}]})


def _sdk(response):
    return NS(chat=NS(completions=NS(create=lambda **kwargs: response)))


def _client(response):
    return OpenAICompatClient(None, "m", "https://example.invalid/v1", label="Example Provider",
                              key_env="EXAMPLE_API_KEY", sdk_client=_sdk(response))


def _reply(content):
    return NS(choices=[NS(message=NS(content=content))])


def test_normal_string_content_still_works():
    assert _client(_reply(PROCEED)).complete_json("s", "u") == PROCEED


@pytest.mark.parametrize(
    "parts",
    [
        [{"type": "text", "text": PROCEED}],
        [NS(type="text", text=PROCEED)],
        [PROCEED],
        [{"type": "text", "text": '{"action": '}, {"type": "text", "text": '"proceed"}'}],
    ],
)
def test_content_given_as_a_list_of_parts_is_joined(parts):
    assert json.loads(_client(_reply(parts)).complete_json("s", "u")) == {"action": "proceed"}


@pytest.mark.parametrize(
    "response, expected_in_message",
    [
        ("<html>Bad gateway</html>", "received str"),
        (b"bytes", "received bytes"),
        ({"choices": []}, "received dict"),
        (None, "received NoneType"),
        (_reply(12345), "received int"),
    ],
)
def test_unexpected_shapes_become_a_safe_llm_error(response, expected_in_message):
    with pytest.raises(LLMError) as info:
        _client(response).complete_json("s", "u")
    assert expected_in_message in str(info.value)
    assert "<html>" not in str(info.value)  # the body is never copied into the message


@pytest.mark.parametrize(
    "response",
    [NS(choices=None), NS(choices=[]), NS(choices=[NS(message=None)]), _reply(None), _reply(""), _reply("  "), _reply([])],
)
def test_missing_or_empty_content_is_an_empty_response_error(response):
    with pytest.raises(LLMError, match="empty response"):
        _client(response).complete_json("s", "u")


def test_real_code_bugs_inside_the_sdk_call_are_still_not_hidden():
    def boom(**kwargs):
        raise AttributeError("a real bug")

    sdk = NS(chat=NS(completions=NS(create=boom)))
    with pytest.raises(AttributeError):
        OpenAICompatClient(None, "m", "https://x.invalid", sdk_client=sdk).complete_json("s", "u")


# ---- the original symptom: a provider answering with the wrong shape must not kill the run
class Scripted:
    def __init__(self, *replies):
        self.replies = list(replies)

    def complete_json(self, system_prompt, user_prompt):
        return self.replies.pop(0)


@pytest.fixture(scope="module")
def eval_df():
    return load_csv(csv_bytes(), "eval.csv", 5 * 1024 * 1024)


def test_bad_clarifier_provider_fails_open_and_the_question_is_still_answered(eval_df):
    bad_clarifier = _client("<html>oops</html>")  # real client, wrong-shaped answer
    outcome = run_analysis(
        "total gross revenue", eval_df, Scripted(PLAN), clarifier_client=bad_clarifier
    )
    assert outcome.status == "success"
    assert "unexpected format" in outcome.clarify_error
    assert "Unexpected internal error" not in outcome.message


def test_bad_reviewer_provider_does_not_fail_the_run(eval_df):
    outcome = run_analysis(
        "total gross revenue", eval_df, Scripted(PLAN), reviewer_client=_client(_reply(None))
    )
    assert outcome.status == "success" and outcome.review_error


def test_eval_record_shows_a_helper_agent_failure(eval_df):
    answerable = next(c for c in CASES if c.kind == "answerable")

    def factory(role):
        return _client("<html>oops</html>") if role == "clarifier" else Scripted(PLAN, PLAN, PLAN)

    record = run_case(answerable, eval_df, CONFIGS["clarifier"], factory, config="clarifier")
    assert record.clarify_error and "unexpected format" in record.clarify_error


# ---- evals.check_providers
def _fake_settings(monkeypatch, make):
    monkeypatch.setattr(check_providers, "load_settings", lambda: object())
    monkeypatch.setattr(check_providers, "describe", lambda s, role: f"fake:{role}")
    monkeypatch.setattr(check_providers, "missing_key_message", lambda s, roles: None)
    monkeypatch.setattr(check_providers, "make_client", make)


def test_check_providers_reports_ok_problem_and_unexpected(monkeypatch, capsys):
    class Good:
        def complete_json(self, s, u):
            return '{"ok": true}'

    class BadShape:
        def complete_json(self, s, u):
            raise LLMError("Gemini rate limit reached.")

    class Broken:
        def complete_json(self, s, u):
            raise AttributeError("kaboom")

    clients = {"planner": Good(), "clarifier": BadShape(), "reviewer": Broken(), "narrator": Good()}
    _fake_settings(monkeypatch, lambda settings, role: clients[role])
    assert check_providers.main() == 1
    out = capsys.readouterr().out
    assert "[planner  ] fake:planner\n    OK" in out
    assert "PROBLEM: Gemini rate limit reached." in out
    assert "UNEXPECTED ERROR" in out and "AttributeError: kaboom" in out


def test_check_providers_all_ok_returns_zero(monkeypatch, capsys):
    class Good:
        def complete_json(self, s, u):
            return '{"ok": true}'

    _fake_settings(monkeypatch, lambda settings, role: Good())
    assert check_providers.main() == 0


def test_check_providers_sends_no_user_data(monkeypatch):
    seen = []

    class Spy:
        def complete_json(self, system, user):
            seen.append(system + user)
            return '{"ok": true}'

    _fake_settings(monkeypatch, lambda settings, role: Spy())
    check_providers.main()
    assert len(seen) == 4 and all(len(s) < 120 for s in seen)
