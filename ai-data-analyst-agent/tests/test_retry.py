"""Retry with backoff, retry-after, and two Groq API keys. No network, no real waiting."""

import groq
import httpx
import pytest

from analyst import llm
from analyst.config import ConfigError, load_settings
from analyst.llm import FallbackClient, GroqClient, LLMError, RetryingClient, _retry_after_seconds
from analyst.providers import api_keys_for, describe, make_client
from evals import check_providers as cp


class Step:
    """A fake client that follows a script: a string = success, an LLMError = failure."""

    def __init__(self, *script, name="client"):
        self.script = list(script)
        self.name = name
        self.calls = 0

    def complete_json(self, system_prompt, user_prompt):
        self.calls += 1
        result = self.script.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _temp(message="rate limited", retry_after=None):
    return LLMError(message, retryable=True, retry_after=retry_after)


def _retrying(clients, sleeps, **kwargs):
    return RetryingClient(clients, sleep=sleeps.append, **kwargs)


# ============================================================ RetryingClient
def test_first_try_success_needs_no_wait():
    sleeps = []
    client = _retrying([Step("ok")], sleeps)
    assert client.complete_json("s", "u") == "ok"
    assert sleeps == [] and client.retries == 0


def test_temporary_failures_are_retried_with_doubling_backoff():
    sleeps = []
    step = Step(_temp(), _temp(), "ok")
    client = _retrying([step], sleeps, max_attempts=4, base_delay=2, max_delay=30)
    assert client.complete_json("s", "u") == "ok"
    assert sleeps == [2, 4] and step.calls == 3 and client.retries == 2


def test_backoff_is_capped():
    sleeps = []
    client = _retrying([Step(_temp(), _temp(), _temp(), "ok")], sleeps, max_attempts=4, base_delay=20, max_delay=30)
    client.complete_json("s", "u")
    assert sleeps == [20, 30, 30]


def test_a_wrong_key_or_model_is_raised_at_once_and_never_retried():
    sleeps = []
    step = Step(LLMError("Groq rejected the API key.", retryable=False))
    with pytest.raises(LLMError, match="rejected the API key"):
        _retrying([step], sleeps).complete_json("s", "u")
    assert step.calls == 1 and sleeps == []


def test_a_real_bug_is_not_hidden():
    with pytest.raises(AttributeError):
        _retrying([Step(AttributeError("bug"))], []).complete_json("s", "u")


def test_giving_up_after_all_attempts_keeps_the_error_retryable():
    sleeps = []
    step = Step(_temp("Groq rate limit reached."), _temp("Groq rate limit reached."), _temp("Groq rate limit reached."))
    with pytest.raises(LLMError) as info:
        _retrying([step], sleeps, max_attempts=3).complete_json("s", "u")
    assert info.value.retryable is True
    assert "Groq rate limit reached." in str(info.value) and "tried 3 time(s)" in str(info.value)
    assert step.calls == 3 and len(sleeps) == 2


def test_the_providers_retry_after_is_used_instead_of_backoff():
    sleeps = []
    client = _retrying([Step(_temp(retry_after=5), "ok")], sleeps, base_delay=2)
    client.complete_json("s", "u")
    assert sleeps == [5]


def test_a_very_long_retry_after_stops_at_once_instead_of_waiting():
    sleeps = []
    step = Step(_temp("Groq rate limit reached.", retry_after=400), "never used")
    with pytest.raises(LLMError) as info:
        _retrying([step], sleeps, max_delay=30).complete_json("s", "u")
    assert step.calls == 1 and sleeps == []
    assert "6 min 40 s" in str(info.value) and "not retried" in str(info.value)
    assert info.value.retryable is True and info.value.retry_after == 400


def test_retry_after_of_zero_does_not_sleep():
    sleeps = []
    _retrying([Step(_temp(retry_after=0), "ok")], sleeps).complete_json("s", "u")
    assert sleeps == []


# ---------------------------------------------------------------- two keys
def test_second_key_is_tried_at_once_without_waiting():
    sleeps = []
    key1, key2 = Step(_temp()), Step("ok")
    client = _retrying([key1, key2], sleeps, max_attempts=3)
    assert client.complete_json("s", "u") == "ok"
    assert (key1.calls, key2.calls) == (1, 1) and sleeps == [] and client.retries == 1


def test_waits_only_after_every_key_has_failed_once():
    sleeps, order = [], []

    class Logged(Step):
        def complete_json(self, s, u):
            order.append(self.name)
            return super().complete_json(s, u)

    key1, key2 = Logged(_temp(), "ok", name="key1"), Logged(_temp(), name="key2")
    client = _retrying([key1, key2], sleeps, max_attempts=3, base_delay=2)
    assert client.complete_json("s", "u") == "ok"
    assert order == ["key1", "key2", "key1"] and sleeps == [2]


def test_a_wrong_first_key_is_shown_not_hidden_by_the_second():
    key1 = Step(LLMError("Groq rejected the API key.", retryable=False))
    key2 = Step("ok")
    with pytest.raises(LLMError, match="rejected the API key"):
        _retrying([key1, key2], []).complete_json("s", "u")
    assert key2.calls == 0


def test_construction_rules():
    with pytest.raises(ValueError):
        RetryingClient([])
    assert RetryingClient([Step()], max_attempts=0).max_attempts == 1


def test_fallback_provider_takes_over_after_retries_are_used_up():
    primary = _retrying([Step(_temp(), _temp())], [], max_attempts=2)
    fallback = Step("from fallback")
    client = FallbackClient(primary, fallback, "Other:model")
    assert client.complete_json("s", "u") == "from fallback" and client.used_fallback


def test_fallback_also_takes_over_when_the_wait_would_be_too_long():
    primary = _retrying([Step(_temp(retry_after=3600))], [], max_delay=30)
    assert FallbackClient(primary, Step("B")).complete_json("s", "u") == "B"


# ============================================================ retry-after parsing
def _status_error(cls, status, headers=None):
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls("error", response=response, body=None)


@pytest.mark.parametrize(
    "header, expected",
    [("7", 7.0), ("0.5", 0.5), ("0", 0.0), ("Wed, 21 Oct 2026 07:28:00 GMT", None), ("-3", None), ("", None), (None, None)],
)
def test_retry_after_header_parsing(header, expected):
    headers = {} if header is None else {"retry-after": header}
    assert _retry_after_seconds(_status_error(groq.RateLimitError, 429, headers)) == expected


def test_error_without_a_response_has_no_retry_after():
    assert _retry_after_seconds(RuntimeError("x")) is None


class _SDK:
    def __init__(self, error):
        self.error = error
        self.chat = type("C", (), {"completions": type("P", (), {"create": self._create})()})()

    def _create(self, **kwargs):
        raise self.error


def test_groq_rate_limit_carries_the_wait_time():
    client = GroqClient(None, "m", sdk_client=_SDK(_status_error(groq.RateLimitError, 429, {"retry-after": "7"})))
    with pytest.raises(LLMError) as info:
        client.complete_json("s", "u")
    assert info.value.retryable and info.value.retry_after == 7.0


def test_groq_server_error_carries_the_wait_time_too():
    client = GroqClient(None, "m", sdk_client=_SDK(_status_error(groq.InternalServerError, 503, {"retry-after": "3"})))
    with pytest.raises(LLMError) as info:
        client.complete_json("s", "u")
    assert info.value.retryable and info.value.retry_after == 3.0


def test_openai_compatible_rate_limit_carries_the_wait_time():
    openai = pytest.importorskip("openai")
    err = _status_error(openai.RateLimitError, 429, {"retry-after": "11"})
    client = llm.OpenAICompatClient(None, "m", "https://x.invalid", sdk_client=_SDK(err))
    with pytest.raises(LLMError) as info:
        client.complete_json("s", "u")
    assert info.value.retryable and info.value.retry_after == 11.0


def test_real_retry_flow_groq_429_then_success_waits_the_headers_time():
    calls = []

    class Flaky:
        chat = type("C", (), {"completions": None})()

    class Completions:
        def create(self, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise _status_error(groq.RateLimitError, 429, {"retry-after": "4"})
            message = type("M", (), {"content": '{"ok": true}'})()
            return type("R", (), {"choices": [type("Ch", (), {"message": message})()]})()

    sdk = type("S", (), {"chat": type("C", (), {"completions": Completions()})()})()
    sleeps = []
    client = RetryingClient([GroqClient(None, "m", sdk_client=sdk)], sleep=sleeps.append)
    assert client.complete_json("s", "u") == '{"ok": true}'
    assert sleeps == [4.0] and len(calls) == 2


# ============================================================ config
KEYS = {"GROQ_API_KEY": "k1-secret", "GROQ_API_KEY_2": "k2-secret"}


def test_retry_defaults_and_values():
    default = load_settings({"GROQ_API_KEY": "k"})
    assert (default.llm_max_attempts, default.llm_retry_base_seconds, default.llm_retry_max_seconds) == (3, 2.0, 30.0)
    custom = load_settings({"GROQ_API_KEY": "k", "LLM_MAX_ATTEMPTS": "5", "LLM_RETRY_BASE_SECONDS": "0.5", "LLM_RETRY_MAX_SECONDS": "90"})
    assert (custom.llm_max_attempts, custom.llm_retry_base_seconds, custom.llm_retry_max_seconds) == (5, 0.5, 90.0)


@pytest.mark.parametrize(
    "name, value",
    [("LLM_MAX_ATTEMPTS", "0"), ("LLM_MAX_ATTEMPTS", "7"), ("LLM_MAX_ATTEMPTS", "two"), ("LLM_MAX_ATTEMPTS", "2.5"),
     ("LLM_RETRY_BASE_SECONDS", "-1"), ("LLM_RETRY_BASE_SECONDS", "abc"), ("LLM_RETRY_MAX_SECONDS", "999")],
)
def test_invalid_retry_settings_are_clear_errors(name, value):
    with pytest.raises(ConfigError, match=name):
        load_settings({"GROQ_API_KEY": "k", name: value})


def test_second_key_is_optional_hidden_and_placeholder_safe():
    assert load_settings({"GROQ_API_KEY": "k"}).groq_api_key_2 is None
    assert load_settings({"GROQ_API_KEY": "k", "GROQ_API_KEY_2": "your_groq_api_key_here"}).groq_api_key_2 is None
    settings = load_settings(KEYS)
    assert settings.groq_api_key_2 == "k2-secret"
    assert "k1-secret" not in repr(settings) and "k2-secret" not in repr(settings)


def test_api_keys_for_drops_duplicates_and_blanks():
    assert api_keys_for(load_settings(KEYS), "groq") == ["k1-secret", "k2-secret"]
    assert api_keys_for(load_settings({"GROQ_API_KEY": "same", "GROQ_API_KEY_2": "same"}), "groq") == ["same"]
    assert api_keys_for(load_settings({"GROQ_API_KEY": "k"}), "gemini") == []


# ============================================================ make_client wiring
class Rec:
    made = []

    def __init__(self, key, model, **kwargs):
        self.key, self.model, self.kwargs = key, model, kwargs
        Rec.made.append(self)


@pytest.fixture(autouse=True)
def _recorder(monkeypatch):
    Rec.made = []
    monkeypatch.setattr(llm, "GroqClient", Rec)


def test_default_settings_wrap_the_client_in_retries_and_turn_the_sdk_retry_off():
    client = make_client(load_settings({"GROQ_API_KEY": "k", "GROQ_MODEL": "m"}), "planner")
    assert isinstance(client, RetryingClient) and client.max_attempts == 3
    assert [c.key for c in client.clients] == ["k"] and client.clients[0].kwargs == {"max_retries": 0}


def test_two_keys_become_two_clients_in_order_for_the_same_model():
    client = make_client(load_settings({**KEYS, "PLANNER_MODEL": "openai/gpt-oss-20b"}), "planner")
    assert [c.key for c in client.clients] == ["k1-secret", "k2-secret"]
    assert {c.model for c in client.clients} == {"openai/gpt-oss-20b"}


def test_every_key_gets_at_least_one_attempt():
    one_attempt = make_client(load_settings({**KEYS, "LLM_MAX_ATTEMPTS": "1"}), "planner")
    assert isinstance(one_attempt, RetryingClient) and one_attempt.max_attempts == 2


def test_one_key_and_one_attempt_is_exactly_the_old_plain_client():
    client = make_client(load_settings({"GROQ_API_KEY": "k", "LLM_MAX_ATTEMPTS": "1"}), "planner")
    assert isinstance(client, Rec) and client.kwargs == {}


def test_same_key_twice_is_one_key():
    client = make_client(load_settings({"GROQ_API_KEY": "s", "GROQ_API_KEY_2": "s"}), "planner")
    assert len(client.clients) == 1


def test_each_role_can_use_its_own_groq_model_with_the_same_keys():
    settings = load_settings({**KEYS, "PLANNER_MODEL": "model-a", "CLARIFIER_MODEL": "model-b",
                              "REVIEWER_MODEL": "model-c", "NARRATOR_MODEL": "model-c"})
    models = {role: make_client(settings, role).clients[0].model for role in ("planner", "clarifier", "reviewer", "narrator")}
    assert models == {"planner": "model-a", "clarifier": "model-b", "reviewer": "model-c", "narrator": "model-c"}


def test_fallback_gets_its_own_retrying_chain():
    settings = load_settings({**KEYS, "PLANNER_MODEL": "main", "PLANNER_FALLBACK_PROVIDER": "groq",
                              "PLANNER_FALLBACK_MODEL": "backup"})
    client = make_client(settings, "planner")
    assert isinstance(client, FallbackClient)
    assert client._primary.clients[0].model == "main" and client._fallback.clients[0].model == "backup"
    assert [c.key for c in client._fallback.clients] == ["k1-secret", "k2-secret"]


def test_other_providers_get_retries_too_and_their_sdk_retry_is_off():
    pytest.importorskip("openai")
    settings = load_settings({"GROQ_API_KEY": "k", "CLARIFIER_PROVIDER": "gemini", "CLARIFIER_MODEL": "m",
                              "GEMINI_API_KEY": "g"})
    client = make_client(settings, "clarifier")
    assert isinstance(client, RetryingClient) and client.clients[0]._client.max_retries == 0


def test_describe_mentions_two_keys_only_when_there_are_two():
    assert "[2 API keys]" in describe(load_settings({**KEYS, "PLANNER_MODEL": "m"}), "planner")
    assert "API keys" not in describe(load_settings({"GROQ_API_KEY": "k", "PLANNER_MODEL": "m"}), "planner")


def test_a_missing_key_is_still_a_clear_error(monkeypatch):
    monkeypatch.setattr(llm, "GroqClient", GroqClient)  # the real one checks the key
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        make_client(load_settings({}), "planner")


# ============================================================ diagnostics
def _http(handler):
    return lambda: httpx.Client(transport=httpx.MockTransport(handler))


MODELS = {"data": [{"id": "openai/gpt-oss-20b", "active": True}, {"id": "openai/gpt-oss-120b", "active": True},
                   {"id": "old/retired-model", "active": False}, {"id": "whisper-large-v3"}]}


def test_models_report_lists_active_models_and_checks_each_role():
    settings = load_settings({"GROQ_API_KEY": "SECRETKEY", "PLANNER_MODEL": "openai/gpt-oss-20b",
                              "CLARIFIER_MODEL": "llama-3.1-8b-instant"})
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json=MODELS)

    out = "\n".join(cp.groq_models_report(settings, "SECRETKEY", _http(handler)))
    assert seen["auth"] == "Bearer SECRETKEY" and "SECRETKEY" not in out
    assert "openai/gpt-oss-120b" in out and "whisper-large-v3" in out and "old/retired-model" not in out
    assert "[planner  ] openai/gpt-oss-20b: available" in out
    assert "[clarifier] llama-3.1-8b-instant: NOT in your account's model list" in out


@pytest.mark.parametrize("response, expected", [(httpx.Response(401, text="no"), "HTTP 401"),
                                                (httpx.Response(200, text="<html>"), "not with a model list"),
                                                (httpx.Response(200, json={"oops": 1}), "not with a model list")])
def test_models_report_handles_bad_answers_without_showing_the_key(response, expected):
    out = "\n".join(cp.groq_models_report(load_settings({"GROQ_API_KEY": "SECRETKEY"}), "SECRETKEY", _http(lambda r: response)))
    assert expected in out and "SECRETKEY" not in out


def test_models_report_handles_network_errors():
    def handler(request):
        raise httpx.ConnectError("internal detail")

    out = "\n".join(cp.groq_models_report(load_settings({"GROQ_API_KEY": "k"}), "k", _http(handler)))
    assert "ConnectError" in out and "internal detail" not in out


class _Good:
    def complete_json(self, s, u):
        return '{"ok": true}'


def _patch_main(monkeypatch, settings):
    monkeypatch.setattr(cp, "load_settings", lambda: settings)
    monkeypatch.setattr(cp, "make_client", lambda s, role: _Good())


def test_keys_flag_tests_each_key_separately(monkeypatch, capsys):
    class FakeGroq:
        def __init__(self, key, model, **kwargs):
            self.key = key

        def complete_json(self, s, u):
            if self.key == "k2-secret":
                raise LLMError("Groq rate limit reached. Wait a little and try again.", retryable=True)
            return '{"ok": true}'

    _patch_main(monkeypatch, load_settings({**KEYS, "PLANNER_MODEL": "gpt-oss-20b"}))
    monkeypatch.setattr(cp.llm, "GroqClient", FakeGroq)
    assert cp.main(["--keys"]) == 1
    out = capsys.readouterr().out
    assert "key 1: OK" in out and "key 2: PROBLEM: Groq rate limit reached" in out
    assert "share one rate limit" in out and "k1-secret" not in out and "k2-secret" not in out


def test_models_flag_prints_the_report(monkeypatch, capsys):
    _patch_main(monkeypatch, load_settings({"GROQ_API_KEY": "k"}))
    monkeypatch.setattr(cp, "groq_models_report", lambda settings, key: ["    REPORT LINE"])
    assert cp.main(["--models"]) == 0
    assert "REPORT LINE" in capsys.readouterr().out


def test_flags_are_off_by_default(monkeypatch):
    _patch_main(monkeypatch, load_settings(KEYS))
    monkeypatch.setattr(cp, "groq_models_report", lambda *a: pytest.fail("not asked for"))
    monkeypatch.setattr(cp.llm, "GroqClient", lambda *a, **k: pytest.fail("not asked for"))
    assert cp.main() == 0


# ============================================================ eval report and template
def test_eval_record_counts_retries_inside_planner_and_fallback_clients():
    import json as _json

    from analyst.data_loader import load_csv
    from evals.cases import CASES
    from evals.dataset import csv_bytes
    from evals.runner import CONFIGS, run_case

    df = load_csv(csv_bytes(), "eval.csv", 5 * 1024 * 1024)
    plan = _json.dumps({"operation": "aggregate", "aggregations": [{"function": "sum", "column": "gross_revenue"}]})
    case = next(c for c in CASES if c.kind == "answerable")

    def factory(role):
        flaky = RetryingClient([Step(_temp(), _temp(), plan)], sleep=lambda s: None, max_attempts=3)
        return FallbackClient(flaky, Step("never used")) if role == "planner" else Step("{}")

    record = run_case(case, df, CONFIGS["baseline"], factory, config="baseline")
    assert record.llm_calls == 1 and record.llm_retries == 2 and record.passed


def test_env_example_models_are_current_and_retry_is_configured():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    env = dict(l.split("=", 1) for l in lines if "=" in l)
    allowed = {"openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"}  # Groq's non-Enterprise chat models
    models = {k: v for k, v in env.items() if k.endswith("_MODEL")}
    assert models and set(models.values()) <= allowed, models  # e.g. llama-3.1-8b-instant is Enterprise-only now
    # Only three Groq chat models exist and qwen is a preview model, so the roles share
    # two production models: the big one for the planner, the small one for the rest.
    assert len({env[f"{r}_MODEL"] for r in ("PLANNER", "CLARIFIER", "REVIEWER", "NARRATOR")}) >= 2
    assert {"LLM_MAX_ATTEMPTS", "LLM_RETRY_BASE_SECONDS", "LLM_RETRY_MAX_SECONDS", "GROQ_API_KEY_2"} <= set(env)
    assert all(env[f"{r}_PROVIDER"] == "groq" for r in ("PLANNER", "CLARIFIER", "REVIEWER", "NARRATOR"))