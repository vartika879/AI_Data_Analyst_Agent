"""Fallback provider: config, client switching and error classification. No network."""

import pytest

from analyst import llm
from analyst.config import ConfigError, load_settings
from analyst.llm import FallbackClient, LLMError, OpenAICompatClient
from analyst.providers import describe, make_client
from tests.test_openai_client import FakeSDK, _client, _sdk_error

BASE = {"GROQ_API_KEY": "g-key", "LLM_MAX_ATTEMPTS": "1"}  # retries are tested in test_retry.py


def _env(**extra):
    return {**BASE, **extra}


class Scripted:
    """A fake client: returns text, or raises the given error."""

    def __init__(self, result="ok"):
        self.result = result
        self.calls = 0

    def complete_json(self, system_prompt, user_prompt):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


# ---------------------------------------------------------------- config


def test_no_fallback_by_default():
    settings = load_settings(_env())
    assert all(settings.fallback_for(r) is None for r in ("planner", "clarifier", "reviewer", "narrator"))


def test_fallback_is_read_per_role():
    settings = load_settings(
        _env(PLANNER_FALLBACK_PROVIDER="Gemini", PLANNER_FALLBACK_MODEL="flash", GEMINI_API_KEY="k")
    )
    assert settings.fallback_for("planner") == ("gemini", "flash")
    assert settings.fallback_for("reviewer") is None


def test_groq_fallback_defaults_to_the_groq_model():
    settings = load_settings(_env(GROQ_MODEL="main", CLARIFIER_FALLBACK_PROVIDER="groq"))
    assert settings.fallback_for("clarifier") == ("groq", "main")


def test_fallback_model_needs_a_provider():
    with pytest.raises(ConfigError, match="PLANNER_FALLBACK_PROVIDER"):
        load_settings(_env(PLANNER_FALLBACK_MODEL="m"))


def test_non_groq_fallback_needs_a_model():
    with pytest.raises(ConfigError, match="REVIEWER_FALLBACK_MODEL"):
        load_settings(_env(REVIEWER_FALLBACK_PROVIDER="gemini"))


def test_unknown_fallback_provider_is_rejected():
    with pytest.raises(ConfigError, match="NARRATOR_FALLBACK_PROVIDER"):
        load_settings(_env(NARRATOR_FALLBACK_PROVIDER="skynet", NARRATOR_FALLBACK_MODEL="m"))


def test_gemini_key_is_read_and_never_shown():
    settings = load_settings(_env(GEMINI_API_KEY="gem_secret"))
    assert settings.provider_keys == {"gemini": "gem_secret"}
    assert "gem_secret" not in repr(settings)


# ---------------------------------------------------------------- FallbackClient


def test_primary_answer_is_used_when_it_works():
    primary, fallback = Scripted("A"), Scripted("B")
    client = FallbackClient(primary, fallback, "Gemini:flash")
    assert client.complete_json("s", "u") == "A"
    assert fallback.calls == 0 and client.used_fallback is False


def test_temporary_error_switches_to_the_fallback():
    primary = Scripted(LLMError("Groq rate limit reached.", retryable=True))
    fallback = Scripted("B")
    client = FallbackClient(primary, fallback, "Gemini:flash")
    assert client.complete_json("s", "u") == "B"
    assert client.used_fallback is True and client.fallback_label == "Gemini:flash"


def test_wrong_key_or_model_is_not_hidden_by_the_fallback():
    primary = Scripted(LLMError("Groq rejected the API key."))  # retryable=False
    fallback = Scripted("B")
    client = FallbackClient(primary, fallback)
    with pytest.raises(LLMError, match="rejected the API key"):
        client.complete_json("s", "u")
    assert fallback.calls == 0


def test_both_failing_gives_one_readable_error():
    primary = Scripted(LLMError("Groq rate limit reached.", retryable=True))
    fallback = Scripted(LLMError("Gemini rate limit reached.", retryable=True))
    client = FallbackClient(primary, fallback, "Gemini:flash")
    with pytest.raises(LLMError) as info:
        client.complete_json("s", "u")
    message = str(info.value)
    assert "Groq rate limit" in message and "Gemini rate limit" in message
    assert info.value.retryable is True and client.used_fallback is False


def test_non_llm_errors_are_not_swallowed():
    client = FallbackClient(Scripted(RuntimeError("bug")), Scripted("B"))
    with pytest.raises(RuntimeError, match="bug"):
        client.complete_json("s", "u")


# ---------------------------------------------------------------- which errors are temporary


@pytest.mark.parametrize(
    "name,attrs,expected",
    [
        ("RateLimitError", {}, True),
        ("APIConnectionError", {}, True),
        ("APITimeoutError", {}, True),
        ("APIStatusError", {"status_code": 503}, True),
        ("APIStatusError", {"status_code": 400}, False),
        ("AuthenticationError", {}, False),
        ("PermissionDeniedError", {}, False),
        ("NotFoundError", {}, False),
    ],
)
def test_openai_errors_are_classified(name, attrs, expected):
    client = _client(FakeSDK(error=_sdk_error(name, **attrs)))
    with pytest.raises(LLMError) as info:
        client.complete_json("s", "u")
    assert info.value.retryable is expected


def test_empty_answer_counts_as_temporary():
    with pytest.raises(LLMError) as info:
        _client(FakeSDK(content="  ")).complete_json("s", "u")
    assert info.value.retryable is True


def test_plain_llm_error_is_not_retryable_by_default():
    assert LLMError("x").retryable is False


# ---------------------------------------------------------------- make_client


def test_make_client_without_fallback_returns_the_plain_client(monkeypatch):
    class Recorder:
        def __init__(self, key, model):
            self.model = model

    monkeypatch.setattr(llm, "GroqClient", Recorder)
    client = make_client(load_settings(_env(GROQ_MODEL="main")), "planner")
    assert isinstance(client, Recorder)


def test_make_client_wraps_when_fallback_is_usable(monkeypatch):
    class Recorder:
        def __init__(self, key, model):
            self.model = model

    monkeypatch.setattr(llm, "GroqClient", Recorder)
    settings = load_settings(
        _env(
            GROQ_MODEL="main",
            PLANNER_FALLBACK_PROVIDER="groq",
            PLANNER_FALLBACK_MODEL="other-groq-model",
        )
    )
    client = make_client(settings, "planner")
    assert isinstance(client, FallbackClient)
    assert client._primary.model == "main"
    assert client._fallback.model == "other-groq-model"
    assert describe(settings, "planner") == "groq:main (fallback groq:other-groq-model)"


def test_fallback_without_its_key_is_simply_not_used(monkeypatch):
    class Recorder:
        def __init__(self, key, model):
            pass

    monkeypatch.setattr(llm, "GroqClient", Recorder)
    settings = load_settings(_env(PLANNER_FALLBACK_PROVIDER="gemini", PLANNER_FALLBACK_MODEL="flash"))
    assert isinstance(make_client(settings, "planner"), Recorder)
    assert "fallback" not in describe(settings, "planner")


def test_fallback_identical_to_main_is_ignored(monkeypatch):
    class Recorder:
        def __init__(self, key, model):
            pass

    monkeypatch.setattr(llm, "GroqClient", Recorder)
    settings = load_settings(_env(GROQ_MODEL="main", PLANNER_FALLBACK_PROVIDER="groq"))
    assert isinstance(make_client(settings, "planner"), Recorder)


def test_gemini_client_uses_the_gemini_endpoint():
    pytest.importorskip("openai")
    settings = load_settings(
        _env(CLARIFIER_PROVIDER="gemini", CLARIFIER_MODEL="some-model", GEMINI_API_KEY="t")
    )
    client = make_client(settings, "clarifier")
    assert isinstance(client, OpenAICompatClient)
    assert "generativelanguage.googleapis.com" in str(client._client.base_url)
