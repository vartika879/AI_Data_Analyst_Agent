"""Per-role provider settings and client selection. No network, no real API."""

import pytest

from analyst import llm
from analyst.config import ConfigError, load_settings
from analyst.llm import LLMError, OpenAICompatClient
from analyst.providers import BASE_URLS, describe, make_client, missing_key_message, provider_for

GROQ = {"GROQ_API_KEY": "g-key", "LLM_MAX_ATTEMPTS": "1"}  # retries are tested in test_retry.py


def _env(**extra):
    return {**GROQ, **extra}


# ---------------------------------------------------------------- config


def test_every_role_defaults_to_groq():
    settings = load_settings(_env())
    for role in ("planner", "clarifier", "reviewer", "narrator"):
        assert provider_for(settings, role) == "groq"
    assert settings.provider_keys == {}


def test_default_models_are_unchanged():
    settings = load_settings(_env(GROQ_MODEL="main", GROQ_REVIEWER_MODEL="rev"))
    assert settings.model_for("planner") == "main"
    assert settings.model_for("reviewer") == "rev"
    assert settings.model_for("narrator") == "main"
    assert settings.reviewer_model == "rev"


def test_role_can_use_another_provider_with_its_own_model():
    settings = load_settings(
        _env(REVIEWER_PROVIDER="gemini", REVIEWER_MODEL="g-model", GEMINI_API_KEY="gem-key")
    )
    assert provider_for(settings, "reviewer") == "gemini"
    assert settings.model_for("reviewer") == "g-model"
    assert settings.reviewer_model == "g-model"
    assert provider_for(settings, "planner") == "groq"
    assert describe(settings, "reviewer") == "gemini:g-model"


def test_provider_name_is_case_insensitive():
    settings = load_settings(_env(NARRATOR_PROVIDER=" Cerebras ", NARRATOR_MODEL="c-model"))
    assert provider_for(settings, "narrator") == "cerebras"


def test_unknown_provider_is_rejected():
    with pytest.raises(ConfigError, match="REVIEWER_PROVIDER"):
        load_settings(_env(REVIEWER_PROVIDER="skynet", REVIEWER_MODEL="m"))


def test_non_groq_provider_needs_a_model():
    with pytest.raises(ConfigError, match="NARRATOR_MODEL"):
        load_settings(_env(NARRATOR_PROVIDER="openrouter"))


def test_model_override_works_for_groq_too():
    settings = load_settings(_env(PLANNER_MODEL="override"))
    assert settings.model_for("planner") == "override"


def test_placeholder_keys_are_ignored():
    settings = load_settings(_env(GEMINI_API_KEY="your_gemini_api_key_here", CEREBRAS_API_KEY="real"))
    assert settings.provider_keys == {"cerebras": "real"}


def test_settings_repr_never_shows_any_key():
    settings = load_settings(_env(GEMINI_API_KEY="secret-gem-key", REVIEWER_PROVIDER="gemini", REVIEWER_MODEL="m"))
    text = repr(settings)
    assert "secret-gem-key" not in text
    assert "g-key" not in text


# ---------------------------------------------------------------- missing keys


def test_missing_key_message_for_groq_keeps_the_old_wording():
    settings = load_settings({})
    assert missing_key_message(settings, ["planner"]) == "Groq API key is not set. Add GROQ_API_KEY to your .env file."


def test_missing_key_message_names_the_right_provider():
    settings = load_settings(_env(REVIEWER_PROVIDER="gemini", REVIEWER_MODEL="m"))
    assert missing_key_message(settings, ["planner"]) is None
    message = missing_key_message(settings, ["planner", "reviewer"])
    assert "Gemini" in message and "GEMINI_API_KEY" in message


def test_roles_that_do_not_run_are_not_checked():
    settings = load_settings(_env(NARRATOR_PROVIDER="cerebras", NARRATOR_MODEL="m"))
    assert missing_key_message(settings, ["planner", "reviewer"]) is None


# ---------------------------------------------------------------- make_client


def test_groq_role_builds_a_groq_client(monkeypatch):
    made = []

    class Recorder:
        def __init__(self, key, model):
            made.append((key, model))

    monkeypatch.setattr(llm, "GroqClient", Recorder)
    make_client(load_settings(_env(GROQ_MODEL="main")), "planner")
    assert made == [("g-key", "main")]


def test_other_provider_builds_an_openai_compatible_client():
    settings = load_settings(
        _env(REVIEWER_PROVIDER="cerebras", REVIEWER_MODEL="c-model", CEREBRAS_API_KEY="c-key")
    )
    try:
        import openai  # noqa: F401
    except ImportError:
        with pytest.raises(LLMError, match="pip install openai"):
            make_client(settings, "reviewer")
        return
    client = make_client(settings, "reviewer")
    assert isinstance(client, OpenAICompatClient)
    assert client._model == "c-model"


def test_other_provider_without_key_is_a_clear_error():
    settings = load_settings(_env(REVIEWER_PROVIDER="openrouter", REVIEWER_MODEL="m"))
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        make_client(settings, "reviewer")


def test_every_non_groq_provider_has_a_base_url():
    assert set(BASE_URLS) == {"gemini", "cerebras", "openrouter"}
    assert all(url.startswith("https://") for url in BASE_URLS.values())
