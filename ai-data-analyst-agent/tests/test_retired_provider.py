"""GitHub Models was retired on 2026-07-30: naming it must be a clear startup error."""

from pathlib import Path

import pytest

from analyst.config import ROLES, ConfigError, load_settings
from evals import check_providers, runner

BASE = {"GROQ_API_KEY": "g-key"}


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("value", ["github", "GitHub", " GITHUB "])
def test_github_as_a_role_provider_is_a_clear_error(role, value):
    variable = f"{role.upper()}_PROVIDER"
    with pytest.raises(ConfigError) as info:
        load_settings({**BASE, variable: value, f"{role.upper()}_MODEL": "m"})
    message = str(info.value)
    assert variable in message and "retired" in message and "July 30, 2026" in message
    assert "groq" in message and "gemini" in message  # tells the user what to use instead


@pytest.mark.parametrize("role", ROLES)
def test_github_as_a_fallback_provider_is_a_clear_error(role):
    variable = f"{role.upper()}_FALLBACK_PROVIDER"
    with pytest.raises(ConfigError, match="retired"):
        load_settings({**BASE, variable: "github", f"{role.upper()}_FALLBACK_MODEL": "m"})


def test_a_leftover_github_token_is_ignored_not_used():
    settings = load_settings({**BASE, "GITHUB_TOKEN": "ghp_leftover"})
    assert settings.provider_keys == {}
    assert "ghp_leftover" not in repr(settings)


def test_other_providers_still_work():
    settings = load_settings({**BASE, "CLARIFIER_PROVIDER": "gemini", "CLARIFIER_MODEL": "m", "GEMINI_API_KEY": "k"})
    assert settings.role_providers["clarifier"] == "gemini"


def test_the_env_example_template_loads_and_names_no_retired_provider():
    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    env = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    settings = load_settings(env)  # raises if the template names a retired or unknown provider
    assert "github" not in set(settings.role_providers.values())
    assert "github" not in set(settings.role_fallback_providers.values())
    assert "GITHUB_TOKEN" not in env


def _broken_config():
    return load_settings({**BASE, "CLARIFIER_PROVIDER": "github", "CLARIFIER_MODEL": "m"})


def test_eval_runner_shows_a_message_not_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr(runner, "load_settings", _broken_config)
    assert runner.main(["--limit", "1"]) == 1
    out = capsys.readouterr().out
    assert "Configuration problem" in out and "retired" in out


def test_check_providers_shows_a_message_not_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr(check_providers, "load_settings", _broken_config)
    assert check_providers.main() == 1
    out = capsys.readouterr().out
    assert "Configuration problem" in out and "retired" in out
