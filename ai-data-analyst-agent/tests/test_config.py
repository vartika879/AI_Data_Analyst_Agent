import pytest

from analyst.config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_MAX_UPLOAD_MB,
    ConfigError,
    load_settings,
)


def test_defaults_when_environment_is_empty():
    settings = load_settings({})
    assert settings.groq_api_key is None
    assert settings.has_api_key is False
    assert settings.groq_model == DEFAULT_GROQ_MODEL
    assert settings.max_upload_mb == DEFAULT_MAX_UPLOAD_MB


def test_values_are_read_from_environment():
    settings = load_settings(
        {"GROQ_API_KEY": "abc123", "GROQ_MODEL": "my-model", "MAX_UPLOAD_MB": "2.5"}
    )
    assert settings.has_api_key is True
    assert settings.groq_model == "my-model"
    assert settings.max_upload_mb == 2.5


def test_max_upload_bytes_is_calculated_correctly():
    settings = load_settings({"MAX_UPLOAD_MB": "1"})
    assert settings.max_upload_bytes == 1024 * 1024


def test_non_numeric_upload_limit_raises_config_error():
    with pytest.raises(ConfigError):
        load_settings({"MAX_UPLOAD_MB": "abc"})


def test_zero_or_negative_upload_limit_raises_config_error():
    with pytest.raises(ConfigError):
        load_settings({"MAX_UPLOAD_MB": "0"})
    with pytest.raises(ConfigError):
        load_settings({"MAX_UPLOAD_MB": "-3"})


def test_placeholder_api_key_is_treated_as_missing():
    settings = load_settings({"GROQ_API_KEY": "your_groq_api_key_here"})
    assert settings.has_api_key is False


def test_repr_does_not_leak_api_key():
    settings = load_settings({"GROQ_API_KEY": "super-secret-key"})
    assert "super-secret-key" not in repr(settings)
    assert "super-secret-key" not in str(settings)