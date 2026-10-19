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


def test_reviewer_defaults():
    settings = load_settings({"GROQ_MODEL": "main-model"})
    assert settings.reviewer_enabled is True
    assert settings.reviewer_model == "main-model"


def test_reviewer_model_can_be_overridden():
    settings = load_settings({"GROQ_MODEL": "main", "GROQ_REVIEWER_MODEL": "cheap"})
    assert settings.reviewer_model == "cheap"


@pytest.mark.parametrize("value", ["false", "0", "no", "False"])
def test_reviewer_can_be_disabled(value):
    assert load_settings({"ENABLE_REVIEWER": value}).reviewer_enabled is False


def test_invalid_reviewer_flag_raises_config_error():
    with pytest.raises(ConfigError, match="ENABLE_REVIEWER"):
        load_settings({"ENABLE_REVIEWER": "maybe"})

def test_narrator_defaults():
    settings = load_settings({"GROQ_MODEL": "main-model"})
    assert settings.narrator_enabled is True
    assert settings.narrator_model == "main-model"


def test_narrator_model_can_be_overridden():
    settings = load_settings({"GROQ_MODEL": "main", "GROQ_NARRATOR_MODEL": "writer"})
    assert settings.narrator_model == "writer"


@pytest.mark.parametrize("value", ["false", "0", "no", "False"])
def test_narrator_can_be_disabled(value):
    assert load_settings({"ENABLE_NARRATOR": value}).narrator_enabled is False


def test_invalid_narrator_flag_raises_config_error():
    with pytest.raises(ConfigError, match="ENABLE_NARRATOR"):
        load_settings({"ENABLE_NARRATOR": "maybe"})


def test_clarifier_defaults():
    settings = load_settings({"GROQ_MODEL": "main-model"})
    assert settings.clarifier_enabled is True
    assert settings.clarifier_model == "main-model"


def test_clarifier_model_can_be_overridden():
    settings = load_settings({"GROQ_MODEL": "main", "GROQ_CLARIFIER_MODEL": "small"})
    assert settings.clarifier_model == "small"


@pytest.mark.parametrize("value", ["false", "0", "no", "False"])
def test_clarifier_can_be_disabled(value):
    assert load_settings({"ENABLE_CLARIFIER": value}).clarifier_enabled is False


def test_invalid_clarifier_flag_raises_config_error():
    with pytest.raises(ConfigError, match="ENABLE_CLARIFIER"):
        load_settings({"ENABLE_CLARIFIER": "maybe"})

def test_privacy_mode_defaults_to_strict():
    assert load_settings({}).privacy_mode == "strict"


def test_privacy_mode_can_be_standard():
    assert load_settings({"PRIVACY_MODE": "standard"}).privacy_mode == "standard"


def test_invalid_privacy_mode_is_rejected():
    with pytest.raises(ConfigError, match="PRIVACY_MODE"):
        load_settings({"PRIVACY_MODE": "disabled"})


def test_legacy_boolean_privacy_values_map_to_modes():
    assert load_settings({"PRIVACY_MODE": "true"}).privacy_mode == "strict"
    assert load_settings({"PRIVACY_MODE": "false"}).privacy_mode == "standard"
