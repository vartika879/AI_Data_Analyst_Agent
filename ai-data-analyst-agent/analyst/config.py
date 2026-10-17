"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field
from typing import Mapping

from dotenv import load_dotenv

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_MAX_UPLOAD_MB = 5.0
PLACEHOLDER_API_KEY = "your_groq_api_key_here"

TRUE_VALUES = {"true", "1", "yes", "on"}
FALSE_VALUES = {"false", "0", "no", "off"}


class ConfigError(ValueError):
    """Raised when a configuration value is invalid."""


@dataclass(frozen=True)
class Settings:
    # repr=False: Settings ko print/log karne par key kabhi nahi dikhegi.
    groq_api_key: str | None = field(default=None, repr=False)
    groq_model: str = DEFAULT_GROQ_MODEL
    max_upload_mb: float = DEFAULT_MAX_UPLOAD_MB
    groq_reviewer_model: str | None = None  # NEW (Phase 10)
    reviewer_enabled: bool = True  # NEW (Phase 10)

    @property
    def has_api_key(self) -> bool:
        return self.groq_api_key is not None

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def reviewer_model(self) -> str:  # NEW (Phase 10)
        return self.groq_reviewer_model or self.groq_model


def _parse_max_upload_mb(raw: str | None) -> float:
    if raw is None or raw.strip() == "":
        return DEFAULT_MAX_UPLOAD_MB
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError("MAX_UPLOAD_MB must be a number.") from None
    if value <= 0:
        raise ConfigError("MAX_UPLOAD_MB must be greater than 0.")
    return value


def _parse_bool(raw: str | None, name: str, default: bool) -> bool:  # NEW (Phase 10)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ConfigError(f"{name} must be true or false.")


def _clean_api_key(raw: str | None) -> str | None:
    if raw is None:
        return None
    key = raw.strip()
    if key == "" or key == PLACEHOLDER_API_KEY:
        return None
    return key


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from environment variables.

    If `environ` is None, we load a local .env file and read os.environ.
    Tests pass a plain dict instead, so they never touch the real environment.
    """
    if environ is None:
        load_dotenv()
        environ = os.environ

    model = environ.get("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL
    reviewer_model = environ.get("GROQ_REVIEWER_MODEL", "").strip() or None

    return Settings(
        groq_api_key=_clean_api_key(environ.get("GROQ_API_KEY")),
        groq_model=model,
        max_upload_mb=_parse_max_upload_mb(environ.get("MAX_UPLOAD_MB")),
        groq_reviewer_model=reviewer_model,
        reviewer_enabled=_parse_bool(environ.get("ENABLE_REVIEWER"), "ENABLE_REVIEWER", True),
    )