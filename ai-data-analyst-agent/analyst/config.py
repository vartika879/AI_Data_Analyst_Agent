"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field
from typing import Mapping

from dotenv import load_dotenv

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_MAX_UPLOAD_MB = 5.0
DEFAULT_LLM_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BASE_SECONDS = 2.0
DEFAULT_RETRY_MAX_SECONDS = 30.0
PLACEHOLDER_API_KEY = "your_groq_api_key_here"

# Every agent role can use its own provider. Key = provider name, value = the
# environment variable that holds that provider's API key.
ROLES = ("planner", "clarifier", "reviewer", "narrator")
PROVIDER_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Providers that no longer exist. Naming one is a clear error at startup.
RETIRED_PROVIDERS = {
    "github": "GitHub Models was retired by GitHub on July 30, 2026 and no longer answers requests",
}

TRUE_VALUES = {"true", "1", "yes", "on"}
FALSE_VALUES = {"false", "0", "no", "off"}


class ConfigError(ValueError):
    """Raised when a configuration value is invalid."""


@dataclass(frozen=True)
class Settings:
    # repr=False: Settings ko print/log karne par key kabhi nahi dikhegi.
    groq_api_key: str | None = field(default=None, repr=False)
    # Optional second Groq key, tried when the first one hits a temporary error.
    groq_api_key_2: str | None = field(default=None, repr=False)
    groq_model: str = DEFAULT_GROQ_MODEL
    max_upload_mb: float = DEFAULT_MAX_UPLOAD_MB
    groq_reviewer_model: str | None = None
    reviewer_enabled: bool = True
    groq_narrator_model: str | None = None
    narrator_enabled: bool = True
    groq_clarifier_model: str | None = None
    clarifier_enabled: bool = True
    privacy_mode: str = "strict"  # strict or standard
    # Per-role provider ("groq" by default) and optional per-role model name.
    role_providers: dict = field(default_factory=lambda: {role: "groq" for role in ROLES})
    role_models: dict = field(default_factory=dict)
    # API keys of the non-Groq providers. repr=False so they never show in logs.
    provider_keys: dict = field(default_factory=dict, repr=False)
    # Optional Plan B per role: used only when the main provider has a temporary
    # problem (rate limit, timeout, server error). Empty by default.
    role_fallback_providers: dict = field(default_factory=dict)
    role_fallback_models: dict = field(default_factory=dict)
    # Retry after temporary errors (rate limit, timeout, server error).
    llm_max_attempts: int = DEFAULT_LLM_MAX_ATTEMPTS
    llm_retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS
    llm_retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS

    @property
    def has_api_key(self) -> bool:
        return self.groq_api_key is not None

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def reviewer_model(self) -> str:
        return self.role_models.get("reviewer") or self.groq_reviewer_model or self.groq_model

    @property
    def narrator_model(self) -> str:
        return self.role_models.get("narrator") or self.groq_narrator_model or self.groq_model

    @property
    def clarifier_model(self) -> str:
        return self.role_models.get("clarifier") or self.groq_clarifier_model or self.groq_model

    def model_for(self, role: str) -> str:
        """The model name an agent role will use (planner included)."""
        if role == "planner":
            return self.role_models.get("planner") or self.groq_model
        return {
            "reviewer": self.reviewer_model,
            "narrator": self.narrator_model,
            "clarifier": self.clarifier_model,
        }[role]


    def fallback_for(self, role: str) -> tuple[str, str] | None:
        """(provider, model) of the fallback for a role, or None if not configured."""
        provider = self.role_fallback_providers.get(role)
        if not provider:
            return None
        return provider, self.role_fallback_models.get(role) or self.groq_model


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


def _parse_bool(raw: str | None, name: str, default: bool) -> bool:
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ConfigError(f"{name} must be true or false.")



def _parse_privacy_mode(raw: str | None) -> str:
    """Parse the privacy mode. Legacy true/false values remain accepted."""
    if raw is None or raw.strip() == "":
        return "strict"
    value = raw.strip().lower()
    if value in {"strict", "true", "1", "yes", "on"}:
        return "strict"
    if value in {"standard", "false", "0", "no", "off"}:
        return "standard"
    raise ConfigError("PRIVACY_MODE must be 'strict' or 'standard'.")

def _clean_api_key(raw: str | None) -> str | None:
    if raw is None:
        return None
    key = raw.strip()
    if key == "" or key == PLACEHOLDER_API_KEY:
        return None
    if key.lower().startswith("your_") and key.lower().endswith("_here"):
        return None  # an untouched placeholder from .env.example
    return key


def _parse_number(raw: str | None, name: str, default, low, high, kind=float):
    if raw is None or raw.strip() == "":
        return default
    try:
        value = kind(raw.strip())
    except ValueError:
        raise ConfigError(f"{name} must be a number, for example {default}.") from None
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}.")
    return value


def _reject_retired(variable: str, provider: str) -> None:
    if provider in RETIRED_PROVIDERS:
        allowed = ", ".join(PROVIDER_KEY_ENV)
        raise ConfigError(
            f"{variable}={provider}: {RETIRED_PROVIDERS[provider]}. "
            f"Change it in your .env file to one of: {allowed}."
        )


def _parse_roles(environ: Mapping[str, str]) -> tuple[dict, dict]:
    """Read <ROLE>_PROVIDER and <ROLE>_MODEL for every agent role."""
    providers: dict[str, str] = {}
    models: dict[str, str] = {}
    for role in ROLES:
        prefix = role.upper()
        provider = environ.get(f"{prefix}_PROVIDER", "").strip().lower() or "groq"
        _reject_retired(f"{prefix}_PROVIDER", provider)
        if provider not in PROVIDER_KEY_ENV:
            allowed = ", ".join(PROVIDER_KEY_ENV)
            raise ConfigError(f"{prefix}_PROVIDER must be one of: {allowed}.")
        model = environ.get(f"{prefix}_MODEL", "").strip()
        if provider != "groq" and not model:
            raise ConfigError(f"{prefix}_MODEL is required when {prefix}_PROVIDER is {provider}.")
        providers[role] = provider
        if model:
            models[role] = model
    return providers, models


def _parse_fallbacks(environ: Mapping[str, str]) -> tuple[dict, dict]:
    """Read <ROLE>_FALLBACK_PROVIDER and <ROLE>_FALLBACK_MODEL (both optional)."""
    providers: dict[str, str] = {}
    models: dict[str, str] = {}
    for role in ROLES:
        prefix = role.upper()
        provider = environ.get(f"{prefix}_FALLBACK_PROVIDER", "").strip().lower()
        model = environ.get(f"{prefix}_FALLBACK_MODEL", "").strip()
        if not provider:
            if model:
                raise ConfigError(
                    f"{prefix}_FALLBACK_MODEL is set but {prefix}_FALLBACK_PROVIDER is empty."
                )
            continue
        _reject_retired(f"{prefix}_FALLBACK_PROVIDER", provider)
        if provider not in PROVIDER_KEY_ENV:
            allowed = ", ".join(PROVIDER_KEY_ENV)
            raise ConfigError(f"{prefix}_FALLBACK_PROVIDER must be one of: {allowed}.")
        if provider != "groq" and not model:
            raise ConfigError(
                f"{prefix}_FALLBACK_MODEL is required when {prefix}_FALLBACK_PROVIDER is {provider}."
            )
        providers[role] = provider
        if model:
            models[role] = model
    return providers, models


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
    narrator_model = environ.get("GROQ_NARRATOR_MODEL", "").strip() or None
    clarifier_model = environ.get("GROQ_CLARIFIER_MODEL", "").strip() or None
    role_providers, role_models = _parse_roles(environ)
    fallback_providers, fallback_models = _parse_fallbacks(environ)
    provider_keys = {}
    for name, env_name in PROVIDER_KEY_ENV.items():
        if name == "groq":
            continue
        key = _clean_api_key(environ.get(env_name))
        if key is not None:
            provider_keys[name] = key

    return Settings(
        groq_api_key=_clean_api_key(environ.get("GROQ_API_KEY")),
        groq_api_key_2=_clean_api_key(environ.get("GROQ_API_KEY_2")),
        llm_max_attempts=_parse_number(
            environ.get("LLM_MAX_ATTEMPTS"), "LLM_MAX_ATTEMPTS", DEFAULT_LLM_MAX_ATTEMPTS, 1, 6, int
        ),
        llm_retry_base_seconds=_parse_number(
            environ.get("LLM_RETRY_BASE_SECONDS"), "LLM_RETRY_BASE_SECONDS", DEFAULT_RETRY_BASE_SECONDS, 0, 60
        ),
        llm_retry_max_seconds=_parse_number(
            environ.get("LLM_RETRY_MAX_SECONDS"), "LLM_RETRY_MAX_SECONDS", DEFAULT_RETRY_MAX_SECONDS, 0, 300
        ),
        groq_model=model,
        max_upload_mb=_parse_max_upload_mb(environ.get("MAX_UPLOAD_MB")),
        groq_reviewer_model=reviewer_model,
        reviewer_enabled=_parse_bool(environ.get("ENABLE_REVIEWER"), "ENABLE_REVIEWER", True),
        groq_narrator_model=narrator_model,
        narrator_enabled=_parse_bool(environ.get("ENABLE_NARRATOR"), "ENABLE_NARRATOR", True),
        groq_clarifier_model=clarifier_model,
        clarifier_enabled=_parse_bool(environ.get("ENABLE_CLARIFIER"), "ENABLE_CLARIFIER", True),
        privacy_mode=_parse_privacy_mode(environ.get("PRIVACY_MODE")),
        role_providers=role_providers,
        role_models=role_models,
        provider_keys=provider_keys,
        role_fallback_providers=fallback_providers,
        role_fallback_models=fallback_models,
    )