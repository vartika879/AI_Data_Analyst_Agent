"""Choose the LLM provider for each agent role (planner, clarifier, reviewer, narrator).

By default every role uses Groq, exactly as before. Set <ROLE>_PROVIDER and
<ROLE>_MODEL in .env to send a role to another provider.

Optionally set <ROLE>_FALLBACK_PROVIDER and <ROLE>_FALLBACK_MODEL: if the main
provider has a temporary problem (rate limit, timeout, server error), the
request is retried once on the fallback. Wrong keys/models are never hidden.
"""

from analyst import llm
from analyst.config import PROVIDER_KEY_ENV, Settings

LABELS = {"groq": "Groq", "gemini": "Gemini", "cerebras": "Cerebras", "openrouter": "OpenRouter"}

# OpenAI-compatible endpoints. Check the provider's docs if a URL stops working.
BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "cerebras": "https://api.cerebras.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def provider_for(settings: Settings, role: str) -> str:
    return settings.role_providers.get(role, "groq")


def api_key_for(settings: Settings, provider: str) -> str | None:
    if provider == "groq":
        return settings.groq_api_key
    return settings.provider_keys.get(provider)


def api_keys_for(settings: Settings, provider: str) -> list[str]:
    """All usable API keys of a provider (Groq may have two). No duplicates, no blanks."""
    if provider == "groq":
        candidates = [settings.groq_api_key, settings.groq_api_key_2]
    else:
        candidates = [settings.provider_keys.get(provider)]
    keys: list[str] = []
    for key in candidates:
        if key and key not in keys:
            keys.append(key)
    return keys


def _active_fallback(settings: Settings, role: str) -> tuple[str, str] | None:
    """The fallback (provider, model) for a role, only if it can really be used.

    No fallback if it is not configured, its API key is missing, or it is the very
    same provider and model as the main one.
    """
    fallback = settings.fallback_for(role)
    if fallback is None:
        return None
    provider, model = fallback
    if not api_key_for(settings, provider):
        return None
    if (provider, model) == (provider_for(settings, role), settings.model_for(role)):
        return None
    return provider, model


def describe(settings: Settings, role: str) -> str:
    """Short label such as 'gemini:gemini-2.0-flash', for the UI and eval output."""
    label = f"{provider_for(settings, role)}:{settings.model_for(role)}"
    key_count = len(api_keys_for(settings, provider_for(settings, role)))
    if key_count > 1:
        label += f" [{key_count} API keys]"
    fallback = _active_fallback(settings, role)
    if fallback is not None:
        label += f" (fallback {fallback[0]}:{fallback[1]})"
    return label


def missing_key_message(settings: Settings, roles) -> str | None:
    """A readable message if any of these roles has no API key, else None."""
    for role in roles:
        provider = provider_for(settings, role)
        if not api_key_for(settings, provider):
            env_name = PROVIDER_KEY_ENV[provider]
            return f"{LABELS[provider]} API key is not set. Add {env_name} to your .env file."
    return None


def _build_client(settings: Settings, provider: str, model: str, key=None, wrapped=False):
    """One client for one key. `wrapped` = a RetryingClient will do the retrying, so
    the SDK's own retry is switched off (otherwise the waits would add up)."""
    key = key or api_key_for(settings, provider)
    extra = {"max_retries": 0} if wrapped else {}
    if provider == "groq":
        # Looked up at call time (not imported by name) so tests can replace it.
        return llm.GroqClient(key, model, **extra)
    return llm.OpenAICompatClient(
        key,
        model,
        BASE_URLS[provider],
        label=LABELS[provider],
        key_env=PROVIDER_KEY_ENV[provider],
        **extra,
    )


def _build_chain(settings: Settings, provider: str, model: str):
    """The client for one (provider, model): plain, or with retries and key failover.

    One key and LLM_MAX_ATTEMPTS=1 gives the plain client, exactly as before. With
    several keys at least one attempt per key is allowed, so the second key is used.
    """
    keys = api_keys_for(settings, provider) or [None]
    if settings.llm_max_attempts <= 1 and len(keys) == 1:
        return _build_client(settings, provider, model, keys[0])
    return llm.RetryingClient(
        [_build_client(settings, provider, model, key, wrapped=True) for key in keys],
        max_attempts=max(settings.llm_max_attempts, len(keys)),
        base_delay=settings.llm_retry_base_seconds,
        max_delay=settings.llm_retry_max_seconds,
    )


def make_client(settings: Settings, role: str):
    """Build the LLM client for one role. Raises LLMError if the key is missing.

    If a usable fallback is configured, the client switches to it on temporary
    errors only. Without a fallback the plain client is returned, as before.
    """
    primary = _build_chain(settings, provider_for(settings, role), settings.model_for(role))
    fallback = _active_fallback(settings, role)
    if fallback is None:
        return primary
    provider, model = fallback
    return llm.FallbackClient(
        primary,
        _build_chain(settings, provider, model),
        fallback_label=f"{LABELS[provider]}:{model}",
    )
