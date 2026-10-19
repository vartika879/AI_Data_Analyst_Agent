"""Thin wrapper around the Groq API.

Every failure becomes an LLMError whose message is safe to show to the user.
We never put the raw exception text or the API key into those messages.
"""

import time
from typing import Protocol

import groq
from groq import Groq

MAX_OUTPUT_TOKENS = 2000  # gpt-oss counts its hidden reasoning in this budget too


class LLMError(RuntimeError):
    """Raised when the LLM call fails. The message is safe to show to the user.

    `retryable` is True for temporary problems (rate limit, timeout, network,
    server error). Only those may switch to a fallback provider. A wrong API key
    or model name is never retryable: the user must see and fix it.
    """

    def __init__(
        self, message: str = "", *, retryable: bool = False, retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after  # seconds the provider asked us to wait, if it said


class LLMClient(Protocol):
    """Anything with this method can act as the LLM (real Groq, or a test fake)."""

    def complete_json(self, system_prompt: str, user_prompt: str) -> str: ...


class GroqClient:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        timeout_seconds: float = 30.0,
        sdk_client=None,
        max_retries: int = 1,
    ) -> None:
        if sdk_client is None:
            if not api_key:
                raise LLMError(
                    "Groq API key is not configured. Add GROQ_API_KEY to your .env file."
                )
            sdk_client = Groq(api_key=api_key, timeout=timeout_seconds, max_retries=max_retries)
        # Note: we keep the SDK client, not the raw key, as an attribute.
        self._client = sdk_client
        self._model = model

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
            )
        except groq.AuthenticationError as exc:
            raise LLMError(
                "Groq rejected the API key. Check GROQ_API_KEY in your .env file."
            ) from exc
        except groq.RateLimitError as exc:
            raise LLMError(
                "Groq rate limit reached. Wait a little and try again.",
                retryable=True,
                retry_after=_retry_after_seconds(exc),
            ) from exc
        except groq.NotFoundError as exc:
            raise LLMError(
                "Groq could not find the model. Check GROQ_MODEL in your .env file."
            ) from exc
        except groq.APIConnectionError as exc:  # also covers timeouts
            raise LLMError(
                "Could not reach Groq (network problem or timeout). Try again.", retryable=True
            ) from exc
        except groq.APIStatusError as exc:
            raise LLMError(
                f"Groq returned an error (HTTP {exc.status_code}).",
                retryable=_is_temporary_status(exc.status_code),
                retry_after=_retry_after_seconds(exc),
            ) from exc
        except groq.GroqError as exc:
            raise LLMError("The Groq request failed.") from exc

        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise LLMError("The AI model returned an empty response.", retryable=True)
        return content


def _retry_after_seconds(exc) -> float | None:
    """Seconds from a `retry-after` header on the error response, else None."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        value = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None  # missing, or an HTTP date we do not parse
    return value if value >= 0 else None


def _format_wait(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60} min {seconds % 60} s" if seconds >= 60 else f"{seconds} s"


def _is_temporary_status(status) -> bool:
    return isinstance(status, int) and (status >= 500 or status == 408)


def _openai_error_retryable(exc: Exception) -> bool:
    """True if an `openai` SDK error is temporary (rate limit, network, server)."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    if names & {"AuthenticationError", "PermissionDeniedError", "NotFoundError"}:
        return False
    if names & {"RateLimitError", "APIConnectionError", "InternalServerError"}:
        return True
    return _is_temporary_status(getattr(exc, "status_code", None))


def _openai_error_message(exc: Exception, label: str, key_env: str) -> str | None:
    """Safe user message for an error raised by the `openai` SDK, else None.

    Works by class name so it needs no import of `openai`. Errors that do not come
    from the SDK (real bugs) return None and are re-raised untouched.
    """
    mro = type(exc).__mro__
    if not any(cls.__module__.split(".")[0] == "openai" for cls in mro):
        return None
    names = {cls.__name__ for cls in mro}
    if "AuthenticationError" in names or "PermissionDeniedError" in names:
        return f"{label} rejected the API key. Check {key_env} in your .env file."
    if "RateLimitError" in names:
        return f"{label} rate limit reached. Wait a little and try again."
    if "NotFoundError" in names:
        return f"{label} could not find the model. Check the model name in your .env file."
    if "APIConnectionError" in names:  # also covers timeouts
        return f"Could not reach {label} (network problem or timeout). Try again."
    if "APIStatusError" in names:
        return f"{label} returned an error (HTTP {getattr(exc, 'status_code', '?')})."
    return f"The {label} request failed."


def _part_text(part) -> str:
    """Text of one content part (some providers send a list of parts, not a string)."""
    if isinstance(part, str):
        return part
    text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
    return text if isinstance(text, str) else ""


def _response_text(response, label: str) -> str:
    """Text of an OpenAI-style chat response, or a user-safe LLMError.

    Providers differ a little: the SDK can hand back a plain string (when the server
    sent something that is not JSON, for example an HTML error page), a message with
    no content, or content as a list of parts. None of these may crash the app.
    """
    if isinstance(response, (str, bytes)) or not hasattr(response, "choices"):
        raise LLMError(
            f"{label} returned a response in an unexpected format "
            f"(received {type(response).__name__}). Check the provider, its base URL "
            "and the model name in your .env file."
        )
    choices = response.choices
    message = getattr(choices[0], "message", None) if choices else None
    content = getattr(message, "content", None)
    if isinstance(content, list):
        content = "".join(_part_text(part) for part in content)
    if content is not None and not isinstance(content, str):
        raise LLMError(
            f"{label} returned content in an unexpected format (received {type(content).__name__}).",
        )
    if not content or not content.strip():
        raise LLMError("The AI model returned an empty response.", retryable=True)
    return content


class OpenAICompatClient:
    """Client for any provider that offers an OpenAI-compatible chat API
    (Gemini, Cerebras, OpenRouter, ...). Same interface as GroqClient."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str,
        label: str = "The AI provider",
        key_env: str = "the API key variable",
        timeout_seconds: float = 30.0,
        sdk_client=None,
        max_retries: int = 1,
    ) -> None:
        if sdk_client is None:
            if not api_key:
                raise LLMError(f"{label} API key is not configured. Add {key_env} to your .env file.")
            try:
                import openai
            except ImportError:
                raise LLMError("The 'openai' package is not installed. Run: pip install openai") from None
            sdk_client = openai.OpenAI(
                api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=max_retries
            )
        # Keep the SDK client, not the raw key, as an attribute.
        self._client = sdk_client
        self._model = model
        self._label = label
        self._key_env = key_env

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            message = _openai_error_message(exc, self._label, self._key_env)
            if message is None:
                raise
            raise LLMError(
                message,
                retryable=_openai_error_retryable(exc),
                retry_after=_retry_after_seconds(exc),
            ) from exc

        return _response_text(response, self._label)


class RetryingClient:
    """Try again after a temporary failure, optionally cycling through several clients.

    Several clients usually means several API keys of the same provider and model.
    - A failure that is not temporary (wrong key or model) is raised at once.
    - After a temporary failure the next client (key) is tried immediately.
    - Only when every client has failed once do we wait: the provider's `retry-after`,
      else exponential backoff (`base_delay`, doubling, capped at `max_delay`).
    - If the provider asks for a wait longer than `max_delay` (a daily limit, for
      example) we stop at once instead of waiting. The error stays "retryable", so a
      configured fallback provider can still take over.
    Keys from the same Groq organization share one limit, so a second key only helps
    if it belongs to a different organization.
    """

    def __init__(
        self,
        clients,
        *,
        max_attempts: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 30.0,
        sleep=time.sleep,
    ) -> None:
        if not clients:
            raise ValueError("RetryingClient needs at least one client.")
        self.clients = list(clients)
        self.max_attempts = max(1, int(max_attempts))
        self._base_delay = max(0.0, float(base_delay))
        self._max_delay = max(0.0, float(max_delay))
        self._sleep = sleep
        self.retries = 0  # how many times a request had to be tried again

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        count = len(self.clients)
        last: LLMError | None = None
        for attempt in range(self.max_attempts):
            try:
                return self.clients[attempt % count].complete_json(system_prompt, user_prompt)
            except LLMError as exc:
                if not exc.retryable:
                    raise
                last = exc
            if attempt == self.max_attempts - 1:
                break
            self.retries += 1
            if count > 1 and (attempt + 1) % count != 0:
                continue  # another key is next: no need to wait
            wait = (
                last.retry_after
                if last.retry_after is not None
                else min(self._max_delay, self._base_delay * 2 ** (attempt // count))
            )
            if wait > self._max_delay:
                raise LLMError(
                    f"{last} The provider asks to wait about {_format_wait(wait)}, longer than "
                    f"the {self._max_delay:g} s this app waits, so the request was not retried.",
                    retryable=True,
                    retry_after=wait,
                ) from last
            if wait > 0:
                self._sleep(wait)
        raise LLMError(
            f"{last} It was tried {self.max_attempts} time(s).",
            retryable=True,
            retry_after=last.retry_after,
        ) from last


class FallbackClient:
    """Try the primary client; on a temporary failure, try the fallback client.

    Same interface as the other clients, so agents never know the difference.
    Wrong keys or model names (not retryable) are raised as they are.
    """

    def __init__(self, primary, fallback, fallback_label: str = "fallback") -> None:
        self._primary = primary
        self._fallback = fallback
        self.fallback_label = fallback_label
        self.used_fallback = False  # becomes True once the fallback answered a request

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        try:
            return self._primary.complete_json(system_prompt, user_prompt)
        except LLMError as first:
            if not first.retryable:
                raise
            first_error = first
        try:
            answer = self._fallback.complete_json(system_prompt, user_prompt)
        except LLMError as second:
            raise LLMError(
                f"{first_error} The fallback ({self.fallback_label}) also failed: {second}",
                retryable=second.retryable,
            ) from second
        self.used_fallback = True
        return answer
