"""Thin wrapper around the Groq API.

Every failure becomes an LLMError whose message is safe to show to the user.
We never put the raw exception text or the API key into those messages.
"""

from typing import Protocol

import groq
from groq import Groq

MAX_OUTPUT_TOKENS = 2000  # gpt-oss counts its hidden reasoning in this budget too


class LLMError(RuntimeError):
    """Raised when the LLM call fails. The message is safe to show to the user."""


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
    ) -> None:
        if sdk_client is None:
            if not api_key:
                raise LLMError(
                    "Groq API key is not configured. Add GROQ_API_KEY to your .env file."
                )
            sdk_client = Groq(api_key=api_key, timeout=timeout_seconds, max_retries=1)
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
                "Groq rate limit reached. Wait a little and try again."
            ) from exc
        except groq.NotFoundError as exc:
            raise LLMError(
                "Groq could not find the model. Check GROQ_MODEL in your .env file."
            ) from exc
        except groq.APIConnectionError as exc:  # also covers timeouts
            raise LLMError(
                "Could not reach Groq (network problem or timeout). Try again."
            ) from exc
        except groq.APIStatusError as exc:
            raise LLMError(f"Groq returned an error (HTTP {exc.status_code}).") from exc
        except groq.GroqError as exc:
            raise LLMError("The Groq request failed.") from exc

        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise LLMError("The AI model returned an empty response.")
        return content