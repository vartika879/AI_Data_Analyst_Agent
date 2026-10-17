from types import SimpleNamespace

import groq
import httpx
import pytest

from analyst.llm import GroqClient, LLMError


class FakeSDK:
    """Mimics the part of the Groq SDK that we use: client.chat.completions.create."""

    def __init__(self, content="{}", choices=True, error=None):
        self.calls = []
        self._content = content
        self._choices = choices
        self._error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        if not self._choices:
            return SimpleNamespace(choices=[])
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _response(status: int) -> httpx.Response:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return httpx.Response(status, request=request)


def test_missing_api_key_raises_llm_error():
    with pytest.raises(LLMError, match="API key"):
        GroqClient(api_key=None, model="m")
    with pytest.raises(LLMError, match="API key"):
        GroqClient(api_key="", model="m")


def test_complete_json_returns_content_and_sends_expected_request():
    sdk = FakeSDK(content='{"operation": "sort"}')
    client = GroqClient(api_key=None, model="my-model", sdk_client=sdk)
    out = client.complete_json("system text", "user text")
    assert out == '{"operation": "sort"}'
    call = sdk.calls[0]
    assert call["model"] == "my-model"
    assert call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}
    assert call["messages"][0] == {"role": "system", "content": "system text"}
    assert call["messages"][1] == {"role": "user", "content": "user text"}


def test_empty_content_raises():
    client = GroqClient(api_key=None, model="m", sdk_client=FakeSDK(content="   "))
    with pytest.raises(LLMError, match="empty"):
        client.complete_json("s", "u")


def test_no_choices_raises():
    client = GroqClient(api_key=None, model="m", sdk_client=FakeSDK(choices=False))
    with pytest.raises(LLMError, match="empty"):
        client.complete_json("s", "u")


def test_rate_limit_becomes_llm_error():
    err = groq.RateLimitError("too many", response=_response(429), body=None)
    client = GroqClient(api_key=None, model="m", sdk_client=FakeSDK(error=err))
    with pytest.raises(LLMError, match="rate limit"):
        client.complete_json("s", "u")


def test_auth_error_message_does_not_leak_details():
    err = groq.AuthenticationError(
        "bad key gsk_secret123", response=_response(401), body=None
    )
    client = GroqClient(api_key=None, model="m", sdk_client=FakeSDK(error=err))
    with pytest.raises(LLMError) as exc_info:
        client.complete_json("s", "u")
    assert "gsk_secret123" not in str(exc_info.value)
    assert "API key" in str(exc_info.value)


def test_connection_error_becomes_llm_error():
    request = httpx.Request("POST", "https://example.invalid")
    err = groq.APIConnectionError(request=request)
    client = GroqClient(api_key=None, model="m", sdk_client=FakeSDK(error=err))
    with pytest.raises(LLMError, match="Could not reach"):
        client.complete_json("s", "u")