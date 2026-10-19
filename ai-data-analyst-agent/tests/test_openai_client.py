"""OpenAICompatClient with a fake SDK. No network, no real API."""

from types import SimpleNamespace

import pytest

from analyst.llm import MAX_OUTPUT_TOKENS, LLMError, OpenAICompatClient


class FakeSDK:
    """Mimics client.chat.completions.create of the OpenAI SDK."""

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


_CLASSES: dict = {}


def _cls(name: str, *bases):
    """A class named like the openai SDK one, living in module 'openai'."""
    if name not in _CLASSES:
        _CLASSES[name] = type(name, bases or (Exception,), {"__module__": "openai"})
    return _CLASSES[name]


def _sdk_error(name: str, **attrs) -> Exception:
    """An exception that looks like one raised by the openai SDK."""
    api_error = _cls("APIError")
    connection = _cls("APIConnectionError", api_error)
    status = _cls("APIStatusError", api_error)
    parents = {
        "APIConnectionError": api_error,
        "APITimeoutError": connection,
        "APIStatusError": api_error,
        "AuthenticationError": status,
        "PermissionDeniedError": status,
        "NotFoundError": status,
        "RateLimitError": status,
    }
    err = _cls(name, parents[name])("secret details that must never be shown")
    for key, value in attrs.items():
        setattr(err, key, value)
    return err


def _client(sdk):
    return OpenAICompatClient(None, "my-model", "https://example.invalid/v1", label="Gemini",
                              key_env="GEMINI_API_KEY", sdk_client=sdk)


def test_missing_api_key_names_the_variable():
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        OpenAICompatClient(None, "m", "https://example.invalid/v1", label="Gemini", key_env="GEMINI_API_KEY")


def test_returns_content_and_sends_expected_request():
    sdk = FakeSDK(content='{"operation": "sort"}')
    assert _client(sdk).complete_json("system text", "user text") == '{"operation": "sort"}'
    call = sdk.calls[0]
    assert call["model"] == "my-model"
    assert call["temperature"] == 0
    assert call["max_tokens"] == MAX_OUTPUT_TOKENS
    assert call["response_format"] == {"type": "json_object"}
    assert call["messages"][0] == {"role": "system", "content": "system text"}
    assert call["messages"][1] == {"role": "user", "content": "user text"}


def test_empty_content_raises():
    with pytest.raises(LLMError, match="empty"):
        _client(FakeSDK(content="   ")).complete_json("s", "u")


def test_no_choices_raises():
    with pytest.raises(LLMError, match="empty"):
        _client(FakeSDK(choices=False)).complete_json("s", "u")


def test_rate_limit_becomes_a_safe_message():
    with pytest.raises(LLMError, match="Gemini rate limit") as info:
        _client(FakeSDK(error=_sdk_error("RateLimitError"))).complete_json("s", "u")
    assert "secret details" not in str(info.value)


def test_auth_error_points_to_the_key_variable():
    with pytest.raises(LLMError, match="GEMINI_API_KEY") as info:
        _client(FakeSDK(error=_sdk_error("AuthenticationError"))).complete_json("s", "u")
    assert "secret details" not in str(info.value)


def test_missing_model_is_explained():
    with pytest.raises(LLMError, match="could not find the model"):
        _client(FakeSDK(error=_sdk_error("NotFoundError"))).complete_json("s", "u")


def test_timeout_is_a_network_message():
    with pytest.raises(LLMError, match="Could not reach Gemini"):
        _client(FakeSDK(error=_sdk_error("APITimeoutError"))).complete_json("s", "u")


def test_other_http_status_shows_only_the_code():
    err = _sdk_error("APIStatusError", status_code=503)
    with pytest.raises(LLMError, match="HTTP 503") as info:
        _client(FakeSDK(error=err)).complete_json("s", "u")
    assert "secret details" not in str(info.value)


def test_real_bugs_are_not_hidden():
    # An error that did not come from the SDK must not be turned into an LLMError.
    with pytest.raises(KeyError):
        _client(FakeSDK(error=KeyError("bug"))).complete_json("s", "u")
