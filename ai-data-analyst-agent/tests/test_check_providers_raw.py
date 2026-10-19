"""evals.check_providers --raw: the raw HTTP probe. No network (httpx MockTransport)."""

from types import SimpleNamespace as NS

import httpx
import pytest

from evals import check_providers as cp

KEY = "ghp_SECRET_TOKEN_123"


def _factory(handler):
    return lambda: httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def _probe(handler):
    return "\n".join(cp.probe_endpoint("https://api.example.invalid/v1", KEY, "some-model", _factory(handler)))


def test_html_200_is_explained_as_not_json():
    out = _probe(lambda r: httpx.Response(200, text="<html>Sign in</html>", headers={"content-type": "text/html"}))
    assert "HTTP 200" in out and "text/html" in out and "<html>Sign in</html>" in out
    assert "NOT with JSON" in out


def test_redirect_shows_where_it_goes():
    out = _probe(lambda r: httpx.Response(302, headers={"location": "https://github.com/login"}))
    assert "HTTP 302" in out and "redirects to: https://github.com/login" in out and "redirected" in out


@pytest.mark.parametrize(
    "status, expected",
    [(401, "API key was rejected"), (403, "API key was rejected"), (404, "model id"), (400, "model id"), (429, "Rate limit"), (503, "server problem")],
)
def test_error_statuses_get_a_matching_hint(status, expected):
    out = _probe(lambda r: httpx.Response(status, json={"error": {"message": "nope"}}))
    assert f"HTTP {status}" in out and expected in out


def test_valid_json_answer_says_the_provider_works():
    out = _probe(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]}))
    assert "provider and URL work" in out


def test_request_matches_what_the_app_sends_and_key_goes_only_in_the_header():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.content.decode()
        return httpx.Response(200, json={})

    _probe(handler)
    assert seen["url"] == "https://api.example.invalid/v1/chat/completions"
    assert seen["auth"] == f"Bearer {KEY}"
    assert KEY not in seen["body"]
    for field in ("some-model", "json_object", "temperature"):
        assert field in seen["body"]


def test_the_key_is_never_printed_even_if_the_provider_echoes_it():
    out = _probe(lambda r: httpx.Response(401, text=f"bad token {KEY}", headers={"location": f"https://x/?t={KEY}"}))
    assert KEY not in out and "***" in out


def test_network_error_shows_only_the_error_type():
    def handler(request):
        raise httpx.ConnectError("secret internal detail")

    out = _probe(handler)
    assert "ConnectError" in out and "secret internal detail" not in out


def test_main_raw_probes_each_non_groq_provider_model_once(monkeypatch, capsys):
    class Good:
        def complete_json(self, s, u):
            return '{"ok": true}'

    providers = {"planner": "groq", "clarifier": "cerebras", "reviewer": "gemini", "narrator": "cerebras"}
    probed = []
    monkeypatch.setattr(cp, "load_settings", lambda: NS(model_for=lambda role: "mistral" if role in ("clarifier", "narrator") else f"m-{role}"))
    monkeypatch.setattr(cp, "describe", lambda s, role: f"{providers[role]}:x")
    monkeypatch.setattr(cp, "missing_key_message", lambda s, roles: None)
    monkeypatch.setattr(cp, "make_client", lambda s, role: Good())
    monkeypatch.setattr(cp, "provider_for", lambda s, role: providers[role])
    monkeypatch.setattr(cp, "api_key_for", lambda s, provider: f"key-{provider}")
    monkeypatch.setattr(cp, "probe_endpoint", lambda url, key, model: probed.append((url, key, model)) or ["    probed"])

    assert cp.main(["--raw"]) == 0
    assert [(m) for _, _, m in probed] == ["mistral", "m-reviewer"]  # cerebras once, gemini once, no groq
    assert "key-cerebras" not in capsys.readouterr().out  # the key is passed on, never printed


def test_main_without_raw_makes_no_raw_request(monkeypatch):
    class Good:
        def complete_json(self, s, u):
            return '{"ok": true}'

    monkeypatch.setattr(cp, "load_settings", lambda: object())
    monkeypatch.setattr(cp, "describe", lambda s, role: role)
    monkeypatch.setattr(cp, "missing_key_message", lambda s, roles: None)
    monkeypatch.setattr(cp, "make_client", lambda s, role: Good())
    monkeypatch.setattr(cp, "probe_endpoint", lambda *a, **k: pytest.fail("raw probe should not run"))
    assert cp.main() == 0
