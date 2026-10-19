"""Check every agent role's provider with one tiny request. Sends NO user data.

    python -m evals.check_providers          # one request per role through the real clients
    python -m evals.check_providers --raw    # also show the provider's raw HTTP answer
    python -m evals.check_providers --models # which Groq models your account can use
    python -m evals.check_providers --keys   # test each Groq API key on its own

For each role (planner, clarifier, reviewer, narrator) it prints OK, or the safe error
message, or (for an unexpected error) the full traceback so you can see what is wrong.
`--raw` is for providers that answer in an unexpected format: it sends the same tiny
request with plain HTTP and prints the status, content type, any redirect and the first
300 characters of the body. The API key is never printed.
"""

import json
import sys
import traceback

import httpx

from analyst.config import ConfigError, load_settings
from analyst.llm import MAX_OUTPUT_TOKENS, LLMError
from analyst import llm
from analyst.providers import (
    BASE_URLS,
    api_keys_for,
    api_key_for,
    describe,
    make_client,
    missing_key_message,
    provider_for,
)

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

ROLES = ("planner", "clarifier", "reviewer", "narrator")
SYSTEM = "Reply with one JSON object only."
USER = 'Return exactly {"ok": true}'
BODY_PREVIEW_CHARS = 300


def _hint(status: int, content_type: str, looks_json: bool) -> str:
    if 300 <= status < 400:
        return "The server redirected the request. Check the base URL; the token may also be rejected."
    if status in (401, 403):
        return "The API key was rejected. Check the key for this provider in your .env file."
    if status in (400, 404, 422):
        return "The request was refused. Check the model id exactly as the provider's catalog lists it."
    if status == 429:
        return "Rate limit reached. Wait a little, or use another provider for this role."
    if status >= 500:
        return "The provider had a server problem. Try again later."
    if 200 <= status < 300 and not looks_json:
        return ("The provider answered successfully but NOT with JSON (often an HTML page from a "
                "proxy, firewall or wrong URL). This is why the app reports an unexpected format.")
    if 200 <= status < 300:
        return "The answer is JSON, so the provider and URL work. Compare it with the format the app expects."
    return "Unexpected status."


def probe_endpoint(base_url: str, key: str, model: str, client_factory=None) -> list[str]:
    """Send the same tiny request the app sends, with plain HTTP. Returns lines to print."""
    factory = client_factory or (lambda: httpx.Client(timeout=30, follow_redirects=False))
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
    }
    try:
        with factory() as client:
            response = client.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
                json=body,
            )
    except httpx.HTTPError as exc:
        return [f"    POST {url}", f"    network error ({type(exc).__name__}): could not reach the provider."]

    text = response.text.replace(key, "***") if key else response.text
    content_type = response.headers.get("content-type", "(none)")
    try:
        json.loads(text)
        looks_json = True
    except ValueError:
        looks_json = False
    lines = [f"    POST {url}", f"    HTTP {response.status_code}   content-type: {content_type}"]
    location = response.headers.get("location")
    if location:
        lines.append(f"    redirects to: {location[:120].replace(key, '***') if key else location[:120]}")
    lines.append(f"    body (first {BODY_PREVIEW_CHARS} chars): {text[:BODY_PREVIEW_CHARS]!r}")
    lines.append(f"    -> {_hint(response.status_code, content_type, looks_json)}")
    return lines


def groq_models_report(settings, key: str, client_factory=None) -> list[str]:
    """Which models this Groq key can use right now, and whether each role's model is among them."""
    factory = client_factory or (lambda: httpx.Client(timeout=30))
    try:
        with factory() as client:
            response = client.get(GROQ_MODELS_URL, headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError as exc:
        return [f"    network error ({type(exc).__name__}): could not reach Groq."]
    if response.status_code != 200:
        return [f"    Groq answered HTTP {response.status_code}. Check the API key.  (key hidden)"]
    try:
        entries = response.json()["data"]
        active = sorted(m["id"] for m in entries if m.get("active", True))
    except (ValueError, KeyError, TypeError):
        return ["    Groq answered, but not with a model list."]
    lines = ["    Models your account can use now:"] + [f"      {model_id}" for model_id in active]
    for role in ROLES:
        if provider_for(settings, role) != "groq":
            continue
        model = settings.model_for(role)
        status = "available" if model in active else "NOT in your account's model list: change it in .env"
        lines.append(f"    [{role:<9}] {model}: {status}")
    return lines


def main(argv=()) -> int:
    raw = "--raw" in argv
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration problem: {exc}")
        return 1
    problems = 0
    for role in ROLES:
        label = describe(settings, role)
        missing = missing_key_message(settings, [role])
        if missing:
            print(f"[{role:<9}] {label}\n    NOT READY: {missing}")
            problems += 1
            continue
        try:
            reply = make_client(settings, role).complete_json(SYSTEM, USER)
            json.loads(reply)  # must be real JSON, like the agents need
            print(f"[{role:<9}] {label}\n    OK")
        except LLMError as exc:
            print(f"[{role:<9}] {label}\n    PROBLEM: {exc}")
            problems += 1
        except json.JSONDecodeError:
            print(f"[{role:<9}] {label}\n    PROBLEM: the model answered, but not with valid JSON.")
            problems += 1
        except Exception:  # an unexpected error: show everything
            print(f"[{role:<9}] {label}\n    UNEXPECTED ERROR:")
            print(traceback.format_exc())
            problems += 1

    if "--keys" in argv:
        print("\nGroq keys, one tiny request each:")
        planner_model = settings.model_for("planner")
        keys = api_keys_for(settings, "groq")
        if not keys:
            print("    No Groq key is set.")
        for number, key in enumerate(keys, start=1):
            try:
                json.loads(llm.GroqClient(key, planner_model, max_retries=0).complete_json(SYSTEM, USER))
                print(f"    key {number}: OK  (model {planner_model})")
            except LLMError as exc:
                print(f"    key {number}: PROBLEM: {exc}")
                problems += 1
            except json.JSONDecodeError:
                print(f"    key {number}: answered, but not with valid JSON.")
                problems += 1
        if len(keys) > 1:
            print("    Note: keys of the same Groq organization share one rate limit.")

    if "--models" in argv:
        print("\nGroq model list (from Groq's own /models endpoint):")
        key = api_key_for(settings, "groq")
        if not key:
            print("    No Groq key is set.")
        else:
            print("\n".join(groq_models_report(settings, key)))

    if raw:
        print("\nRaw HTTP check (same tiny request, no user data):")
        seen = set()
        for role in ROLES:
            provider = provider_for(settings, role)
            model = settings.model_for(role)
            key = api_key_for(settings, provider)
            if provider == "groq" or not key or (provider, model) in seen:
                continue  # Groq uses its own SDK; skip missing keys and repeats
            seen.add((provider, model))
            print(f"[{provider}:{model}]")
            print("\n".join(probe_endpoint(BASE_URLS[provider], key, model)))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
