"""UI checks for the privacy panel, using Streamlit's AppTest. Synthetic data, no real API.

AppTest cannot upload a file, so the uploader line of app.py is replaced by a fake
uploader in a temporary copy. Everything else in app.py runs as written.
"""

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from analyst.data_loader import load_csv  # noqa: E402
from analyst.llm import LLMError  # noqa: E402
from analyst.planner import build_user_prompt  # noqa: E402
from analyst.privacy import PrivacyPolicy  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"
CSV = b"employee name,city,salary\nAsha Verma,Zanzibarville,100\nRavi Rao,Pune,200\nCharu Jain,Zanzibarville,300\n"

FAKE_UPLOADER = (
    "class _Fake:\n"
    "    name = 't.csv'\n"
    "    def getvalue(self):\n"
    f"        return {CSV!r}\n"
    "uploaded = _Fake()\n"
)


@pytest.fixture
def calls(monkeypatch):
    """Replace GroqClient with a recorder that fails if it is ever asked for anything."""
    recorded = []

    class Recorder:
        def __init__(self, *args, **kwargs):
            pass

        def complete_json(self, system_prompt, user_prompt):
            recorded.append(user_prompt)
            raise LLMError("blocked in test")

    # Isolate the test from the developer's real .env: no provider/model/fallback
    # settings and no real API keys may leak in, or a role could go to a real provider.
    monkeypatch.setattr("analyst.config.load_dotenv", lambda *args, **kwargs: False)
    for role in ("PLANNER", "CLARIFIER", "REVIEWER", "NARRATOR"):
        for suffix in ("PROVIDER", "MODEL", "FALLBACK_PROVIDER", "FALLBACK_MODEL"):
            monkeypatch.delenv(f"{role}_{suffix}", raising=False)
    for key_name in ("GEMINI_API_KEY", "GITHUB_TOKEN", "CEREBRAS_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key_name, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setenv("PRIVACY_MODE", "true")
    monkeypatch.setattr("analyst.llm.GroqClient", Recorder)
    return recorded


@pytest.fixture
def app(tmp_path, calls):
    source = re.sub(r"uploaded = st\.file_uploader\([^\n]*\)", lambda m: FAKE_UPLOADER, APP.read_text(encoding="utf-8"))
    path = tmp_path / "app_under_test.py"
    path.write_text(source, encoding="utf-8")
    at = AppTest.from_file(str(path), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def _json_blobs(at) -> str:
    return " ".join(json.dumps(j.value) for j in at.json).lower()


def _analyze(at):
    next(b for b in at.button if b.label == "Analyze").click().run()


def test_panel_states_that_the_question_is_sent_as_typed(app):
    text = " ".join(m.value for m in app.markdown).lower()
    assert "exactly as you typed" in text


def test_panel_json_is_the_real_planner_payload(app):
    app.text_input(key="question_input").set_value("average salary by city").run()
    df = load_csv(CSV, "t.csv", 5 * 1024 * 1024)
    expected = json.loads(build_user_prompt("average salary by city", df, None, PrivacyPolicy.strict()))
    assert expected in [j.value if isinstance(j.value, dict) else json.loads(j.value) for j in app.json]


def test_panel_never_shows_sample_values_or_hidden_columns(app):
    app.multiselect(key="hidden_columns").set_value(["employee name"]).run()
    app.text_input(key="question_input").set_value("average salary by city").run()
    blobs = _json_blobs(app)
    for secret in ("employee name", "asha verma", "zanzibarville", "pune"):
        assert secret not in blobs
    assert "city" in blobs


def test_panel_warns_when_the_question_names_a_hidden_column(app):
    app.multiselect(key="hidden_columns").set_value(["salary"]).run()
    app.text_input(key="question_input").set_value("Show SALARY by city").run()
    assert any("will not be sent" in w.value for w in app.warning)


def test_clicking_analyze_with_a_hidden_question_calls_no_llm(app, calls):
    app.multiselect(key="hidden_columns").set_value(["salary"]).run()
    app.text_input(key="question_input").set_value("Show SALARY by city").run()
    _analyze(app)
    assert calls == []
    assert any("not sent" in e.value for e in app.error)


def test_clicking_analyze_with_a_normal_question_reaches_the_llm_without_data(app, calls):
    app.multiselect(key="hidden_columns").set_value(["employee name"]).run()
    app.text_input(key="question_input").set_value("average salary by city").run()
    _analyze(app)
    assert calls, "the question should have been sent"
    sent = " ".join(calls).lower()
    for secret in ("employee name", "asha verma", "zanzibarville", "pune"):
        assert secret not in sent


def test_privacy_off_panel_warns_that_samples_are_sent(app):
    app.toggle[0].set_value(False).run()
    assert any("Privacy mode is off" in w.value for w in app.warning)
    assert "zanzibarville" in _json_blobs(app)