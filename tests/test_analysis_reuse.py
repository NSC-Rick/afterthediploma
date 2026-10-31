"""Unified analysis execution: shared results, dedup, invalidation.

The acceptance criterion: run AI-assisted analysis once, navigate to AI
Comparison, and see both result sets immediately with no further API call.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from modules import ai_client

APP = str(Path(__file__).resolve().parent.parent / "app.py")

FAKE_AI_SKILLS = [
    {"skill": "MIG welding", "category": "core", "status": "documented",
     "evidence": "MIG named in program", "rationale": ""},
    {"skill": "CNC machining exposure", "category": "adjacent", "status": "recommended",
     "evidence": "", "rationale": "regional manufacturing demand"},
]


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    return at


def _btn(at: AppTest, key: str):
    matches = [b for b in at.button if b.key == key]
    assert matches, f"button {key!r} not found"
    return matches[0]


def _demo_analysis(at: AppTest):
    _btn(at, "home-demo").click()
    at.run()
    _btn(at, "sel-confirm").click()
    at.run()


@pytest.fixture
def spy_ai(monkeypatch):
    """Replace the real API call with a counting fake."""
    calls = {"n": 0}

    def fake(text, api_key=None, model=None):
        calls["n"] += 1
        return list(FAKE_AI_SKILLS)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(ai_client, "analyze_curriculum", fake)
    return calls


# ------------------------------------------------------------- acceptance
def test_ai_analysis_once_then_compare_no_second_call(spy_ai):
    at = _app()
    _demo_analysis(at)
    assert spy_ai["n"] == 0  # AI disabled by default -> rules only

    at.session_state["use_ai"] = True
    at.run()
    assert spy_ai["n"] == 1  # fingerprint changed -> single unified run
    assert at.session_state["analysis"]["ai_status"] == "ok"

    at.radio(key="nav").set_value("AI Comparison")
    at.run()
    assert spy_ai["n"] == 1, "comparison must reuse stored results"
    assert not at.exception
    assert len(at.metric) > 0  # diff metrics rendered


def test_navigation_and_reruns_make_no_extra_calls(spy_ai):
    at = _app()
    _demo_analysis(at)
    at.session_state["use_ai"] = True
    at.run()
    assert spy_ai["n"] == 1

    for page in ["Executive Dashboard", "1 · Curriculum Intelligence",
                 "2 · Career Pathways", "3 · Regional Employment",
                 "4 · Earnings Intelligence", "5 · Curriculum Advisor",
                 "AI Comparison", "Curriculum Library", "Home"]:
        at.radio(key="nav").set_value(page)
        at.run()
        assert spy_ai["n"] == 1, f"navigating to {page} made an extra API call"
    at.run()  # bare rerun
    at.run()
    assert spy_ai["n"] == 1


def test_model_change_invalidates_results(spy_ai, monkeypatch):
    at = _app()
    _demo_analysis(at)
    at.session_state["use_ai"] = True
    at.run()
    assert spy_ai["n"] == 1

    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    at.run()
    assert spy_ai["n"] == 2  # model changed -> deliberate fresh run


def test_profile_switch_invalidates_results(spy_ai):
    at = _app()
    _demo_analysis(at)
    at.session_state["use_ai"] = True
    at.run()
    assert spy_ai["n"] == 1

    # switch to the sample profile -> new curriculum -> re-analyzes
    at.radio(key="nav").set_value("Home")
    at.run()
    _btn(at, "home-analyze").click()
    at.run()
    _btn(at, "sel-sample").click()
    at.run()
    _btn(at, "sel-confirm").click()
    at.run()
    assert spy_ai["n"] == 2
    assert at.session_state["active_profile_id"] == "SAMPLE-WELD-001"


# ------------------------------------------------------------- ai disabled
def test_compare_shows_deterministic_when_ai_off():
    at = _app()
    _demo_analysis(at)
    at.radio(key="nav").set_value("AI Comparison")
    at.run()
    assert not at.exception
    text = " ".join(i.value for i in at.info)
    assert "AI enhancement" in text or "disabled" in text.lower() \
        or "OPENAI_API_KEY" in text
    assert len(at.dataframe) > 0  # deterministic results shown


def test_enable_ai_from_compare_page(spy_ai):
    at = _app()
    _demo_analysis(at)
    at.radio(key="nav").set_value("AI Comparison")
    at.run()
    _btn(at, "cmp-enable-ai").click()
    at.run()
    assert spy_ai["n"] == 1
    assert at.session_state["analysis"]["ai_status"] == "ok"


# ------------------------------------------------------------- error path
def test_ai_error_keeps_deterministic(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def boom(text, api_key=None, model=None):
        raise ai_client.AIClientError("auth failed", "auth")

    monkeypatch.setattr(ai_client, "analyze_curriculum", boom)
    at = _app()
    _demo_analysis(at)
    at.session_state["use_ai"] = True
    at.run()
    a = at.session_state["analysis"]
    assert a["ai_status"] == "error"
    assert "auth failed" in a["ai_error"]
    assert len(a["skills"]) > 0  # deterministic results intact

    at.radio(key="nav").set_value("AI Comparison")
    at.run()
    assert not at.exception
    text = " ".join(w.value for w in at.warning)
    assert "failed" in text.lower()
