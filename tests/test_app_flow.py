"""User-journey tests: initial state, selection gating, navigation, switching.

Uses Streamlit's AppTest harness to drive the real UI headlessly.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from modules import scp

APP = str(Path(__file__).resolve().parent.parent / "app.py")

PRE_OPTIONS = {"Home", "Select Curriculum", "Curriculum Library", "Data & Methodology"}
MODULE_OPTIONS = {
    "Executive Dashboard",
    "1 · Curriculum Intelligence",
    "2 · Career Pathways",
    "3 · Regional Employment",
    "4 · Earnings Intelligence",
    "5 · Curriculum Advisor",
}


@pytest.fixture(autouse=True)
def _no_disk_writes(monkeypatch):
    """App flows save profiles via scp.save_profile - keep tests off disk."""
    monkeypatch.setattr(scp, "save_profile", lambda *a, **k: None)


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    return at


def _btn(at: AppTest, key: str):
    matches = [b for b in at.button if b.key == key]
    assert matches, f"button {key!r} not found"
    return matches[0]


def _analyze_via_demo(at: AppTest):
    _btn(at, "home-demo").click()
    at.run()
    _btn(at, "sel-confirm").click()
    at.run()


# ------------------------------------------------------------- initial state
def test_initial_state_is_home_without_analysis():
    at = _app()
    assert not at.exception
    options = set(at.radio(key="nav").options)
    assert options == PRE_OPTIONS
    assert "Executive Dashboard" not in options
    assert len(at.metric) == 0  # no dashboard metrics before analysis


def test_welding_not_auto_selected():
    at = _app()
    assert at.session_state["active_profile_id"] is None
    assert at.session_state["analysis"] is None


# ------------------------------------------------------------- gated modules
def test_intelligence_modules_require_selection():
    at = _app()
    for gated in MODULE_OPTIONS:
        assert gated not in set(at.radio(key="nav").options)


# ------------------------------------------------------------- demo journey
def test_demo_journey_unlocks_analysis():
    at = _app()
    _analyze_via_demo(at)
    assert at.radio(key="nav").value == "Executive Dashboard"
    options = set(at.radio(key="nav").options)
    assert MODULE_OPTIONS.issubset(options)
    assert len(at.metric) > 0  # dashboard metrics rendered
    assert not at.exception


def test_review_step_before_analysis():
    at = _app()
    _btn(at, "home-demo").click()
    at.run()
    assert at.radio(key="nav").value == "Select Curriculum"
    # review summary renders institution + source type before any analysis
    text = " ".join(m.value for m in at.markdown)
    assert "Lyndon Institute" in text
    assert "Source type" in text
    assert at.session_state["analysis"] is None


# ------------------------------------------------------------- navigation
def test_return_home_keeps_analysis():
    at = _app()
    _analyze_via_demo(at)
    at.radio(key="nav").set_value("Home")
    at.run()
    assert set(at.radio(key="nav").options).issuperset(MODULE_OPTIONS)
    assert at.session_state["analysis"] is not None
    # home page shows continue option while analysis is active
    text = " ".join(
        [m.value for m in at.markdown] + [i.value for i in at.info]
    )
    assert "already active" in text


def test_explore_mode_lists_profiles():
    at = _app()
    _btn(at, "home-explore").click()
    at.run()
    assert at.radio(key="nav").value == "Select Curriculum"
    sel = at.selectbox(key="sel-explore")
    assert any("LI-WELD-001" in str(o) for o in sel.options)


def test_switching_profiles_replaces_analysis():
    at = _app()
    _analyze_via_demo(at)
    assert at.session_state["active_profile_id"] == "LI-WELD-001"
    # switch: home -> analyze -> create profile from sample -> confirm
    at.radio(key="nav").set_value("Home")
    at.run()
    _btn(at, "home-analyze").click()
    at.run()
    _btn(at, "sel-sample").click()
    at.run()
    _btn(at, "sel-confirm").click()
    at.run()
    assert at.session_state["active_profile_id"] == "SAMPLE-WELD-001"
    assert at.radio(key="nav").value == "Executive Dashboard"
    assert not at.exception


def test_modules_render_after_analysis():
    at = _app()
    _analyze_via_demo(at)
    for page in MODULE_OPTIONS - {"Executive Dashboard"}:
        at.radio(key="nav").set_value(page)
        at.run()
        assert not at.exception, f"{page} raised: {at.exception}"


# ------------------------------------------------------------- geo config
def test_geographic_reference_configurable():
    at = _app()
    _analyze_via_demo(at)
    assert "Lyndon" in at.session_state["geo_ref"]["name"]
    at.session_state["geo_ref"]["name"] = "Test reference"
    at.session_state["geo_ref"]["latitude"] = 44.0
    at.session_state["geo_ref"]["longitude"] = -72.5
    at.radio(key="nav").set_value("3 · Regional Employment")
    at.run()
    text = " ".join(
        [m.value for m in at.markdown] + [c.value for c in at.caption]
    )
    assert "Test reference" in text
