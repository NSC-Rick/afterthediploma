"""Tests for the Regional Intelligence engine."""

from pathlib import Path

import pytest
import pymupdf
from streamlit.testing.v1 import AppTest

from modules import employers, intel
from modules.data_sources import REFERENCE_POINT, get_provider
from modules.report import build_report
from tests.test_report import _payload, GEO

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_layers_partition_exclusively():
    assert intel.layer_for(0) == "L1"
    assert intel.layer_for(24.9) == "L1"
    assert intel.layer_for(25) == "L2"
    assert intel.layer_for(49.9) == "L2"
    assert intel.layer_for(50) == "L3"
    assert intel.layer_for(100) == "L3"
    assert intel.layer_for(101) is None  # beyond max radius


def test_scope_derives_from_curriculum():
    provider = get_provider()
    from modules import scp, skills, careers
    p = scp.load_profile(Path("data/profiles/LI-WELD-001.json"))
    sk = scp.apply_profile_overrides(
        p, skills.extract_skills_rules(scp.corpus_text(p), provider.skills_taxonomy()),
        provider.skills_taxonomy())
    ca = careers.match_occupations(sk, provider.occupations())
    scope = intel.research_scope({"skills": sk}, sk, ca)
    assert "MIG welding (GMAW)" in scope["skills"] or len(scope["skills"]) > 5
    assert scope["primary_occupations"] and scope["adjacent_occupations"]
    assert "51-4121.00" in scope["soc_codes"]
    assert "metal fabrication" in scope["industries"]


def test_research_assigns_single_layer_per_observation():
    res = intel.run_research(
        {"soc_codes": ["51-4121.00"], "skills": [], "industries": []}, GEO)
    seen = {}
    for key, obs in res["by_layer"].items():
        for o in obs:
            assert o.obs_id not in seen
            seen[o.obs_id] = key
            assert o.layer == key
    total = sum(len(v) for v in res["by_layer"].values()) + len(res["unlocated"])
    assert total == len(res["observations"])
    # unlocated obs (VT DOL, no coords) is retained, not forced into a layer
    assert any(o.obs_id == "OBS-011" for o in res["unlocated"])


def test_summaries_are_geographically_distinct():
    res = intel.run_research(
        {"soc_codes": ["51-4121.00"], "skills": [], "industries": []}, GEO)
    s1 = intel.summarize_layer("L1", res["by_layer"]["L1"], res["scope"])
    s3 = intel.summarize_layer("L3", res["by_layer"]["L3"], res["scope"])
    assert s1 != s3 and "0-25" in s1 and "50-100" in s3


def test_empty_layer_states_insufficiency():
    s = intel.summarize_layer("L2", [], {})
    assert "No evidence-backed" in s or "not" in s.lower()


def test_pdf_omits_or_includes_intel_section():
    analysis, profile, ds = _payload()
    pdf_no = build_report(analysis, profile, ds, GEO)
    t_no = " ".join(p.get_text() for p in
                    pymupdf.open(stream=pdf_no, filetype="pdf"))
    assert "not been collected" in t_no
    res = intel.run_research({"soc_codes": []}, GEO)
    ds2 = dict(ds, intel=res)
    pdf_yes = build_report(analysis, profile, ds2, GEO)
    t_yes = " ".join(p.get_text() for p in
                     pymupdf.open(stream=pdf_yes, filetype="pdf"))
    assert "Regional Intelligence Outlook" in t_yes and "Immediate Region" in t_yes


def test_module_page_no_research_without_button():
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    [b for b in at.button if b.key == "home-demo"][0].click(); at.run()
    [b for b in at.button if b.key == "sel-confirm"][0].click(); at.run()
    at.radio(key="nav").set_value("Regional Intelligence"); at.run()
    assert not at.exception
    assert at.session_state.get("intel_result") is None  # nothing collected
    # deliberate run
    [b for b in at.button if b.key == "intel-run"][0].click(); at.run()
    res = at.session_state["intel_result"]
    assert res and len(res["by_layer"]) == 3
    at.radio(key="nav").set_value("Executive Dashboard"); at.run()
    at.radio(key="nav").set_value("Regional Intelligence"); at.run()
    assert at.session_state["intel_result"] is res  # reused, no rerun
