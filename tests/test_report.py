"""Tests for the Curriculum Opportunity Report (PDF) service."""

from pathlib import Path

import pandas as pd
import pytest
import pymupdf

from modules import careers, earnings, employers, recommendations, report, scp, skills
from modules.data_sources import REFERENCE_POINT, get_provider
from modules.curriculum import load_sample_curriculum


GEO = {
    "name": REFERENCE_POINT["name"],
    "latitude": REFERENCE_POINT["latitude"],
    "longitude": REFERENCE_POINT["longitude"],
}


def _payload(profile_id: str = "LI-WELD-001", use_ai: bool = False):
    """Assemble the same inputs the app passes to build_report."""
    provider = get_provider()
    profile = scp.load_profile(Path(f"data/profiles/{profile_id}.json"))
    corpus = scp.corpus_text(profile)
    rules_df = skills.extract_skills_rules(corpus, provider.skills_taxonomy())
    final = scp.apply_profile_overrides(profile, rules_df, provider.skills_taxonomy())
    ca = careers.match_occupations(final, provider.occupations())
    ca_ids = recommendations.careers_with_skill_ids(ca, provider.occupations())
    emp = employers.employers_with_distance(provider.employers())
    recs = recommendations.generate_recommendations(final, ca_ids, emp)
    analysis = {
        "skills": final,
        "careers": ca,
        "careers_ids": ca_ids,
        "skills_rules": rules_df,
        "skills_ai": rules_df if use_ai else None,
        "careers_rules": ca,
        "careers_ai": ca if use_ai else None,
        "method": "Deterministic rules engine"
        + (" + AI enhancement" if use_ai else ""),
        "used_ai": use_ai,
        "ai_status": "ok" if use_ai else "disabled",
        "ai_error": None,
        "fingerprint": "testfp",
        "meta": {"model": "gpt-4o-mini" if use_ai else None,
                 "prompt_version": "1.0", "scp_version": "1.0",
                 "ran_at": "2026-10-31T09:00:00"},
    }
    datasets = {
        "wages": earnings.pathway_wages(ca, provider.wages()),
        "employers": emp,
        "recommendations": recs,
        "sources": provider.sources(),
        "job_listings": provider.job_listings(),
    }
    return analysis, profile, datasets


def _text(pdf: bytes) -> str:
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return "\n".join(p.get_text() for p in doc)


# ------------------------------------------------------------- generation
def test_report_is_valid_pdf():
    analysis, profile, ds = _payload()
    pdf = report.build_report(analysis, profile, ds, GEO)
    assert pdf[:5] == b"%PDF-"
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        assert doc.page_count >= 4  # multi-section professional document


def test_report_contains_identity_and_provenance():
    pdf = report.build_report(*_payload(), GEO)
    text = _text(pdf)
    assert "Lyndon Institute" in text
    assert "LI-WELD-001" in text
    assert "public program description" in text  # source type labeled
    assert "Bureau of Labor Statistics" in text  # wage source
    flat = " ".join(text.split())
    assert "not verified as currently hiring" in flat  # employer caveat
    assert "not confirmed student competencies" in flat.lower()


def test_skill_statuses_clearly_distinguished():
    pdf = report.build_report(*_payload(), GEO)
    text = _text(pdf)
    for label in ["Documented in public source", "Reasonably inferred",
                  "Recommended for consideration"]:
        assert label in text


def test_figures_match_application_data():
    analysis, profile, ds = _payload()
    text = _text(report.build_report(analysis, profile, ds, GEO))
    welder = analysis["careers"][
        analysis["careers"].title == "Welder / Welding Fabricator"
    ].iloc[0]
    assert f"{welder['compatibility_score']}/100" in text
    wage = ds["wages"][ds["wages"].soc_code == "51-4121.00"].iloc[0]
    assert f"${int(wage['median_annual']):,}" in text  # VT median $60,890


def test_report_without_ai_works():
    analysis, profile, ds = _payload(use_ai=False)
    assert analysis["skills_ai"] is None
    pdf = report.build_report(analysis, profile, ds, GEO)
    assert pdf[:5] == b"%PDF-"


def test_report_second_curriculum_profile():
    analysis, profile, ds = _payload("SAMPLE-WELD-001")
    pdf = report.build_report(analysis, profile, ds, GEO)
    text = _text(pdf)
    assert "SAMPLE-WELD-001" in text
    assert "Representative sample welding curriculum" in text


def test_report_handles_minimal_data():
    """Empty recs, no profile, empty skills -> still a valid labeled PDF."""
    provider = get_provider()
    empty_skills = pd.DataFrame(
        columns=["skill_id", "skill_name", "category", "status",
                 "evidence", "confidence", "rationale"])
    ca = careers.match_occupations(
        skills.extract_skills_rules("x" * 100, provider.skills_taxonomy()),
        provider.occupations())
    analysis = {
        "skills": empty_skills, "careers": ca,
        "careers_ids": ca, "skills_rules": empty_skills,
        "skills_ai": None, "careers_rules": ca, "careers_ai": None,
        "method": "Deterministic rules engine", "used_ai": False,
        "ai_status": "disabled", "ai_error": None, "fingerprint": "x",
        "meta": {},
    }
    ds = {
        "wages": earnings.pathway_wages(ca, provider.wages()),
        "employers": provider.employers().assign(distance_miles=999),
        "recommendations": pd.DataFrame(),
        "sources": provider.sources(),
        "job_listings": None,
    }
    pdf = report.build_report(analysis, None, ds, GEO)
    assert pdf[:5] == b"%PDF-"
    text = _text(pdf)
    assert "Not specified" in text  # missing institution, no crash


def test_report_no_invented_openings():
    pdf = report.build_report(*_payload(), GEO)
    text = " ".join(_text(pdf).split()).lower()
    assert "no job openings are asserted" in text
    assert "currently hiring" in text  # only inside the caveat
