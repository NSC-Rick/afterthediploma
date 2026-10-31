"""Regression: cross-curriculum pathway contamination must not recur.

A pathway passes an eligibility test (>=1 non-transferable required skill
present) before receiving an alignment score.
"""

from pathlib import Path

import pytest

from modules import careers, scp, skills
from modules.data_sources import get_provider

WELDING_TITLES = {"Welder / Welding Fabricator", "Welding Machine Setter, Operator, Tender",
                  "Structural Metal Fabricator and Fitter", "Sheet Metal Worker",
                  "Tool and Die Maker", "Certified Welding Inspector (AWS CWI)"}
NURSING_TITLES = {"Nursing Assistant (CNA/LNA)", "Home Health Aide", "Medical Assistant",
                  "Licensed Practical Nurse (LPN)", "Registered Nurse (RN)"}
MFG_TITLES = {"CNC Machine Tool Operator", "Mechanical Assembly Technician",
              "Industrial Maintenance Technician", "Quality Control Inspector",
              "Manufacturing / Production Supervisor",
              "Engineering Technician (Mechanical / Manufacturing)"}


def _careers(profile_id):
    p = get_provider()
    prof = scp.load_profile(Path(f"data/profiles/{profile_id}.json"))
    sk = scp.apply_profile_overrides(
        prof,
        skills.extract_skills_rules(scp.corpus_text(prof), p.skills_taxonomy()),
        p.skills_taxonomy(),
    )
    return careers.match_occupations(sk, p.occupations())


def test_lna_excludes_welding_occupations():
    titles = set(_careers("LI-LNA-001")["title"])
    assert titles.isdisjoint(WELDING_TITLES)
    assert titles.isdisjoint(MFG_TITLES)


def test_welding_excludes_nursing_occupations():
    titles = set(_careers("LI-WELD-001")["title"])
    assert titles.isdisjoint(NURSING_TITLES)


def test_lna_keeps_legitimate_healthcare_pathways():
    ca = _careers("LI-LNA-001")
    assert "Nursing Assistant (CNA/LNA)" in set(
        ca[ca.pathway == "primary"]["title"])
    # LPN and RN are extended - not immediate employment options
    extended = set(ca[ca.pathway == "extended"]["title"])
    assert {"Licensed Practical Nurse (LPN)", "Registered Nurse (RN)"} <= extended
    assert "Nursing Assistant (CNA/LNA)" not in extended


def test_transferable_alone_does_not_qualify():
    """A curriculum carrying only transferable skills cannot produce pathways."""
    p = get_provider()
    tax = p.skills_taxonomy()
    trans = tax[tax.category == "transferable"]
    fake = trans.assign(status="documented", evidence="test", confidence="",
                        rationale="")[["skill_id", "skill_name", "category",
                                       "status", "evidence", "confidence", "rationale"]]
    ca = careers.match_occupations(fake, p.occupations())
    assert len(ca) == 0


def test_scores_recomputed_after_filtering():
    ca = _careers("LI-LNA-001")
    assert (ca["compatibility_score"] > 0).all()
    assert set(ca.pathway) <= {"primary", "adjacent", "extended"}
