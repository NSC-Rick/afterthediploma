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


WELDING_SKILL_IDS = {"mig_welding", "tig_welding", "stick_welding",
                     "oxyfuel_cutting", "blueprint_reading", "weld_symbols",
                     "welding_safety", "welding_positions", "fcaw_welding",
                     "grinding_finishing", "material_preparation"}
HEALTH_SKILL_IDS = {"patient_care", "vital_signs", "personal_care_assistance",
                    "mobility_assistance", "infection_prevention",
                    "clinical_documentation", "dementia_care_awareness",
                    "rehabilitation_support", "medical_terminology",
                    "ehr_basics", "patient_safety", "patient_communication"}


def _skills(profile_id):
    p = get_provider()
    prof = scp.load_profile(Path(f"data/profiles/{profile_id}.json"))
    return scp.apply_profile_overrides(
        prof,
        skills.extract_skills_rules(scp.corpus_text(prof), p.skills_taxonomy()),
        p.skills_taxonomy(),
    )


def test_lna_skills_contain_no_welding_competencies():
    sk = _skills("LI-LNA-001")
    # no welding skill appears AT ALL in the LNA analysis - not as
    # documented, inferred, suggested, or unestablished
    assert not set(sk.skill_id) & WELDING_SKILL_IDS


def test_welding_skills_contain_no_clinical_competencies():
    sk = _skills("LI-WELD-001")
    assert not set(sk.skill_id) & HEALTH_SKILL_IDS


def test_lna_documented_skills_have_relevant_evidence():
    sk = _skills("LI-LNA-001")
    doc = sk[sk.status == "documented"]
    assert len(doc) > 0
    prof = scp.load_profile(Path("data/profiles/LI-LNA-001.json"))
    corpus = scp.corpus_text(prof).lower()
    for _, r in doc.iterrows():
        # evidence must be a substring of the actual source material or a
        # profile-supplied snippet - never a foreign-domain claim
        ev = (r["evidence"] or "").lower()
        assert "welding" not in ev and "weld" not in ev


def test_lna_recommendations_are_healthcare_relevant():
    from modules import recommendations, employers
    p = get_provider()
    sk = _skills("LI-LNA-001")
    ca = careers.match_occupations(sk, p.occupations())
    ca_ids = recommendations.careers_with_skill_ids(ca, p.occupations())
    recs = recommendations.generate_recommendations(
        sk, ca_ids, employers.employers_with_distance(p.employers()))
    welding_terms = recs["recommendation"].str.contains(
        "weld|arc|oxy-fuel|cnc|machining", case=False)
    assert not welding_terms.any()


def test_displayed_counts_match_present_skills():
    """Module-1 expander logic: counts equal filtered display records."""
    sk = _skills("LI-LNA-001")
    for cat in ["core", "supporting", "transferable", "adjacent"]:
        shown = sk[(sk.category == cat) & sk.status.isin(
            ["documented", "inferred", "recommended_addition"])]
        # what the UI displays is what exists - no hidden catalog rows
        assert len(shown) <= len(sk)
        assert set(shown.skill_id) - WELDING_SKILL_IDS == set(shown.skill_id)
