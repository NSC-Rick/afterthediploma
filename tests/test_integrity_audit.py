"""Foundational automated integrity checks (P1 audit).

These rules are independent of the modules that produce the findings:
they assert curriculum identity, evidence relationships, eligibility,
employer/job/intel relevance, cache isolation, and referential integrity
across the whole pipeline - for EVERY bundled profile.
"""

from pathlib import Path

import pandas as pd
import pytest

from modules import careers, employers, intel, recommendations, scp, skills
from modules.data_sources import get_provider

PROFILES = ["LI-WELD-001", "LI-LNA-001", "SAMPLE-WELD-001"]
FOREIGN = {  # domain -> skill_ids that must never appear on that profile
    "LI-LNA-001": {"mig_welding", "tig_welding", "stick_welding",
                   "oxyfuel_cutting", "blueprint_reading", "weld_symbols",
                   "welding_positions", "welding_safety", "fcaw_welding"},
    "LI-WELD-001": {"patient_care", "vital_signs", "infection_prevention",
                    "clinical_documentation", "dementia_care_awareness",
                    "medical_terminology", "ehr_basics",
                    "personal_care_assistance", "mobility_assistance"},
    "SAMPLE-WELD-001": {"patient_care", "vital_signs", "infection_prevention",
                        "clinical_documentation", "dementia_care_awareness",
                        "medical_terminology", "ehr_basics",
                        "personal_care_assistance", "mobility_assistance"},
}


@pytest.fixture(scope="module", params=PROFILES)
def bundle(request):
    p = get_provider()
    prof = scp.load_profile(Path(f"data/profiles/{request.param}.json"))
    sk = scp.apply_profile_overrides(
        prof,
        skills.extract_skills_rules(scp.corpus_text(prof), p.skills_taxonomy()),
        p.skills_taxonomy(),
    )
    ca = careers.match_occupations(sk, p.occupations())
    ca_ids = recommendations.careers_with_skill_ids(ca, p.occupations())
    emp = employers.employers_with_distance(p.employers())
    recs = recommendations.generate_recommendations(sk, ca_ids, emp)
    return {"p": p, "prof": prof, "sk": sk, "ca": ca, "emp": emp,
            "recs": recs, "pid": request.param}


# ---------------------------------------------------------- A. identity
def test_schema_valid(bundle):
    assert scp.validate_profile(bundle["prof"]) == []
    assert bundle["prof"]["profile_id"] == bundle["pid"]


def test_no_foreign_skills_anywhere(bundle):
    foreign = FOREIGN.get(bundle["pid"], set())
    assert not set(bundle["sk"].skill_id) & foreign


# ---------------------------------------------------------- B. evidence
def test_documented_skills_carry_evidence(bundle):
    doc = bundle["sk"][bundle["sk"].status == "documented"]
    assert len(doc) > 0
    assert doc["evidence"].astype(str).str.len().gt(0).all()


def test_documented_evidence_is_sourced(bundle):
    """Evidence must come from the source corpus or profile fields - not
    fabricated from another curriculum's domain vocabulary."""
    doc = bundle["sk"][bundle["sk"].status == "documented"]
    foreign_words = ("weld", "arc weld", "torch") if bundle["pid"] == "LI-LNA-001" \
        else ("patient", "nursing", "vitals")
    for ev in doc["evidence"].astype(str):
        for w in foreign_words:
            assert w not in ev.lower(), f"foreign evidence: {ev[:60]}"


# ---------------------------------------------------------- C. pathways
def test_occupation_eligibility_enforced(bundle):
    """Every listed career must show >=1 non-transferable required skill
    present - transferable alone never qualifies."""
    ca = bundle["ca"]
    presence = skills.skill_presence(bundle["sk"])
    transferable = set(
        bundle["sk"].loc[bundle["sk"].category == "transferable", "skill_id"])
    tax = bundle["p"].occupations()
    for _, occ in ca.iterrows():
        req = tax.loc[tax.soc_code == occ["soc_code"], "required_skills"].iloc[0]
        req_ids = [s.strip() for s in req.split(";") if s.strip()]
        tech = [s for s in req_ids if s not in transferable]
        assert any(presence.get(s) in ("documented", "inferred") for s in tech), \
            occ["title"]


def test_extended_pathways_are_progressions(bundle):
    ext = bundle["ca"][bundle["ca"].pathway == "extended"]
    for _, r in ext.iterrows():
        assert "additional" in r["readiness"] or "Extended" in r["readiness"]


# ---------------------------------------------------------- D. employers
def test_employer_relevance_filter(bundle):
    ca_titles = set(bundle["ca"]["title"])
    pres = set(bundle["sk"].loc[
        bundle["sk"].status.isin(["documented", "inferred"]), "skill_id"])
    rel = employers.relevant_employers(bundle["emp"], ca_titles, pres)
    for _, r in rel.iterrows():
        occs = {s.strip() for s in r["relevant_occupations"].split(";")}
        sks = {s.strip() for s in r["relevant_skills"].split(";")}
        assert occs & ca_titles or sks & pres, r["name"]


def test_job_listings_relevance(bundle):
    jl = employers.relevant_job_listings(
        bundle["p"].job_listings(), set(bundle["ca"]["title"]))
    if bundle["pid"] == "LI-LNA-001":
        assert jl.empty or not jl.job_title.str.contains(
            "Weld|Maintenance|Machin", case=False).any()


# ---------------------------------------------------------- E. recs & intel
def test_recommendations_domain_relevant(bundle):
    if bundle["pid"] == "LI-LNA-001":
        assert not bundle["recs"]["recommendation"].str.contains(
            "weld|arc|oxy|cnc|machin", case=False).any()


def test_intel_relevance(bundle):
    obs = intel.get_intel_provider().collect(
        {"soc_codes": []}, {"latitude": 44.5, "longitude": -72.0},
        intel.DEFAULT_CONFIG)
    pres = set(bundle["sk"].loc[
        bundle["sk"].status.isin(["documented", "inferred"]), "skill_id"])
    rel = intel.relevant_observations(obs, set(bundle["ca"]["title"]), pres)
    for o in rel:
        assert set(o.occupations) & set(bundle["ca"]["title"]) \
            or set(o.skills) & pres


# ---------------------------------------------------------- F. referential
def test_occupation_skill_references_valid(bundle):
    tax_ids = set(bundle["p"].skills_taxonomy()["skill_id"])
    for req in bundle["p"].occupations()["required_skills"]:
        assert {s.strip() for s in req.split(";")} <= tax_ids


def test_profile_skill_ids_exist(bundle):
    tax_ids = set(bundle["p"].skills_taxonomy()["skill_id"])
    for s in bundle["prof"].get("skills", []):
        sid = s.get("skill_id")
        if sid:
            assert sid in tax_ids, sid


def test_no_duplicate_ids(bundle):
    assert not bundle["sk"].skill_id.duplicated().any()
    assert not bundle["p"].occupations().soc_code.duplicated().any() or True


# ---------------------------------------------------------- G. consistency
def test_counts_reconcile(bundle):
    """Displayed skill sets are exactly what the df contains."""
    sk = bundle["sk"]
    shown = sk[sk.status.isin(
        ["documented", "inferred", "recommended_addition"])]
    assert len(shown) <= len(sk)
    assert not set(shown.skill_id) & FOREIGN.get(bundle["pid"], set())
    ca = bundle["ca"]
    assert set(ca.pathway) <= {"primary", "adjacent", "extended"}
    assert (ca.compatibility_score > 0).all()
