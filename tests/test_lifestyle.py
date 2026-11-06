"""Career Lifestyle Profile: evidence-scored dimensions, null handling."""

from modules import lifestyle
from tests.test_report import _payload
import pandas as pd


def _profiles(pid):
    from modules import earnings, employers
    from modules.data_sources import get_provider
    a, prof, ds = _payload(pid)
    ca, p = a["careers"], get_provider()
    pres = set(a["skills"].loc[
        a["skills"].status.isin(["documented", "inferred"]), "skill_id"])
    emp = employers.relevant_employers(
        employers.employers_with_distance(p.employers()),
        set(ca["title"]), pres)
    pw = earnings.pathway_wages(ca, p.wages())
    return {r["soc_code"]: lifestyle.build_profile(
        r, ca, pw[pw.soc_code == r["soc_code"]].iloc[0].to_dict()
        if (pw.soc_code == r["soc_code"]).any() else {},
        emp, "Vermont") for _, r in ca.iterrows()}


def test_schema_and_versioning():
    prof = list(_profiles("LI-WELD-001").values())[0]
    assert prof["profile_version"] == "1.0"
    assert prof["methodology_version"] == "1.0"
    assert len(prof["dimensions"]) == 8
    ids = {d["id"] for d in prof["dimensions"]}
    assert ids == {"FIN", "LOC", "WLB", "GRW", "STB", "AUT", "FUL", "COM"}


def test_scores_bounded_and_scorable_subset():
    for prof in _profiles("LI-WELD-001").values():
        for d in prof["dimensions"]:
            if d["score"] is not None:
                assert 0 <= d["score"] <= 100
                assert d["id"] in lifestyle.SCORABLE
                assert d["band"] in {
                    "More constrained", "Some opportunities",
                    "Moderate opportunities", "Broad opportunities",
                    "Extensive opportunities"}


def test_missing_data_returns_null_not_midpoint():
    prof = list(_profiles("LI-LNA-001").values())[0]
    wlb = next(d for d in prof["dimensions"] if d["id"] == "WLB")
    assert wlb["score"] is None
    assert wlb["evidence_status"] == "insufficient_evidence"
    assert wlb["limitations"]  # explains what is missing


def test_ful_com_never_scored():
    for prof in _profiles("LI-AUTO-001").values():
        for d in prof["dimensions"]:
            if d["id"] in ("FUL", "COM"):
                assert d["score"] is None


def test_fin_is_derived_and_documented():
    profs = _profiles("LI-WELD-001")
    prof = profs["51-4121.00"]  # Welder - has published median
    fin = next(d for d in prof["dimensions"] if d["id"] == "FIN")
    assert fin["score"] is not None
    assert fin["evidence_status"] == "derived"
    assert any("living_wage_benchmark" in i for i in fin["indicators"])
    # derived value is reproducible
    assert abs(fin["score"] - min(100, 60890 / 48800 * 60)) < 0.5


def test_radar_suppression_when_few_scored():
    prof = list(_profiles("LI-LNA-001").values())[0]
    q = lifestyle.quantitative_dims(prof)
    assert all(d["score"] is not None for d in q)
    assert len(q) >= 1


def test_caveat_and_reflection_present():
    prof = list(_profiles("LI-LNA-001").values())[0]
    assert "not an absolute assessment" in lifestyle.CAVEAT
    assert len(prof["student_reflection"]) >= 2


def test_profiles_differ_across_curricula():
    weld = _profiles("LI-WELD-001")
    lna = _profiles("LI-LNA-001")
    assert set(weld) != set(lna)  # different occupations entirely
