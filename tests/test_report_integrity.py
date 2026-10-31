"""PDF report integrity: no cross-curriculum terminology, reconciled counts."""

import re

import pymupdf
import pytest

from modules.report import build_report
from tests.test_report import GEO, _payload


def _text_of(pdf: bytes) -> str:
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return " ".join(" ".join(p.get_text().split()) for p in doc)


def test_lna_pdf_contains_no_welding_terminology():
    analysis, profile, ds = _payload("LI-LNA-001")
    text = _text_of(build_report(analysis, profile, ds, GEO)).lower()
    for term in ["welding", "mig", "tig", "oxy-fuel", "fabricat",
                 "blueprint", "flux-cored", "smaw", "gmaw"]:
        assert term not in text, f"foreign term in LNA report: {term}"


def test_welding_pdf_contains_no_clinical_terminology():
    analysis, profile, ds = _payload()
    text = _text_of(build_report(analysis, profile, ds, GEO)).lower()
    for term in ["patient care", "nursing", "vital signs", "dementia",
                 "clinical documentation"]:
        assert term not in text, f"foreign term in welding report: {term}"


@pytest.mark.parametrize("pid", ["LI-WELD-001", "LI-LNA-001"])
def test_pdf_skill_counts_match_analysis(pid):
    analysis, profile, ds = _payload(pid)
    text = _text_of(build_report(analysis, profile, ds, GEO))
    sk = analysis["skills"]
    n_doc = int((sk.status == "documented").sum())
    n_inf = int((sk.status == "inferred").sum())
    assert f"documents {n_doc} skills" in text
    # domain-scope line: doc + inf + rec + und == len(sk)
    m = re.search(r"(\d+) documented \+ (\d+) inferred \+ (\d+) suggested "
                  r"additions \+ (\d+) not established", text)
    assert m, "domain-scope summary line missing"
    assert sum(int(g) for g in m.groups()) == len(sk)
    assert f"{len(sk)} skills tracked" in text


def test_pdf_exec_summary_derives_from_analysis():
    analysis, profile, ds = _payload("LI-LNA-001")
    text = _text_of(build_report(analysis, profile, ds, GEO))
    top = analysis["careers"].nlargest(1, "compatibility_score").iloc[0]
    assert top["title"] in text
    assert f"{top['compatibility_score']}/100" in text


def test_pdf_wage_coverage_disclosed():
    analysis, profile, ds = _payload("LI-LNA-001")
    text = _text_of(build_report(analysis, profile, ds, GEO))
    # healthcare wage rows lack p25/median/p75 - must be disclosed, not hidden
    assert "Coverage limitation" in text or "not publish" in text


def test_pdf_zero_intel_is_coverage_caveat_not_absence():
    from modules import intel as intel_mod
    analysis, profile, ds = _payload("LI-LNA-001")
    res = intel_mod.run_research({"soc_codes": []}, GEO)
    pdf = build_report(analysis, profile, dict(ds, intel=res), GEO)
    text = _text_of(pdf)
    assert "NOT a verified absence" in text or "limited coverage" in text


def test_pdf_engine_mode_stated():
    analysis, profile, ds = _payload("LI-LNA-001", use_ai=False)
    text = _text_of(build_report(analysis, profile, ds, GEO))
    assert "Deterministic engine" in text
