"""Dashboard pathway tooltip: dynamic content per curriculum, all three layers."""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location("appmod", ROOT / "app.py")
_appmod = importlib.util.module_from_spec(spec)
try:  # app.py executes Streamlit calls on import; tolerate them
    spec.loader.exec_module(_appmod)
except Exception:
    pass
_pathway_tooltips = _appmod._pathway_tooltips

from tests.test_report import _payload


@pytest.mark.parametrize("profile_id", ["LI-WELD-001", "LI-LNA-001"])
def test_tooltips_cover_all_categories(profile_id):
    analysis, _, _ = _payload(profile_id)
    tips = _pathway_tooltips(analysis["careers"])
    assert len(tips) == 3 and all(isinstance(t, str) for t in tips)
    ca = analysis["careers"]
    for tip, cat in zip(tips, ["primary", "adjacent", "extended"]):
        sub = ca[ca.pathway == cat]
        for _, r in sub.iterrows():
            assert r["title"] in tip
            assert f"{r['compatibility_score']}/100" in tip


def test_lna_tooltip_names():
    analysis, _, _ = _payload("LI-LNA-001")
    tips = _pathway_tooltips(analysis["careers"])
    assert "Nursing Assistant (CNA/LNA)" in tips[0]          # primary
    assert "Licensed Practical Nurse" in tips[2] or \
        "Registered Nurse (RN)" in tips[2]                  # extended


def test_welding_tooltip_names():
    analysis, _, _ = _payload()
    tips = _pathway_tooltips(analysis["careers"])
    assert "Welder / Welding Fabricator" in tips[0]


@pytest.mark.parametrize("profile_id", ["LI-WELD-001", "LI-LNA-001"])
def test_tooltip_entry_count_equals_category_count(profile_id):
    """Regression: 'Pathways: N' must list exactly N careers - not one."""
    analysis, _, _ = _payload(profile_id)
    ca = analysis["careers"]
    tips = _pathway_tooltips(ca)
    for tip, cat in zip(tips, ["primary", "adjacent", "extended"]):
        n_rows = int((ca.pathway == cat).sum())
        n_listed = tip.count("/100")
        assert n_listed == n_rows, (
            f"{cat}: bar shows {n_rows} but tooltip lists {n_listed}")


@pytest.mark.parametrize("profile_id", ["LI-WELD-001", "LI-LNA-001"])
def test_tooltips_sorted_by_score(profile_id):
    analysis, _, _ = _payload(profile_id)
    for tip in _pathway_tooltips(analysis["careers"]):
        scores = [float(s) for s in re.findall(r"(\d+(?:\.\d+)?)/100", tip)]
        assert scores == sorted(scores, reverse=True)


def test_tooltips_not_hardcoded():
    """Two curricula produce different tooltip content."""
    t1 = _pathway_tooltips(_payload("LI-WELD-001")[0]["careers"])
    t2 = _pathway_tooltips(_payload("LI-LNA-001")[0]["careers"])
    assert t1 != t2
    # zero-score cross-domain occupations may appear in adjacent/extended
    # layers; the primary layer must stay curriculum-relevant
    assert "Nursing Assistant" not in t1[0]
