"""Tests for career pathway matching and score explainability."""

import pytest

from modules import careers
from modules.curriculum import load_sample_curriculum
from modules.data_sources import get_provider
from modules.skills import extract_skills_rules


@pytest.fixture(scope="module")
def scored():
    provider = get_provider()
    text = load_sample_curriculum()
    sk = extract_skills_rules(text, provider.skills_taxonomy())
    return careers.match_occupations(sk, provider.occupations())


def test_all_pathway_categories_present(scored):
    assert set(scored["pathway"]) == {"primary", "adjacent", "extended"}


def test_scores_bounded(scored):
    assert scored["compatibility_score"].between(0, 100).all()


def test_welder_is_top_primary(scored):
    primary = scored[scored.pathway == "primary"].sort_values(
        "compatibility_score", ascending=False
    )
    assert "Welder" in primary.iloc[0]["title"]


def test_score_reproducible_from_breakdown(scored):
    for _, row in scored.iterrows():
        expected = round(
            100 * (row["n_documented"] * 1.0 + row["n_inferred"] * 0.5)
            / row["n_required"],
            1,
        )
        assert abs(row["compatibility_score"] - expected) < 0.05


def test_explanation_discloses_limitations(scored):
    text = careers.explain_score(scored.iloc[0])
    assert "not a validated employment probability" in text
