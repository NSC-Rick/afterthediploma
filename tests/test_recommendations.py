"""Tests for the curriculum improvement advisor."""

import pytest

from modules import careers, employers, recommendations
from modules.curriculum import load_sample_curriculum
from modules.data_sources import get_provider
from modules.skills import extract_skills_rules


@pytest.fixture(scope="module")
def recs_df():
    provider = get_provider()
    text = load_sample_curriculum()
    sk = extract_skills_rules(text, provider.skills_taxonomy())
    ca = careers.match_occupations(sk, provider.occupations())
    ca_ids = recommendations.careers_with_skill_ids(ca, provider.occupations())
    emp = employers.employers_with_distance(provider.employers())
    return recommendations.generate_recommendations(sk, ca_ids, emp)


def test_recommendations_generated(recs_df):
    assert len(recs_df) >= 5


def test_all_groups_labeled(recs_df):
    assert set(recs_df["group"]).issubset(
        {"Quick wins", "Moderate enhancements", "Strategic opportunities"}
    )


def test_recommendations_are_advisory(recs_df):
    assert (recs_df["data_status"].str.contains("advisory")).all()
    for _, r in recs_df.iterrows():
        assert r["confidence_limitations"], "missing confidence/limitations"


def test_cnc_recommendation_fires_for_sample(recs_df):
    assert recs_df["recommendation"].str.contains("CNC").any()
