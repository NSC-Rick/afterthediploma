"""Tests for wage intelligence: sourcing, labeling, and joins."""

import pytest

from modules import careers, earnings
from modules.curriculum import load_sample_curriculum
from modules.data_sources import get_provider
from modules.skills import extract_skills_rules


@pytest.fixture(scope="module")
def pathway_wage_df():
    provider = get_provider()
    text = load_sample_curriculum()
    sk = extract_skills_rules(text, provider.skills_taxonomy())
    ca = careers.match_occupations(sk, provider.occupations())
    return earnings.pathway_wages(ca, provider.wages())


def test_every_occupation_has_wage_row(pathway_wage_df):
    assert pathway_wage_df["median_annual"].notna().all()


def test_vermont_scope_preferred(pathway_wage_df):
    welders = pathway_wage_df[pathway_wage_df.soc_code == "51-4121.00"].iloc[0]
    assert welders["wage_geographic_scope"] == "Vermont"
    assert welders["median_annual"] == 60890  # real BLS OEWS figure via O*NET


def test_national_fallback_labeled(pathway_wage_df):
    fab = pathway_wage_df[pathway_wage_df.soc_code == "51-2041.00"].iloc[0]
    # Vermont estimate unavailable for this SOC -> national fallback, labeled
    assert fab["wage_geographic_scope"] == "United States"


def test_percentile_ordering(pathway_wage_df):
    df = pathway_wage_df.dropna(subset=["pct10", "pct90"])
    assert (df["pct10"] < df["entry_proxy"]).all()
    assert (df["entry_proxy"] < df["median_annual"]).all()
    assert (df["median_annual"] < df["experienced_proxy"]).all()


def test_wages_carry_source_metadata(pathway_wage_df):
    assert pathway_wage_df["source"].str.contains("Bureau of Labor Statistics").all()
    assert (pathway_wage_df["data_status"] == "Verified source data").all()


def test_wage_range_summary(pathway_wage_df):
    lo, hi = earnings.wage_range_summary(pathway_wage_df)
    assert lo and hi and lo < hi
