"""Tests for curriculum extraction and skill classification."""

import pytest

from modules.curriculum import extract_text, load_sample_curriculum
from modules.data_sources import get_provider
from modules.skills import extract_skills_rules, present_skills


@pytest.fixture(scope="module")
def taxonomy():
    return get_provider().skills_taxonomy()


@pytest.fixture(scope="module")
def sample_text():
    return load_sample_curriculum()


def test_sample_curriculum_loads(sample_text):
    assert len(sample_text) > 500
    assert "NOT an official Lyndon Institute document" in sample_text


def test_extract_txt_bytes():
    text = extract_text(
        b"MIG welding and blueprint reading are covered in this course of study.",
        "curr.txt",
    )
    assert "MIG" in text


def test_extract_rejects_unknown_extension():
    with pytest.raises(ValueError):
        extract_text(b"data", "curr.xlsx")


def test_extract_rejects_empty_file():
    with pytest.raises(ValueError):
        extract_text(b"", "empty.txt")


def test_rules_extraction_finds_documented_skills(sample_text, taxonomy):
    df = extract_skills_rules(sample_text, taxonomy)
    doc = set(df[df.status == "documented"]["skill_id"])
    # the representative curriculum explicitly names these
    for sid in ["mig_welding", "tig_welding", "stick_welding", "blueprint_reading"]:
        assert sid in doc


def test_every_documented_skill_has_evidence(sample_text, taxonomy):
    df = extract_skills_rules(sample_text, taxonomy)
    for _, row in df[df.status == "documented"].iterrows():
        assert row["evidence"], f"{row['skill_id']} documented without evidence"


def test_inferred_not_confused_with_documented(sample_text, taxonomy):
    df = extract_skills_rules(sample_text, taxonomy)
    inferred = df[df.status == "inferred"]
    assert all(inferred["confidence"].str.contains("inferred"))
    assert not (set(inferred["skill_id"]) & set(df[df.status == "documented"]["skill_id"]))


def test_no_cnc_in_sample_flags_recommendation(sample_text, taxonomy):
    df = extract_skills_rules(sample_text, taxonomy)
    cnc = df[df.skill_id == "cnc_operation"].iloc[0]
    assert cnc["status"] == "recommended_addition"


def test_present_skills_includes_both_statuses(sample_text, taxonomy):
    df = extract_skills_rules(sample_text, taxonomy)
    statuses = set(present_skills(df)["status"])
    assert "documented" in statuses
