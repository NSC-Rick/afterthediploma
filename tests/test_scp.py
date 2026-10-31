"""Tests for the Standard Curriculum Profile (SCP) architecture."""

import json

import pytest

from modules import scp, skills
from modules.data_sources import get_provider
from pathlib import Path


@pytest.fixture(scope="module")
def li_profile():
    return scp.load_profile(Path("data/profiles/LI-WELD-001.json"))


@pytest.fixture(scope="module")
def taxonomy():
    return get_provider().skills_taxonomy()


# ---------------------------------------------------------------- schema
def test_li_profile_validates(li_profile):
    assert scp.validate_profile(li_profile) == []


def test_schema_version_enforced(li_profile):
    bad = dict(li_profile, scp_version="9.9")
    assert scp.validate_profile(bad)


def test_missing_required_field_fails(li_profile):
    bad = {k: v for k, v in li_profile.items() if k != "institution"}
    assert any("institution" in e for e in scp.validate_profile(bad))


def test_invalid_skill_status_fails(li_profile):
    bad = json.loads(json.dumps(li_profile))
    bad["skills"][0]["status"] = "definitely_mastered"
    assert scp.validate_profile(bad)


def test_source_type_enum_enforced(li_profile):
    bad = json.loads(json.dumps(li_profile))
    bad["source"]["source_type"] = " vibes"
    assert scp.validate_profile(bad)


# ---------------------------------------------------------------- provenance
def test_source_type_is_public_description(li_profile):
    assert li_profile["source"]["source_type"] == "public_program_description"
    assert "not represent the complete" in li_profile["source"]["notes"].lower() \
        or "does NOT represent" in li_profile["source"]["notes"]


def test_original_source_preserved(li_profile):
    assert "stick arc welding" in li_profile["source"]["original_text"]


def test_hours_not_invented(li_profile):
    assert li_profile["program"]["instructional_hours"] is None
    for u in li_profile["instructional_units"]:
        assert u["hours"] is None


def test_unknown_status_used_for_undocumented(li_profile):
    safety = next(s for s in li_profile["skills"] if s["skill_id"] == "welding_safety")
    assert safety["status"] == "unknown"


# ---------------------------------------------------------------- drafting
def test_draft_from_text_preserves_source():
    draft = scp.draft_from_text("Some curriculum text about welding.", "manual_entry")
    assert draft["source"]["original_text"] == "Some curriculum text about welding."
    assert draft["source"]["source_type"] == "manual_entry"
    assert scp.validate_profile(draft) == []


def test_draft_invents_nothing():
    draft = scp.draft_from_text("short doc", "uploaded_document")
    assert draft["program"]["instructional_hours"] is None
    assert draft["credentials"] == []
    assert draft["assessments"] == []


# ---------------------------------------------------------------- import/export
def test_export_import_round_trip(li_profile):
    exported = scp.export_profile(li_profile)
    reimported = scp.import_profile(exported)
    assert reimported["profile_id"] == "LI-WELD-001"
    assert scp.validate_profile(reimported) == []


def test_import_rejects_invalid():
    with pytest.raises(ValueError):
        scp.import_profile('{"profile_id": "X"}')


def test_import_rejects_non_json():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        scp.import_profile("not json at all")


# ---------------------------------------------------------------- analysis wiring
def test_corpus_contains_documented_content(li_profile):
    corpus = scp.corpus_text(li_profile)
    assert "MIG" in corpus
    assert "AWS D1.1" in corpus


def test_profile_overrides_mark_unknown_absent(li_profile, taxonomy):
    corpus = scp.corpus_text(li_profile)
    df = skills.extract_skills_rules(corpus, taxonomy)
    df = scp.apply_profile_overrides(li_profile, df, taxonomy)
    safety = df[df.skill_id == "welding_safety"].iloc[0]
    assert safety["status"] == "not_documented"  # educator marked unknown


def test_profile_documented_skills_surface(li_profile, taxonomy):
    corpus = scp.corpus_text(li_profile)
    df = skills.extract_skills_rules(corpus, taxonomy)
    df = scp.apply_profile_overrides(li_profile, df, taxonomy)
    doc = set(df[df.status == "documented"]["skill_id"])
    for sid in ["stick_welding", "mig_welding", "tig_welding", "blueprint_reading"]:
        assert sid in doc
