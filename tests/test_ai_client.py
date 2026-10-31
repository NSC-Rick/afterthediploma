"""Tests for the OpenAI Responses API client and engine comparison.

No test here calls the real API - the client is exercised through its
schema validation, error mapping, and the no-key code path only.
"""

import os

import pandas as pd
import pytest

from modules import ai_client, compare


# ------------------------------------------------------------- availability
def test_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert ai_client.openai_available() is False


def test_available_with_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert ai_client.openai_available() is True


def test_model_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    assert ai_client.configured_model() == "gpt-4.1-mini"
    monkeypatch.delenv("OPENAI_MODEL")
    assert ai_client.configured_model() == ai_client.DEFAULT_MODEL


# ------------------------------------------------------------- connectivity
def test_connection_without_key_is_auth_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = ai_client.test_connection()
    assert result["ok"] is False
    assert result["category"] == "auth"


def test_analyze_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ai_client.AIClientError) as e:
        ai_client.analyze_curriculum("some text")
    assert e.value.category == "auth"


# ------------------------------------------------------------- error mapping
def _exc(name: str) -> Exception:
    """Fake exception carrying an SDK-style class name."""
    return type(name, (Exception,), {})("fake")


def test_error_mapping_categories():
    cases = {
        "AuthenticationError": "auth",
        "PermissionDeniedError": "billing",
        "InsufficientQuotaError": "billing",
        "RateLimitError": "ratelimit",
        "NotFoundError": "model",
        "APITimeoutError": "network",
        "APIConnectionError": "network",
        "SomeRandomError": "unknown",
    }
    for cls_name, expected in cases.items():
        err = ai_client._map_error(_exc(cls_name))
        assert err.category == expected, cls_name
        assert "sk-" not in str(err)  # never leak key material


# ------------------------------------------------------------- schema validation
def test_validate_good_payload():
    payload = {
        "skills": [
            {"skill": "MIG welding", "category": "core", "status": "documented",
             "evidence": "MIG unit", "rationale": ""},
            {"skill": "CNC", "category": "adjacent", "status": "recommended",
             "evidence": "", "rationale": "gap"},
        ]
    }
    assert len(ai_client._validate(payload)) == 2


def test_validate_rejects_bad_shapes():
    with pytest.raises(ai_client.AIClientError):
        ai_client._validate({"skills": "nope"})
    with pytest.raises(ai_client.AIClientError):
        ai_client._validate({"wrong": []})
    with pytest.raises(ai_client.AIClientError):  # all entries invalid
        ai_client._validate({"skills": [{"skill": "", "category": "x", "status": "y"}]})


def test_validate_filters_invalid_entries():
    payload = {
        "skills": [
            {"skill": "MIG welding", "category": "core", "status": "documented"},
            {"skill": "Fake", "category": "nonsense", "status": "documented"},
            {"skill": "Also fake", "category": "core", "status": "mastered"},
        ]
    }
    cleaned = ai_client._validate(payload)
    assert len(cleaned) == 1 and cleaned[0]["skill"] == "MIG welding"


# ------------------------------------------------------------- comparisons
def _skills_df(rows):
    return pd.DataFrame(
        rows, columns=["skill_id", "skill_name", "category", "status",
                       "evidence", "confidence", "rationale"]
    )


def test_diff_skills_change_types():
    rules = _skills_df([
        ("mig_welding", "MIG welding", "core", "documented", "e", "", ""),
        ("cnc_operation", "CNC", "adjacent", "not_documented", "", "", ""),
        ("shop_math", "Shop math", "supporting", "inferred", "", "", ""),
    ])
    ai = _skills_df([
        ("mig_welding", "MIG welding", "core", "documented", "e", "", ""),
        ("cnc_operation", "CNC", "adjacent", "documented", "ai-ev", "", "ai-r"),
        ("shop_math", "Shop math", "supporting", "not_documented", "", "", ""),
    ])
    d = compare.diff_skills(rules, ai).set_index("skill_id")
    assert d.loc["mig_welding", "change"] == "agreement"
    assert d.loc["cnc_operation", "change"] == "ai_upgraded"
    assert d.loc["shop_math", "change"] == "ai_downgraded"
    assert d.loc["cnc_operation", "ai_evidence"] == "ai-ev"


def test_diff_careers_delta():
    base = dict(title="T", pathway="primary", matched_documented="", matched_inferred="",
                missing_skills="x", n_required=1, n_documented=0, n_inferred=0,
                training_notes="", why_relevant="")
    rules = pd.DataFrame([
        dict(base, soc_code="A", compatibility_score=50.0),
        dict(base, soc_code="B", compatibility_score=80.0),
    ])
    ai = pd.DataFrame([
        dict(base, soc_code="A", compatibility_score=60.0),
        dict(base, soc_code="B", compatibility_score=80.0),
    ])
    d = compare.diff_careers(rules, ai).set_index("soc_code")
    assert d.loc["A", "score_delta"] == 10.0
    assert d.loc["B", "score_delta"] == 0.0


def test_diff_recommendations_tags_origin():
    r = pd.DataFrame([{"recommendation": "Add CNC", "group": "Moderate enhancements"},
                      {"recommendation": "Add CAD", "group": "Moderate enhancements"}])
    a = pd.DataFrame([{"recommendation": "Add CNC", "group": "Moderate enhancements"}])
    d = compare.diff_recommendations(r, a).set_index("recommendation")
    assert d.loc["Add CNC", "produced_by"] == "both engines"
    assert d.loc["Add CAD", "produced_by"] == "rules only"


# ------------------------------------------------------------- invariants
def test_reference_data_not_in_comparison_surface():
    """AI comparison diffs never include wage or employer fields."""
    import inspect
    src = inspect.getsource(compare)
    assert "median_annual" not in src and "distance_miles" not in src
