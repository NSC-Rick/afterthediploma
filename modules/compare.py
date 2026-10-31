"""Side-by-side comparison: deterministic vs AI-assisted analysis.

Produces diff tables for skills, career pathway scores, and curriculum
recommendations generated from the SAME curriculum input. Verified wage
and employer reference data are never part of the comparison - they are
invariant across engines.
"""

from __future__ import annotations

import pandas as pd


def diff_skills(rules_df: pd.DataFrame, ai_df: pd.DataFrame) -> pd.DataFrame:
    """Per-skill status comparison between engines.

    change_type values:
      agreement        - same status from both engines
      ai_upgraded      - AI found stronger evidence (e.g. absent -> documented)
      ai_downgraded    - AI found weaker evidence than the rules engine
      ai_only          - skill appears only in AI output
      rules_only       - skill appears only in rules output
    """
    strength = {
        "documented": 3,
        "inferred": 2,
        "recommended_addition": 1,
        "recommended": 1,
        "not_documented": 0,
    }
    r = rules_df.set_index("skill_id")[["skill_name", "category", "status"]]
    a = ai_df.set_index("skill_id")[["skill_name", "category", "status", "evidence", "rationale"]]
    joined = r.join(a, how="outer", lsuffix="_rules", rsuffix="_ai", sort=False)

    rows = []
    for sid, row in joined.iterrows():
        rs = row.get("status_rules")
        as_ = row.get("status_ai")
        rs = "not_documented" if pd.isna(rs) else rs
        as_ = "not_documented" if pd.isna(as_) else as_
        name = row.get("skill_name_rules")
        if pd.isna(name):
            name = row.get("skill_name_ai")
        cat = row.get("category_rules")
        if pd.isna(cat):
            cat = row.get("category_ai")
        if pd.isna(row.get("status_rules")):
            change = "ai_only"
        elif pd.isna(row.get("status_ai")):
            change = "rules_only"
        elif rs == as_:
            change = "agreement"
        elif strength.get(as_, 0) > strength.get(rs, 0):
            change = "ai_upgraded"
        else:
            change = "ai_downgraded"
        rows.append(
            {
                "skill_id": sid,
                "skill": name,
                "category": cat,
                "rules_status": rs,
                "ai_status": as_,
                "change": change,
                "ai_evidence": "" if pd.isna(row.get("evidence")) else row.get("evidence"),
                "ai_rationale": "" if pd.isna(row.get("rationale")) else row.get("rationale"),
            }
        )
    return pd.DataFrame(rows)


def diff_careers(rules_ca: pd.DataFrame, ai_ca: pd.DataFrame) -> pd.DataFrame:
    """Compatibility-score deltas between engines per occupation."""
    r = rules_ca.set_index(["soc_code", "title"])[
        ["pathway", "compatibility_score", "missing_skills"]
    ]
    a = ai_ca.set_index(["soc_code", "title"])[["compatibility_score", "missing_skills"]]
    joined = r.join(a, how="left", lsuffix="_rules", rsuffix="_ai")
    joined["score_delta"] = (
        joined["compatibility_score_ai"] - joined["compatibility_score_rules"]
    )
    return joined.reset_index().rename(
        columns={
            "compatibility_score_rules": "rules_score",
            "compatibility_score_ai": "ai_score",
            "missing_skills_rules": "rules_missing",
            "missing_skills_ai": "ai_missing",
        }
    )[["soc_code", "title", "pathway", "rules_score", "ai_score", "score_delta",
        "ai_missing"]]


def diff_recommendations(recs_rules: pd.DataFrame, recs_ai: pd.DataFrame) -> pd.DataFrame:
    """Recommendation sets from each engine, tagged by which produced them."""
    r_titles = set(recs_rules["recommendation"]) if len(recs_rules) else set()
    a_titles = set(recs_ai["recommendation"]) if len(recs_ai) else set()
    rows = []
    for _, row in recs_rules.iterrows():
        rows.append(
            {
                "recommendation": row["recommendation"],
                "group": row["group"],
                "produced_by": "both engines" if row["recommendation"] in a_titles else "rules only",
            }
        )
    for _, row in recs_ai.iterrows():
        if row["recommendation"] not in r_titles:
            rows.append(
                {
                    "recommendation": row["recommendation"],
                    "group": row["group"],
                    "produced_by": "AI only",
                }
            )
    return pd.DataFrame(rows)
