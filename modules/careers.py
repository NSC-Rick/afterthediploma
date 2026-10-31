"""Career pathway matching with transparent, explainable scoring.

Compatibility score (indicative, NOT a validated employment probability):

    score = 100 * sum(weight of matched required skills)
                  / sum(weight of all required skills)

    weight = 1.0 for skills documented in the curriculum
           = 0.5 for skills inferred from the curriculum
           = 0.0 for skills absent

Every score ships with the matched/missing skill breakdown so a reader
can reproduce it by hand.
"""

from __future__ import annotations

import pandas as pd

from .skills import STATUS_DOCUMENTED, STATUS_INFERRED, skill_presence

DOCUMENTED_WEIGHT = 1.0
INFERRED_WEIGHT = 0.5

PATHWAY_ORDER = {"primary": 0, "adjacent": 1, "extended": 2}
PATHWAY_LABELS = {
    "primary": "Primary pathway",
    "adjacent": "Adjacent pathway",
    "extended": "Extended pathway",
}


def match_occupations(
    skills_df: pd.DataFrame, occupations: pd.DataFrame
) -> pd.DataFrame:
    """Score every occupation in the dataset against the extracted skills."""
    presence = skill_presence(skills_df)
    id_to_name = skills_df.drop_duplicates("skill_id").set_index("skill_id")["skill_name"]

    rows = []
    for _, occ in occupations.iterrows():
        required = [s.strip() for s in str(occ["required_skills"]).split(";") if s.strip()]
        matched_doc, matched_inf, missing = [], [], []
        total = 0.0
        earned = 0.0
        for sid in required:
            weight = (
                DOCUMENTED_WEIGHT
                if presence.get(sid) == STATUS_DOCUMENTED
                else INFERRED_WEIGHT
                if presence.get(sid) == STATUS_INFERRED
                else 0.0
            )
            total += DOCUMENTED_WEIGHT
            earned += weight
            name = id_to_name.get(sid, sid)
            if weight == DOCUMENTED_WEIGHT:
                matched_doc.append(name)
            elif weight == INFERRED_WEIGHT:
                matched_inf.append(name)
            else:
                missing.append(name)
        score = round(100.0 * earned / total, 1) if total else 0.0
        rows.append(
            {
                "soc_code": occ["soc_code"],
                "title": occ["title"],
                "pathway": occ["pathway"],
                "compatibility_score": score,
                "matched_documented": "; ".join(matched_doc),
                "matched_inferred": "; ".join(matched_inf),
                "missing_skills": "; ".join(missing),
                "n_required": len(required),
                "n_documented": len(matched_doc),
                "n_inferred": len(matched_inf),
                "training_notes": occ["training_notes"],
                "why_relevant": occ["why_relevant"],
            }
        )

    out = pd.DataFrame(rows)
    out["_o"] = out["pathway"].map(PATHWAY_ORDER)
    return (
        out.sort_values(["_o", "compatibility_score"], ascending=[True, False])
        .drop(columns="_o")
        .reset_index(drop=True)
    )


def explain_score(row: pd.Series) -> str:
    """Human-readable explanation of one occupation's score."""
    return (
        f"{row['n_documented']} required skill(s) documented, "
        f"{row['n_inferred']} inferred, "
        f"{row['n_required'] - row['n_documented'] - row['n_inferred']} absent "
        f"of {row['n_required']} total. Score = "
        f"({row['n_documented']}*1.0 + {row['n_inferred']}*0.5) / {row['n_required']} "
        f"= {row['compatibility_score']}/100. "
        "Indicative alignment only - not a validated employment probability."
    )
