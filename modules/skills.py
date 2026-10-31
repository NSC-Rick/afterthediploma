"""Skill extraction and classification.

Two engines produce the same result shape:

* ``extract_skills_rules`` - deterministic keyword/evidence matching
  against the bundled skills taxonomy. Always available, reproducible,
  and the default in Demonstration Mode.
* AI-assisted extraction via ``modules.ai_client`` (used only in Live
  Analysis Mode) - results are validated and mapped onto the taxonomy.

Skills are reported as one of:
  * ``documented``            - explicit textual evidence in the curriculum
  * ``inferred``              - reasonably implied by documented content,
                                NOT a confirmed student competency
  * ``recommended_addition``  - not documented, plausibly valuable
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

STATUS_DOCUMENTED = "documented"
STATUS_INFERRED = "inferred"
STATUS_RECOMMENDED = "recommended_addition"
STATUS_NOT_FOUND = "not_documented"

# When these skills are documented, the listed skill can reasonably be
# inferred even if never named in the document. Each rule keeps a
# human-readable rationale so inference stays transparent.
INFERENCE_RULES: list[dict] = [
    {
        "inferred": "welding_safety",
        "requires_any": ["mig_welding", "tig_welding", "stick_welding", "fcaw_welding"],
        "rationale": "Any accredited hands-on welding instruction necessarily "
        "includes safety protocol even when the document does not state it.",
    },
    {
        "inferred": "material_preparation",
        "requires_any": ["mig_welding", "tig_welding", "stick_welding", "fcaw_welding"],
        "rationale": "Arc welding practice requires joint and surface preparation; "
        "the skill is implied by any welding process instruction.",
    },
    {
        "inferred": "weld_symbols",
        "requires_any": ["blueprint_reading"],
        "rationale": "Weld-symbol interpretation is a standard component of "
        "welding blueprint instruction.",
    },
    {
        "inferred": "measuring_layout",
        "requires_any": ["blueprint_reading"],
        "rationale": "Blueprint interpretation is applied through measurement "
        "and layout work in the shop.",
    },
    {
        "inferred": "grinding_finishing",
        "requires_any": ["material_preparation"],
        "rationale": "Material preparation in welding programs typically covers "
        "grinding and surface finishing.",
    },
    {
        "inferred": "hand_power_tools",
        "requires_any": ["material_preparation", "grinding_finishing"],
        "rationale": "Preparation and finishing work is performed with hand and "
        "power tools.",
    },
    {
        "inferred": "workplace_safety",
        "requires_any": ["welding_safety"],
        "rationale": "Welding safety instruction sits inside general shop and "
        "workplace safety practice.",
    },
    {
        "inferred": "quality_inspection",
        "requires_any": ["weld_symbols", "metallurgy_basics"],
        "rationale": "Symbol and metallurgy content is typically paired with "
        "visual weld inspection practice.",
    },
    {
        "inferred": "troubleshooting",
        "requires_any": ["equipment_setup_maintenance"],
        "rationale": "Equipment setup and maintenance instruction develops "
        "process troubleshooting ability.",
    },
    {
        "inferred": "fabrication_fitting",
        "requires_any": ["blueprint_reading", "material_preparation"],
        "rationale": "Curricula that pair blueprint reading with material "
        "preparation typically include fit-up and fabrication practice.",
    },
]


@dataclass
class SkillHit:
    skill_id: str
    skill_name: str
    category: str
    status: str
    evidence: str
    confidence: str
    rationale: str


def _find_evidence(text: str, keyword_pattern: str) -> str | None:
    """Return the line/sentence containing the first keyword match."""
    match = re.search(keyword_pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    start = text.rfind("\n", 0, match.start()) + 1
    end = text.find("\n", match.end())
    snippet = text[start : end if end != -1 else len(text)].strip(" -\t")
    return snippet[:200] + ("..." if len(snippet) > 200 else "")


def extract_skills_rules(curriculum_text: str, taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Deterministically extract skills using the taxonomy keyword table."""
    documented_ids: set[str] = set()
    hits: list[SkillHit] = []

    for _, row in taxonomy.iterrows():
        evidence = _find_evidence(curriculum_text, row["keywords"])
        if evidence:
            documented_ids.add(row["skill_id"])
            hits.append(
                SkillHit(
                    skill_id=row["skill_id"],
                    skill_name=row["skill_name"],
                    category=row["category"],
                    status=STATUS_DOCUMENTED,
                    evidence=evidence,
                    confidence="High - explicit curriculum text",
                    rationale="",
                )
            )

    # Inference pass: rules only fire when the skill itself is undocumented.
    inferred_ids: set[str] = set()
    for rule in INFERENCE_RULES:
        sid = rule["inferred"]
        if sid in documented_ids or sid in inferred_ids:
            continue
        if any(t in documented_ids for t in rule["requires_any"]):
            inferred_ids.add(sid)
            row = taxonomy.loc[taxonomy["skill_id"] == sid].iloc[0]
            hits.append(
                SkillHit(
                    skill_id=sid,
                    skill_name=row["skill_name"],
                    category=row["category"],
                    status=STATUS_INFERRED,
                    evidence="",
                    confidence="Moderate - inferred, not documented",
                    rationale=rule["rationale"],
                )
            )

    # Recommended additions: flagged taxonomy skills absent from the document.
    for _, row in taxonomy.iterrows():
        if row["skill_id"] in documented_ids or row["skill_id"] in inferred_ids:
            continue
        status = (
            STATUS_RECOMMENDED
            if str(row.get("recommend_if_missing", "")).lower() == "true"
            else STATUS_NOT_FOUND
        )
        hits.append(
            SkillHit(
                skill_id=row["skill_id"],
                skill_name=row["skill_name"],
                category=row["category"],
                status=status,
                evidence="",
                confidence="" if status == STATUS_NOT_FOUND else "Suggestion - not in document",
                rationale=(
                    "Flagged as a potentially valuable addition based on "
                    "occupation requirements in the reference dataset."
                    if status == STATUS_RECOMMENDED
                    else ""
                ),
            )
        )

    df = pd.DataFrame([h.__dict__ for h in hits])
    cat_order = {"core": 0, "supporting": 1, "transferable": 2, "adjacent": 3}
    df["_o"] = df["category"].map(cat_order)
    return df.sort_values(["_o", "status", "skill_name"]).drop(columns="_o").reset_index(drop=True)


def merge_ai_results(
    rules_df: pd.DataFrame,
    ai_skills: list[dict],
    taxonomy: pd.DataFrame,
) -> pd.DataFrame:
    """Merge validated AI extraction output with the deterministic results.

    AI results are mapped onto the taxonomy by name when possible so that
    downstream career matching stays consistent.
    """
    name_to_row = taxonomy.set_index(taxonomy["skill_name"].str.lower())
    merged = rules_df.set_index("skill_id")
    status_map = {
        "documented": STATUS_DOCUMENTED,
        "inferred": STATUS_INFERRED,
        "recommended": STATUS_RECOMMENDED,
        "recommended_addition": STATUS_RECOMMENDED,
    }
    for item in ai_skills:
        canonical = name_to_row.loc[name_to_row.index == str(item.get("skill", "")).lower()]
        if len(canonical):
            r = canonical.iloc[0]
            sid, sname, cat = r["skill_id"], r["skill_name"], r["category"]
        else:  # AI proposed a skill outside the taxonomy - keep it visibly labeled
            sid = re.sub(r"[^a-z0-9]+", "_", str(item.get("skill", "unknown")).lower())
            sname, cat = str(item.get("skill", "Unknown")), "adjacent"
        status = status_map.get(str(item.get("status", "")).lower())
        if not status:
            continue
        confidence = (
            "High - explicit curriculum text (AI)"
            if status == STATUS_DOCUMENTED
            else "Moderate - inferred, not documented (AI)"
            if status == STATUS_INFERRED
            else "Suggestion - not in document (AI)"
        )
        row = {
            "skill_name": sname,
            "category": cat,
            "status": status,
            "evidence": str(item.get("evidence", ""))[:220],
            "confidence": confidence,
            "rationale": str(item.get("rationale", "")),
        }
        existing = merged.loc[sid] if sid in merged.index else None
        # documented beats inferred beats recommended when both engines agree
        precedence = {STATUS_DOCUMENTED: 0, STATUS_INFERRED: 1, STATUS_RECOMMENDED: 2}
        if existing is None or precedence[status] < precedence.get(existing["status"], 9):
            merged.loc[sid] = row
            merged.loc[sid].name = sid

    out = merged.reset_index().rename(columns={"index": "skill_id"})
    cat_order = {"core": 0, "supporting": 1, "transferable": 2, "adjacent": 3}
    out["_o"] = out["category"].map(cat_order).fillna(9)
    return out.sort_values(["_o", "status", "skill_name"]).drop(columns="_o").reset_index(drop=True)


def present_skills(skills_df: pd.DataFrame) -> pd.DataFrame:
    """Skills the curriculum demonstrably or plausibly develops."""
    return skills_df[skills_df["status"].isin([STATUS_DOCUMENTED, STATUS_INFERRED])]


def skill_presence(skills_df: pd.DataFrame) -> dict[str, str]:
    """Map of skill_id -> status for documented/inferred skills."""
    return present_skills(skills_df).set_index("skill_id")["status"].to_dict()
