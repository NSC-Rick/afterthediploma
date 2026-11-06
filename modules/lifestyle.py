"""Career Lifestyle Profile v1.0 — evidence-scored lifestyle dimensions.

Scored only where documented inputs exist; otherwise the dimension returns
``insufficient_evidence`` with an explanation. FUL and COM are narrative
only (no objective score methodology exists yet).

Methodology v1.0:
- FIN (derived): occupation's best-scope median/mean annual wage vs a
  single-adult living-wage benchmark; ratio r -> score = min(100, r*60).
  Benchmark: MIT Living Wage Calculator 2025, Vermont, 1 adult ≈ $48,800
  annual pre-tax requirement (illustrative constant - user-updatable).
- LOC (derived): count of curriculum-relevant regional employers within
  100 mi naming this occupation -> capped scale 0-100 (>=8 -> 100).
- GRW (derived): number of extended-pathway occupations sharing required
  skills with this occupation -> capped scale (>=4 -> 100).
- WLB, STB, AUT: no authoritative schedule/demand/autonomy dataset is
  connected -> always ``insufficient_evidence`` in v1.
- FUL, COM: narrative-only, never scored.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math

PROFILE_VERSION = "1.0"
METHOD_VERSION = "1.0"

# MIT Living Wage Calculator (livingwage.mit.edu), Vermont, 1-adult household,
# 2025 release - annual pre-tax basic-needs requirement. Labeled benchmark;
# replace when an updated table is ingested.
LIVING_WAGE_VT_1ADULT = 48_800
LIVING_WAGE_SOURCE = (
    "MIT Living Wage Calculator 2025, Vermont, 1 adult household "
    "(illustrative constant - verify before stakeholder use)")

SCORABLE = {"FIN", "LOC", "WLB", "GRW", "STB", "AUT"}
DESCRIPTIVE = {"FUL", "COM"}

DIMENSIONS = [
    ("FIN", "Financial Independence"),
    ("LOC", "Housing & Location"),
    ("WLB", "Work-Life Balance"),
    ("GRW", "Career Growth"),
    ("STB", "Stability & Security"),
    ("AUT", "Autonomy & Flexibility"),
    ("FUL", "Personal Fulfillment"),
    ("COM", "Community Contribution"),
]


@dataclass
class Dimension:
    id: str
    label: str
    score: float | None
    band: str | None
    evidence_status: str
    summary: str
    indicators: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    limitations: list = field(default_factory=list)


def _band(score: float | None) -> str | None:
    """Application-defined descriptive bands - NOT official classifications."""
    if score is None:
        return None
    if score < 25:
        return "More constrained"
    if score < 45:
        return "Some opportunities"
    if score < 65:
        return "Moderate opportunities"
    if score < 85:
        return "Broad opportunities"
    return "Extensive opportunities"


def _clamp(v: float) -> float:
    return round(max(0.0, min(100.0, v)), 1)


def _fin(wage_row: dict) -> Dimension:
    """Score = living-wage ratio mapped to 0-100 (ratio 1.67+ -> 100)."""
    wage = wage_row.get("median_annual") or wage_row.get("mean_annual")
    basis = "median" if wage_row.get("median_annual") else "mean"
    if not wage:
        return Dimension("FIN", "Financial Independence", None, None,
                         "insufficient_evidence",
                         "No published wage benchmark for this occupation - "
                         "independence cannot be assessed from earnings data.",
                         limitations=["Occupational wage data unavailable"])
    ratio = wage / LIVING_WAGE_VT_1ADULT
    score = _clamp(ratio * 60)
    scope = wage_row.get("wage_geographic_scope", "United States")
    return Dimension(
        "FIN", "Financial Independence", score, _band(score), "derived",
        f"Published {basis} wage ${wage:,.0f} ({scope}) is about "
        f"{ratio:.1f}x the Vermont single-adult basic-needs estimate "
        f"(${LIVING_WAGE_VT_1ADULT:,.0f}). Earnings alone do not guarantee "
        "independence - household costs, taxes, and experience level matter.",
        indicators=[f"published_{basis}_wage={wage:,.0f}",
                    f"living_wage_benchmark={LIVING_WAGE_VT_1ADULT:,.0f}",
                    f"ratio={ratio:.2f}"],
        sources=[wage_row.get("source", ""), LIVING_WAGE_SOURCE],
        limitations=["Single-adult benchmark; larger households need more",
                     "Gross annual figure - taxes and benefits vary"])


def _loc(occ_title: str, emp_df) -> Dimension:
    if emp_df is None or not len(emp_df):
        return Dimension("LOC", "Housing & Location", None, None,
                         "insufficient_evidence",
                         "Regional employer mapping unavailable for this "
                         "occupation.",
                         limitations=["No employer relevance data"])
    n = int(emp_df["relevant_occupations"].str.contains(
        str(occ_title).split(" (")[0], regex=False).sum())
    score = _clamp(n / 8 * 100)
    return Dimension(
        "LOC", "Housing & Location", score, _band(score), "derived",
        f"{n} relevant employer{'s' if n != 1 else ''} within the study "
        "region list this pathway - more options means more flexibility in "
        "where you could live and work.",
        indicators=[f"relevant_employers_within_100mi={n}"],
        sources=["Bundled regional employer dataset"],
        limitations=["Reference employers only - not current job openings",
                     "Counts reflect the demonstration dataset, not the "
                     "full labor market"])


def _grw(occ_row, careers_df) -> Dimension:
    ext = careers_df[careers_df.pathway == "extended"]
    score = _clamp(len(ext) / 4 * 100) if len(ext) else None
    if score is None:
        return Dimension("GRW", "Career Growth", None, None,
                         "insufficient_evidence",
                         "No progression pathways are documented for this "
                         "occupation.",
                         limitations=["No extended pathways identified"])
    titles = "; ".join(ext["title"].head(3))
    return Dimension(
        "GRW", "Career Growth", score, _band(score), "derived",
        f"{len(ext)} documented extended pathway{'s' if len(ext) != 1 else ''} "
        f"exist for this curriculum (e.g., {titles}) - progression typically "
        "requires additional credentials or experience.",
        indicators=[f"extended_pathways={len(ext)}"],
        sources=["Occupational pathway analysis"],
        limitations=["Progression count is indicative, not a guarantee"])


def _insufficient(dim_id, label, reason) -> Dimension:
    return Dimension(dim_id, label, None, None, "insufficient_evidence",
                     reason, limitations=[reason])


NARRATIVES = {
    "FUL": ("Personal fulfillment is different for everyone. Think about "
            "which parts of this work you might find motivating - the tasks, "
            "the people, or the outcomes."),
    "COM": "This occupation supports its community through the work itself.",
}


def build_profile(occ, careers_df, wage_row: dict | None, emp_df,
                  geo_name: str = "Vermont") -> dict:
    """One profile per occupation + geography; reproducible from inputs."""
    title = occ["title"] if hasattr(occ, "get") else str(occ)
    dims: list[Dimension] = [
        _fin(wage_row or {}),
        _loc(title, emp_df),
        _insufficient("WLB", "Work-Life Balance",
                      "No authoritative schedule/hours dataset is connected; "
                      "typical schedules vary by employer - ask locally."),
        _grw(occ, careers_df),
        _insufficient("STB", "Stability & Security",
                      "No occupational demand projections dataset is "
                      "connected; do not assume demand."),
        _insufficient("AUT", "Autonomy & Flexibility",
                      "No authoritative autonomy/self-employment indicator "
                      "is connected for this occupation."),
    ]
    dims.append(Dimension(
        "FUL", "Personal Fulfillment", None, None, "inferred",
        NARRATIVES["FUL"]
        + " Consider what part of this work interests you most.",
        limitations=["Subjective - no objective measure exists"]))
    dims.append(Dimension(
        "COM", "Community Contribution", None, None, "inferred",
        f"People in this occupation serve their community by doing work "
        f"that keeps local businesses and services running - ask what "
        f"problems this job solves for people near {geo_name}.",
        limitations=["Descriptive only - contributions are not ranked"]))

    return {
        "occupation_title": title,
        "occupation_id": occ.get("soc_code", "") if hasattr(occ, "get") else "",
        "geography": geo_name,
        "profile_version": PROFILE_VERSION,
        "methodology_version": METHOD_VERSION,
        "dimensions": [asdict(d) for d in dims],
        "student_reflection": [
            "What expenses would you need to cover to live independently in "
            "your preferred community?",
            "Which parts of this job match how you like to work - with your "
            "hands, with people, or solving problems?",
            "What could you do in the next year to make this pathway stronger "
            "for you - skills, credentials, experience?",
        ],
        "overall_limitations": [
            "Scores are application-defined indicators, not scientific "
            "measurements.",
            "Regional employer data is a reference dataset, not live "
            "hiring information.",
            "Individual experiences vary by employer and circumstances.",
        ],
    }


def quantitative_dims(profile: dict) -> list[dict]:
    return [d for d in profile["dimensions"]
            if d["score"] is not None and d["id"] in SCORABLE]


def descriptive_dims(profile: dict) -> list[dict]:
    return [d for d in profile["dimensions"] if d["id"] in DESCRIPTIVE]


CAVEAT = ("About this profile: This visualization illustrates general "
          "lifestyle characteristics associated with this career. It is "
          "intended as a guide for exploration, not an absolute assessment. "
          "Individual experiences vary based on employer, location, "
          "experience, personal circumstances, and choices.")
