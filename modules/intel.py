"""Regional Intelligence engine.

Derives a curriculum-driven research scope from the completed analysis,
assigns geolocated observations to exactly one of three geographic
layers, and produces concise evidence-backed summaries per layer.

Source adapters: `IntelligenceProvider` is the contract. The bundled
`BundledIntelProvider` returns curated demonstration observations that
are CLEARLY labeled as such - this app does not fabricate live
retrieval. A live adapter (news API, RSS, employer feeds) plugs into
`get_intel_provider()` later; paid services require explicit approval
and credentials (see README).

Cost control: `run_research` is invoked ONLY from an explicit user
action; results are cached in session_state keyed by research
fingerprint and reused by every consumer (module page, PDF, rerun).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import date

import pandas as pd

LAYERS = [
    {"key": "L1", "label": "Immediate Region", "band": "0-25 miles",
     "min": 0, "max": 25,
     "question": "What is happening closest to our students?"},
    {"key": "L2", "label": "Extended Region", "band": "25-50 miles",
     "min": 25, "max": 50,
     "question": "What additional career possibilities emerge?"},
    {"key": "L3", "label": "Broader Region", "band": "50-100 miles",
     "min": 50, "max": 100,
     "question": "What should the institution be preparing for?"},
]
MAX_RADIUS = 100


def layer_for(distance_miles: float) -> str | None:
    for l in LAYERS:
        if l["min"] <= distance_miles < l["max"] or (
            l["key"] == "L3" and distance_miles == 100
        ):
            return l["key"]
    return None


def research_scope(analysis: dict, skills_df, careers_df) -> dict:
    """Derive the research scope dynamically from the selected curriculum."""
    present = skills_df[skills_df.status.isin(["documented", "inferred"])]
    prim = careers_df[careers_df.pathway == "primary"]
    adj = careers_df[careers_df.pathway == "adjacent"]
    industries = sorted(
        {
            i
            for occs in careers_df.get("relevant_industries", pd.Series(dtype=str))
            for i in str(occs).split(";")
            if i.strip()
        }
        or {"manufacturing", "metal fabrication", "construction",
            "industrial maintenance", "utilities"}
    )
    return {
        "skills": present["skill_name"].tolist(),
        "primary_occupations": prim["title"].tolist(),
        "adjacent_occupations": adj["title"].tolist(),
        "soc_codes": careers_df["soc_code"].unique().tolist(),
        "industries": industries,
    }


def fingerprint(scope: dict, geo: dict, config: dict) -> str:
    parts = "|".join([
        ",".join(sorted(scope.get("soc_codes", []))),
        f"{geo['latitude']:.4f},{geo['longitude']:.4f}",
        config.get("provider", "?"), str(config.get("max_docs", "")),
    ])
    return hashlib.sha256(parts.encode()).hexdigest()[:12]


# ------------------------------------------------------------- observation
@dataclass
class Observation:
    obs_id: str
    source_url: str
    source_type: str          # employer_site | news | econdev | municipal | industry
    source_published: str
    collected_at: str
    organization: str
    industry: str
    city: str
    state: str
    latitude: float | None
    longitude: float | None
    distance_miles: float | None
    layer: str | None         # L1|L2|L3|None when unlocatable
    category: str             # expansion | equipment | workforce | skills | infrastructure | partnership | certification | contraction
    observation: str          # factual statement only
    skills: list[str]
    occupations: list[str]
    curriculum_implication: str
    evidence_class: str       # Verified source data | Demonstration example | Unverified
    confidence: str           # high | moderate | low
    review_status: str = "unreviewed"


class IntelligenceProvider:
    """Contract any future live adapter must satisfy."""

    def collect(self, scope: dict, geo: dict, config: dict) -> list[Observation]:
        raise NotImplementedError


class BundledIntelProvider(IntelligenceProvider):
    """Curated demonstration observations - labeled, never live claims."""

    FILE = "regional_intel.csv"

    def collect(self, scope, geo, config):
        from .data_sources import DATA_DIR
        from .employers import haversine_miles

        df = pd.read_csv(DATA_DIR / self.FILE).head(int(config.get("max_docs", 25)))
        out = []
        for _, r in df.iterrows():
            lat, lon = r["latitude"], r["longitude"]
            dist = None
            if pd.notna(lat) and pd.notna(lon):
                dist = round(haversine_miles(
                    geo["latitude"], geo["longitude"], float(lat), float(lon)), 1)
            out.append(Observation(
                obs_id=r["obs_id"], source_url=r["source_url"],
                source_type=r["source_type"],
                source_published=r["source_published"],
                collected_at=str(date.today()),
                organization=r["organization"], industry=r["industry"],
                city=r["city"], state=r["state"],
                latitude=lat if pd.notna(lat) else None,
                longitude=lon if pd.notna(lon) else None,
                distance_miles=dist, layer=layer_for(dist) if dist is not None else None,
                category=r["category"], observation=r["observation"],
                skills=[s.strip() for s in str(r["skills"]).split(";") if s.strip()],
                occupations=[s.strip() for s in str(r["occupations"]).split(";") if s.strip()],
                curriculum_implication=r["curriculum_implication"],
                evidence_class=r["evidence_class"], confidence=r["confidence"],
            ))
        return out


DEFAULT_CONFIG = {"provider": "bundled", "max_docs": 25}


def get_intel_provider(config: dict | None = None) -> IntelligenceProvider:
    config = config or DEFAULT_CONFIG
    # Live adapters register here (news API, RSS, employer feeds).
    # Until one is configured, the clearly-labeled bundled set is used.
    return BundledIntelProvider()


def run_research(scope: dict, geo: dict, config: dict | None = None) -> dict:
    """ONE collection operation covering all three layers."""
    config = {**DEFAULT_CONFIG, **(config or {})}
    provider = get_intel_provider(config)
    obs = provider.collect(scope, geo, config)
    by_layer = {l["key"]: [o for o in obs if o.layer == l["key"]] for l in LAYERS}
    unlocated = [o for o in obs if o.layer is None]
    return {
        "fingerprint": fingerprint(scope, geo, config),
        "collected_at": str(date.today()),
        "provider": config["provider"],
        "geo": geo,
        "scope": scope,
        "observations": obs,
        "by_layer": by_layer,
        "unlocated": unlocated,
    }


def summarize_layer(layer_key: str, observations: list[Observation],
                    scope: dict) -> str:
    """Deterministic ~100-150 word briefing (no extra AI call).

    Summarizes evidence-backed findings only; states insufficiency plainly.
    """
    layer = next(l for l in LAYERS if l["key"] == layer_key)
    if not observations:
        return (f"No evidence-backed developments were established within "
                f"{layer['band']} during this collection. That absence is itself "
                f"useful signal - targeted outreach to regional employers and "
                f"development organizations is recommended before drawing "
                f"conclusions about {layer['label'].lower()} conditions.")
    top = sorted(observations, key=lambda o: {"high": 0, "moderate": 1, "low": 2}.get(
        o.confidence, 3))[:3]
    orgs = ", ".join(dict.fromkeys(o.organization for o in top))
    inds = ", ".join(dict.fromkeys(o.industry for o in top))
    sk = ", ".join(dict.fromkeys(s for o in top for s in o.skills) or ["—"])
    picks = " ".join(f"{o.organization}: {o.observation}" for o in top[:2])
    return (
        f"{len(observations)} observation(s) within {layer['band']} relevant to this "
        f"curriculum's skills and pathways. Strongest evidence: {picks} "
        f"Industries represented: {inds}. Skills implicated: {sk}. "
        f"For educators: verify these developments directly with {orgs.split(',')[0]} "
        f"and related employers before curriculum changes; isolated observations are "
        f"leads for investigation, not established trends."
    )
