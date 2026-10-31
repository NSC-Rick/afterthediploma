"""Regional employment intelligence: distance math and employer matching.

Distances are STRAIGHT-LINE (great-circle, haversine) miles from the
geographic reference point. They are a screening radius, not driving
distance - the UI states this distinction wherever distances appear.
"""

from __future__ import annotations

import math

import pandas as pd

from .data_sources import DataProvider, REFERENCE_POINT

EARTH_RADIUS_MILES = 3958.8


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two coordinates in miles."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * math.asin(math.sqrt(a))


def employers_with_distance(
    employers: pd.DataFrame,
    ref_lat: float | None = None,
    ref_lon: float | None = None,
) -> pd.DataFrame:
    """Attach straight-line distance from the reference point to each employer."""
    lat = ref_lat if ref_lat is not None else REFERENCE_POINT["latitude"]
    lon = ref_lon if ref_lon is not None else REFERENCE_POINT["longitude"]
    df = employers.copy()
    df["distance_miles"] = df.apply(
        lambda r: round(haversine_miles(lat, lon, float(r["latitude"]), float(r["longitude"])), 1),
        axis=1,
    )
    return df.sort_values("distance_miles").reset_index(drop=True)


def employers_within_radius(
    employers: pd.DataFrame, radius_miles: float
) -> pd.DataFrame:
    """Return employers whose straight-line distance is within the radius."""
    df = employers_with_distance(employers)
    return df[df["distance_miles"] <= radius_miles].reset_index(drop=True)


def filter_employers(
    employers: pd.DataFrame,
    industries: list[str] | None = None,
    occupation: str | None = None,
) -> pd.DataFrame:
    df = employers
    if industries:
        df = df[df["industry"].isin(industries)]
    if occupation:
        df = df[df["relevant_occupations"].str.contains(occupation, regex=False)]
    return df.reset_index(drop=True)


def get_job_listings(provider: DataProvider) -> pd.DataFrame:
    return provider.job_listings()


def _set(v) -> set[str]:
    return {s.strip() for s in str(v).split(";") if s.strip()}


def relevant_employers(
    df: pd.DataFrame, career_titles: set[str], skill_ids: set[str]
) -> pd.DataFrame:
    """Curriculum-scoped employer relevance.

    An employer is relevant when its declared relevant_occupations intersect
    the curriculum's eligible career titles, OR its relevant_skills intersect
    the curriculum's present skills. Employers failing both checks may be
    real regional organizations but are NOT relevant to this curriculum.
    """
    if df.empty:
        return df
    mask = df.apply(
        lambda r: bool(
            _set(r.get("relevant_occupations", "")) & career_titles
            or _set(r.get("relevant_skills", "")) & skill_ids
        ),
        axis=1,
    )
    return df[mask].reset_index(drop=True)


def relevant_job_listings(df: pd.DataFrame, career_titles: set[str]) -> pd.DataFrame:
    """Keep listings whose title shares a distinctive token (len>4) with an
    eligible career title - a conservative relevance heuristic; irrelevant
    illustrative listings are dropped rather than shown."""
    if df.empty:
        return df
    toks = {
        w.lower() for t in career_titles for w in str(t).replace("/", " ").split()
        if len(w) > 4
    }

    def hit(title: str) -> bool:
        words = {w.lower() for w in str(title).replace("/", " ").split()}
        return bool(words & toks)

    return df[df["job_title"].map(hit)].reset_index(drop=True)
