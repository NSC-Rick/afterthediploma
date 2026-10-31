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
