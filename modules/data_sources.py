"""Data access layer and source-transparency primitives.

All reference data flows through provider classes so that a live
intelligence feed (an API, a database, a scheduled ingestion job) can
replace the bundled CSV files without any change to the analysis
modules or the user interface.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class DataStatus(str, enum.Enum):
    """Provenance labels used across every display surface."""

    VERIFIED = "Verified source data"
    DERIVED = "Derived analysis"
    AI = "AI-generated recommendation"
    DEMO = "Demonstration example"
    UNVERIFIED = "Unavailable or unverified"


@dataclass(frozen=True)
class SourceInfo:
    """Metadata attached to every externally sourced claim."""

    name: str
    geographic_scope: str
    publication_period: str = ""
    retrieved_date: str = ""
    status: str = DataStatus.DEMO.value
    url: str = ""


# Geographic reference for regional employer intelligence.
REFERENCE_POINT = {
    "name": "Lyndon Institute, Lyndon Center, Vermont",
    "latitude": 44.5387,
    "longitude": -72.0163,
    "source": SourceInfo(
        name="OpenStreetMap Nominatim geocode of campus address",
        geographic_scope="Lyndon Center, VT 05850",
        retrieved_date="2026-10-31",
        status=DataStatus.VERIFIED.value,
        url="https://nominatim.openstreetmap.org/",
    ),
}

GEOGRAPHIC_SCOPES = ["United States", "Vermont"]

BLS_SOURCE = SourceInfo(
    name="U.S. Bureau of Labor Statistics OEWS via O*NET OnLine (CC-BY 4.0)",
    geographic_scope="United States and Vermont",
    publication_period="O*NET 2025 wage data (BLS OEWS)",
    retrieved_date="2026-10-31",
    status=DataStatus.VERIFIED.value,
    url="https://www.onetonline.org/",
)


def _load_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Bundled data file missing: {path}")
    return pd.read_csv(path)


class DataProvider:
    """Interface that any future live provider must satisfy."""

    def occupations(self) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError

    def wages(self) -> pd.DataFrame:
        raise NotImplementedError

    def employers(self) -> pd.DataFrame:
        raise NotImplementedError

    def job_listings(self) -> pd.DataFrame:
        raise NotImplementedError

    def skills_taxonomy(self) -> pd.DataFrame:
        raise NotImplementedError

    def sources(self) -> list[SourceInfo]:
        raise NotImplementedError


class BundledDataProvider(DataProvider):
    """Serves the curated demonstration datasets shipped with the app."""

    def occupations(self) -> pd.DataFrame:
        return _load_csv("occupations.csv")

    def wages(self) -> pd.DataFrame:
        return _load_csv("wages.csv")

    def employers(self) -> pd.DataFrame:
        return _load_csv("employers.csv")

    def job_listings(self) -> pd.DataFrame:
        return _load_csv("job_listings.csv")

    def skills_taxonomy(self) -> pd.DataFrame:
        return _load_csv("skills_taxonomy.csv")

    def sources(self) -> list[SourceInfo]:
        return [
            REFERENCE_POINT["source"],
            BLS_SOURCE,
            SourceInfo(
                name="Bundled employer dataset (real employers identified from public "
                "business listings and company websites; compiled for demonstration)",
                geographic_scope="Northeast Vermont / northern New Hampshire",
                retrieved_date="2026-10-31",
                status=DataStatus.DEMO.value,
            ),
            SourceInfo(
                name="Bundled occupation and skills-alignment dataset (curated for demonstration)",
                geographic_scope="United States",
                retrieved_date="2026-10-31",
                status=DataStatus.DEMO.value,
            ),
            SourceInfo(
                name="Representative sample welding curriculum (fictional; not an official "
                "Lyndon Institute document)",
                geographic_scope="Not applicable",
                retrieved_date="2026-10-31",
                status=DataStatus.DEMO.value,
            ),
        ]


def get_provider() -> DataProvider:
    """Return the active provider. Swap this for a live provider later."""
    return BundledDataProvider()
