"""Earnings intelligence built on published BLS OEWS percentile wages.

Convention used throughout:
  * "Entry proxy"      = 25th percentile annual wage
  * "Median"           = 50th percentile annual wage
  * "Experienced proxy"= 75th percentile annual wage

A percentile is a position in the wage distribution - NOT a guaranteed
entry-level or experienced wage. Vermont figures are preferred; when a
state estimate is suppressed/unavailable the national figure is used and
the geographic scope column makes that explicit.
"""

from __future__ import annotations

import pandas as pd

PREFERRED_SCOPE = "Vermont"
FALLBACK_SCOPE = "United States"


def _select_wage_row(wages: pd.DataFrame, soc_code: str) -> pd.Series | None:
    rows = wages[wages["soc_code"] == soc_code]
    if rows.empty:
        return None
    for scope in (PREFERRED_SCOPE, FALLBACK_SCOPE):
        hit = rows[rows["geographic_scope"] == scope]
        if not hit.empty:
            return hit.iloc[0]
    return rows.iloc[0]


def pathway_wages(careers: pd.DataFrame, wages: pd.DataFrame) -> pd.DataFrame:
    """Attach best-available wage rows to each career pathway."""
    rows = []
    for _, occ in careers.iterrows():
        w = _select_wage_row(wages, occ["soc_code"])
        if w is None:
            rows.append(
                {
                    "soc_code": occ["soc_code"],
                    "title": occ["title"],
                    "pathway": occ["pathway"],
                    "wage_geographic_scope": "Unavailable",
                    "entry_proxy": None,
                    "median_annual": None,
                    "experienced_proxy": None,
                    "pct10": None,
                    "pct90": None,
                    "source": "No wage data",
                    "reporting_period": "",
                    "data_status": "Unavailable or unverified",
                }
            )
            continue
        rows.append(
            {
                "soc_code": occ["soc_code"],
                "title": occ["title"],
                "pathway": occ["pathway"],
                "wage_geographic_scope": w["geographic_scope"],
                "entry_proxy": w["pct25_annual"],
                "median_annual": w["median_annual"],
                "experienced_proxy": w["pct75_annual"],
                "pct10": w["pct10_annual"],
                "pct90": w["pct90_annual"],
                "source": w["source"],
                "reporting_period": w["reporting_period"],
                "data_status": w["data_status"],
            }
        )
    return pd.DataFrame(rows)


def wage_range_summary(pathway_wage_df: pd.DataFrame) -> tuple[int | None, int | None]:
    """Lowest and highest wage figures across all pathways (p10..p90)."""
    vals = pd.concat(
        [pathway_wage_df["pct10"].dropna(), pathway_wage_df["pct90"].dropna()]
    )
    if vals.empty:
        return None, None
    return int(vals.min()), int(vals.max())


def compare_scopes(wages: pd.DataFrame, soc_codes: list[str]) -> pd.DataFrame:
    """Median wage comparison, Vermont vs United States, for the given SOCs."""
    rows = wages[wages["soc_code"].isin(soc_codes)]
    pivot = rows.pivot_table(
        index=["soc_code", "occupation_title"],
        columns="geographic_scope",
        values="median_annual",
        aggfunc="first",
    ).reset_index()
    return pivot
