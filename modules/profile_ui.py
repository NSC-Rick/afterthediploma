"""Reusable Streamlit editor for Standard Curriculum Profiles.

Used by the Curriculum Library and by Module 1 so the review screen is
identical regardless of how the profile was created. Returns an updated
profile dict; validation and persistence are the caller's job.
"""

from __future__ import annotations

import copy

import pandas as pd
import streamlit as st

from .scp import SOURCE_TYPES

_SKILL_CATS = ["core", "supporting", "transferable", "adjacent", "unknown"]
_SKILL_STATUSES = ["documented", "inferred", "unknown", "recommended"]
_PROFICIENCY = ["awareness", "introductory", "competent", "advanced", "unknown"]
_CRED_STATUSES = ["documented", "inferred", "unknown"]


def _df_or_empty(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    return df.reindex(columns=columns)


def _records(df: pd.DataFrame, columns: list[str], intcols: tuple[str, ...] = ()) -> list[dict]:
    df = df.reindex(columns=columns)
    out = []
    for rec in df.to_dict("records"):
        clean = {}
        for k, v in rec.items():
            if pd.isna(v):
                clean[k] = None if k in intcols else ""
            elif k in intcols:
                try:
                    clean[k] = int(v) if float(v) == int(float(v)) else float(v)
                except (TypeError, ValueError):
                    clean[k] = None
            else:
                clean[k] = str(v)
        if any(v not in (None, "") for v in clean.values()):
            out.append(clean)
    return out


def profile_editor(profile: dict, key_prefix: str) -> dict:
    """Render editable profile fields and return an updated profile dict."""
    p = copy.deepcopy(profile)

    st.markdown("#### Institution & program")
    c1, c2 = st.columns(2)
    p["institution"]["name"] = c1.text_input(
        "Institution name", p["institution"].get("name", ""), key=f"{key_prefix}-inst"
    )
    p["institution"]["location"] = c2.text_input(
        "Location", p["institution"].get("location", ""), key=f"{key_prefix}-loc"
    )
    c1, c2 = st.columns(2)
    p["program"]["name"] = c1.text_input(
        "Program name", p["program"].get("name", ""), key=f"{key_prefix}-prog"
    )
    p["program"]["level"] = c2.text_input(
        "Level (e.g. secondary CTE)", p["program"].get("level", ""), key=f"{key_prefix}-lvl"
    )
    p["program"]["description"] = st.text_area(
        "Program description", p["program"].get("description", ""),
        height=100, key=f"{key_prefix}-desc"
    )
    c1, c2 = st.columns(2)
    p["program"]["duration"] = c1.text_input(
        "Duration (as documented)", p["program"].get("duration", ""), key=f"{key_prefix}-dur"
    )
    hours = c2.text_input(
        "Instructional hours (blank = unknown)",
        "" if p["program"].get("instructional_hours") is None
        else str(p["program"]["instructional_hours"]),
        key=f"{key_prefix}-hrs",
        help="Left blank when the source does not state hours - they are never invented.",
    )
    p["program"]["instructional_hours"] = (
        float(hours) if hours.strip().replace(".", "", 1).isdigit() else None
    )

    st.markdown("#### Source")
    src_types = list(SOURCE_TYPES.keys())
    p["source"]["source_type"] = st.selectbox(
        "Source type", src_types,
        index=src_types.index(p["source"].get("source_type", "manual_entry")),
        format_func=lambda s: SOURCE_TYPES[s], key=f"{key_prefix}-stype",
    )
    c1, c2 = st.columns(2)
    p["source"]["title"] = c1.text_input(
        "Source title", p["source"].get("title", ""), key=f"{key_prefix}-stitle"
    )
    p["source"]["url"] = c2.text_input(
        "Source URL", p["source"].get("url", ""), key=f"{key_prefix}-surl"
    )
    p["source"]["notes"] = st.text_area(
        "Source notes", p["source"].get("notes", ""), height=68, key=f"{key_prefix}-snotes"
    )

    st.markdown("#### Instructional units")
    units_df = _df_or_empty(
        p.get("instructional_units", []),
        ["sequence", "title", "description", "hours", "source_evidence"],
    )
    edited_units = st.data_editor(
        units_df, num_rows="dynamic", width="stretch",
        key=f"{key_prefix}-units",
        column_config={
            "sequence": st.column_config.NumberColumn("Seq", step=1),
            "hours": st.column_config.NumberColumn("Hours", help="Blank when undocumented"),
        },
    )
    p["instructional_units"] = _records(
        edited_units, ["sequence", "title", "description", "hours", "source_evidence"],
        intcols=("sequence", "hours"),
    )

    st.markdown("#### Learning outcomes")
    out_df = _df_or_empty(p.get("learning_outcomes", []), ["outcome", "source_evidence"])
    edited_out = st.data_editor(
        out_df, num_rows="dynamic", width="stretch", key=f"{key_prefix}-out"
    )
    p["learning_outcomes"] = _records(edited_out, ["outcome", "source_evidence"])

    st.markdown("#### Skills")
    skills_df = _df_or_empty(
        p.get("skills", []),
        ["skill", "skill_id", "category", "status", "proficiency_level", "evidence", "notes"],
    )
    edited_sk = st.data_editor(
        skills_df, num_rows="dynamic", width="stretch",
        key=f"{key_prefix}-skills",
        column_config={
            "category": st.column_config.SelectboxColumn("Category", options=_SKILL_CATS),
            "status": st.column_config.SelectboxColumn("Status", options=_SKILL_STATUSES),
            "proficiency_level": st.column_config.SelectboxColumn(
                "Proficiency", options=_PROFICIENCY
            ),
        },
    )
    p["skills"] = _records(
        edited_sk,
        ["skill", "skill_id", "category", "status", "proficiency_level", "evidence", "notes"],
    )
    for s in p["skills"]:  # schema requires enums; blank cells default honestly
        s["status"] = s["status"] if s["status"] in _SKILL_STATUSES else "unknown"
        s["category"] = s["category"] if s["category"] in _SKILL_CATS else "unknown"
        s["proficiency_level"] = (
            s["proficiency_level"] if s["proficiency_level"] in _PROFICIENCY else "unknown"
        )

    st.markdown("#### Credentials")
    cred_df = _df_or_empty(p.get("credentials", []), ["name", "type", "status", "evidence"])
    edited_cred = st.data_editor(
        cred_df, num_rows="dynamic", width="stretch", key=f"{key_prefix}-cred",
        column_config={
            "status": st.column_config.SelectboxColumn("Status", options=_CRED_STATUSES)
        },
    )
    p["credentials"] = _records(edited_cred, ["name", "type", "status", "evidence"])
    for c in p["credentials"]:
        c["status"] = c["status"] if c["status"] in _CRED_STATUSES else "unknown"

    st.markdown("#### Assessments")
    asmt_df = _df_or_empty(p.get("assessments", []), ["name", "type", "status", "evidence"])
    edited_asmt = st.data_editor(
        asmt_df, num_rows="dynamic", width="stretch", key=f"{key_prefix}-asmt",
        column_config={
            "status": st.column_config.SelectboxColumn("Status", options=_CRED_STATUSES)
        },
    )
    p["assessments"] = _records(edited_asmt, ["name", "type", "status", "evidence"])
    for a in p["assessments"]:
        a["status"] = a["status"] if a["status"] in _CRED_STATUSES else "unknown"

    st.markdown("#### Review")
    c1, c2 = st.columns(2)
    p["metadata"]["editor"] = c1.text_input(
        "Reviewed by", p["metadata"].get("editor", ""), key=f"{key_prefix}-editor"
    )
    p["metadata"]["review_status"] = c2.selectbox(
        "Review status", ["draft", "educator_reviewed", "verified"],
        index=["draft", "educator_reviewed", "verified"].index(
            p["metadata"].get("review_status", "draft")
        ),
        key=f"{key_prefix}-revstatus",
    )
    return p
