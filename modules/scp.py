"""Standard Curriculum Profile (SCP) v1.0.

Any supported curriculum source (PDF, DOCX, TXT, pasted text, public
program description) is converted into a consistent, validated,
editable profile before career-intelligence analysis runs.

Invariants enforced here:
  * The original source text is preserved verbatim under
    ``profile["source"]["original_text"]`` and never mutated.
  * Instructional hours, credentials, proficiency levels, and
    assessments are never invented - fields the source does not
    document stay null / "unknown".
  * Profiles are validated against ``data/scp_schema.json`` (SCP v1.0).
"""

from __future__ import annotations

import copy
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

from .data_sources import DATA_DIR

SCP_VERSION = "1.0"
SCHEMA_PATH = DATA_DIR / "scp_schema.json"
PROFILES_DIR = DATA_DIR / "profiles"

SOURCE_TYPES = {
    "complete_curriculum": "Complete curriculum",
    "course_catalog": "Course catalog",
    "public_program_description": "Public program description",
    "manual_entry": "Manually entered content",
    "uploaded_document": "Uploaded document",
}

SKILL_STATUSES = ["documented", "inferred", "unknown", "recommended"]
PROFICIENCY_LEVELS = ["awareness", "introductory", "competent", "advanced", "unknown"]


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_profile(profile: dict) -> list[str]:
    """Validate a profile against the SCP schema. Returns a list of errors."""
    errors: list[str] = []
    try:
        import jsonschema

        validator = jsonschema.Draft7Validator(load_schema())
        errors = [f"{'/'.join(map(str, e.absolute_path)) or '/'}: {e.message}"
                  for e in validator.iter_errors(profile)]
    except ImportError:  # pragma: no cover - jsonschema is a declared dep
        errors = _structural_validate(profile)
    return sorted(errors)


def _structural_validate(profile: dict) -> list[str]:
    """Minimal fallback validation when jsonschema is unavailable."""
    errors = []
    for key in ("scp_version", "profile_id", "institution", "program", "source", "skills", "metadata"):
        if key not in profile:
            errors.append(f"/{key}: required property missing")
    if profile.get("scp_version") != SCP_VERSION:
        errors.append("/scp_version: must be '1.0'")
    for s in profile.get("skills", []):
        if s.get("status") not in SKILL_STATUSES:
            errors.append(f"/skills/{s.get('skill','?')}/status: invalid")
    return errors


def new_profile_id(prefix: str = "CUSTOM") -> str:
    existing = {p.stem for p in PROFILES_DIR.glob("*.json")}
    n = len(existing) + 1
    return f"{prefix}-{n:03d}"


def blank_profile(profile_id: str, source_type: str = "manual_entry") -> dict:
    return {
        "scp_version": SCP_VERSION,
        "profile_id": profile_id,
        "institution": {"name": "", "location": "", "institution_type": "", "website": ""},
        "program": {
            "name": "", "cip_code": None, "level": "", "description": "",
            "duration": "", "instructional_hours": None,
        },
        "source": {
            "source_type": source_type, "title": "", "url": "",
            "retrieved_date": str(date.today()), "notes": "",
            "original_text": "", "evidence": [],
        },
        "instructional_units": [],
        "learning_outcomes": [],
        "skills": [],
        "credentials": [],
        "assessments": [],
        "metadata": {
            "created_date": str(date.today()),
            "last_modified": str(date.today()),
            "editor": "",
            "review_status": "draft",
            "data_status": "Derived analysis - unreviewed draft",
            "notes": "",
        },
    }


def draft_from_text(
    text: str,
    source_type: str,
    title: str = "",
    profile_id: str | None = None,
) -> dict:
    """Build a draft SCP from raw curriculum text.

    The document is stored verbatim; structured fields stay empty (or
    "unknown") until the educator review screen or extraction fills them.
    Nothing is invented.
    """
    p = blank_profile(profile_id or new_profile_id("DRAFT"), source_type)
    p["source"]["title"] = title
    p["source"]["original_text"] = text
    # First non-empty line is a pragmatic title guess - clearly a draft.
    first = next((ln.strip(" #-\t") for ln in text.splitlines() if ln.strip()), "")
    p["program"]["name"] = title or (first[:80] if first else "Untitled curriculum")
    p["metadata"]["notes"] = (
        "Draft profile generated from source text; review and edit before use."
    )
    return p


def corpus_text(profile: dict) -> str:
    """Serialize profile content into a text corpus for skill extraction.

    Derived from the profile's structured fields; the verbatim source is
    included so keyword evidence still traces to the original document.
    """
    parts = [
        profile["program"].get("name", ""),
        profile["program"].get("description", ""),
        profile["source"].get("original_text", ""),
    ]
    for u in profile.get("instructional_units", []):
        parts += [u.get("title", ""), u.get("description", "")]
    for o in profile.get("learning_outcomes", []):
        parts.append(o.get("outcome", ""))
    for s in profile.get("skills", []):
        if s.get("status") in ("documented", "inferred"):
            parts.append(s.get("skill", ""))
    for c in profile.get("credentials", []):
        if c.get("status") == "documented":
            parts.append(c.get("name", ""))
    return "\n".join(x for x in parts if x)


def map_skill_name(skill_name: str, taxonomy: pd.DataFrame) -> str | None:
    """Best-effort map of a free-text skill name to a taxonomy skill_id."""
    name = skill_name.lower().strip()
    for _, row in taxonomy.iterrows():
        canon = str(row["skill_name"]).lower()
        if name == canon or name in canon or canon in name:
            return row["skill_id"]
    for _, row in taxonomy.iterrows():  # keyword hit on the name itself
        if re.search(str(row["keywords"]), name, flags=re.IGNORECASE):
            return row["skill_id"]
    return None


def apply_profile_overrides(
    profile: dict, skills_df: pd.DataFrame, taxonomy: pd.DataFrame
) -> pd.DataFrame:
    """Apply educator-reviewed SCP skill statuses onto the engine output.

    * documented  -> forced documented (with profile evidence)
    * inferred    -> forced inferred (if not already documented)
    * unknown     -> forced out of the present set (educator cannot confirm)
    * recommended -> kept as a suggested addition
    """
    df = skills_df.copy()
    precedence = {"documented": 0, "inferred": 1, "recommended_addition": 2}
    for s in profile.get("skills", []):
        sid = s.get("skill_id") or map_skill_name(s.get("skill", ""), taxonomy)
        if not sid:
            continue
        status = s.get("status")
        if status == "unknown":
            df.loc[df.skill_id == sid, "status"] = "not_documented"
            continue
        mapped = {
            "documented": "documented",
            "inferred": "inferred",
            "recommended": "recommended_addition",
        }.get(status)
        if not mapped:
            continue
        row = df[df.skill_id == sid]
        if row.empty:
            continue
        cur = row.iloc[0]["status"]
        if precedence.get(mapped, 9) <= precedence.get(cur, 9) or (
            mapped == "inferred" and cur == "not_documented"
        ) or cur == "not_documented":
            df.loc[df.skill_id == sid, "status"] = mapped
            if s.get("evidence") and not row.iloc[0].get("evidence"):
                df.loc[df.skill_id == sid, "evidence"] = str(s["evidence"])[:220]
            if mapped == "inferred" and not s.get("evidence"):
                df.loc[df.skill_id == sid, "confidence"] = (
                    "Moderate - inferred in reviewed profile"
                )
    return df


# ------------------------------------------------------------- library io
def save_profile(profile: dict, directory: Path | None = None) -> Path:
    errors = validate_profile(profile)
    if errors:
        raise ValueError("Profile failed SCP v1.0 validation:\n" + "\n".join(errors))
    directory = directory or PROFILES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    profile["metadata"]["last_modified"] = str(date.today())
    path = directory / f"{profile['profile_id']}.json"
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return path


def load_profile(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def list_profiles(directory: Path | None = None) -> list[dict]:
    directory = directory or PROFILES_DIR
    out = []
    for f in sorted(directory.glob("*.json")):
        try:
            p = load_profile(f)
            p["_path"] = f
            out.append(p)
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def import_profile(json_text: str) -> dict:
    profile = json.loads(json_text)
    errors = validate_profile(profile)
    if errors:
        raise ValueError(
            "Imported file is not a valid SCP v1.0 profile:\n" + "\n".join(errors[:10])
        )
    return profile


def export_profile(profile: dict) -> str:
    clean = {k: v for k, v in profile.items() if not k.startswith("_")}
    return json.dumps(clean, indent=2)
