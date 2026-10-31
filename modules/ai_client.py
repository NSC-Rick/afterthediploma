"""Live Analysis Mode: OpenAI-backed curriculum skill extraction.

The client is deliberately defensive: every response is JSON-parsed and
validated against the expected schema before it can influence the
analysis. Any failure raises ``AIClientError``; callers fall back to the
deterministic rules engine and surface a warning.

Live curriculum analysis is a *document* capability - it says nothing
about live employment-market verification, which is a separate concern.
"""

from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_MODEL = "gpt-4o-mini"
TIMEOUT_SECONDS = 45
MAX_CHARS = 60_000  # stay well inside context limits

VALID_STATUSES = {"documented", "inferred", "recommended"}
VALID_CATEGORIES = {"core", "supporting", "transferable", "adjacent"}


class AIClientError(Exception):
    """Raised for any failure in live AI analysis (auth, network, schema)."""


def openai_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "skills": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "skill": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": sorted(VALID_CATEGORIES),
                        },
                        "status": {
                            "type": "string",
                            "enum": sorted(VALID_STATUSES),
                        },
                        "evidence": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["skill", "category", "status"],
                },
            }
        },
        "required": ["skills"],
    }


_SYSTEM_PROMPT = """You are a curriculum analysis engine for a career and technical
education (CTE) intelligence platform. Analyze the provided curriculum
document and extract skills a student would plausibly develop.

Rules you MUST follow:
- status "documented": the skill is explicitly described in the text.
  Copy a short verbatim evidence quote into "evidence".
- status "inferred": the skill is reasonably implied by documented
  content but never stated. Explain why in "rationale". Never present
  an inferred skill as a confirmed student competency.
- status "recommended": the skill is absent but plausibly valuable for
  welding-adjacent careers (e.g., CNC, CAD, industrial maintenance,
  quality assurance, robotics, certifications).
- category must be one of: core, supporting, transferable, adjacent.
- Do not invent employers, wages, certifications, or outcomes.
- Return ONLY valid JSON: {"skills": [...]}."""


def _validate(payload: Any) -> list[dict]:
    if not isinstance(payload, dict) or not isinstance(payload.get("skills"), list):
        raise AIClientError("AI response missing 'skills' array")
    cleaned = []
    for item in payload["skills"]:
        if not isinstance(item, dict):
            continue
        if item.get("status") not in VALID_STATUSES or item.get("category") not in VALID_CATEGORIES:
            continue
        if not str(item.get("skill", "")).strip():
            continue
        cleaned.append(item)
    if not cleaned:
        raise AIClientError("AI response contained no valid skill entries")
    return cleaned


def analyze_curriculum(
    curriculum_text: str,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    """Return a validated list of skill dicts extracted by the model."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise AIClientError("No OpenAI API key configured (OPENAI_API_KEY).")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AIClientError("The 'openai' package is not installed.") from exc

    client = OpenAI(api_key=key, timeout=TIMEOUT_SECONDS)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Curriculum document:\n\n" + curriculum_text[:MAX_CHARS]
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 - normalize all client errors
        raise AIClientError(f"OpenAI request failed: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AIClientError("AI response was not valid JSON") from exc

    return _validate(payload)
