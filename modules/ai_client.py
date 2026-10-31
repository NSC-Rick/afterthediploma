"""Live Analysis Mode: OpenAI-backed curriculum skill extraction.

Uses the OpenAI **Responses API** with a strict JSON-Schema output
format. Every response is parsed and validated before it can influence
the analysis. Failures raise ``AIClientError`` (with a user-facing
category); callers fall back to the deterministic rules engine.

Live curriculum analysis is a *document* capability - it never modifies
verified wage, employer, or other reference data.
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
    """Any failure in live AI analysis (auth, billing, rate limit, model, schema).

    ``category`` is a short stable code for the UI:
    auth / billing / ratelimit / model / network / schema / unknown.
    """

    def __init__(self, message: str, category: str = "unknown"):
        super().__init__(message)
        self.category = category


def openai_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def configured_model() -> str:
    return os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL


def _schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "skills": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
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
                    "required": ["skill", "category", "status", "evidence", "rationale"],
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
  related careers (e.g., CNC, CAD, industrial maintenance, quality
  assurance, robotics, certifications).
- category must be one of: core, supporting, transferable, adjacent.
- Do not invent employers, wages, certifications, or outcomes.
- Return ONLY valid JSON matching the required schema."""


def _validate(payload: Any) -> list[dict]:
    if not isinstance(payload, dict) or not isinstance(payload.get("skills"), list):
        raise AIClientError("AI response missing 'skills' array", "schema")
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
        raise AIClientError("AI response contained no valid skill entries", "schema")
    return cleaned


def _map_error(exc: Exception) -> AIClientError:
    """Translate SDK exceptions into categorized, safe user messages.

    Keyed on exception class name so it stays correct across SDK versions.
    The API key is never included in any message.
    """
    name = type(exc).__name__
    if name == "AuthenticationError":
        return AIClientError(
            "OpenAI authentication failed - check that OPENAI_API_KEY is valid "
            "and has not been revoked.", "auth",
        )
    if name in ("PermissionDeniedError", "InsufficientQuotaError"):
        return AIClientError(
            "The API key lacks permission or the account has insufficient "
            "quota/billing for this request.", "billing",
        )
    if name == "RateLimitError":
        return AIClientError(
            "OpenAI rate limit reached - wait a moment and try again.", "ratelimit",
        )
    if name == "NotFoundError":
        return AIClientError(
            f"The configured model is unavailable to this account "
            f"('{configured_model()}'). Set OPENAI_MODEL to an available model.",
            "model",
        )
    if name in ("APITimeoutError", "APIConnectionError"):
        return AIClientError(
            "Could not reach OpenAI (network timeout or connection error). "
            "Check connectivity and retry.", "network",
        )
    if name == "BadRequestError":
        return AIClientError(
            f"The model rejected the request parameters "
            f"('{configured_model()}'): {exc}", "model",
        )
    return AIClientError(f"OpenAI request failed: {exc}", "unknown")


def _client(key: str):
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise AIClientError("The 'openai' package is not installed.", "unknown") from exc
    return OpenAI(api_key=key, timeout=TIMEOUT_SECONDS)


def test_connection(model: str | None = None) -> dict:
    """Safe connectivity check: one small authenticated Responses call.

    Costs a handful of tokens; confirms the key, the account, and the
    configured model all work. Never returns or logs the key.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return {"ok": False, "message": "No OPENAI_API_KEY configured.", "category": "auth"}
    model = model or configured_model()
    try:
        resp = _client(key).responses.create(
            model=model,
            input="Reply with the single word: ok",
            max_output_tokens=16,
        )
        return {
            "ok": True,
            "message": f"Connected to OpenAI - model '{model}' responded "
            f"({(resp.output_text or '').strip()[:20]!r}).",
            "category": "ok",
        }
    except Exception as exc:  # noqa: BLE001 - normalized by design
        err = _map_error(exc)
        return {"ok": False, "message": err.args[0], "category": err.category}


def analyze_curriculum(
    curriculum_text: str,
    api_key: str | None = None,
    model: str | None = None,
) -> list[dict]:
    """Validated skill extraction via the Responses API (strict JSON schema)."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise AIClientError("No OpenAI API key configured (OPENAI_API_KEY).", "auth")
    model = model or configured_model()

    try:
        resp = _client(key).responses.create(
            model=model,
            instructions=_SYSTEM_PROMPT,
            input="Curriculum document:\n\n" + curriculum_text[:MAX_CHARS],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "skill_extraction",
                    "schema": _schema(),
                    "strict": True,
                }
            },
            max_output_tokens=8000,
        )
        raw = resp.output_text or ""
    except Exception as exc:  # noqa: BLE001 - normalized by design
        raise _map_error(exc) from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AIClientError("AI response was not valid JSON", "schema") from exc

    return _validate(payload)
