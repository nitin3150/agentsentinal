import asyncio
import json
import logging
import os
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from agentsentinal.models import RiskCategory, RiskFlag, RiskLevel

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
# OpenRouter uses the MODEL env var; override with SEMANTIC_MODEL to pin a specific model.
DEFAULT_OPENROUTER_MODEL = os.getenv("SEMANTIC_MODEL") or os.getenv("MODEL", "openai/gpt-4o-mini")
SEMANTIC_TIMEOUT = float(os.getenv("SEMANTIC_TIMEOUT", "30"))

SEMANTIC_PROMPT = """You are an AI agent auditor. Analyse the system prompt below and return JSON only.

SYSTEM PROMPT UNDER REVIEW:
\"\"\"
{system_prompt}
\"\"\"

CONTEXT:
- Framework: {framework}
- Tool count: {tool_count}
- Static analysis already found {ambiguous_count} ambiguous phrase(s), {constraint_count} explicit constraint(s).

Return a single JSON object with EXACTLY these keys and no others:
{{
  "persona_clarity_score":   <int 1-10, how clearly the agent's role/identity is defined>,
  "scope_definition_score":  <int 1-10, how well the agent's refusal boundaries are defined>,
  "tone_consistency_score":  <int 1-10, internal consistency of the instructions>,
  "ambiguous_phrases":       [<strings — semantic-level ambiguities NOT already on the static list>],
  "risk_flags": [
    {{
      "category": "<one of: hallucination_prone | persona_drift | scope_overflow | ambiguous_instructions>",
      "description": "<one sentence>",
      "severity":    "<low | medium | high>",
      "suggestion":  "<one sentence, actionable>"
    }}
  ],
  "estimated_baseline_score": <int 0-100, predicted pass rate on a standard eval suite>
}}

Output ONLY the JSON. No prose, no markdown fences, no trailing commas."""


class SemanticAnalysis(BaseModel):
    persona_clarity_score: int = Field(ge=1, le=10, default=5)
    scope_definition_score: int = Field(ge=1, le=10, default=5)
    tone_consistency_score: int = Field(ge=1, le=10, default=5)
    ambiguous_phrases: list[str] = Field(default_factory=list)
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    estimated_baseline_score: int = Field(ge=0, le=100, default=50)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _tolerant_json_load(text: str) -> Optional[dict]:
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Drop trailing commas before } or ].
    cleaned2 = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    try:
        return json.loads(cleaned2)
    except json.JSONDecodeError:
        return None


def _clamp(value: Any, low: int, high: int, default: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, v))


def _parse_severity(value: Any) -> RiskLevel:
    if isinstance(value, str):
        try:
            return RiskLevel(value.lower())
        except ValueError:
            pass
    return RiskLevel.MEDIUM


def _parse_category(value: Any) -> RiskCategory:
    if isinstance(value, str):
        try:
            return RiskCategory(value.lower())
        except ValueError:
            pass
    return RiskCategory.HALLUCINATION_PRONE


def _parse_payload(payload: dict) -> SemanticAnalysis:
    flags_raw = payload.get("risk_flags") or []
    flags: list[RiskFlag] = []
    for item in flags_raw:
        if not isinstance(item, dict):
            continue
        try:
            flags.append(RiskFlag(
                category=_parse_category(item.get("category")),
                description=str(item.get("description", "")).strip() or "Semantic risk reported by LLM.",
                location="system_prompt:semantic",
                severity=_parse_severity(item.get("severity")),
                suggestion=str(item.get("suggestion", "")).strip() or "Review and tighten the prompt.",
            ))
        except Exception:
            continue

    ambiguous_raw = payload.get("ambiguous_phrases") or []
    ambiguous = [str(x) for x in ambiguous_raw if isinstance(x, (str, int, float))]

    return SemanticAnalysis(
        persona_clarity_score=_clamp(payload.get("persona_clarity_score"), 1, 10, 5),
        scope_definition_score=_clamp(payload.get("scope_definition_score"), 1, 10, 5),
        tone_consistency_score=_clamp(payload.get("tone_consistency_score"), 1, 10, 5),
        ambiguous_phrases=ambiguous,
        risk_flags=flags,
        estimated_baseline_score=_clamp(payload.get("estimated_baseline_score"), 0, 100, 50),
    )


async def _call_openrouter(rendered_prompt: str) -> Optional[str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return None
    try:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=DEFAULT_OPENROUTER_MODEL,
                messages=[{"role": "user", "content": rendered_prompt}],
                temperature=0,
            ),
            timeout=SEMANTIC_TIMEOUT,
        )
        return response.choices[0].message.content
    except asyncio.TimeoutError:
        logger.warning("Semantic analysis (OpenRouter) timed out after %.0fs", SEMANTIC_TIMEOUT)
        return None
    except Exception as exc:
        logger.warning("Semantic analysis (OpenRouter) failed: %s: %s", type(exc).__name__, exc)
        return None


async def _call_gemini(rendered_prompt: str) -> Optional[str]:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(DEFAULT_GEMINI_MODEL)
        resp = await asyncio.wait_for(
            model.generate_content_async(rendered_prompt),
            timeout=SEMANTIC_TIMEOUT,
        )
        return getattr(resp, "text", None)
    except asyncio.TimeoutError:
        logger.warning("Semantic analysis timed out after %.0fs", SEMANTIC_TIMEOUT)
        return None
    except Exception as exc:
        logger.warning("Semantic analysis failed: %s: %s", type(exc).__name__, exc)
        return None


async def analyze_semantic(
    system_prompt: str,
    framework: str = "unknown",
    tool_count: int = 0,
    static_findings: Optional[dict[str, Any]] = None,
) -> SemanticAnalysis:
    """One Gemini Flash call. Returns SemanticAnalysis. Never raises."""
    if not system_prompt:
        return SemanticAnalysis()

    static_findings = static_findings or {}
    rendered = SEMANTIC_PROMPT.format(
        system_prompt=system_prompt[:6000],  # cap context size
        framework=framework,
        tool_count=tool_count,
        ambiguous_count=len(static_findings.get("ambiguous_phrases", []) or []),
        constraint_count=static_findings.get("constraint_count", 0),
    )

    raw = await _call_openrouter(rendered) or await _call_gemini(rendered)
    if not raw:
        return SemanticAnalysis()

    payload = _tolerant_json_load(raw)
    if not isinstance(payload, dict):
        logger.warning("Semantic analysis returned unparseable JSON")
        return SemanticAnalysis()

    try:
        return _parse_payload(payload)
    except Exception as exc:
        logger.warning("Semantic payload parsing failed: %s: %s", type(exc).__name__, exc)
        return SemanticAnalysis()
