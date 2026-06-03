import asyncio
import json
import logging
import os
import re
from typing import Any, Optional

from agentsentinel.models import RiskCategory, RiskFlag, RiskLevel
from agentsentinel.models.policies import PolicyAnalysis, PolicyViolation

logger = logging.getLogger(__name__)

from agentsentinel.utils.llm import call_llm, DEFAULT_LLM_MODEL

POLICY_TIMEOUT = float(os.getenv("POLICY_TIMEOUT", "30"))

POLICY_PROMPT = """You are an AI compliance auditor. Check whether the agent configuration below violates any rules in the provided policy document.

AGENT SYSTEM PROMPT:
\"\"\"
{system_prompt}
\"\"\"

AGENT TOOLS (JSON):
{tool_definitions}

POLICY DOCUMENT:
\"\"\"
{policy_text}
\"\"\"

Return a single JSON object with EXACTLY these keys and no others:
{{
  "compliance_score": <int 0-100, 100 = fully compliant, 0 = completely non-compliant>,
  "violations": [
    {{
      "description": "<one sentence — what rule is violated and where in the agent config>",
      "policy_reference": "<quote or section from the policy that is violated>",
      "severity": "<low | medium | high>",
      "suggestion": "<one sentence, actionable fix>"
    }}
  ]
}}

If the agent is fully compliant, return an empty violations array and compliance_score of 100.
Output ONLY the JSON. No prose, no markdown fences, no trailing commas."""


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


def _parse_payload(payload: dict) -> PolicyAnalysis:
    violations_raw = payload.get("violations") or []
    violations: list[PolicyViolation] = []
    flags: list[RiskFlag] = []

    for item in violations_raw:
        if not isinstance(item, dict):
            continue
        severity = _parse_severity(item.get("severity"))
        description = str(item.get("description", "")).strip() or "Policy violation detected."
        policy_ref = str(item.get("policy_reference", "")).strip() or "See policy document."
        suggestion = str(item.get("suggestion", "")).strip() or "Review and update agent configuration."

        violations.append(PolicyViolation(
            description=description,
            policy_reference=policy_ref,
            severity=severity,
            suggestion=suggestion,
        ))
        flags.append(RiskFlag(
            category=RiskCategory.POLICY_VIOLATION,
            description=description,
            location="policy_check",
            severity=severity,
            suggestion=suggestion,
        ))

    return PolicyAnalysis(
        compliance_score=_clamp(payload.get("compliance_score"), 0, 100, 100),
        violations=violations,
        risk_flags=flags,
    )




async def analyze_policy(
    system_prompt: str,
    tool_definitions: list,
    policy_text: str,
) -> PolicyAnalysis:
    """LLM call to check agent config against policy text. Never raises."""
    if not policy_text:
        return PolicyAnalysis()

    tool_json = json.dumps(
        [t if isinstance(t, dict) else {"name": getattr(t, "name", str(t))} for t in tool_definitions],
        indent=2,
    )[:2000]

    rendered = POLICY_PROMPT.format(
        system_prompt=system_prompt[:4000],
        tool_definitions=tool_json,
        policy_text=policy_text[:6000],
    )

    raw = await call_llm(rendered, timeout=POLICY_TIMEOUT)
    if not raw:
        return PolicyAnalysis()

    payload = _tolerant_json_load(raw)
    if not isinstance(payload, dict):
        logger.warning("Policy analysis returned unparseable JSON")
        return PolicyAnalysis()

    try:
        return _parse_payload(payload)
    except Exception as exc:
        logger.warning("Policy payload parsing failed: %s: %s", type(exc).__name__, exc)
        return PolicyAnalysis()
