import asyncio
import difflib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

import agentsentinel
from agentsentinel.models.policies import ComplianceViolation, ComplianceAnalysis, ComplianceStandardResult
from agentsentinel.models.agent import RiskLevel

logger = logging.getLogger(__name__)

SUPPORTED_STANDARDS: set[str] = {"hipaa", "soc2", "owasp", "pii"}

_COMPLIANCE_DIR = Path(agentsentinel.__file__).parent / "compliance"


@dataclass
class ComplianceRule:
    id: str
    description: str
    severity: str
    suggestion: str
    required_patterns: list[str] = field(default_factory=list)
    forbidden_patterns: list[str] = field(default_factory=list)


def resolve_standards(requested: list[str]) -> list[str]:
    """Validate and expand the user-supplied standards list.

    OR semantics: "All"/"all" expands to all supported standards.
    Exact match → included. Close match (cutoff 0.7) → warning + skipped.
    No match → ValueError.
    """
    if not requested:
        return []

    if any(s.lower() == "all" for s in requested):
        return list(SUPPORTED_STANDARDS)

    result: list[str] = []
    for s in requested:
        lower = s.lower()
        if lower in SUPPORTED_STANDARDS:
            result.append(lower)
            continue
        close = difflib.get_close_matches(lower, SUPPORTED_STANDARDS, n=1, cutoff=0.7)
        if close:
            logger.warning(
                "Unknown compliance standard '%s' — did you mean '%s'? Skipping.",
                s,
                close[0],
            )
        else:
            raise ValueError(
                f"Unsupported compliance standard: '{s}'. Supported: {sorted(SUPPORTED_STANDARDS)}"
            )
    return result


def load_rules(standard: str) -> list[ComplianceRule]:
    """Load compliance rules from YAML for a given standard."""
    yaml_path = _COMPLIANCE_DIR / f"{standard.lower()}.yaml"
    if not yaml_path.exists():
        raise ValueError(f"No rule file found for standard: '{standard}'")
    with yaml_path.open() as fh:
        data = yaml.safe_load(fh)
    return [
        ComplianceRule(
            id=raw["id"],
            description=raw["description"],
            severity=raw.get("severity", "medium"),
            suggestion=raw.get("suggestion", ""),
            required_patterns=[p.lower() for p in raw.get("required_patterns", [])],
            forbidden_patterns=[p.lower() for p in raw.get("forbidden_patterns", [])],
        )
        for raw in data.get("rules", [])
    ]


def _check_rules_static(
    system_prompt: str,
    tool_definitions: list,
    rules: list[ComplianceRule],
) -> tuple[list[ComplianceViolation], list[ComplianceRule]]:
    """Rule-based pattern check (OR semantics for both lists).

    Returns:
        violations: rules with a confirmed forbidden pattern match
        ambiguous: rules whose required patterns were all absent (LLM will confirm)
    """
    text = system_prompt.lower()
    tool_names = " ".join(
        t.get("name", "") if isinstance(t, dict) else getattr(t, "name", "")
        for t in tool_definitions
    ).lower()
    combined = f"{text} {tool_names}"

    violations: list[ComplianceViolation] = []
    ambiguous: list[ComplianceRule] = []

    for rule in rules:
        # If required patterns present and any found, rule passes entirely — skip further checks.
        if rule.required_patterns and any(p in combined for p in rule.required_patterns):
            continue

        matched_forbidden = False
        for pattern in rule.forbidden_patterns:
            if pattern in combined:
                violations.append(ComplianceViolation(
                    rule_id=rule.id,
                    description=rule.description,
                    severity=RiskLevel(rule.severity),
                    suggestion=rule.suggestion,
                ))
                matched_forbidden = True
                break

        # Only flag as ambiguous if required patterns were absent AND no forbidden match occurred.
        if rule.required_patterns and not matched_forbidden:
            ambiguous.append(rule)

    return violations, ambiguous


COMPLIANCE_TIMEOUT = float(os.getenv("COMPLIANCE_TIMEOUT", "30"))

_LLM_PROMPT = """You are an AI compliance auditor. Rule-based analysis flagged the rules below as potentially violated for the {standard} standard.

AGENT SYSTEM PROMPT:
\"\"\"
{system_prompt}
\"\"\"

TOOL NAMES: {tool_names}

FLAGGED RULES:
{flagged_rules}

For each rule, decide if it is a REAL violation given the full context of the system prompt.
A rule flagged for a missing required pattern may still be satisfied if the concept is expressed differently.

Return ONLY this JSON:
{{
  "confirmed_violations": [
    {{
      "rule_id": "<rule id>",
      "confirmed": true or false,
      "description": "<one sentence explanation if confirmed, empty string if not>",
      "suggestion": "<actionable fix if confirmed, empty string if not>"
    }}
  ]
}}
No prose, no markdown fences."""


async def _llm_call(prompt: str) -> str | None:
    """Try Groq first, Gemini fallback. Returns raw LLM text or None."""
    groq_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GRO_MODEL", "llama-3.3-70b-versatile").removeprefix("groq/")
    if groq_key:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=groq_key,
                base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            )
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                ),
                timeout=COMPLIANCE_TIMEOUT,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            logger.warning("Compliance LLM call (Groq) failed: %s", exc)

    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            resp = await asyncio.wait_for(
                client.aio.models.generate_content(model=gemini_model, contents=prompt),
                timeout=COMPLIANCE_TIMEOUT,
            )
            return resp.text
        except Exception as exc:
            logger.warning("Compliance LLM call (Gemini) failed: %s", exc)

    return None


def _tolerant_json_load(text: str) -> dict | None:
    """Strip markdown fences, then attempt JSON parse with trailing-comma recovery."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    recovered = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    try:
        return json.loads(recovered)
    except json.JSONDecodeError:
        return None


async def _confirm_with_llm(
    system_prompt: str,
    tool_definitions: list,
    ambiguous_rules: list[ComplianceRule],
    standard: str,
) -> list[ComplianceViolation]:
    """LLM pass to confirm or dismiss rule-based ambiguous findings."""
    if not ambiguous_rules:
        return []

    tool_names = ", ".join(
        t.get("name", "") if isinstance(t, dict) else getattr(t, "name", "")
        for t in tool_definitions
    ) or "(none)"

    flagged_text = "\n".join(
        f"- [{r.id}] {r.description} (required patterns: {r.required_patterns})"
        for r in ambiguous_rules
    )

    prompt = _LLM_PROMPT.format(
        standard=standard.upper(),
        system_prompt=system_prompt[:4000],
        tool_names=tool_names[:500],
        flagged_rules=flagged_text,
    )

    raw = await _llm_call(prompt)
    if not raw:
        return []

    payload = _tolerant_json_load(raw)
    if not isinstance(payload, dict):
        logger.warning("Compliance LLM returned unparseable JSON for standard '%s'", standard)
        return []

    rule_map = {r.id: r for r in ambiguous_rules}
    violations: list[ComplianceViolation] = []

    for item in payload.get("confirmed_violations", []):
        if not item.get("confirmed"):
            continue
        rule_id = item.get("rule_id", "")
        rule = rule_map.get(rule_id)
        if rule is None:
            continue
        violations.append(ComplianceViolation(
            rule_id=rule_id,
            description=item.get("description") or rule.description,
            severity=RiskLevel(rule.severity),
            suggestion=item.get("suggestion") or rule.suggestion,
        ))

    return violations


async def _analyze_single_standard(
    standard: str,
    system_prompt: str,
    tool_definitions: list,
) -> tuple[str, ComplianceStandardResult | None]:
    """Analyze one standard. Returns (standard, result) or (standard, None) on load error."""
    try:
        rules = load_rules(standard)
    except ValueError as exc:
        logger.warning("Skipping standard '%s': %s", standard, exc)
        return standard, None

    static_violations, ambiguous = _check_rules_static(system_prompt, tool_definitions, rules)
    llm_violations = await _confirm_with_llm(system_prompt, tool_definitions, ambiguous, standard)

    all_violations = static_violations + llm_violations
    return standard, ComplianceStandardResult(
        compliant=len(all_violations) == 0,
        violations=all_violations,
    )


async def analyze_compliance(
    system_prompt: str,
    tool_definitions: list,
    standards: list[str],
) -> ComplianceAnalysis:
    """Hybrid compliance analysis: rule-based first, LLM confirmation for ambiguous rules."""
    resolved = resolve_standards(standards)
    if not resolved:
        return ComplianceAnalysis()

    pairs = await asyncio.gather(
        *[_analyze_single_standard(s, system_prompt, tool_definitions) for s in resolved]
    )

    results: dict[str, ComplianceStandardResult] = {
        standard: result
        for standard, result in pairs
        if result is not None
    }

    return ComplianceAnalysis(
        standards_checked=list(results.keys()),
        results=results,
    )
