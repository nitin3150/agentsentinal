import difflib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

import agentsentinel

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


# Placeholder — implemented fully in analyze_compliance below
async def analyze_compliance(
    system_prompt: str,
    tool_definitions: list,
    standards: list[str],
):
    """Hybrid compliance analysis: rule-based first, LLM confirmation for ambiguous rules.
    Full implementation added in a later step."""
    from agentsentinel.models.policies import ComplianceAnalysis
    resolved = resolve_standards(standards)
    if not resolved:
        return ComplianceAnalysis()
    return ComplianceAnalysis(standards_checked=resolved)
