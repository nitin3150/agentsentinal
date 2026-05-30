# Compliance Analysis Design

**Date:** 2026-05-30  
**Status:** Approved

## Problem

Compliance checking exists as a skeleton: `ComplianceTest` is wired into the orchestrator but its result is discarded, `check_compliance` has a `self` call bug, `hippaa.yaml` is empty, and `ComplianceViolation` has no fields that convey actionable information. Nothing surfaces in `InspectedAgentProfile`.

## Goals

- Support HIPAA, SOC2, OWASP, PII compliance checks
- User passes `compliance=["hipaa", "soc2"]` or `compliance=["All"]`
- Unknown standard → `ValueError`; typo (edit distance ≤2) → warning, skip
- Hybrid checking: rule-based first pass, LLM confirmation for ambiguous/violated rules
- Per-standard results in `InspectedAgentProfile.compliance_results`

## Data Models (`models/policies.py`)

```python
class ComplianceViolation(BaseModel):
    rule_id: str
    description: str
    severity: RiskLevel
    suggestion: str

class ComplianceStandardResult(BaseModel):
    compliant: bool
    violations: list[ComplianceViolation] = []

class ComplianceAnalysis(BaseModel):
    standards_checked: list[str] = []
    results: dict[str, ComplianceStandardResult] = {}
```

`InspectedAgentProfile` gains:
```python
compliance_results: dict[str, ComplianceStandardResult] = {}
```

## YAML Rule Format (`compliance/<standard>.yaml`)

```yaml
name: HIPAA
version: "2024"
rules:
  - id: hipaa-001
    description: "Agent must not store or log PHI without explicit consent mention"
    forbidden_patterns: ["store patient", "log health", "save medical"]
    required_patterns: []
    severity: high
    suggestion: "Add explicit PHI handling policy to system prompt"
```

Four files: `hipaa.yaml`, `soc2.yaml`, `owasp.yaml`, `pii.yaml`. Each contains 5–10 rules covering the standard's core requirements for AI agents.

## Standards Registry

```python
SUPPORTED_STANDARDS = {"hipaa", "soc2", "owasp", "pii"}

def resolve_standards(requested: list[str]) -> list[str]:
    # "All" expands to list(SUPPORTED_STANDARDS)
    # exact match → include
    # edit distance ≤2 → warning + skip
    # no match → raise ValueError
```

## `analyze_compliance` Function (`analyzers/compliances.py`)

Full rewrite. Signature matches other analyzers:

```python
async def analyze_compliance(
    system_prompt: str,
    tool_definitions: list,
    standards: list[str],
) -> ComplianceAnalysis
```

**Per standard:**
1. Load YAML rules via `load_rules(standard)`
2. Rule-based pass: check `required_patterns` present and `forbidden_patterns` absent in `system_prompt` + tool names
3. Collect rule-based violations
4. If any violations found OR patterns were ambiguous (no strong signal) → LLM confirmation pass (reuse `_call_groq` / `_call_gemini` pattern from `policy.py`)
5. LLM receives: system prompt, tool list, standard name, flagged rules — returns confirmed/dismissed violations
6. Build `ComplianceStandardResult(compliant=len(violations)==0, violations=[...])`

## Orchestrator Wiring (`orchestrator.py`)

- Replace `ComplianceTest().analyze_compliances(...)` call with `await analyze_compliance(system_prompt, tool_definitions, compliance)`
- Pass `compliance_res` to `aggregate()` as `compliance=compliance_res`

## Aggregator (`aggregator.py`)

```python
def aggregate(..., compliance: Optional[ComplianceAnalysis] = None):
    if compliance is not None:
        profile_kwargs["compliance_results"] = compliance.results
```

## Error Handling

| Situation | Behavior |
|-----------|----------|
| Unknown standard (no close match) | `ValueError` raised before analysis starts |
| Typo (edit distance ≤2) | `logger.warning`, standard skipped |
| YAML missing or malformed | `logger.warning`, standard skipped |
| LLM unavailable | Falls back to rule-based result only |
| `compliance=[]` | Skip entirely, `compliance_results={}` |

## Files Changed

| File | Change |
|------|--------|
| `compliance/hipaa.yaml` | Populate rules |
| `compliance/soc2.yaml` | New file with rules |
| `compliance/owasp.yaml` | New file with rules |
| `compliance/pii.yaml` | New file with rules |
| `models/policies.py` | Add `ComplianceViolation`, `ComplianceStandardResult`, `ComplianceAnalysis` |
| `models/agent.py` | Add `compliance_results` to `InspectedAgentProfile` |
| `analyzers/compliances.py` | Full rewrite: `resolve_standards`, `load_rules`, `analyze_compliance` |
| `orchestrator.py` | Replace `ComplianceTest` call, wire result to `aggregate()` |
| `aggregator.py` | Add `compliance` param, map to `profile_kwargs` |
