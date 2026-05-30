# Compliance Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement per-standard compliance analysis (HIPAA, SOC2, OWASP, PII) with hybrid rule-based + LLM checking, wired end-to-end into `InspectedAgentProfile`.

**Architecture:** YAML files define rules per standard; `resolve_standards()` validates user input with typo detection; `analyze_compliance()` runs rule-based pattern matching then calls an LLM to confirm ambiguous/violated rules; results surface as `compliance_results: dict[str, ComplianceStandardResult]` on `InspectedAgentProfile`.

**Tech Stack:** Python, Pydantic, PyYAML, difflib (stdlib), OpenAI-compatible async client (Groq), Google GenAI (Gemini fallback)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/agentsentinel/compliance/hipaa.yaml` | Modify | HIPAA rules for AI agents |
| `src/agentsentinel/compliance/soc2.yaml` | Create | SOC2 rules for AI agents |
| `src/agentsentinel/compliance/owasp.yaml` | Create | OWASP LLM Top 10 rules |
| `src/agentsentinel/compliance/pii.yaml` | Create | PII/data privacy rules |
| `src/agentsentinel/models/policies.py` | Modify | Add `ComplianceViolation`, `ComplianceStandardResult`, `ComplianceAnalysis` models |
| `src/agentsentinel/models/agent.py` | Modify | Add `COMPLIANCE_VIOLATION` to `RiskCategory`, `compliance_results` to `InspectedAgentProfile` |
| `src/agentsentinel/core/agents/inspector/analyzers/compliances.py` | Rewrite | `resolve_standards`, `load_rules`, `_check_rules_static`, `_confirm_with_llm`, `analyze_compliance` |
| `src/agentsentinel/core/agents/inspector/orchestrator.py` | Modify | Replace `ComplianceTest` call, pass result to `aggregate()` |
| `src/agentsentinel/core/agents/inspector/aggregator.py` | Modify | Accept `compliance` param, map to `profile_kwargs` |
| `tests/test_compliance.py` | Create | Unit tests for all compliance logic |

---

## Task 1: Data Models

**Files:**
- Modify: `src/agentsentinel/models/policies.py`
- Modify: `src/agentsentinel/models/agent.py`
- Test: `tests/test_compliance.py`

- [ ] **Step 1: Write failing model tests**

```python
# tests/test_compliance.py
import pytest
from agentsentinel.models.policies import (
    ComplianceViolation,
    ComplianceStandardResult,
    ComplianceAnalysis,
)
from agentsentinel.models.agent import RiskCategory, RiskLevel, InspectedAgentProfile


def test_compliance_violation_fields():
    v = ComplianceViolation(
        rule_id="hipaa-001",
        description="PHI stored without consent",
        severity=RiskLevel.HIGH,
        suggestion="Add consent clause",
    )
    assert v.rule_id == "hipaa-001"
    assert v.severity == RiskLevel.HIGH


def test_compliance_standard_result_compliant_default():
    r = ComplianceStandardResult(compliant=True)
    assert r.violations == []


def test_compliance_standard_result_non_compliant():
    v = ComplianceViolation(
        rule_id="hipaa-001",
        description="PHI stored without consent",
        severity=RiskLevel.HIGH,
        suggestion="Add consent clause",
    )
    r = ComplianceStandardResult(compliant=False, violations=[v])
    assert len(r.violations) == 1


def test_compliance_analysis_defaults():
    a = ComplianceAnalysis()
    assert a.standards_checked == []
    assert a.results == {}


def test_riskcat_has_compliance_violation():
    assert RiskCategory.COMPLIANCE_VIOLATION == "compliance_violation"


def test_inspected_profile_has_compliance_results():
    p = InspectedAgentProfile(agent_id="test")
    assert p.compliance_results == {}
```

- [ ] **Step 2: Run to confirm failures**

```bash
cd /Users/nitingoyal/Developer/agentsentinal
pytest tests/test_compliance.py -v 2>&1 | head -40
```

Expected: ImportError or AttributeError — models not yet updated.

- [ ] **Step 3: Update `models/policies.py`**

Replace the existing `ComplianceViolation` class and add the two new models. Full file content:

```python
from pydantic import BaseModel, Field
from agentsentinel.models.agent import RiskFlag, RiskLevel


class PolicyViolation(BaseModel):
    description: str
    policy_reference: str
    severity: RiskLevel = RiskLevel.MEDIUM
    suggestion: str


class PolicyAnalysis(BaseModel):
    compliance_score: int = Field(ge=0, le=100, default=100)
    violations: list[PolicyViolation] = Field(default_factory=list)
    risk_flags: list[RiskFlag] = Field(default_factory=list)


class ComplianceViolation(BaseModel):
    rule_id: str
    description: str
    severity: RiskLevel = RiskLevel.MEDIUM
    suggestion: str


class ComplianceStandardResult(BaseModel):
    compliant: bool
    violations: list[ComplianceViolation] = Field(default_factory=list)


class ComplianceAnalysis(BaseModel):
    standards_checked: list[str] = Field(default_factory=list)
    results: dict[str, ComplianceStandardResult] = Field(default_factory=dict)
```

- [ ] **Step 4: Add `COMPLIANCE_VIOLATION` to `RiskCategory` in `models/agent.py`**

In `models/agent.py`, add one line to the `RiskCategory` enum (after `POLICY_VIOLATION`):

```python
COMPLIANCE_VIOLATION = "compliance_violation"
```

- [ ] **Step 5: Add `compliance_results` field to `InspectedAgentProfile` in `models/agent.py`**

At the end of `InspectedAgentProfile`, after the `extraction_confidence` field, add:

```python
# Compliance
compliance_results: dict[str, "ComplianceStandardResult"] = Field(default_factory=dict)
```

Also add the import at the top of `models/agent.py` — but `ComplianceStandardResult` lives in `models/policies.py` which imports from `models/agent.py`, creating a circular import. Use `TYPE_CHECKING` guard + string annotation:

```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agentsentinel.models.policies import ComplianceStandardResult
```

Add `from __future__ import annotations` as the very first line of `models/agent.py`.

- [ ] **Step 6: Run tests — expect pass**

```bash
pytest tests/test_compliance.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentsentinel/models/policies.py src/agentsentinel/models/agent.py tests/test_compliance.py
git commit -m "feat: add ComplianceViolation, ComplianceStandardResult, ComplianceAnalysis models"
```

---

## Task 2: YAML Compliance Rule Files

**Files:**
- Modify: `src/agentsentinel/compliance/hipaa.yaml`
- Create: `src/agentsentinel/compliance/soc2.yaml`
- Create: `src/agentsentinel/compliance/owasp.yaml`
- Create: `src/agentsentinel/compliance/pii.yaml`

No tests needed — YAML correctness is validated in Task 3 via `load_rules`.

- [ ] **Step 1: Write `src/agentsentinel/compliance/hipaa.yaml`**

```yaml
name: HIPAA
version: "2024"
rules:
  - id: hipaa-001
    description: "Agent must not store or log PHI without explicit consent mention"
    required_patterns: []
    forbidden_patterns: ["store patient data", "log health records", "save medical records", "store phi", "log phi"]
    severity: high
    suggestion: "Add explicit PHI handling and consent policy to the system prompt"

  - id: hipaa-002
    description: "Agent must apply minimum necessary data access principle"
    required_patterns: ["minimum necessary", "only necessary data", "least privilege", "need to know"]
    forbidden_patterns: []
    severity: medium
    suggestion: "Specify data minimization — state that the agent only accesses data necessary for the task"

  - id: hipaa-003
    description: "Agent must not transmit PHI over unencrypted channels"
    required_patterns: []
    forbidden_patterns: ["send patient data", "transmit health", "email medical", "share phi without"]
    severity: high
    suggestion: "Prohibit unencrypted transmission of health data in the system prompt"

  - id: hipaa-004
    description: "Agent must mention access control or authorization checks"
    required_patterns: ["access control", "authorized users only", "authentication required", "role-based", "permission"]
    forbidden_patterns: []
    severity: medium
    suggestion: "Add explicit mention of access control requirements for health data operations"

  - id: hipaa-005
    description: "Agent must not expose PHI in logs or error messages"
    required_patterns: []
    forbidden_patterns: ["log patient name", "print health data", "debug phi", "output medical record"]
    severity: high
    suggestion: "Explicitly prohibit logging or printing PHI in any output or debug messages"
```

- [ ] **Step 2: Write `src/agentsentinel/compliance/soc2.yaml`**

```yaml
name: SOC2
version: "2024"
rules:
  - id: soc2-001
    description: "Agent must mention data security or encryption"
    required_patterns: ["encrypt", "secure", "data security", "tls", "at rest", "in transit"]
    forbidden_patterns: []
    severity: high
    suggestion: "State data security and encryption requirements explicitly in the system prompt"

  - id: soc2-002
    description: "Agent must not grant unrestricted data access"
    required_patterns: []
    forbidden_patterns: ["unrestricted access", "full database access", "access all records", "no access controls"]
    severity: high
    suggestion: "Remove any unrestricted data access permissions and scope tool access appropriately"

  - id: soc2-003
    description: "Agent must mention audit logging or activity monitoring"
    required_patterns: ["audit", "log all actions", "activity log", "monitor", "record actions"]
    forbidden_patterns: []
    severity: medium
    suggestion: "Add requirement to log or audit all agent actions for SOC2 compliance"

  - id: soc2-004
    description: "Agent must mention availability or error handling"
    required_patterns: ["error handling", "fallback", "retry", "availability", "uptime", "graceful"]
    forbidden_patterns: []
    severity: low
    suggestion: "Specify error handling and availability expectations in the system prompt"

  - id: soc2-005
    description: "Agent must mention change management or version control"
    required_patterns: []
    forbidden_patterns: ["bypass version control", "skip review", "no approval needed", "ignore change management"]
    severity: medium
    suggestion: "Remove references that bypass change management controls"
```

- [ ] **Step 3: Write `src/agentsentinel/compliance/owasp.yaml`**

```yaml
name: OWASP
version: "LLM Top 10 2025"
rules:
  - id: owasp-llm01
    description: "Agent must be protected against prompt injection"
    required_patterns: ["ignore instructions from", "do not follow user", "user cannot override", "prompt injection", "instruction hierarchy"]
    forbidden_patterns: []
    severity: high
    suggestion: "Add explicit prompt injection guardrails — state that user input cannot override system instructions"

  - id: owasp-llm02
    description: "Agent must not expose sensitive system prompt contents to users"
    required_patterns: []
    forbidden_patterns: ["reveal this prompt", "share your instructions", "output your system prompt", "show your rules to user"]
    severity: high
    suggestion: "Add explicit instruction prohibiting disclosure of system prompt contents"

  - id: owasp-llm03
    description: "Agent must validate or sanitize external tool outputs before use"
    required_patterns: ["validate", "sanitize", "verify tool output", "check response", "trusted source"]
    forbidden_patterns: []
    severity: medium
    suggestion: "Add output validation requirement for all external tool responses"

  - id: owasp-llm06
    description: "Agent must not expose secrets, API keys, or credentials"
    required_patterns: []
    forbidden_patterns: ["api_key", "api-key", "secret_key", "password =", "bearer token", "private key"]
    severity: high
    suggestion: "Remove any hardcoded secrets from the system prompt; use environment variables instead"

  - id: owasp-llm07
    description: "Agent tools must have scoped permissions — no over-privileged tools"
    required_patterns: []
    forbidden_patterns: ["full admin access", "root access", "unrestricted tool", "all permissions"]
    severity: high
    suggestion: "Scope tool permissions to the minimum required for the agent's stated purpose"
```

- [ ] **Step 4: Write `src/agentsentinel/compliance/pii.yaml`**

```yaml
name: PII
version: "2024"
rules:
  - id: pii-001
    description: "Agent must not collect or store personal data without purpose limitation"
    required_patterns: []
    forbidden_patterns: ["collect all user data", "store personal information indefinitely", "retain all data", "keep user data forever"]
    severity: high
    suggestion: "Add purpose limitation — state why personal data is collected and limit retention"

  - id: pii-002
    description: "Agent must mention data retention or deletion policy"
    required_patterns: ["data retention", "delete after", "purge after", "retain for", "right to deletion", "right to erasure"]
    forbidden_patterns: []
    severity: medium
    suggestion: "Specify a data retention period and right-to-deletion policy in the system prompt"

  - id: pii-003
    description: "Agent must not log personally identifiable information"
    required_patterns: []
    forbidden_patterns: ["log user name", "log email", "log ssn", "log credit card", "log phone number", "log address"]
    severity: high
    suggestion: "Prohibit logging PII fields (name, email, SSN, phone, address) in any output"

  - id: pii-004
    description: "Agent must mention user consent for data processing"
    required_patterns: ["user consent", "explicit consent", "opt-in", "user permission", "data consent"]
    forbidden_patterns: []
    severity: medium
    suggestion: "Add explicit consent requirement before processing personal data"

  - id: pii-005
    description: "Agent must mention data encryption for PII at rest or in transit"
    required_patterns: ["encrypt pii", "encrypt personal", "secure storage", "encrypted at rest", "encrypted in transit", "tls", "encryption"]
    forbidden_patterns: []
    severity: medium
    suggestion: "State that PII must be encrypted both at rest and in transit"
```

- [ ] **Step 5: Commit**

```bash
git add src/agentsentinel/compliance/
git commit -m "feat: add compliance rule YAML files for HIPAA, SOC2, OWASP, PII"
```

---

## Task 3: Standards Registry + Rule Loader

**Files:**
- Rewrite: `src/agentsentinel/core/agents/inspector/analyzers/compliances.py`
- Test: `tests/test_compliance.py`

- [ ] **Step 1: Add failing tests for `resolve_standards` and `load_rules`**

Append to `tests/test_compliance.py`:

```python
from agentsentinel.core.agents.inspector.analyzers.compliances import (
    resolve_standards,
    load_rules,
    SUPPORTED_STANDARDS,
)


# --- resolve_standards ---

def test_resolve_all_expands_to_all_supported():
    result = resolve_standards(["All"])
    assert set(result) == SUPPORTED_STANDARDS


def test_resolve_all_case_insensitive():
    result = resolve_standards(["all"])
    assert set(result) == SUPPORTED_STANDARDS


def test_resolve_single_valid():
    assert resolve_standards(["hipaa"]) == ["hipaa"]


def test_resolve_multiple_valid():
    result = resolve_standards(["hipaa", "soc2"])
    assert set(result) == {"hipaa", "soc2"}


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unsupported compliance standard: 'completelyunknown'"):
        resolve_standards(["completelyunknown"])


def test_resolve_typo_warns_and_skips(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        result = resolve_standards(["hiipa"])
    assert result == []
    assert "hiipa" in caplog.text


def test_resolve_empty_list_returns_empty():
    assert resolve_standards([]) == []


# --- load_rules ---

def test_load_rules_hipaa_returns_rules():
    rules = load_rules("hipaa")
    assert len(rules) > 0
    assert all(hasattr(r, "id") for r in rules)


def test_load_rules_unknown_raises():
    with pytest.raises(ValueError, match="No rule file found for standard: 'fake'"):
        load_rules("fake")


def test_load_rules_rule_has_required_fields():
    rules = load_rules("hipaa")
    r = rules[0]
    assert r.id
    assert r.description
    assert r.severity in ("low", "medium", "high")
    assert isinstance(r.required_patterns, list)
    assert isinstance(r.forbidden_patterns, list)
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/test_compliance.py -v -k "resolve or load_rules" 2>&1 | head -30
```

Expected: ImportError — `resolve_standards` not yet defined.

- [ ] **Step 3: Write `compliances.py` — registry, rule loader (no LLM yet)**

Full file content (replaces the existing file):

```python
import difflib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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

    "All" / "all" expands to all supported standards.
    Exact match → included.
    Close match (difflib cutoff 0.7) → warning + skipped.
    No match → ValueError.
    """
    if not requested:
        return []

    expanded = requested
    if any(s.lower() == "all" for s in requested):
        return list(SUPPORTED_STANDARDS)

    result: list[str] = []
    for s in expanded:
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
            raise ValueError(f"Unsupported compliance standard: '{s}'. Supported: {sorted(SUPPORTED_STANDARDS)}")
    return result


def load_rules(standard: str) -> list[ComplianceRule]:
    """Load compliance rules from YAML for a given standard."""
    yaml_path = _COMPLIANCE_DIR / f"{standard.lower()}.yaml"
    if not yaml_path.exists():
        raise ValueError(f"No rule file found for standard: '{standard}'")
    with yaml_path.open() as fh:
        data = yaml.safe_load(fh)
    rules = []
    for raw in data.get("rules", []):
        rules.append(ComplianceRule(
            id=raw["id"],
            description=raw["description"],
            severity=raw.get("severity", "medium"),
            suggestion=raw.get("suggestion", ""),
            required_patterns=[p.lower() for p in raw.get("required_patterns", [])],
            forbidden_patterns=[p.lower() for p in raw.get("forbidden_patterns", [])],
        ))
    return rules
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_compliance.py -v -k "resolve or load_rules"
```

Expected: all registry + loader tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentsentinel/core/agents/inspector/analyzers/compliances.py tests/test_compliance.py
git commit -m "feat: add standards registry and YAML rule loader for compliance"
```

---

## Task 4: Rule-Based Static Checker

**Files:**
- Modify: `src/agentsentinel/core/agents/inspector/analyzers/compliances.py`
- Test: `tests/test_compliance.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_compliance.py`:

```python
from agentsentinel.core.agents.inspector.analyzers.compliances import _check_rules_static
from agentsentinel.models.agent import RiskLevel


def test_static_check_finds_forbidden_pattern():
    from agentsentinel.core.agents.inspector.analyzers.compliances import ComplianceRule
    rule = ComplianceRule(
        id="test-001",
        description="Must not store patient data",
        severity="high",
        suggestion="Fix it",
        required_patterns=[],
        forbidden_patterns=["store patient data"],
    )
    violations, ambiguous = _check_rules_static(
        "You are a helpful agent. You store patient data for analysis.",
        [],
        [rule],
    )
    assert len(violations) == 1
    assert violations[0].rule_id == "test-001"
    assert len(ambiguous) == 0


def test_static_check_finds_missing_required_pattern():
    from agentsentinel.core.agents.inspector.analyzers.compliances import ComplianceRule
    rule = ComplianceRule(
        id="test-002",
        description="Must mention encryption",
        severity="medium",
        suggestion="Add encryption",
        required_patterns=["encrypt"],
        forbidden_patterns=[],
    )
    violations, ambiguous = _check_rules_static(
        "You are a helpful assistant.",
        [],
        [rule],
    )
    assert len(violations) == 0
    assert len(ambiguous) == 1
    assert ambiguous[0].id == "test-002"


def test_static_check_passes_compliant_prompt():
    from agentsentinel.core.agents.inspector.analyzers.compliances import ComplianceRule
    rule = ComplianceRule(
        id="test-003",
        description="Must mention encryption",
        severity="medium",
        suggestion="Add encryption",
        required_patterns=["encrypt"],
        forbidden_patterns=["store patient data"],
    )
    violations, ambiguous = _check_rules_static(
        "You must encrypt all data. Never store patient data.",
        [],
        [rule],
    )
    assert violations == []
    assert ambiguous == []
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/test_compliance.py -v -k "static_check" 2>&1 | head -20
```

Expected: ImportError — `_check_rules_static` not defined.

- [ ] **Step 3: Add `_check_rules_static` to `compliances.py`**

Add after `load_rules`. Import `ComplianceViolation` and `RiskLevel` at the top of the file:

```python
from agentsentinel.models.policies import ComplianceViolation
from agentsentinel.models.agent import RiskLevel
```

Then add:

```python
def _check_rules_static(
    system_prompt: str,
    tool_definitions: list,
    rules: list[ComplianceRule],
) -> tuple[list[ComplianceViolation], list[ComplianceRule]]:
    """Check rules via pattern matching.

    Returns:
        violations: rules with confirmed forbidden pattern match
        ambiguous: rules with missing required patterns (need LLM confirmation)
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
        # Forbidden pattern hit → definite violation
        for pattern in rule.forbidden_patterns:
            if pattern in combined:
                violations.append(ComplianceViolation(
                    rule_id=rule.id,
                    description=rule.description,
                    severity=RiskLevel(rule.severity),
                    suggestion=rule.suggestion,
                ))
                break

        # Required pattern missing → ambiguous (LLM will confirm)
        if rule.required_patterns:
            found = any(p in combined for p in rule.required_patterns)
            if not found:
                ambiguous.append(rule)

    return violations, ambiguous
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_compliance.py -v -k "static_check"
```

Expected: all 3 static checker tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentsentinel/core/agents/inspector/analyzers/compliances.py tests/test_compliance.py
git commit -m "feat: add rule-based static compliance checker"
```

---

## Task 5: LLM Confirmation Pass

**Files:**
- Modify: `src/agentsentinel/core/agents/inspector/analyzers/compliances.py`
- Test: `tests/test_compliance.py`

- [ ] **Step 1: Add failing tests (mocked LLM)**

Append to `tests/test_compliance.py`:

```python
import asyncio
from unittest.mock import AsyncMock, patch
from agentsentinel.core.agents.inspector.analyzers.compliances import _confirm_with_llm, ComplianceRule


def test_confirm_with_llm_returns_violations_for_confirmed():
    rule = ComplianceRule(
        id="hipaa-002",
        description="Must mention data minimization",
        severity="medium",
        suggestion="Add data minimization principle",
        required_patterns=["minimum necessary"],
        forbidden_patterns=[],
    )
    mock_response = '{"confirmed_violations": [{"rule_id": "hipaa-002", "confirmed": true, "description": "No data minimization mention", "suggestion": "Add data minimization principle"}]}'

    with patch(
        "agentsentinel.core.agents.inspector.analyzers.compliances._llm_call",
        new=AsyncMock(return_value=mock_response),
    ):
        result = asyncio.run(_confirm_with_llm("You are a helpful agent.", [], [rule], "hipaa"))

    assert len(result) == 1
    assert result[0].rule_id == "hipaa-002"


def test_confirm_with_llm_dismisses_false_positives():
    rule = ComplianceRule(
        id="hipaa-002",
        description="Must mention data minimization",
        severity="medium",
        suggestion="Add data minimization principle",
        required_patterns=["minimum necessary"],
        forbidden_patterns=[],
    )
    mock_response = '{"confirmed_violations": [{"rule_id": "hipaa-002", "confirmed": false, "description": "", "suggestion": ""}]}'

    with patch(
        "agentsentinel.core.agents.inspector.analyzers.compliances._llm_call",
        new=AsyncMock(return_value=mock_response),
    ):
        result = asyncio.run(_confirm_with_llm("You are a helpful agent.", [], [rule], "hipaa"))

    assert result == []


def test_confirm_with_llm_empty_ambiguous_skips_llm():
    result = asyncio.run(_confirm_with_llm("anything", [], [], "hipaa"))
    assert result == []
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/test_compliance.py -v -k "confirm_with_llm" 2>&1 | head -20
```

Expected: ImportError — `_confirm_with_llm` not defined.

- [ ] **Step 3: Add LLM helpers to `compliances.py`**

Add at top of file (after existing imports):

```python
import asyncio
import json
import os
import re

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
A rule flagged for missing a required pattern may still be satisfied if the concept is expressed differently.

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


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


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

    try:
        payload = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
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
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_compliance.py -v -k "confirm_with_llm"
```

Expected: all 3 LLM confirmation tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentsentinel/core/agents/inspector/analyzers/compliances.py tests/test_compliance.py
git commit -m "feat: add LLM confirmation pass for ambiguous compliance rules"
```

---

## Task 6: `analyze_compliance` Orchestrator Function

**Files:**
- Modify: `src/agentsentinel/core/agents/inspector/analyzers/compliances.py`
- Test: `tests/test_compliance.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_compliance.py`:

```python
from agentsentinel.core.agents.inspector.analyzers.compliances import analyze_compliance
from agentsentinel.models.policies import ComplianceAnalysis


def test_analyze_compliance_empty_standards_returns_empty():
    result = asyncio.run(analyze_compliance("some prompt", [], []))
    assert isinstance(result, ComplianceAnalysis)
    assert result.standards_checked == []
    assert result.results == {}


def test_analyze_compliance_unknown_raises():
    with pytest.raises(ValueError):
        asyncio.run(analyze_compliance("prompt", [], ["unknownxyz"]))


def test_analyze_compliance_all_expands_and_runs():
    with patch(
        "agentsentinel.core.agents.inspector.analyzers.compliances._llm_call",
        new=AsyncMock(return_value='{"confirmed_violations": []}'),
    ):
        result = asyncio.run(analyze_compliance(
            "You must encrypt all data. Access control required. Audit all actions.",
            [],
            ["All"],
        ))
    assert set(result.standards_checked) == {"hipaa", "soc2", "owasp", "pii"}
    assert set(result.results.keys()) == {"hipaa", "soc2", "owasp", "pii"}


def test_analyze_compliance_detects_forbidden_pattern():
    with patch(
        "agentsentinel.core.agents.inspector.analyzers.compliances._llm_call",
        new=AsyncMock(return_value='{"confirmed_violations": []}'),
    ):
        result = asyncio.run(analyze_compliance(
            "store patient data for analysis",
            [],
            ["hipaa"],
        ))
    hipaa = result.results["hipaa"]
    assert not hipaa.compliant
    assert any(v.rule_id == "hipaa-001" for v in hipaa.violations)
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/test_compliance.py -v -k "analyze_compliance" 2>&1 | head -20
```

Expected: ImportError — `analyze_compliance` not defined.

- [ ] **Step 3: Add `analyze_compliance` to `compliances.py`**

Add at end of file:

```python
from agentsentinel.models.policies import ComplianceAnalysis, ComplianceStandardResult


async def analyze_compliance(
    system_prompt: str,
    tool_definitions: list,
    standards: list[str],
) -> ComplianceAnalysis:
    """Hybrid compliance analysis: rule-based first, LLM confirmation for ambiguous rules."""
    resolved = resolve_standards(standards)
    if not resolved:
        return ComplianceAnalysis()

    results: dict[str, ComplianceStandardResult] = {}

    for standard in resolved:
        try:
            rules = load_rules(standard)
        except ValueError as exc:
            logger.warning("Skipping standard '%s': %s", standard, exc)
            continue

        static_violations, ambiguous = _check_rules_static(system_prompt, tool_definitions, rules)
        llm_violations = await _confirm_with_llm(system_prompt, tool_definitions, ambiguous, standard)

        all_violations = static_violations + llm_violations
        results[standard] = ComplianceStandardResult(
            compliant=len(all_violations) == 0,
            violations=all_violations,
        )

    return ComplianceAnalysis(
        standards_checked=list(results.keys()),
        results=results,
    )
```

Note: Move the `from agentsentinel.models.policies import ComplianceViolation` import added in Task 4 to also include `ComplianceAnalysis, ComplianceStandardResult`. Update that import line to:

```python
from agentsentinel.models.policies import ComplianceViolation, ComplianceAnalysis, ComplianceStandardResult
```

- [ ] **Step 4: Run all compliance tests**

```bash
pytest tests/test_compliance.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentsentinel/core/agents/inspector/analyzers/compliances.py tests/test_compliance.py
git commit -m "feat: add analyze_compliance orchestrator function"
```

---

## Task 7: Wire Into Orchestrator + Aggregator

**Files:**
- Modify: `src/agentsentinel/core/agents/inspector/orchestrator.py`
- Modify: `src/agentsentinel/core/agents/inspector/aggregator.py`
- Test: `tests/test_compliance.py`

- [ ] **Step 1: Add integration test**

Append to `tests/test_compliance.py`:

```python
import asyncio
from agentsentinel.models import AgentProfile
from agentsentinel.core.agents.inspector.orchestrator import InspectorAgent


def test_orchestrator_compliance_surfaces_in_profile():
    with patch(
        "agentsentinel.core.agents.inspector.analyzers.compliances._llm_call",
        new=AsyncMock(return_value='{"confirmed_violations": []}'),
    ), patch(
        "agentsentinel.core.agents.inspector.analyzers.semantic.analyze_semantic",
        new=AsyncMock(return_value=None),
    ), patch(
        "agentsentinel.core.agents.inspector.analyzers.policy.analyze_policy",
        new=AsyncMock(return_value=None),
    ):
        profile = AgentProfile(
            system_prompt="store patient data for analysis. You help users.",
        )
        inspector = InspectorAgent(semantic_enabled=False)
        result = asyncio.run(inspector.inspect(profile, compliance=["hipaa"]))

    assert "hipaa" in result.compliance_results
    assert not result.compliance_results["hipaa"].compliant


def test_orchestrator_empty_compliance_leaves_results_empty():
    with patch(
        "agentsentinel.core.agents.inspector.analyzers.semantic.analyze_semantic",
        new=AsyncMock(return_value=None),
    ), patch(
        "agentsentinel.core.agents.inspector.analyzers.policy.analyze_policy",
        new=AsyncMock(return_value=None),
    ):
        profile = AgentProfile(system_prompt="You are a helpful assistant.")
        inspector = InspectorAgent(semantic_enabled=False)
        result = asyncio.run(inspector.inspect(profile))

    assert result.compliance_results == {}
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/test_compliance.py -v -k "orchestrator_compliance" 2>&1 | head -30
```

Expected: FAIL — compliance result not wired through yet.

- [ ] **Step 3: Update `orchestrator.py`**

Replace the compliance block (lines 79–86) with:

```python
compliance_res = None
if compliance:
    try:
        from agentsentinel.core.agents.inspector.analyzers.compliances import analyze_compliance
        compliance_res = await analyze_compliance(
            extraction.system_prompt,
            extraction.tool_definitions,
            compliance,
        )
        logger.info("Compliance result: %s", compliance_res)
    except ValueError as exc:
        logger.error("Compliance standard error: %s", exc)
        raise
    except Exception as exc:
        logger.warning("Compliance analysis failed: %s", exc)
        extraction.warnings.append(f"Compliance analysis failed: {exc}")
```

Also update the `aggregate()` call at the bottom (line 147) to pass `compliance`:

```python
return aggregate(
    agent_id=str(uuid4()),
    extraction=extraction,
    prompt=prompt,
    tools=tools,
    memory=memory,
    framework=framework,
    semantic=semantic,
    policy=policy,
    compliance=compliance_res,
)
```

Remove the old `from agentsentinel.core.agents.inspector.analyzers.compliances import ComplianceTest` import at the top of the file.

- [ ] **Step 4: Update `aggregator.py`**

Add `ComplianceAnalysis` to the import at the top:

```python
from agentsentinel.models.policies import ComplianceAnalysis, PolicyAnalysis
```

Update the `aggregate` function signature:

```python
def aggregate(
    agent_id: str,
    extraction: ExtractionResult,
    prompt: Optional[PromptAnalysis],
    tools: Optional[ToolsAnalysis],
    memory: Optional[MemoryAnalysis],
    framework: Optional[FrameworkAnalysis],
    semantic: Optional[SemanticAnalysis],
    policy: Optional[PolicyAnalysis] = None,
    compliance: Optional[ComplianceAnalysis] = None,
) -> InspectedAgentProfile:
```

Add compliance block after the existing `if policy is not None:` block:

```python
if compliance is not None:
    profile_kwargs["compliance_results"] = compliance.results
```

- [ ] **Step 5: Run all tests**

```bash
pytest tests/test_compliance.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Run full test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: all existing tests still PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentsentinel/core/agents/inspector/orchestrator.py \
        src/agentsentinel/core/agents/inspector/aggregator.py \
        tests/test_compliance.py
git commit -m "feat: wire compliance analysis end-to-end into InspectedAgentProfile"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|-----------------|------|
| HIPAA, SOC2, OWASP, PII standards | Task 2 |
| `"All"` keyword expands to all | Task 3 (`resolve_standards`) |
| Unknown standard → ValueError | Task 3 |
| Typo → warning + skip | Task 3 |
| Hybrid rule-based + LLM | Tasks 4 + 5 |
| Per-standard results in `InspectedAgentProfile` | Tasks 1 + 7 |
| LLM reuses Groq/Gemini pattern | Task 5 |
| YAML missing → warning + skip | Task 6 (`analyze_compliance`) |
| Empty compliance list → no-op | Task 6 |

**Placeholder scan:** None found.

**Type consistency:**
- `ComplianceViolation` defined Task 1, used Tasks 4/5/6 ✓
- `ComplianceStandardResult` defined Task 1, used Tasks 6/7 ✓
- `ComplianceAnalysis` defined Task 1, used Tasks 6/7 ✓
- `ComplianceRule` dataclass defined Task 3, used Tasks 4/5/6 ✓
- `_llm_call` defined Task 5, mocked in tests Tasks 5/6/7 ✓
- `analyze_compliance` import in orchestrator Task 7 matches definition Task 6 ✓
