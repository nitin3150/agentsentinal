# Agent Sentinel — Production Readiness Platform for AI Agents

Agent Sentinel inspects, improves, and stress-tests AI agents before they ship. It performs static + semantic analysis of an agent's system prompt, tool definitions, memory, and framework structure, produces a risk report, rewrites the prompt to fix every flagged issue, and runs adversarial prompt campaigns to verify the fixes hold under pressure.

## Repository Structure

```
agentsentinel/
├── src/agentsentinel/
│   ├── sentinel.py                  # AgentSentinel — main entry point
│   ├── compliance/                  # YAML rule files per standard
│   │   ├── hipaa.yaml
│   │   ├── soc2.yaml
│   │   ├── owasp.yaml
│   │   └── pii.yaml
│   ├── core/agents/
│   │   ├── intake/                  # Framework detection & profile extraction
│   │   │   ├── agent_intake.py      # AgentIntake orchestrator
│   │   │   └── detectors/
│   │   │       ├── langgraph.py     # LangGraph detector
│   │   │       └── filepath.py      # Source file detector
│   │   ├── inspector/               # Static + semantic analysis
│   │   │   ├── orchestrator.py      # InspectorAgent
│   │   │   ├── aggregator.py        # Combines analyzer outputs
│   │   │   └── analyzers/
│   │   │       ├── prompt.py        # Constraint, ambiguity, injection checks
│   │   │       ├── tools.py         # Tool quality scoring
│   │   │       ├── memory.py        # Memory backend risk detection
│   │   │       ├── framework.py     # Graph depth, loops, HITL, cycle detection
│   │   │       ├── semantic.py      # LLM-powered semantic analysis
│   │   │       ├── policy.py        # Policy PDF compliance check
│   │   │       └── compliances.py   # HIPAA / SOC2 / OWASP / PII rule engine
│   │   ├── improver/                # DSPy-based prompt rewriter
│   │   │   ├── prompt_improver.py   # PromptImprover (parallel + sequential fixes)
│   │   │   ├── signatures.py        # DSPy fix signatures per risk category
│   │   │   ├── policy_guard.py      # Final policy compliance gate
│   │   │   └── evaluations.py       # DSPy optimizer metric
│   │   └── tester/                  # Adversarial testing pipeline
│   │       ├── tester.py            # TestAgent orchestrator
│   │       ├── adversarial_prompts_generator.py
│   │       ├── runner.py            # Runs prompts against live agent
│   │       ├── evaluator.py         # Scores each response
│   │       └── report.py            # Generates audit_report.json + .md
│   ├── models/
│   │   ├── agent.py                 # AgentProfile, InspectedAgentProfile, RiskFlag
│   │   ├── policies.py              # ComplianceViolation, ComplianceAnalysis
│   │   ├── intake.py                # ExtractionResult
│   │   └── prompt.py                # ImprovementResult
│   └── utils/
│       ├── policies.py              # PDF policy parser
│       └── logger.py
├── demo/                            # Example agents (LangGraph, CrewAI, ADK, etc.)
├── tests/
├── main.py
├── pyproject.toml
└── .env
```

## Quick Start

```bash
git clone https://github.com/goyalnitin148/agentsentinel.git
cd agentsentinel
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set GROQ_API_KEY, GROQ_MODEL, OPENROUTER_API_KEY, MODEL
```

## Core Workflow

```mermaid
flowchart TD
    A[Your Agent / Source File] --> B[AgentIntake\nFramework detection + profile extraction]
    B --> C[InspectorAgent\nStatic analyzers run first\nprompt · tools · memory · framework]
    C --> D[Concurrent LLM analyses\nsemantic · policy · compliance]
    D --> E[InspectedAgentProfile\nrisk_flags · scores · compliance_results]
    E --> F[TestAgent\nAdversarial stress test]
    F --> G{Pass rate ≥ threshold?}
    G -->|yes| H[audit_report.json + .md\nAudit complete]
    G -->|no| I[PromptImprover\nDSPy-based rewrite]
    I --> J[Re-inspect with improved prompt]
    J --> F
    J -.->|max iterations reached| H
```

## Four Operations

### 1. `inspect(agent)` — risk analysis

Extracts the agent's system prompt and tools, runs four static analyzers synchronously, then fires three LLM-powered analyzers concurrently:

| Analyzer | Type | What it checks |
|---|---|---|
| `prompt` | static | Ambiguous phrases, missing constraints, injection surface |
| `tools` | static | Quality score per tool, missing fields |
| `memory` | static | Memory backend type, TTL, scope, data-leak risks |
| `framework` | static | Graph depth, loops, conditional edges, human-in-loop |
| `semantic` | LLM | Persona clarity, scope definition, tone, hallucination risk |
| `policy` | LLM | Violations against a supplied policy PDF |
| `compliance` | LLM + rules | HIPAA / SOC2 / OWASP LLM Top 10 / PII rules |

Returns an `InspectedAgentProfile` with `risk_flags`, scores, `policy_violations`, and `compliance_results`.

```python
from agentsentinel.sentinel import AgentSentinel

sentinel = AgentSentinel()
profile = sentinel.inspect(
    agent,                             # compiled LangGraph graph (or other framework)
    system_prompt="...",               # optional override
    policies="sample_policies.pdf",    # optional policy PDF
    compliance=["hipaa", "soc2"],      # optional — or "All" for all standards
    source_code="...",                 # optional — pass source for live agents
)
print(profile.overall_risk)            # low / medium / high
print(profile.risk_flags)
print(profile.compliance_results)      # per-standard PASS/FAIL + violations
```

### 2. `improve(profile)` — prompt rewriting

Takes the `InspectedAgentProfile` and rewrites the system prompt + tool definitions to fix every flagged risk using DSPy `ChainOfThought` signatures. Sequential fixes (injection → persona) run first; remaining fixes run in parallel and are merged.

Risk categories fixed:

- `INJECTION_VULNERABLE` — adds input-validation guardrails
- `PERSONA_DRIFT` — anchors role and persona
- `CONSTRAINT_MISSING` — adds policy- and regulation-grounded constraints
- `AMBIGUOUS_INSTRUCTIONS` — rewrites vague phrases
- `SCOPE_OVERFLOW` — narrows agent boundaries
- `HALLUCINATION_PRONE` — adds grounding and abstention rules
- `MEMORY_RISK` — adds memory-handling constraints
- `POLICY_VIOLATION` — resolves detected policy violations
- `TOOL_QUALITY_LOW` — rewrites low-scoring tool descriptions and parameters

```python
result = sentinel.improve(profile, policies="sample_policies.pdf")
print(result.improved_prompt)
print(result.change_log)
```

### 3. `stress_test(agent, profile)` — adversarial stress test

Three-step pipeline:

1. **Generate** — DSPy generates adversarial prompts targeting every risk flag in the profile → `adversarial_prompts.json`
2. **Run** — fires each prompt against the live agent (multithreaded) → `agent_responses.json`
3. **Evaluate** — DSPy scores each response for policy compliance → `audit_report.json` + `audit_report.md`

```python
report = sentinel.stress_test(agent, profile, policies="sample_policies.pdf")
print(report["summary"])   # pass_rate_pct, passed, failed, total
```

### 4. `audit(agent)` — full automated loop

Runs the complete pipeline with an optimization loop. If stress test pass rate is below `pass_threshold`, it rewrites the prompt, re-inspects, and tests again — up to `max_iterations` times.

```python
result = sentinel.audit(
    agent,
    policies="sample_policies.pdf",
    compliance=["hipaa", "soc2", "owasp", "pii"],  # or ["All"]
    pass_threshold=85.0,    # % pass rate to consider audit complete (default: 80)
    max_iterations=3,       # max optimize → re-test cycles (default: 3)
)

print(result["profile"])    # final InspectedAgentProfile
print(result["report"])     # final stress test report
print(result["iteration"])  # how many optimization cycles ran
```

## Compliance Standards

| Standard | Rules | What it checks |
|---|---|---|
| `hipaa` | 5 rules | PHI handling, minimum necessary access, encryption, audit trails |
| `soc2` | 5 rules | Data security, access control, audit logging, availability |
| `owasp` | 5 rules | LLM Top 10 2025 — prompt injection, insecure output, data leakage |
| `pii` | 5 rules | Consent, retention policy, encryption, scope of collection |

Pass `compliance=["All"]` to check all four standards at once. Rule-based checks run first; ambiguous cases are confirmed by LLM. All standards are checked concurrently.

## Risk Categories

| Category | Description |
|---|---|
| `injection_vulnerable` | System prompt can be overridden by user input |
| `constraint_missing` | No explicit do/don't boundaries defined |
| `ambiguous_instructions` | Vague phrasing that allows misinterpretation |
| `scope_overflow` | Agent can act beyond its intended domain |
| `tool_quality_low` | Tools lack descriptions, typed params, or error handling |
| `persona_drift` | Persona not anchored — model can be role-played out of it |
| `memory_risk` | Memory pattern may leak data across sessions |
| `hallucination_prone` | No grounding or abstention requirements |
| `policy_violation` | Prompt or tools conflict with supplied policy document |
| `compliance_violation` | Prompt violates a regulatory compliance rule |

## Supported Frameworks

| Framework | Status |
|---|---|
| LangGraph | Full support — live object + source file |
| LangChain | Partial — pass `system_prompt` and `tool_definitions` explicitly |
| CrewAI | Demo available (`demo/crewai_agent.py`) |
| Google ADK | Demo available (`demo/google_adk_agent.py`) |
| LlamaIndex | Demo available (`demo/llamaindex_agent.py`) |

For unsupported frameworks, pass `system_prompt`, `tool_definitions`, and optionally `source_code` directly to `inspect()`.

## Environment Variables

```bash
# LLM for semantic analysis, compliance, improver, and tester (via Groq)
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile   # or any Groq-hosted model

# Fallback LLM for compliance checks (via Google Gemini)
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash        # optional, defaults to gemini-2.0-flash

# LLM for demo agents (via OpenRouter)
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL=stepfun/step-3.5-flash

# Optional
COMPLIANCE_TIMEOUT=30                # seconds per compliance LLM call (default: 30)
```

## Running Tests

```bash
uv run pytest
```

## License

MIT
