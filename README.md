# Agent Sentinel

Agent Sentinel inspects, optimizes, and stress-tests AI agents before they ship. It analyzes an agent source file or live object, surfaces risk and compliance issues, rewrites the prompt when needed, and runs adversarial tests to verify the result.

## What’s Included

- `inspect` for source-level risk analysis and compliance checks
- `optimize` for prompt rewriting from a saved inspection profile
- `stress-test` for adversarial prompt campaigns against a live agent
- `audit` for the full inspect → optimize → retest loop
- `main.py` as a simple end-to-end demo script

## Installation

This project is packaged as `agentsentinel-ai`.

```bash
pip install -e .
```

If you want the optional demo integrations, install the extras declared in `pyproject.toml`:

```bash
pip install -e ".[demo]"
```

## Environment

The code uses LiteLLM-compatible model strings. Set one of these model variables:

```bash
LLM_MODEL=groq/llama-3.3-70b-versatile
# or
GROQ_MODEL=groq/llama-3.3-70b-versatile
# or
OPENROUTER_MODEL=openrouter/anthropic/claude-3.5-sonnet
```

Then provide the matching API key for your provider:

```bash
GROQ_API_KEY=your_key
OPENROUTER_API_KEY=your_key
```

Optional settings:

```bash
LLM_TIMEOUT=30
AGENTSENTINEL_LOG_PROMPTS=false
AGENTSENTINEL_SAFE_MODE=true
```

## CLI

The package installs an `agentsentinel` command.

```bash
agentsentinel --help
agentsentinel inspect path/to/agent.py --compliance hipaa,soc2 --policies path/to/policy.pdf
agentsentinel optimize path/to/profile.json --policies path/to/policy.pdf
agentsentinel stress-test path/to/agent.py --profile path/to/profile.json
agentsentinel audit path/to/agent.py --compliance hipaa,soc2,owasp,pii
```

CLI options:

- `inspect` supports `--then-optimize` and `--then-stress-test`
- `stress-test` can re-use a saved profile with `--profile`
- `audit` supports `--threshold`, `--max-iterations`, and `--output-dir`

## Demo Script

`main.py` wires the library into the LangChain demo agent:

```bash
python main.py
```

It loads the demo agent, inspects it, optimizes the profile, and runs the stress test.

## Library Usage

```python
from agentsentinel.sentinel import AgentSentinel

sentinel = AgentSentinel()
profile = sentinel.inspect(
    source="demo/langchain_agent.py",
    domain="personal agent",
    policies="sample_policies.pdf",
    compliance=["hipaa", "soc2"],
)

result = sentinel.optimize(profile, policies="sample_policies.pdf")
report = sentinel.stress_test(agent, profile, policies="sample_policies.pdf")
```

The `audit` helper runs the full loop:

```python
result = sentinel.audit(
    agent,
    policies="sample_policies.pdf",
    compliance=["hipaa", "soc2", "owasp", "pii"],
    pass_threshold=80.0,
    max_iterations=3,
)
```

## Repository Structure

```text
src/agentsentinel/
├── cli/          # Click commands, loaders, and output helpers
├── compliance/   # YAML compliance rules
├── core/         # Intake, inspection, optimization, and testing pipelines
├── models/       # Pydantic models for profiles and results
├── utils/        # Shared helpers such as LLM and policy utilities
└── sentinel.py   # Public orchestration entry point

demo/             # Example agents for LangChain, LangGraph, CrewAI, ADK, and LlamaIndex
tests/            # Unit tests for detectors, CLI, and workflows
```

## Testing

```bash
uv run pytest
```

## License

MIT
