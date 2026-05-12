# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent Sentinal is an enterprise-grade data protection system implementing "Data Protection as Code". Currently a skeleton project with minimal implementation.

## Running the Project

```bash
# Run main entry point (includes OpenTelemetry tracer setup)
python main.py

# Alternative entry point
python sentinel_tracer.py
```

## Dependencies

Install via `pyproject.toml`:
```bash
pip install -r requirements.txt
```

Key dependencies:
- **Web**: FastAPI, uvicorn, pydantic
- **LLM**: langchain, langgraph, dspy-ai, langchain-google-vertexai
- **Observability**: opentelemetry-sdk, opentelemetry-exporter-otlp
- **Data**: sqlalchemy, asyncpg, redis, arq
- **PII Protection**: presidio-analyzer, presidio-anonymizer
- **Evaluation**: ragas, langsmith

## Architecture (Planned)

Per README.md, intended structure:
```
agentsentinal/
├── api/           # API endpoints
├── core/          # Core config
├── services/      # Business logic
│   ├── llm/       # LLM integration (Vertex AI)
│   ├── vector_store/
│   ├── worker/
│   ├── anonymizer/
│   ├── classify/
│   └── validator/
├── schemas/       # Pydantic models
└── main.py
```

Current state: directories are empty placeholders.

## Key Files

- `main.py` - Contains `SentinelTracer` class with OpenTelemetry instrumentation for LLM/agent monitoring
- `sentinel_tracer.py` - Basic entry point
- `.env` - Contains `OPENROUTER_API_KEY` and `MODEL` config

## Two-Tier Classification System (Planned)

Per README.md:
- **Tier 1**: Content classification (PHI, SPI, PCI, PII, IP) using 3-shot prompt engineering
- **Tier 2**: Policy-driven enforcement via LLM validators (Allow/Reject/Flag)
