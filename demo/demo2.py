# demo/test_inspector.py

import asyncio
from agentsentinal.core.agents.inspector import InspectorAgent

BROKEN_DEMO_SYSTEM_PROMPT = """
You are a helpful assistant. Help users with their questions.
Answer questions as needed. Be concise and helpful.
If a user asks something, try to provide a useful answer.
You have access to a search tool and a calculator.
Use your judgment about when to use tools.
"""

TOOL_DEFINITIONS = [
    {
        "name": "search_wikipedia",
        "description": "search",    # deliberately bad — too short
        "parameters": {
            "properties": {
                "query": {}          # deliberately missing type
            }
        }
    },
    {
        "name": "calculator",
        "description": "Performs arithmetic calculations on two numbers.",
        "parameters": {
            "properties": {
                "expression": {"type": "string", "description": "The math expression"},
            }
        }
        # deliberately missing: required fields, error handling
    }
]

async def main():
    inspector = InspectorAgent(semantic_enabled=False)  # offline; flip to True with GOOGLE_API_KEY set

    profile = await inspector.inspect(
        agent_id         = "demo-broken-agent",
        system_prompt    = BROKEN_DEMO_SYSTEM_PROMPT,
        tool_definitions = TOOL_DEFINITIONS,
        framework_hint   = "langgraph",
    )

    print(f"\n{'─'*50}")
    print(f"  AgentProfile for: {profile.agent_id}")
    print(f"{'─'*50}")
    print(f"  Persona clarity:    {profile.persona_clarity_score}/10")
    print(f"  Scope definition:   {profile.scope_definition_score}/10")
    print(f"  Constraint count:   {profile.constraint_count}")
    print(f"  Injection surface:  {profile.injection_surface.value.upper()}")
    print(f"  Avg tool quality:   {profile.avg_tool_quality}/10")
    print(f"  Overall risk:       {profile.overall_risk.value.upper()}")
    print(f"  Estimated baseline: {profile.estimated_baseline_score}/100")

    print(f"\n  Risk flags ({len(profile.risk_flags)} found):")
    for flag in profile.risk_flags:
        print(f"    [{flag.severity.value.upper()}] {flag.category}")
        print(f"      → {flag.description}")
        print(f"      ✓ Fix: {flag.suggestion[:80]}...")

    print(f"\n  Ambiguous phrases found:")
    for phrase in profile.ambiguous_phrases[:5]:
        print(f"    · {phrase}")

if __name__ == "__main__":
    asyncio.run(main())