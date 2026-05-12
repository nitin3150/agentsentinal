from typing import Any, Optional

from agentsentinal.intake.raw_strings import extract_from_strings
from agentsentinal.intake.types import ExtractionResult


def extract(
    agent: Any = None,
    system_prompt: Optional[str] = None,
    tool_definitions: Optional[list[dict[str, Any]]] = None,
    file_path: Optional[str] = None,
    framework_hint: Optional[str] = None,
) -> ExtractionResult:
    """Dispatch to the right intake handler.

    Priority:
      1. system_prompt string  → Format 3 (raw strings)
      2. file_path string      → Format 4 (AST)
      3. agent object          → Format 1 or 2 (LangGraph or LangChain)
    """
    if system_prompt is not None:
        return extract_from_strings(system_prompt, tool_definitions, framework_hint)

    if file_path is not None:
        from agentsentinal.intake.file_path import extract_from_file
        return extract_from_file(file_path, framework_hint)

    if agent is not None:
        return _extract_from_object(agent, framework_hint)

    result = ExtractionResult()
    result.warnings.append("No agent, prompt, or file_path provided")
    return result


def _extract_from_object(agent: Any, framework_hint: Optional[str]) -> ExtractionResult:
    """Pick between LangGraph and LangChain based on agent shape."""
    # LangGraph compiled graph exposes get_graph() and .nodes
    if hasattr(agent, "get_graph") and hasattr(agent, "nodes"):
        from agentsentinal.intake.langgraph import extract_from_langgraph
        return extract_from_langgraph(agent, framework_hint)

    # LangChain AgentExecutor / Runnable
    if hasattr(agent, "agent") or hasattr(agent, "runnable") or hasattr(agent, "invoke"):
        from agentsentinal.intake.langchain import extract_from_langchain
        return extract_from_langchain(agent, framework_hint)

    result = ExtractionResult(framework=framework_hint or "unknown")
    result.warnings.append(f"Unrecognised agent object type: {type(agent).__name__}")
    return result


__all__ = ["extract", "ExtractionResult"]
