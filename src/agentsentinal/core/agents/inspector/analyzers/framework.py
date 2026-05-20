import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from agentsentinal.models import RiskCategory, RiskFlag, RiskLevel


FRAMEWORK_HINTS = {
    "langgraph":  ["langgraph", "StateGraph", "create_react_agent", "compile()", "Pregel"],
    "crewai":     ["crewai", "Crew(", "Agent(role"],
    "autogen":    ["autogen", "ConversableAgent", "GroupChat"],
    "adk":        ["google.adk", "adk.agents", "LlmAgent"],
    "langchain":  ["langchain.agents", "AgentExecutor", "create_agent", "RunnableWithMessageHistory"],
}

HITL_PATTERNS = [
    "interrupt_before", "interrupt_after",
    "interrupt(", "human_input", "HumanMessage(",
    "ask_human", "approval", "confirm_with_user",
]

LOOP_PATTERNS = [
    "while True", "while not",
    "for _ in range(",
    "add_edge", "add_conditional_edges",
]


class FrameworkAnalysis(BaseModel):
    framework: str = "unknown"
    estimated_depth: int = 0
    has_loops: bool = False
    has_conditional_edges: bool = False
    has_human_in_loop: bool = False
    risk_flags: list[RiskFlag] = Field(default_factory=list)


def _detect_framework(agent: Any, source: str) -> str:
    # Prefer runtime-object detection.
    if agent is not None:
        mod = type(agent).__module__ or ""
        for fw, hints in FRAMEWORK_HINTS.items():
            for h in hints:
                if h.lower() in mod.lower():
                    return fw
        if hasattr(agent, "get_graph") and hasattr(agent, "nodes"):
            return "langgraph"
        if type(agent).__name__ in ("AgentExecutor", "RunnableWithMessageHistory"):
            return "langchain"

    if not source:
        return "unknown"

    scores: dict[str, int] = {}
    for fw, hints in FRAMEWORK_HINTS.items():
        scores[fw] = sum(1 for h in hints if h in source)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def _inspect_langgraph_object(agent: Any) -> tuple[int, bool, bool, bool]:
    """Return (node_count, has_conditional, has_loops, has_hitl)."""
    node_count = 0
    has_conditional = False
    has_loops = False
    has_hitl = False

    try:
        graph = agent.get_graph()
        nodes = [n for n in graph.nodes if n not in ("__start__", "__end__")]
        node_count = len(nodes)

        edges = getattr(graph, "edges", []) or []
        adjacency: dict[str, list[str]] = {}
        for edge in edges:
            src = getattr(edge, "source", None)
            tgt = getattr(edge, "target", None)
            # LangGraph exposes conditional edges via data attribute or bool flag
            data = getattr(edge, "data", None)
            is_conditional = bool(
                getattr(edge, "conditional", False)
                or callable(data)
                or isinstance(data, str) and data
            )
            if is_conditional:
                has_conditional = True
            if src is not None:
                adjacency.setdefault(src, [])
                if tgt is not None:
                    adjacency[src].append(tgt)

        # DFS back-edge detection (proper cycle check)
        visited: set[str] = set()
        in_stack: set[str] = set()

        def _dfs(node: str) -> bool:
            visited.add(node)
            in_stack.add(node)
            for neighbour in adjacency.get(node, []):
                if neighbour not in visited:
                    if _dfs(neighbour):
                        return True
                elif neighbour in in_stack:
                    return True
            in_stack.discard(node)
            return False

        for start in list(adjacency):
            if start not in visited:
                if _dfs(start):
                    has_loops = True
                    break
    except Exception:
        pass

    # Detect interrupt_before / interrupt_after on the compiled graph if available.
    for attr in ("interrupt_before_nodes", "interrupt_after_nodes",
                 "interrupt_before", "interrupt_after"):
        v = getattr(agent, attr, None)
        if v:
            has_hitl = True
            break

    return node_count, has_conditional, has_loops, has_hitl


def _inspect_source(source: str) -> tuple[bool, bool, bool]:
    """Return (has_conditional, has_loops, has_hitl) from text scan."""
    has_conditional = "add_conditional_edges" in source or "conditional_edges" in source
    has_loops = any(p in source for p in ["while True", "while not", "for _ in range("])
    has_hitl = any(p in source for p in HITL_PATTERNS)
    return has_conditional, has_loops, has_hitl


def _build_flags(an: FrameworkAnalysis) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    if an.has_loops and not an.has_human_in_loop:
        flags.append(RiskFlag(
            category=RiskCategory.SCOPE_OVERFLOW,
            description="Agent graph contains loops with no human-in-the-loop interrupts. Risk of unbounded autonomous execution.",
            location=f"framework:{an.framework}",
            severity=RiskLevel.MEDIUM,
            suggestion="Add a max_iterations limit or an interrupt_before checkpoint on critical nodes.",
        ))
    if an.has_conditional_edges and not an.has_human_in_loop and an.framework == "langgraph":
        flags.append(RiskFlag(
            category=RiskCategory.SCOPE_OVERFLOW,
            description="Agent uses conditional routing without HITL — may take unrecoverable autonomous decisions.",
            location=f"framework:{an.framework}",
            severity=RiskLevel.MEDIUM,
            suggestion="Add interrupt_before on irreversible nodes (e.g. send_email, charge_payment).",
        ))
    return flags


def analyze_framework(agent: Any = None, source_code: Optional[str] = None) -> FrameworkAnalysis:
    """Static framework structure analysis."""
    source = source_code or ""
    framework = _detect_framework(agent, source)

    node_count = 0
    has_conditional = False
    has_loops = False
    has_hitl = False

    if framework == "langgraph" and agent is not None:
        node_count, has_conditional, has_loops, has_hitl = _inspect_langgraph_object(agent)

    if source:
        sc, sl, sh = _inspect_source(source)
        has_conditional = has_conditional or sc
        has_loops = has_loops or sl
        has_hitl = has_hitl or sh

    an = FrameworkAnalysis(
        framework=framework,
        estimated_depth=node_count,
        has_conditional_edges=has_conditional,
        has_loops=has_loops,
        has_human_in_loop=has_hitl,
    )
    an.risk_flags = _build_flags(an)
    return an
