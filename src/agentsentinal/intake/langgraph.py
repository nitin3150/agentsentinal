import inspect
from typing import Any, Optional

from agentsentinal.intake.types import ExtractionResult


MIN_PROMPT_LEN = 20
MODEL_NODE_NAMES = ("model", "agent")  # newer / older LangGraph
TOOLS_NODE_NAMES = ("tools",)
SKIP_NODES = ("__start__", "__end__")


def _coerce_prompt(value: Any) -> Optional[str]:
    """Accept str, SystemMessage, ChatPromptTemplate. Return prompt text or None."""
    if value is None:
        return None
    if isinstance(value, str) and len(value) > MIN_PROMPT_LEN:
        return value

    # SystemMessage (langchain_core.messages)
    content = getattr(value, "content", None)
    if isinstance(content, str) and len(content) > MIN_PROMPT_LEN:
        return content

    # ChatPromptTemplate
    messages = getattr(value, "messages", None)
    if messages:
        parts: list[str] = []
        for m in messages:
            t = getattr(m, "content", None) or getattr(getattr(m, "prompt", None), "template", None)
            if isinstance(t, str):
                parts.append(t)
        joined = "\n".join(parts).strip()
        if len(joined) > MIN_PROMPT_LEN:
            return joined

    # Plain ChatPromptValue or PromptTemplate with .template
    template = getattr(value, "template", None)
    if isinstance(template, str) and len(template) > MIN_PROMPT_LEN:
        return template

    return None


def _extract_prompt_from_model_node(model_node: Any) -> Optional[str]:
    """Walk closure of the model node's callable to find the system prompt."""
    bound = getattr(model_node, "bound", None) or model_node
    fn = getattr(bound, "afunc", None) or getattr(bound, "func", None) or bound
    if not callable(fn):
        return None

    # Try named closurevars first (cheap + readable).
    try:
        nonlocals = inspect.getclosurevars(fn).nonlocals
    except Exception:
        nonlocals = {}

    for name in ("system_message", "state_modifier", "prompt", "system_prompt"):
        if name in nonlocals:
            text = _coerce_prompt(nonlocals[name])
            if text:
                return text

    # Fall back to raw cell walk.
    closure = getattr(fn, "__closure__", None) or ()
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        text = _coerce_prompt(value)
        if text:
            return text

    return None


def _extract_tools_from_tools_node(tools_node: Any) -> list[Any]:
    """Pull StructuredTool list off the tools node. Returns [] if absent."""
    bound = getattr(tools_node, "bound", None) or tools_node
    for attr in ("tools_by_name", "_tools_by_name"):
        registry = getattr(bound, attr, None)
        if isinstance(registry, dict) and registry:
            return list(registry.values())

    # Older shapes: nodes['tools'].data._tools_by_name
    data = getattr(tools_node, "data", None)
    for attr in ("tools_by_name", "_tools_by_name"):
        registry = getattr(data, attr, None)
        if isinstance(registry, dict) and registry:
            return list(registry.values())

    # Last resort: any `.tools` attribute returning a list.
    tools_attr = getattr(bound, "tools", None) or getattr(data, "tools", None)
    if isinstance(tools_attr, (list, tuple)):
        return list(tools_attr)
    return []


def _structured_tool_to_dict(tool: Any) -> dict[str, Any]:
    name = getattr(tool, "name", None) or getattr(tool, "__name__", "<unnamed>")
    description = (getattr(tool, "description", "") or getattr(tool, "__doc__", "") or "").strip()
    parameters: dict[str, Any] = {}
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        try:
            schema = args_schema.model_json_schema() if hasattr(args_schema, "model_json_schema") else None
            if isinstance(schema, dict):
                parameters = {
                    "properties": schema.get("properties", {}) or {},
                    "required": schema.get("required", []) or [],
                }
        except Exception:
            parameters = {}
    return {"name": name, "description": description, "parameters": parameters}


def extract_from_langgraph(agent: Any, framework_hint: Optional[str] = None) -> ExtractionResult:
    """Format 1: compiled LangGraph graph (CompiledStateGraph / create_react_agent output)."""
    result = ExtractionResult(framework=framework_hint or "langgraph", source_object=agent)

    nodes: dict[str, Any] = {}
    try:
        nodes = dict(agent.nodes)
    except Exception as exc:
        result.warnings.append(f"Could not read agent.nodes: {exc}")
        result.confidence = result.compute_confidence()
        return result

    # Model node + prompt
    model_node = None
    for candidate in MODEL_NODE_NAMES:
        if candidate in nodes and candidate not in SKIP_NODES:
            model_node = nodes[candidate]
            break
    if model_node is None:
        result.warnings.append("Could not locate model/agent node in graph.")
    else:
        prompt = _extract_prompt_from_model_node(model_node)
        if prompt:
            result.system_prompt = prompt
        else:
            result.warnings.append("Found model node but could not extract a system prompt from its closure.")

    # Tools node
    tools_node = None
    for candidate in TOOLS_NODE_NAMES:
        if candidate in nodes:
            tools_node = nodes[candidate]
            break
    if tools_node is not None:
        structured = _extract_tools_from_tools_node(tools_node)
        result.tool_definitions = [_structured_tool_to_dict(t) for t in structured]
        if not structured:
            result.warnings.append("Found tools node but tools_by_name was empty or missing.")

    # Intentionally not setting source_code: inspect.getsource(type(agent)) returns
    # framework internals (CompiledStateGraph), not the user's code. Memory/framework
    # text scans on that would be misleading. Format 4 (file_path) supplies source.
    result.confidence = result.compute_confidence()
    return result
