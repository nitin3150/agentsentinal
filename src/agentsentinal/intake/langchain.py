from typing import Any, Optional

from agentsentinal.intake.types import ExtractionResult


MIN_PROMPT_LEN = 20
PROMPT_ATTR_NAMES = ("prompt", "system_message", "system_prompt", "template")
RUNNABLE_ATTR_NAMES = ("agent", "runnable", "bound", "first", "_runnable")


def _coerce_prompt(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str) and len(value) > MIN_PROMPT_LEN:
        return value
    content = getattr(value, "content", None)
    if isinstance(content, str) and len(content) > MIN_PROMPT_LEN:
        return content
    template = getattr(value, "template", None)
    if isinstance(template, str) and len(template) > MIN_PROMPT_LEN:
        return template
    messages = getattr(value, "messages", None)
    if messages:
        parts: list[str] = []
        for m in messages:
            t = getattr(m, "content", None)
            if not isinstance(t, str):
                tpl = getattr(m, "prompt", None)
                t = getattr(tpl, "template", None)
            if isinstance(t, str):
                parts.append(t)
        joined = "\n".join(parts).strip()
        if len(joined) > MIN_PROMPT_LEN:
            return joined
    return None


def _walk_for_prompt(obj: Any, depth: int = 0, seen: Optional[set[int]] = None) -> Optional[str]:
    """Recurse through common runnable wrappers looking for a ChatPromptTemplate."""
    if obj is None or depth > 6:
        return None
    seen = seen or set()
    obj_id = id(obj)
    if obj_id in seen:
        return None
    seen.add(obj_id)

    # Direct attribute hits
    for attr in PROMPT_ATTR_NAMES:
        candidate = getattr(obj, attr, None)
        text = _coerce_prompt(candidate)
        if text:
            return text

    # Object itself may be a ChatPromptTemplate / SystemMessage
    text = _coerce_prompt(obj)
    if text:
        return text

    # RunnableSequence has .steps
    steps = getattr(obj, "steps", None)
    if isinstance(steps, (list, tuple)):
        for step in steps:
            text = _walk_for_prompt(step, depth + 1, seen)
            if text:
                return text

    # RunnableSequence first/middle/last attributes
    for attr in RUNNABLE_ATTR_NAMES:
        child = getattr(obj, attr, None)
        if child is not None and child is not obj:
            text = _walk_for_prompt(child, depth + 1, seen)
            if text:
                return text

    return None


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


def _find_tools(agent: Any) -> list[Any]:
    for attr in ("tools", "_tools"):
        tools = getattr(agent, attr, None)
        if isinstance(tools, (list, tuple)) and tools:
            return list(tools)
    inner = getattr(agent, "agent", None)
    if inner is not None:
        for attr in ("tools", "_tools"):
            tools = getattr(inner, attr, None)
            if isinstance(tools, (list, tuple)) and tools:
                return list(tools)
    return []


def extract_from_langchain(agent: Any, framework_hint: Optional[str] = None) -> ExtractionResult:
    """Format 2: LangChain AgentExecutor / RunnableWithMessageHistory / generic Runnable."""
    result = ExtractionResult(framework=framework_hint or "langchain", source_object=agent)

    prompt = _walk_for_prompt(agent)
    if prompt:
        result.system_prompt = prompt
    else:
        result.warnings.append(
            "Could not extract system prompt from LangChain agent. Pass system_prompt= explicitly."
        )

    tools = _find_tools(agent)
    result.tool_definitions = [_structured_tool_to_dict(t) for t in tools]
    if not tools:
        result.warnings.append("Could not locate tools on the LangChain agent.")

    # source_code intentionally left None — see langgraph.py comment.
    result.confidence = result.compute_confidence()
    return result
