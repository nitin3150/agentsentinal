from typing import Any

from pydantic import BaseModel, Field

from agentsentinal.models import RiskCategory, RiskFlag, RiskLevel, ToolProfile


MIN_DESCRIPTION_CHARS = 10

ERROR_HANDLING_HINTS = [
    "error", "fail", "fails", "failure",
    "empty", "timeout", "exception",
    "raise", "raises", "returns none",
    "if not found", "if missing", "unavailable",
    "fallback",
]


class ToolsAnalysis(BaseModel):
    tool_profiles: list[ToolProfile] = Field(default_factory=list)
    tool_count: int = 0
    avg_tool_quality: float = 0.0
    risk_flags: list[RiskFlag] = Field(default_factory=list)


def _normalise_tool(tool: Any) -> dict[str, Any]:
    """Normalise dict-form, StructuredTool, or BaseTool into a common dict."""
    if isinstance(tool, dict):
        params = tool.get("parameters") or tool.get("args_schema") or {}
        properties = params.get("properties", {}) if isinstance(params, dict) else {}
        required = params.get("required", []) if isinstance(params, dict) else []
        return {
            "name": tool.get("name", "<unnamed>"),
            "description": tool.get("description", "") or "",
            "properties": properties,
            "required": list(required) if isinstance(required, (list, tuple)) else [],
        }

    name = getattr(tool, "name", None) or getattr(tool, "__name__", "<unnamed>")
    description = getattr(tool, "description", "") or getattr(tool, "__doc__", "") or ""

    properties: dict[str, Any] = {}
    required: list[str] = []
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        try:
            schema = args_schema.model_json_schema() if hasattr(args_schema, "model_json_schema") else args_schema
            if isinstance(schema, dict):
                properties = schema.get("properties", {}) or {}
                required = list(schema.get("required", []) or [])
        except Exception:
            properties = {}

    return {
        "name": name,
        "description": description.strip(),
        "properties": properties,
        "required": required,
    }


def _score_tool(norm: dict[str, Any]) -> ToolProfile:
    desc = norm["description"]
    properties: dict[str, Any] = norm["properties"]

    has_description = len(desc) >= MIN_DESCRIPTION_CHARS
    param_count = len(properties)

    typed = 0
    for spec in properties.values():
        if isinstance(spec, dict) and ("type" in spec or "anyOf" in spec or "$ref" in spec):
            typed += 1
    has_typed_params = param_count == 0 or typed == param_count

    has_error_handling = any(h in desc.lower() for h in ERROR_HANDLING_HINTS)

    missing: list[str] = []
    if not has_description:
        missing.append("description (too short or empty)")
    if not has_typed_params:
        missing.append("typed parameters")
    if not has_error_handling:
        missing.append("error/empty/timeout behaviour in description")

    # Score: 4 base + components.
    score = 4
    if has_description:
        score += 2
    if len(desc) > 60:
        score += 1
    if has_typed_params:
        score += 2
    if has_error_handling:
        score += 1
    score = max(1, min(10, score))

    return ToolProfile(
        name=norm["name"],
        has_description=has_description,
        has_typed_params=has_typed_params,
        has_error_handling=has_error_handling,
        param_count=param_count,
        missing_fields=missing,
        quality_score=score,
    )


def _build_flags(profiles: list[ToolProfile]) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    for p in profiles:
        if p.quality_score >= 6:
            continue
        severity = RiskLevel.HIGH if p.quality_score < 4 else RiskLevel.MEDIUM
        flags.append(RiskFlag(
            category=RiskCategory.TOOL_QUALITY_LOW,
            description=f"Tool '{p.name}' scored {p.quality_score}/10. Missing: {', '.join(p.missing_fields) or 'unknown'}.",
            location=f"tool:{p.name}",
            severity=severity,
            suggestion="Add a description explaining what the tool does, what it returns, and what happens on failure. Annotate every parameter with an explicit type.",
        ))
    return flags


def analyze_tools(tools: list[Any]) -> ToolsAnalysis:
    """Pure static analysis of tool definitions. No LLM calls."""
    if not tools:
        return ToolsAnalysis(tool_count=0, avg_tool_quality=0.0)

    profiles = [_score_tool(_normalise_tool(t)) for t in tools]
    avg = sum(p.quality_score for p in profiles) / len(profiles)
    return ToolsAnalysis(
        tool_profiles=profiles,
        tool_count=len(profiles),
        avg_tool_quality=round(avg, 2),
        risk_flags=_build_flags(profiles),
    )
