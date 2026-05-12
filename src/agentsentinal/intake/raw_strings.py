from typing import Any, Optional

from agentsentinal.intake.types import ExtractionResult


def extract_from_strings(
    system_prompt: str,
    tool_definitions: Optional[list[dict[str, Any]]] = None,
    framework_hint: Optional[str] = None,
) -> ExtractionResult:
    """Format 3 intake: user passes prompt + tool dicts directly."""
    result = ExtractionResult(
        system_prompt=system_prompt or "",
        tool_definitions=list(tool_definitions or []),
        framework=framework_hint or "unknown",
    )
    if not result.system_prompt:
        result.warnings.append("Empty system_prompt provided")
    result.confidence = result.compute_confidence()
    return result
