from typing import Any, Optional

from pydantic import BaseModel, Field

class ExtractionResult(BaseModel):
    """Normalised intake output. All extractors return this shape."""

    system_prompt: str = ""
    tool_definitions: list[dict[str, Any]] = Field(default_factory=list)
    framework: str = "unknown"
    source_object: Optional[Any] = None  # raw agent ref for framework analyzer
    source_code: Optional[str] = None    # for file_path intake
    warnings: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    model_config = {"arbitrary_types_allowed": True}

    def compute_confidence(self) -> float:
        score = 0.0
        if self.system_prompt and len(self.system_prompt) > 50:
            score += 0.5
        if self.system_prompt and len(self.system_prompt) > 200:
            score += 0.1
        if self.tool_definitions:
            score += 0.3
        if self.framework != "unknown":
            score += 0.1
        return min(score, 1.0)