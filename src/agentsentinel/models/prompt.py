from pydantic import BaseModel

class ChangeLogEntry(BaseModel):
    field: str
    before: str
    after: str
    reason: str

class OptimizedResult(BaseModel):
    """
    Returned by PromptOptimized.forward().

    improved_prompt:
        Drop-in replacement for the original system prompt.

    improved_tool_definitions:
        List of tool dicts in the same format InspectorAgent accepts.
        Only tools that were rewritten are included; the rest are passed
        through unchanged.

    change_log:
        One entry per fix applied, describing what changed and why.

    policy_violations:
        Empty list = clean.  Non-empty = the improved prompt still
        contains policy-violating content and must not be deployed.
    """
    optimized_prompt:           str
    optimized_tool_definitions: list[dict]
    change_log:                list[ChangeLogEntry]
    policy_violations:         list[str]
    diff:                      str = ""