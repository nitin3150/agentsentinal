import ast
from pathlib import Path
from typing import Any, Optional

from agentsentinal.intake.types import ExtractionResult


PROMPT_NAMES = (
    "SYSTEM_PROMPT", "SYS_PROMPT", "PROMPT", "INSTRUCTIONS",
    "AGENT_PROMPT", "SYSTEM_MESSAGE", "ROLE_PROMPT",
)

TOOL_DECORATOR_NAMES = ("tool", "Tool", "function_tool")
TOOL_CONSTRUCTOR_NAMES = ("Tool", "StructuredTool", "FunctionTool")

FRAMEWORK_IMPORT_HINTS = {
    "langgraph":  ["langgraph"],
    "langchain":  ["langchain"],
    "crewai":     ["crewai"],
    "autogen":    ["autogen"],
    "adk":        ["google.adk", "adk"],
}


def _read_source(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _str_from_node(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                parts.append("{...}")  # placeholder for f-string slots
        return "".join(parts)
    return None


def _find_module_prompt(tree: ast.Module) -> Optional[str]:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not any(n in PROMPT_NAMES or n.upper() in PROMPT_NAMES for n in names):
                continue
            text = _str_from_node(node.value)
            if text and len(text.strip()) > 20:
                return text.strip()
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in PROMPT_NAMES or node.target.id.upper() in PROMPT_NAMES:
                text = _str_from_node(node.value) if node.value is not None else None
                if text and len(text.strip()) > 20:
                    return text.strip()
    return None


def _decorator_id(dec: ast.AST) -> Optional[str]:
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Call):
        if isinstance(dec.func, ast.Name):
            return dec.func.id
        if isinstance(dec.func, ast.Attribute):
            return dec.func.attr
    if isinstance(dec, ast.Attribute):
        return dec.attr
    return None


def _annotation_type(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        return ast.unparse(node) if hasattr(ast, "unparse") else None
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _properties_from_args(args: ast.arguments) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for arg in args.args:
        if arg.arg == "self":
            continue
        ann = _annotation_type(arg.annotation)
        spec: dict[str, Any] = {}
        if ann:
            spec["type"] = ann.lower() if ann in ("str", "int", "float", "bool", "list", "dict") else ann
        props[arg.arg] = spec
    return props


def _find_decorated_tools(tree: ast.Module) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(_decorator_id(d) in TOOL_DECORATOR_NAMES for d in node.decorator_list):
            continue
        description = (ast.get_docstring(node) or "").strip()
        props = _properties_from_args(node.args)
        required = [name for name in props.keys()]
        tools.append({
            "name": node.name,
            "description": description,
            "parameters": {"properties": props, "required": required},
        })
    return tools


def _detect_framework(tree: ast.Module) -> str:
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    text = " ".join(imports)
    for fw, hints in FRAMEWORK_IMPORT_HINTS.items():
        if any(h in text for h in hints):
            return fw
    return "unknown"


def extract_from_file(file_path: str, framework_hint: Optional[str] = None) -> ExtractionResult:
    """Format 4: Static parse of a Python source file."""
    path = Path(file_path)
    if not path.is_file():
        result = ExtractionResult(framework=framework_hint or "unknown")
        result.warnings.append(f"File not found or not a regular file: {file_path}")
        result.confidence = result.compute_confidence()
        return result

    source = _read_source(path)
    if source is None:
        result = ExtractionResult(framework=framework_hint or "unknown")
        result.warnings.append(f"Could not read file: {file_path}")
        result.confidence = result.compute_confidence()
        return result

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        result = ExtractionResult(framework=framework_hint or "unknown", source_code=source)
        result.warnings.append(f"Syntax error parsing {path.name}: {exc}")
        result.confidence = result.compute_confidence()
        return result

    framework = framework_hint or _detect_framework(tree)
    prompt = _find_module_prompt(tree) or ""
    tools = _find_decorated_tools(tree)

    result = ExtractionResult(
        system_prompt=prompt,
        tool_definitions=tools,
        framework=framework,
        source_code=source,
    )
    if not prompt:
        result.warnings.append(
            f"No module-level system prompt assignment found (looked for: {', '.join(PROMPT_NAMES)})."
        )
    if not tools:
        result.warnings.append("No @tool-decorated functions found.")
    result.confidence = result.compute_confidence()
    return result
