import ast
import importlib.util
import logging
from pathlib import Path
from typing import Any, Optional

from agentsentinel.models import AgentProfile

logger = logging.getLogger(__name__)


class FilePathDetector:
    def __init__(self, agent: Any) -> None:
        self.path = agent

    def can_handle(self) -> bool:
        if not isinstance(self.path, (str, Path)):
            return False
        p = Path(self.path)
        return p.exists() and p.suffix == '.py'

    def _find_compile_target(self, source: str) -> Optional[str]:
        """AST scan: return variable name assigned from a .compile() call at module level."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            func = node.value.func
            if not (isinstance(func, ast.Attribute) and func.attr == 'compile'):
                continue
            if node.targets and isinstance(node.targets[0], ast.Name):
                return node.targets[0].id
        return None

    def _import_and_get(self, path: Path, var_name: str) -> Optional[Any]:
        """Dynamically import a Python file and return the named module-level variable."""
        spec = importlib.util.spec_from_file_location("_sentinel_agent_module", str(path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return getattr(module, var_name, None)

    def __call__(self) -> AgentProfile:
        path = Path(self.path)
        source = path.read_text()
        compiled_var = self._find_compile_target(source)

        if compiled_var:
            try:
                graph_obj = self._import_and_get(path, compiled_var)
                if graph_obj is not None:
                    from agentsentinel.core.agents.intake.detectors.langgraph import LangGraphDetector
                    detector = LangGraphDetector(graph_obj)
                    if detector.can_handle():
                        logger.info("FilePathDetector: dynamic import succeeded, using LangGraphDetector")
                        result = detector()
                        result.source_object = graph_obj
                        result.source_code = source
                        return result
            except Exception as exc:
                logger.warning("FilePathDetector: dynamic import failed (%s), falling back to static AST", exc)

        logger.info("FilePathDetector: using static AST extraction")
        return self._static_extract(source)

    def _static_extract(self, source: str) -> AgentProfile:
        """Low-confidence fallback: extract what AST can see without executing code."""
        from agentsentinel.core.agents.intake.detectors.langgraph.utils import _PROMPT_VAR_NAMES

        warnings = ["Static AST extraction — dynamic import unavailable (lower confidence)"]
        system_prompt = ""
        tool_definitions: list[dict] = []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return AgentProfile(
                warnings=["Failed to parse Python file"],
                source_code=source,
            )

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not (isinstance(target, ast.Name) and target.id in _PROMPT_VAR_NAMES):
                    continue
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    system_prompt = node.value.value
                    break
            if system_prompt:
                break

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                is_tool = (
                    (isinstance(dec, ast.Name) and dec.id == 'tool') or
                    (isinstance(dec, ast.Attribute) and dec.attr == 'tool')
                )
                if is_tool:
                    tool_definitions.append({
                        'name': node.name,
                        'description': ast.get_docstring(node) or "",
                        'parameters': {},
                    })
                    break

        framework = "langgraph" if "langgraph" in source else "unknown"

        return AgentProfile(
            system_prompt=system_prompt,
            tool_definitions=tool_definitions,
            framework=framework,
            warnings=warnings,
            source_code=source,
        )
