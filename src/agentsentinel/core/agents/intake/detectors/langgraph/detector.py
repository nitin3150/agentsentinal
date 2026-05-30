import logging
from typing import Any, Optional

from agentsentinel.models import AgentProfile

from .utils import SKIP_NODES, _unwrap_fn
from .prompt import val_to_prompt, extract_system_prompt
from .tools import extract_tool_definitions, extract_bound_tools, _find_tools_registry

logger = logging.getLogger(__name__)


class LangGraphDetector:
    MODEL_NODE_NAMES = (
        'model', 'agent', 'assistant', 'chatbot',
        'llm', 'call_model', 'generate', 'reasoner',
    )

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def can_handle(self) -> bool:
        import importlib
        pregel_class = None
        for import_path in ('langgraph.pregel', 'langgraph.pregel.main'):
            try:
                mod = importlib.import_module(import_path)
                pregel_class = getattr(mod, 'Pregel', None)
                if pregel_class is not None:
                    break
            except ImportError:
                continue
        if pregel_class is None:
            return False
        logger.info("Framework: LangGraph")
        return isinstance(self.agent, pregel_class)

    # ── Context schema helpers ────────────────────────────────────────────────

    def _extract_from_context_schema(self) -> Optional[str]:
        for attr in ('context_schema', 'input_schema'):
            schema = getattr(self.agent, attr, None)
            if schema is None:
                continue
            # Dataclass fields
            if hasattr(schema, '__dataclass_fields__'):
                field = schema.__dataclass_fields__.get('system_prompt')
                if field:
                    default = field.default
                    if default and isinstance(default, str) and len(default) > 15:
                        return default
                    if field.default_factory != type(field.default_factory):
                        try:
                            val = field.default_factory()
                            if isinstance(val, str) and len(val) > 15:
                                return val
                        except Exception:
                            pass
            # Pydantic v2 model_fields
            model_fields = getattr(schema, 'model_fields', None)
            if isinstance(model_fields, dict):
                field_info = model_fields.get('system_prompt')
                if field_info is not None:
                    default = getattr(field_info, 'default', None)
                    if default is not None and isinstance(default, str) and len(default) > 15:
                        return default
        return None

    def _extract_from_context_instance(self) -> Optional[str]:
        context = getattr(self.agent, 'config', None)
        if context is None:
            try:
                context_schema = getattr(self.agent, 'context_schema', None)
                if context_schema:
                    context = context_schema()
            except Exception:
                pass
        if context and hasattr(context, 'system_prompt'):
            result = val_to_prompt(getattr(context, 'system_prompt'), min_len=0)
            if result:
                return result
        return None

    def _extract_from_node(self, node: Any) -> Optional[str]:
        if hasattr(node, 'nodes'):
            sub = LangGraphDetector(node)
            return sub().system_prompt or None
        bound = getattr(node, 'bound', None) or node
        fn = getattr(bound, 'afunc', None) or getattr(bound, 'func', None) or bound
        fn = getattr(fn, '__wrapped__', fn)
        if hasattr(fn, 'nodes'):
            sub = LangGraphDetector(fn)
            return sub().system_prompt or None
        return extract_system_prompt(node)

    # ── Orchestration ─────────────────────────────────────────────────────────

    def __call__(self) -> AgentProfile:
        result = AgentProfile(
            system_prompt='',
            framework='langgraph',
            tool_definitions=[],
            warnings=[],
        )

        try:
            nodes = dict(self.agent.nodes)
        except Exception:
            result.warnings.append('Could not read graph nodes')
            return result

        # ── System prompt ────────────────────────────────────────────────────
        model_node = next(
            (nodes[n] for n in self.MODEL_NODE_NAMES if n in nodes), None
        )
        if model_node is not None:
            prompt = self._extract_from_node(model_node)
            if prompt:
                logger.info("System prompt extracted from agent")
                result.system_prompt = prompt
            else:
                logger.debug("System prompt not found in model node, checking context schema")
                result.warnings.append('System prompt not found in closure')
        else:
            for name, node in nodes.items():
                if name in SKIP_NODES or name == 'tools':
                    continue
                prompt = self._extract_from_node(node)
                if prompt:
                    result.system_prompt = prompt
                    break
            if not result.system_prompt:
                result.warnings.append('Model node not found')

        if not result.system_prompt:
            prompt = self._extract_from_context_schema()
            if prompt:
                logger.info("System prompt extracted from context schema")
                result.system_prompt = prompt
            else:
                prompt = self._extract_from_context_instance()
                if prompt:
                    logger.info("System prompt extracted from context instance")
                    result.system_prompt = prompt

        # ── Tool definitions ─────────────────────────────────────────────────
        tools_node = None
        for _name, _node in nodes.items():
            if _name in SKIP_NODES:
                continue
            if _find_tools_registry(_node) is not None:
                tools_node = _node
                break

        if tools_node is not None:
            tools = extract_tool_definitions(tools_node)
            if tools:
                logger.info("Tools found: %d", len(tools))
                result.tool_definitions = tools
            else:
                logger.error("ToolNode present but no tools extracted")
                result.warnings.append('No tools found')
        else:
            bound_tools = extract_bound_tools(nodes)
            if bound_tools:
                result.tool_definitions = bound_tools

        # Subgraph fallback
        if not result.tool_definitions:
            for _name, _node in nodes.items():
                if _name in SKIP_NODES:
                    continue
                _bound = getattr(_node, 'bound', None) or _node
                _fn = getattr(_bound, 'afunc', None) or getattr(_bound, 'func', None) or _bound
                _fn = _unwrap_fn(_fn)
                if hasattr(_fn, 'nodes'):
                    sub = LangGraphDetector(_fn)
                    sub_result = sub()
                    if sub_result.tool_definitions:
                        result.tool_definitions = sub_result.tool_definitions
                        break

        result.source_object = self.agent
        return result
