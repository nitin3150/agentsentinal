import inspect
from agentsentinal.models import AgentProfile
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

SKIP_NODES = {'__start__', '__end__'}

# Known variable names that hold system prompts
_PROMPT_VAR_NAMES = (
    'system_prompt', 'system_message', 'prompt', 'state_modifiers',
    'instructions', 'system', 'sys_prompt', 'SYSTEM', 'SYSTEM_PROMPT',
)


class LangGraphDetector:
    MODEL_NODE_NAMES = ('model', 'agent')

    def __init__(self, agent):
        self.agent = agent

    def can_handle(self) -> bool:
        try:
            from langgraph.pregel.main import Pregel
            return isinstance(self.agent, Pregel)
        except ImportError:
            return hasattr(self.agent, 'nodes')

    # ── Prompt helpers ────────────────────────────────────────────────────────

    def _val_to_prompt(self, val: Any, min_len: int = 0) -> Optional[str]:
        """Extract a prompt string from a single value."""
        if val is None:
            return None
        # Plain string
        if isinstance(val, str) and len(val) > min_len:
            return val
        # BaseMessage (SystemMessage etc.) with .content
        content = getattr(val, 'content', None)
        if isinstance(content, str) and len(content) > min_len:
            return content
        # ChatPromptTemplate — has .messages list of MessagePromptTemplate
        messages = getattr(val, 'messages', None)
        if isinstance(messages, (list, tuple)):
            for msg_tpl in messages:
                if hasattr(msg_tpl, 'prompt') and hasattr(msg_tpl.prompt, 'template'):
                    tmpl = msg_tpl.prompt.template
                    if isinstance(tmpl, str) and len(tmpl) > min_len:
                        return tmpl
        # List/tuple of messages — find first SystemMessage
        if isinstance(val, (list, tuple)):
            for item in val:
                if not hasattr(item, 'content') or not isinstance(item.content, str):
                    continue
                role = getattr(item, 'type', '') or getattr(item, 'role', '')
                if 'system' in str(role).lower():
                    return item.content
        return None

    def _extract_system_prompt(self, model_node: Any) -> Optional[str]:
        bound = getattr(model_node, 'bound', None) or model_node
        fn = getattr(bound, 'afunc', None) or getattr(bound, 'func', None) or bound

        if not callable(fn):
            return None

        # Unwrap: LangGraph calls update_wrapper on its partial, so __wrapped__ is the real fn
        fn = getattr(fn, '__wrapped__', fn)

        # Class-based callable (e.g. AgentNode() instance) — check instance attrs, skip dunders
        if not inspect.isfunction(fn) and not inspect.ismethod(fn) and callable(fn):
            for attr_name, val in vars(fn).items():
                if attr_name.startswith('__'):
                    continue
                result = self._val_to_prompt(val)
                if result:
                    return result

        # Bound method — check __self__ instance attrs, skip dunders
        self_obj = getattr(fn, '__self__', None)
        if self_obj is not None:
            for attr_name, val in vars(self_obj).items():
                if attr_name.startswith('__'):
                    continue
                result = self._val_to_prompt(val)
                if result:
                    return result

        # Closure vars (nonlocals + globals)
        try:
            cv = inspect.getclosurevars(fn)
            lookup = {**cv.globals, **cv.nonlocals}
        except Exception:
            lookup = {}

        # Check known variable names first — no length guard
        for name in _PROMPT_VAR_NAMES:
            val = lookup.get(name)
            if val is None:
                continue
            result = self._val_to_prompt(val, min_len=0)
            if result:
                return result

        # Scan all non-dunder values — catches non-standard names like _MODULE_PROMPT
        for key, val in lookup.items():
            if key.startswith('__'):
                continue
            result = self._val_to_prompt(val, min_len=15)
            if result:
                return result

        return None

    # ── Tool helpers ──────────────────────────────────────────────────────────

    def _convert_tool(self, tool: Any) -> dict:
        schema = {}
        args_schema = getattr(tool, 'args_schema', None)
        if args_schema:
            try:
                schema = args_schema.model_json_schema()
            except Exception:
                pass
        return {
            'name': getattr(tool, 'name', None),
            'description': getattr(tool, 'description', None),
            'parameters': schema,
        }

    def _extract_tool_definitions(self, tools_node: Any) -> list:
        bound = getattr(tools_node, 'bound', None) or tools_node
        for attr in ('tools_by_name', '_tools_by_name'):
            registry = getattr(bound, attr, None)
            if isinstance(registry, dict) and registry:
                return [self._convert_tool(t) for t in registry.values()]

        data = getattr(tools_node, 'data', None)
        for attr in ('tools_by_name', '_tools_by_name'):
            registry = getattr(data, attr, None)
            if isinstance(registry, dict) and registry:
                return [self._convert_tool(t) for t in registry.values()]

        tools_attr = getattr(bound, 'tools', None) or getattr(data, 'tools', None)
        if isinstance(tools_attr, (list, tuple)):
            return [self._convert_tool(t) for t in tools_attr if hasattr(t, 'name')]
        return []

    def _extract_bound_tools(self, nodes: dict) -> list:
        """Find tools bound via model.bind_tools() (no 'tools' node in graph)."""
        for name, node in nodes.items():
            if name in SKIP_NODES:
                continue
            bound = getattr(node, 'bound', None) or node
            fn_raw = getattr(bound, 'afunc', None) or getattr(bound, 'func', None) or bound
            fn = getattr(fn_raw, '__wrapped__', fn_raw)
            if not callable(fn):
                continue
            try:
                cv = inspect.getclosurevars(fn)
                lookup = {**cv.globals, **cv.nonlocals}
            except Exception:
                lookup = {}
            for val in lookup.values():
                kwargs = getattr(val, 'kwargs', {}) or {}
                raw_tools = kwargs.get('tools')
                if not isinstance(raw_tools, (list, tuple)) or not raw_tools:
                    continue
                converted = []
                for t in raw_tools:
                    if hasattr(t, 'name'):  # LangChain tool object
                        converted.append(self._convert_tool(t))
                    elif isinstance(t, dict) and t.get('type') == 'function':
                        # OpenAI function-call schema format
                        fn_spec = t.get('function', {})
                        converted.append({
                            'name': fn_spec.get('name'),
                            'description': fn_spec.get('description'),
                            'parameters': fn_spec.get('parameters', {}),
                        })
                if converted:
                    return converted
        return []

    # ── Main extraction ───────────────────────────────────────────────────────

    def _extract_from_node(self, node: Any,) -> Optional[str]:
        """Try to extract a system prompt from a single node, recursing into subgraphs."""
        if hasattr(node, 'nodes'):
            sub = LangGraphDetector(node)
            return sub().system_prompt or None

        # Check if the underlying callable IS a compiled subgraph
        bound = getattr(node, 'bound', None) or node
        fn = getattr(bound, 'afunc', None) or getattr(bound, 'func', None) or bound
        fn = getattr(fn, '__wrapped__', fn)
        if hasattr(fn, 'nodes'):
            sub = LangGraphDetector(fn)
            return sub().system_prompt or None

        return self._extract_system_prompt(node)

    def __call__(self) -> AgentProfile:
        result = AgentProfile(
            system_prompt    = '',
            framework        = 'langgraph',
            tool_definitions = [],
            warnings         = [],
        )

        try:
            nodes = dict(self.agent.nodes)
        except Exception:
            result.warnings.append('Could not read graph nodes')
            return result

        # ── System prompt ────────────────────────────────────────────────────
        # 1. Try MODEL_NODE_NAMES first
        model_node = next(
            (nodes[n] for n in self.MODEL_NODE_NAMES if n in nodes), None
        )
        if model_node is not None:
            prompt = self._extract_from_node(model_node)
            if prompt:
                logger.info("System prompt extracted from agent")
                result.system_prompt = prompt
            else:
                logger.error("System prompt not found in agent closure")
                result.warnings.append('System prompt not found in closure')
        else:
            # 2. Scan all non-skip, non-tools nodes
            for name, node in nodes.items():
                if name in SKIP_NODES or name == 'tools':
                    continue
                prompt = self._extract_from_node(node)
                if prompt:
                    result.system_prompt = prompt
                    break
            if not result.system_prompt:
                result.warnings.append('Model node not found')

        # ── Tool definitions ─────────────────────────────────────────────────
        tools_node = nodes.get('tools')
        if tools_node is not None:
            tools = self._extract_tool_definitions(tools_node)
            if tools:
                logger.info("Tools found: %d", len(tools))
                result.tool_definitions = tools
            else:
                logger.error("Tools node present but no tools extracted")
                result.warnings.append('No tools found')
        else:
            bound = self._extract_bound_tools(nodes)
            if bound:
                result.tool_definitions = bound

        return result