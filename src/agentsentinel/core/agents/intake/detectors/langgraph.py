import inspect
import functools
from agentsentinel.models import AgentProfile
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

SKIP_NODES = {'__start__', '__end__'}


def _unwrap_fn(fn: Any) -> Any:
    """Recursively unwrap __wrapped__ and functools.partial layers."""
    seen: set = set()
    while True:
        fid = id(fn)
        if fid in seen:
            break
        seen.add(fid)
        if hasattr(fn, '__wrapped__'):
            fn = fn.__wrapped__
        elif isinstance(fn, functools.partial):
            fn = fn.func
        else:
            break
    return fn

# Known variable names that hold system prompts
_PROMPT_VAR_NAMES = (
    'system_prompt', 'system_message', 'prompt', 'state_modifiers', 'state_modifier',
    'instructions', 'system', 'sys_prompt', 'SYSTEM', 'SYSTEM_PROMPT',
)


class LangGraphDetector:
    MODEL_NODE_NAMES = (
        'model', 'agent', 'assistant', 'chatbot',
        'llm', 'call_model', 'generate', 'reasoner',
    )

    def __init__(self, agent):
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
        # PromptTemplate — has .template directly (no .messages)
        template = getattr(val, 'template', None)
        if isinstance(template, str) and len(template) > min_len:
            return template
        # List/tuple of messages — find first SystemMessage
        if isinstance(val, (list, tuple)):
            for item in val:
                if not hasattr(item, 'content') or not isinstance(item.content, str):
                    continue
                role = getattr(item, 'type', '') or getattr(item, 'role', '')
                if 'system' in str(role).lower():
                    return item.content
        # model.bind(system=...) or model.bind(system_prompt=...) stored in .kwargs
        kwargs = getattr(val, 'kwargs', None)
        if isinstance(kwargs, dict):
            for key in ('system', 'system_prompt', 'system_message'):
                sys_val = kwargs.get(key)
                if sys_val is not None:
                    inner = self._val_to_prompt(sys_val, min_len=min_len)
                    if inner:
                        return inner
        return None

    def _extract_system_prompt(self, model_node: Any) -> Optional[str]:
        bound = getattr(model_node, 'bound', None) or model_node

        # RunnableSequence (prompt | llm) — not __call__-able, check .steps before callable guard
        steps = getattr(bound, 'steps', None)
        if isinstance(steps, (list, tuple)) and steps:
            for step in steps:
                result = self._val_to_prompt(step, min_len=15)
                if result:
                    return result
                sub_steps = getattr(step, 'steps', None)
                if isinstance(sub_steps, (list, tuple)):
                    for sub_step in sub_steps:
                        result = self._val_to_prompt(sub_step, min_len=15)
                        if result:
                            return result

        fn = getattr(bound, 'afunc', None) or getattr(bound, 'func', None) or bound

        if not callable(fn):
            return None

        # Unwrap __wrapped__ and functools.partial layers recursively
        fn = _unwrap_fn(fn)

        # Class-based callable — scan instance attrs first, then class hierarchy
        if not inspect.isfunction(fn) and not inspect.ismethod(fn) and callable(fn):
            instance_keys: set = set()
            # 1. Instance __dict__ (self.system_prompt = "...")
            for attr_name, val in vars(fn).items():
                if attr_name.startswith('__'):
                    continue
                instance_keys.add(attr_name)
                result = self._val_to_prompt(val)
                if result:
                    return result
            # 2. Class body attrs not shadowed by instance (class Base: system_prompt = "...")
            for klass in type(fn).__mro__:
                for attr_name, cls_val in vars(klass).items():
                    if attr_name.startswith('__') or attr_name in instance_keys:
                        continue
                    result = self._val_to_prompt(cls_val)
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

        # Function default arguments — def node(state, system_prompt="...")
        if hasattr(fn, '__code__') and hasattr(fn, '__defaults__'):
            defaults = fn.__defaults__ or ()
            kwdefaults = getattr(fn, '__kwdefaults__', None) or {}
            varnames = fn.__code__.co_varnames[:fn.__code__.co_argcount]
            for name, val in zip(reversed(varnames), reversed(defaults)):
                if name in _PROMPT_VAR_NAMES:
                    result = self._val_to_prompt(val, min_len=0)
                    if result:
                        return result
            for name, val in kwdefaults.items():
                if name in _PROMPT_VAR_NAMES:
                    result = self._val_to_prompt(val, min_len=0)
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
            fn = _unwrap_fn(fn_raw)

            # Closure vars scan
            if callable(fn):
                try:
                    cv = inspect.getclosurevars(fn)
                    lookup = {**cv.globals, **cv.nonlocals}
                except Exception:
                    lookup = {}
                converted = self._tools_from_kwargs_lookup(lookup.values())
                if converted:
                    return converted

            # Instance attrs of class-based callables (self.llm = model.bind_tools(...))
            fn_obj = None
            if not inspect.isfunction(fn) and not inspect.ismethod(fn) and callable(fn):
                fn_obj = fn
            self_obj = getattr(fn, '__self__', None)
            if self_obj is not None:
                fn_obj = self_obj
            if fn_obj is not None:
                for attr_val in vars(fn_obj).values():
                    kwargs = getattr(attr_val, 'kwargs', None)
                    if not isinstance(kwargs, dict):
                        continue
                    raw_tools = kwargs.get('tools')
                    if not isinstance(raw_tools, (list, tuple)) or not raw_tools:
                        continue
                    converted = self._tools_from_raw(raw_tools)
                    if converted:
                        return converted
        return []

    def _tools_from_raw(self, raw_tools: list) -> list:
        converted = []
        for t in raw_tools:
            if hasattr(t, 'name'):
                converted.append(self._convert_tool(t))
            elif isinstance(t, dict) and t.get('type') == 'function':
                fn_spec = t.get('function', {})
                converted.append({
                    'name': fn_spec.get('name'),
                    'description': fn_spec.get('description'),
                    'parameters': fn_spec.get('parameters', {}),
                })
        return converted

    def _tools_from_kwargs_lookup(self, values) -> list:
        for val in values:
            kwargs = getattr(val, 'kwargs', {}) or {}
            raw_tools = kwargs.get('tools') if isinstance(kwargs, dict) else None
            if not isinstance(raw_tools, (list, tuple)) or not raw_tools:
                continue
            converted = self._tools_from_raw(raw_tools)
            if converted:
                return converted
        return []

    # ── Main extraction ───────────────────────────────────────────────────────

    def _extract_from_context_schema(self) -> Optional[str]:
        """Extract system prompt from the LangGraph context_schema (e.g., Context dataclass)."""
        context_schema = getattr(self.agent, 'context_schema', None)
        if context_schema is None:
            return None
        
        # Handle dataclass with field defaults
        if hasattr(context_schema, '__dataclass_fields__'):
            fields_dict = getattr(context_schema, '__dataclass_fields__', {})
            system_prompt_field = fields_dict.get('system_prompt')
            if system_prompt_field:
                default = system_prompt_field.default
                if default and isinstance(default, str) and len(default) > 15:
                    return default
                # Check default_factory for callable defaults
                if system_prompt_field.default_factory != type(system_prompt_field.default_factory):
                    try:
                        val = system_prompt_field.default_factory()
                        if isinstance(val, str) and len(val) > 15:
                            return val
                    except Exception:
                        pass
        
        return None

    def _extract_from_context_instance(self) -> Optional[str]:
        """Extract system prompt from an instantiated context object."""
        context = getattr(self.agent, 'config', None)
        if context is None:
            try:
                context_schema = getattr(self.agent, 'context_schema', None)
                if context_schema:
                    context = context_schema()
            except Exception:
                pass
        
        if context and hasattr(context, 'system_prompt'):
            val = getattr(context, 'system_prompt')
            result = self._val_to_prompt(val, min_len=0)
            if result:
                return result
        
        return None

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
                logger.debug("System prompt not found in model node closure, checking context schema")
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
        
        # 3. Fallback: Extract from context_schema (e.g., Context dataclass)
        if not result.system_prompt:
            prompt = self._extract_from_context_schema()
            if prompt:
                logger.info("System prompt extracted from context schema")
                result.system_prompt = prompt
            else:
                # Try instantiated context
                prompt = self._extract_from_context_instance()
                if prompt:
                    logger.info("System prompt extracted from context instance")
                    result.system_prompt = prompt

        # ── Tool definitions ─────────────────────────────────────────────────
        # Scan ALL non-skip nodes for ToolNode pattern — not just 'tools' name
        tools_node = None
        for _name, _node in nodes.items():
            if _name in SKIP_NODES:
                continue
            _candidate = getattr(_node, 'bound', None) or _node
            _data = getattr(_node, 'data', None)
            for _attr in ('tools_by_name', '_tools_by_name'):
                if isinstance(getattr(_candidate, _attr, None), dict):
                    tools_node = _node
                    break
                if _data is not None and isinstance(getattr(_data, _attr, None), dict):
                    tools_node = _node
                    break
            if tools_node is not None:
                break

        if tools_node is not None:
            tools = self._extract_tool_definitions(tools_node)
            if tools:
                logger.info("Tools found: %d", len(tools))
                result.tool_definitions = tools
            else:
                logger.error("ToolNode present but no tools extracted")
                result.warnings.append('No tools found')
        else:
            bound_tools = self._extract_bound_tools(nodes)
            if bound_tools:
                result.tool_definitions = bound_tools

        # Subgraph fallback — if still no tools, recurse into subgraph nodes
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

        return result