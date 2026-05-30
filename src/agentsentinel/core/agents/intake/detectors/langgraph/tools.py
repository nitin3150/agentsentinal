import inspect
from typing import Any, Optional, Sequence

from .utils import SKIP_NODES, _unwrap_fn


def _find_tools_registry(obj: Any, depth: int = 4) -> Optional[dict]:
    """Recursively search for tools_by_name/_tools_by_name through wrapper chains."""
    if obj is None or depth == 0:
        return None
    for attr in ('tools_by_name', '_tools_by_name'):
        registry = getattr(obj, attr, None)
        if isinstance(registry, dict) and registry:
            return registry
    for wrap_attr in ('bound', 'func', 'afunc', 'data'):
        inner = getattr(obj, wrap_attr, None)
        if inner is not None and inner is not obj:
            result = _find_tools_registry(inner, depth - 1)
            if result:
                return result
    return None


def convert_tool(tool: Any) -> dict:
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


def tools_from_raw(raw_tools: Sequence) -> list:
    converted = []
    for t in raw_tools:
        if hasattr(t, 'name'):
            converted.append(convert_tool(t))
        elif isinstance(t, dict) and t.get('type') == 'function':
            fn_spec = t.get('function', {})
            converted.append({
                'name': fn_spec.get('name'),
                'description': fn_spec.get('description'),
                'parameters': fn_spec.get('parameters', {}),
            })
    return converted


def tools_from_kwargs_lookup(values) -> list:
    for val in values:
        kwargs = getattr(val, 'kwargs', {}) or {}
        raw_tools = kwargs.get('tools') if isinstance(kwargs, dict) else None
        if isinstance(raw_tools, (list, tuple)) and raw_tools:
            converted = tools_from_raw(raw_tools)
            if converted:
                return converted
        # RunnableSequence (prompt | model.bind_tools(...)) — scan each step
        steps = getattr(val, 'steps', None)
        if isinstance(steps, (list, tuple)):
            converted = tools_from_kwargs_lookup(steps)
            if converted:
                return converted
    return []


def extract_tool_definitions(tools_node: Any) -> list:
    registry = _find_tools_registry(tools_node)
    if registry:
        return [convert_tool(t) for t in registry.values()]

    bound = getattr(tools_node, 'bound', None) or tools_node
    data = getattr(tools_node, 'data', None)
    tools_attr = getattr(bound, 'tools', None) or (getattr(data, 'tools', None) if data else None)
    if isinstance(tools_attr, (list, tuple)):
        return [convert_tool(t) for t in tools_attr if hasattr(t, 'name')]
    return []


def extract_bound_tools(nodes: dict) -> list:
    """Find tools bound via model.bind_tools() when no ToolNode exists in graph."""
    for name, node in nodes.items():
        if name in SKIP_NODES:
            continue
        bound = getattr(node, 'bound', None) or node
        fn_raw = getattr(bound, 'afunc', None) or getattr(bound, 'func', None) or bound
        fn = _unwrap_fn(fn_raw)

        if callable(fn):
            try:
                import inspect as _inspect
                cv = _inspect.getclosurevars(fn)
                lookup = {**cv.globals, **cv.nonlocals}
            except Exception:
                lookup = {}
            converted = tools_from_kwargs_lookup(lookup.values())
            if converted:
                return converted

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
                converted = tools_from_raw(raw_tools)
                if converted:
                    return converted
    return []
