import inspect
from typing import Any, Optional

from .utils import _PROMPT_VAR_NAMES, _unwrap_fn


def val_to_prompt(val: Any, min_len: int = 0) -> Optional[str]:
    """Extract a prompt string from a single value."""
    if val is None:
        return None
    if isinstance(val, str) and len(val) > min_len:
        return val
    # BaseMessage (SystemMessage etc.) with .content
    content = getattr(val, 'content', None)
    if isinstance(content, str) and len(content) > min_len:
        return content
    # ChatPromptTemplate — .messages list of MessagePromptTemplate
    messages = getattr(val, 'messages', None)
    if isinstance(messages, (list, tuple)):
        for msg_tpl in messages:
            if hasattr(msg_tpl, 'prompt') and hasattr(msg_tpl.prompt, 'template'):
                tmpl = msg_tpl.prompt.template
                if isinstance(tmpl, str) and len(tmpl) > min_len:
                    return tmpl
    # PromptTemplate — .template directly (no .messages)
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
    # model.bind(system=...) kwargs
    kwargs = getattr(val, 'kwargs', None)
    if isinstance(kwargs, dict):
        for key in ('system', 'system_prompt', 'system_message'):
            sys_val = kwargs.get(key)
            if sys_val is not None:
                inner = val_to_prompt(sys_val, min_len=min_len)
                if inner:
                    return inner
    return None


def extract_system_prompt(model_node: Any) -> Optional[str]:
    bound = getattr(model_node, 'bound', None) or model_node

    # RunnableSequence (prompt | llm) — no __call__, check .steps before callable guard
    steps = getattr(bound, 'steps', None)
    if isinstance(steps, (list, tuple)) and steps:
        for step in steps:
            result = val_to_prompt(step, min_len=15)
            if result:
                return result
            sub_steps = getattr(step, 'steps', None)
            if isinstance(sub_steps, (list, tuple)):
                for sub_step in sub_steps:
                    result = val_to_prompt(sub_step, min_len=15)
                    if result:
                        return result

    fn = getattr(bound, 'afunc', None) or getattr(bound, 'func', None) or bound

    if not callable(fn):
        return None

    fn = _unwrap_fn(fn)

    # Class-based callable — instance attrs first, then class hierarchy
    if not inspect.isfunction(fn) and not inspect.ismethod(fn) and callable(fn):
        instance_keys: set = set()
        for attr_name, val in vars(fn).items():
            if attr_name.startswith('__'):
                continue
            instance_keys.add(attr_name)
            result = val_to_prompt(val)
            if result:
                return result
        klass_type = type(fn)
        for klass in getattr(klass_type, '__mro__', (klass_type,)):
            for attr_name, cls_val in vars(klass).items():
                if attr_name.startswith('__') or attr_name in instance_keys:
                    continue
                result = val_to_prompt(cls_val)
                if result:
                    return result

    # Bound method — check __self__ instance attrs
    self_obj = getattr(fn, '__self__', None)
    if self_obj is not None:
        for attr_name, val in vars(self_obj).items():
            if attr_name.startswith('__'):
                continue
            result = val_to_prompt(val)
            if result:
                return result

    # Closure vars (nonlocals + globals)
    try:
        cv = inspect.getclosurevars(fn)
        lookup = {**cv.globals, **cv.nonlocals}
    except Exception:
        lookup = {}

    for name in _PROMPT_VAR_NAMES:
        val = lookup.get(name)
        if val is None:
            continue
        result = val_to_prompt(val, min_len=0)
        if result:
            return result

    for key, val in lookup.items():
        if key.startswith('__'):
            continue
        result = val_to_prompt(val, min_len=15)
        if result:
            return result

    # Function default arguments — def node(state, system_prompt="...")
    if hasattr(fn, '__code__') and hasattr(fn, '__defaults__'):
        defaults = fn.__defaults__ or ()
        kwdefaults = getattr(fn, '__kwdefaults__', None) or {}
        varnames = fn.__code__.co_varnames[:fn.__code__.co_argcount]
        for name, val in zip(reversed(varnames), reversed(defaults)):
            if name in _PROMPT_VAR_NAMES:
                result = val_to_prompt(val, min_len=0)
                if result:
                    return result
        for name, val in kwdefaults.items():
            if name in _PROMPT_VAR_NAMES:
                result = val_to_prompt(val, min_len=0)
                if result:
                    return result

    return None
