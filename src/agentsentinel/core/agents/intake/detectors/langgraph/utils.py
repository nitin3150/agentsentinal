import functools
from typing import Any

SKIP_NODES = {'__start__', '__end__'}

_PROMPT_VAR_NAMES = (
    'system_prompt', 'system_message', 'prompt', 'state_modifiers', 'state_modifier',
    'instructions', 'system', 'sys_prompt', 'SYSTEM', 'SYSTEM_PROMPT',
)


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
