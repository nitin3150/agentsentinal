# Intake Agent Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all known bugs and coverage gaps in the LangGraph intake detector, then add file-path intake via dynamic import + static AST fallback.

**Architecture:** Nine sequential tasks — bugs first, then expanded extraction coverage, then `AgentIntake` logic fixes, then the new `FilePathDetector`. Each task ships passing tests before the next begins.

**Tech Stack:** Python 3.11+, LangGraph, LangChain Core, Pydantic v2, `ast`, `importlib.util`, `inspect`, `functools`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/agentsentinel/core/agents/intake/detectors/langgraph.py` | Modify | All LangGraph runtime extraction |
| `src/agentsentinel/core/agents/intake/detectors/filepath.py` | **Create** | File-path intake: AST scan + dynamic import + static fallback |
| `src/agentsentinel/core/agents/intake/detectors/__init__.py` | Modify | Export `FilePathDetector` |
| `src/agentsentinel/core/agents/intake/agent_intake.py` | Modify | Register `FilePathDetector`, fix tool merge + source_object |
| `src/agentsentinel/models/agent.py` | Modify | Add `source_code: Optional[str]` to `AgentProfile` |
| `tests/test_langgraph_detector.py` | Modify | New tests for each gap fixed; update stale assertions |
| `tests/test_filepath_detector.py` | **Create** | Full coverage of `FilePathDetector` |

---

## Task 1: Critical Bug Fixes

**Files:**
- Modify: `src/agentsentinel/core/agents/intake/detectors/langgraph.py:11-14`
- Modify: `src/agentsentinel/core/agents/intake/agent_intake.py:20`
- Modify: `src/agentsentinel/core/agents/intake/detectors/langgraph.py:23-29`
- Modify: `tests/test_langgraph_detector.py`

### Bug A: `state_modifier` missing from `_PROMPT_VAR_NAMES`

`create_react_agent(model, tools, state_modifier="...")` stores the prompt as a closure variable named `state_modifier` (singular). Current list has `state_modifiers` (plural) — miss.

- [ ] **Step 1: Write failing test**

Add to `tests/test_langgraph_detector.py` inside `TestPromptFromVariableName`:

```python
def test_state_modifier_var(self):
    """create_react_agent uses state_modifier (singular), not state_modifiers."""
    def make():
        state_modifier = "You are a react agent assistant."
        def node(state):
            _ = state_modifier
            return state
        return node
    result = LangGraphDetector(_compile_graph(make()))()
    assert result.system_prompt == "You are a react agent assistant."
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /Users/nitingoyal/Developer/agentsentinal
pytest tests/test_langgraph_detector.py::TestPromptFromVariableName::test_state_modifier_var -v
```

Expected: FAIL — prompt not extracted.

- [ ] **Step 3: Fix `_PROMPT_VAR_NAMES` — add `'state_modifier'` (singular)**

In `src/agentsentinel/core/agents/intake/detectors/langgraph.py`, change lines 11-14:

```python
_PROMPT_VAR_NAMES = (
    'system_prompt', 'system_message', 'prompt', 'state_modifier', 'state_modifiers',
    'instructions', 'system', 'sys_prompt', 'SYSTEM', 'SYSTEM_PROMPT',
)
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
pytest tests/test_langgraph_detector.py::TestPromptFromVariableName::test_state_modifier_var -v
```

Expected: PASS

---

### Bug B: `.strip(".")` corrupts valid prompts

`agent_intake.py:20` strips all trailing periods. `"Do not share PII."` becomes `"Do not share PII"`. Remove the `.strip(".")`.

- [ ] **Step 5: Update stale test that asserts period stripping**

In `tests/test_langgraph_detector.py`, find `test_prompt_stripped_of_newlines_and_dots` and replace it:

```python
def test_prompt_stripped_of_leading_trailing_newlines(self):
    """AgentIntake strips leading/trailing newlines but preserves periods."""
    def make():
        system_prompt = "\nYou are helpful. Be concise.\n"
        def node(state):
            _ = system_prompt
            return state
        return node
    app = _compile_graph(make())
    profile = AgentIntake().extract_profile(app)
    assert not profile.system_prompt.startswith("\n")
    assert not profile.system_prompt.endswith("\n")
    assert profile.system_prompt.endswith(".")  # period preserved
```

- [ ] **Step 6: Fix `agent_intake.py` — remove `.strip(".")`**

In `src/agentsentinel/core/agents/intake/agent_intake.py`, change line 20:

```python
result.system_prompt = result.system_prompt.strip("\n")
```

- [ ] **Step 7: Run updated test**

```bash
pytest tests/test_langgraph_detector.py::TestAgentIntake::test_prompt_stripped_of_leading_trailing_newlines -v
```

Expected: PASS

---

### Bug C: Fragile Pregel import path

`from langgraph.pregel.main import Pregel` is internal and moved across versions. Fallback `hasattr(agent, 'nodes')` is too permissive (matches dicts, networkx graphs).

- [ ] **Step 8: Write test that confirms plain dict is rejected**

This test already exists (`test_false_for_plain_dict`). Run it to confirm baseline:

```bash
pytest tests/test_langgraph_detector.py::TestCanHandle -v
```

Expected: all pass.

- [ ] **Step 9: Fix `can_handle` with multi-path Pregel import**

In `src/agentsentinel/core/agents/intake/detectors/langgraph.py`, replace the `can_handle` method:

```python
def can_handle(self) -> bool:
    pregel_class = None
    for import_path in (
        'langgraph.pregel',
        'langgraph.pregel.main',
    ):
        try:
            import importlib
            mod = importlib.import_module(import_path)
            pregel_class = getattr(mod, 'Pregel', None)
            if pregel_class is not None:
                break
        except ImportError:
            continue

    if pregel_class is not None:
        logger.info("Framework: LangGraph")
        return isinstance(self.agent, pregel_class)

    # No langgraph installed — reject everything
    return False
```

- [ ] **Step 10: Run all can_handle tests**

```bash
pytest tests/test_langgraph_detector.py::TestCanHandle -v
```

Expected: all pass.

- [ ] **Step 11: Commit**

```bash
git add src/agentsentinel/core/agents/intake/detectors/langgraph.py \
        src/agentsentinel/core/agents/intake/agent_intake.py \
        tests/test_langgraph_detector.py
git commit -m "fix: state_modifier spelling, strip-period corruption, fragile Pregel import"
```

---

## Task 2: Extend `_val_to_prompt` — More Value Types

Add extraction for: `PromptTemplate.template`, `model.bind()` system kwargs, function default arguments, Pydantic v2 `model_fields`.

**Files:**
- Modify: `src/agentsentinel/core/agents/intake/detectors/langgraph.py`
- Modify: `tests/test_langgraph_detector.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_langgraph_detector.py` inside `TestPromptValueTypes`:

```python
def test_prompt_template(self):
    """Plain PromptTemplate has .template — not .messages."""
    from langchain_core.prompts import PromptTemplate
    def make():
        prompt = PromptTemplate.from_template(
            "You are a PromptTemplate assistant answering {question}."
        )
        def node(state):
            _ = prompt
            return state
        return node
    result = LangGraphDetector(_compile_graph(make()))()
    assert "You are a PromptTemplate assistant" in result.system_prompt

def test_model_bind_system_kwarg(self):
    """System prompt stored in model.bind(system=...) kwargs dict."""
    class FakeBoundModel:
        kwargs = {"system": "You are a bound-model assistant."}
        def __call__(self, state):
            return state

    def make():
        llm = FakeBoundModel()
        def node(state):
            _ = llm
            return state
        return node
    result = LangGraphDetector(_compile_graph(make()))()
    assert result.system_prompt == "You are a bound-model assistant."

def test_function_default_arg_prompt(self):
    """System prompt as a default argument value on the node function."""
    def node(state, system_prompt="You are a default-arg assistant."):
        return state
    result = LangGraphDetector(_compile_graph(node))()
    assert result.system_prompt == "You are a default-arg assistant."

def test_pydantic_state_model_default(self):
    """System prompt as Pydantic model field default (not dataclass)."""
    from pydantic import BaseModel as PydanticBaseModel

    class Config(PydanticBaseModel):
        system_prompt: str = "You are a Pydantic-state assistant."

    class FakeGraph:
        pass

    agent = _compile_graph(lambda state: state)
    agent.context_schema = Config
    result = LangGraphDetector(agent)()
    assert result.system_prompt == "You are a Pydantic-state assistant."
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
pytest tests/test_langgraph_detector.py::TestPromptValueTypes::test_prompt_template \
       tests/test_langgraph_detector.py::TestPromptValueTypes::test_model_bind_system_kwarg \
       tests/test_langgraph_detector.py::TestPromptValueTypes::test_function_default_arg_prompt \
       tests/test_langgraph_detector.py::TestPromptValueTypes::test_pydantic_state_model_default -v
```

Expected: all FAIL.

- [ ] **Step 3: Extend `_val_to_prompt` for `PromptTemplate.template` and `model.bind` kwargs**

In `src/agentsentinel/core/agents/intake/detectors/langgraph.py`, update `_val_to_prompt` — insert after the `messages` block (after line 59, before `return None`):

```python
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
    # PromptTemplate — has .template directly (no .messages)
    template = getattr(val, 'template', None)
    if isinstance(template, str) and len(template) > min_len:
        return template
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
    # model.bind(system=...) or model.bind(messages=[SystemMessage(...)])
    kwargs = getattr(val, 'kwargs', None)
    if isinstance(kwargs, dict):
        for key in ('system', 'system_prompt', 'system_message'):
            sys_val = kwargs.get(key)
            if sys_val:
                result = self._val_to_prompt(sys_val, min_len=min_len)
                if result:
                    return result
    return None
```

- [ ] **Step 4: Add function default arg extraction to `_extract_system_prompt`**

In `_extract_system_prompt`, after the closure scan block (after line 113, before `return None`), insert:

```python
# Function default arguments (e.g. def node(state, system_prompt="You are..."))
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
```

- [ ] **Step 5: Add Pydantic `model_fields` support to `_extract_from_context_schema`**

In `_extract_from_context_schema`, after the dataclass block (after line 211, before `return None`), insert:

```python
# Pydantic v2 model with model_fields
if hasattr(context_schema, 'model_fields'):
    try:
        from pydantic_core import PydanticUndefinedType
        undefined_types = (PydanticUndefinedType,)
    except ImportError:
        undefined_types = ()
    field = context_schema.model_fields.get('system_prompt')
    if field is not None:
        default = field.default
        if default is not None and not isinstance(default, undefined_types):
            if isinstance(default, str) and len(default) > 15:
                return default
```

- [ ] **Step 6: Run all new tests**

```bash
pytest tests/test_langgraph_detector.py::TestPromptValueTypes -v
```

Expected: all PASS.

- [ ] **Step 7: Run full suite to check no regressions**

```bash
pytest tests/test_langgraph_detector.py -v
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/agentsentinel/core/agents/intake/detectors/langgraph.py \
        tests/test_langgraph_detector.py
git commit -m "feat: extract prompts from PromptTemplate, bind kwargs, fn defaults, Pydantic fields"
```

---

## Task 3: Extend Extraction Paths — `partial`, Inherited Attrs, `RunnableSequence`

**Files:**
- Modify: `src/agentsentinel/core/agents/intake/detectors/langgraph.py`
- Modify: `tests/test_langgraph_detector.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_langgraph_detector.py`:

```python
class TestExtractionPaths:
    def test_functools_partial_unwrap(self):
        """Double-wrapped functools.partial — both layers unwrapped."""
        import functools

        def base_node(state, config=None):
            return state

        def make():
            system_prompt = "You are a partial-wrapped assistant."
            def inner(state):
                _ = system_prompt
                return state
            return functools.partial(functools.partial(inner))

        result = LangGraphDetector(_compile_graph(make()))()
        assert result.system_prompt == "You are a partial-wrapped assistant."

    def test_inherited_class_attr_prompt(self):
        """System prompt defined on parent class body, not instance __dict__."""
        class BaseAgent:
            system_prompt = "You are an inherited-attr assistant."

        class ConcreteAgent(BaseAgent):
            def __call__(self, state):
                return state

        result = LangGraphDetector(_compile_graph(ConcreteAgent()))()
        assert result.system_prompt == "You are an inherited-attr assistant."

    def test_runnable_sequence_prompt_in_steps(self):
        """Node is prompt | llm chain (RunnableSequence) — prompt in .steps[0]."""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnableLambda

        chain = ChatPromptTemplate.from_messages([
            ("system", "You are a RunnableSequence assistant."),
            ("human", "{input}"),
        ]) | RunnableLambda(lambda x: x)

        g = StateGraph(State)
        g.add_node("agent", chain)
        g.set_entry_point("agent")
        g.add_edge("agent", END)
        app = g.compile()

        result = LangGraphDetector(app)()
        assert result.system_prompt == "You are a RunnableSequence assistant."
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_langgraph_detector.py::TestExtractionPaths -v
```

Expected: all FAIL.

- [ ] **Step 3: Add `_unwrap_fn` helper and use it in `_extract_system_prompt`**

In `src/agentsentinel/core/agents/intake/detectors/langgraph.py`, add import at top:

```python
import functools
```

Add `_unwrap_fn` as a module-level function (before `LangGraphDetector` class):

```python
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
```

In `_extract_system_prompt`, replace the single `fn = getattr(fn, '__wrapped__', fn)` line with:

```python
fn = _unwrap_fn(fn)
```

- [ ] **Step 4: Add inherited class attr scan**

In `_extract_system_prompt`, in the class-based callable block (currently `for attr_name, val in vars(fn).items()`), replace it with:

```python
if not inspect.isfunction(fn) and not inspect.ismethod(fn) and callable(fn):
    # Check instance attrs AND class hierarchy attrs
    for klass in type(fn).__mro__:
        for attr_name, val in vars(klass).items():
            if attr_name.startswith('__'):
                continue
            actual_val = getattr(fn, attr_name, val)  # instance shadows class
            result = self._val_to_prompt(actual_val)
            if result:
                return result
```

- [ ] **Step 5: Add `RunnableSequence.steps` traversal**

In `_extract_system_prompt`, after the closure scan (after function default args block added in Task 2), before `return None`:

```python
# RunnableSequence — node is `prompt | llm` chain, prompt in .steps[0]
steps = getattr(bound, 'steps', None) or getattr(fn, 'steps', None)
if isinstance(steps, (list, tuple)):
    for step in steps:
        result = self._val_to_prompt(step, min_len=15)
        if result:
            return result
        # One level of nesting (e.g. RunnableSequence inside RunnableSequence)
        sub_steps = getattr(step, 'steps', None)
        if isinstance(sub_steps, (list, tuple)):
            for sub_step in sub_steps:
                result = self._val_to_prompt(sub_step, min_len=15)
                if result:
                    return result
```

- [ ] **Step 6: Run all extraction path tests**

```bash
pytest tests/test_langgraph_detector.py::TestExtractionPaths -v
```

Expected: all PASS.

- [ ] **Step 7: Run full suite**

```bash
pytest tests/test_langgraph_detector.py -v
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/agentsentinel/core/agents/intake/detectors/langgraph.py \
        tests/test_langgraph_detector.py
git commit -m "feat: unwrap functools.partial, scan inherited class attrs, traverse RunnableSequence.steps"
```

---

## Task 4: Expand Node Name Coverage

**Files:**
- Modify: `src/agentsentinel/core/agents/intake/detectors/langgraph.py:18`
- Modify: `tests/test_langgraph_detector.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_langgraph_detector.py` inside `TestNodeNameRouting`:

```python
def test_assistant_node_name(self):
    def make():
        system_prompt = "Agent via assistant node."
        def node(state):
            _ = system_prompt
            return state
        return node
    result = LangGraphDetector(_compile_graph(make(), node_name="assistant"))()
    assert result.system_prompt == "Agent via assistant node."

def test_llm_node_name(self):
    def make():
        system_prompt = "Agent via llm node."
        def node(state):
            _ = system_prompt
            return state
        return node
    result = LangGraphDetector(_compile_graph(make(), node_name="llm"))()
    assert result.system_prompt == "Agent via llm node."

def test_call_model_node_name(self):
    def make():
        system_prompt = "Agent via call_model node."
        def node(state):
            _ = system_prompt
            return state
        return node
    result = LangGraphDetector(_compile_graph(make(), node_name="call_model"))()
    assert result.system_prompt == "Agent via call_model node."
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_langgraph_detector.py::TestNodeNameRouting::test_assistant_node_name \
       tests/test_langgraph_detector.py::TestNodeNameRouting::test_llm_node_name \
       tests/test_langgraph_detector.py::TestNodeNameRouting::test_call_model_node_name -v
```

Expected: FAIL — falls through to fallback scan (which DOES find them), so these may already pass. If they pass, skip to Step 4.

- [ ] **Step 3: Expand `MODEL_NODE_NAMES`**

In `src/agentsentinel/core/agents/intake/detectors/langgraph.py`, change line 18:

```python
MODEL_NODE_NAMES = (
    'model', 'agent', 'assistant', 'chatbot',
    'llm', 'call_model', 'generate', 'reasoner',
)
```

- [ ] **Step 4: Run full node routing tests**

```bash
pytest tests/test_langgraph_detector.py::TestNodeNameRouting -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentsentinel/core/agents/intake/detectors/langgraph.py \
        tests/test_langgraph_detector.py
git commit -m "feat: expand MODEL_NODE_NAMES to cover assistant/llm/call_model/chatbot/generate/reasoner"
```

---

## Task 5: Tool Extraction Coverage Gaps

Fix: `ToolNode` not named `'tools'`, tools in `RunnableSequence`, `bind_tools` on instance attributes, tools in subgraphs.

**Files:**
- Modify: `src/agentsentinel/core/agents/intake/detectors/langgraph.py`
- Modify: `tests/test_langgraph_detector.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_langgraph_detector.py`:

```python
class TestToolExtractionGaps:
    def test_tool_node_with_non_tools_name(self):
        """ToolNode named 'execute_tools' instead of 'tools' — still extracted."""
        from langchain_core.tools import tool
        from langgraph.prebuilt import ToolNode

        @tool
        def search(query: str) -> str:
            """Search the web for query."""
            return query

        g = StateGraph(State)
        g.add_node("agent", lambda state: state)
        g.add_node("execute_tools", ToolNode([search]))
        g.set_entry_point("agent")
        g.add_edge("agent", "execute_tools")
        g.add_edge("execute_tools", END)
        app = g.compile()

        result = LangGraphDetector(app)()
        assert len(result.tool_definitions) == 1
        assert result.tool_definitions[0]["name"] == "search"

    def test_bind_tools_on_class_instance_attr(self):
        """model.bind_tools(tools) stored as instance attr — not closure var."""
        from langchain_core.tools import tool

        @tool
        def lookup(id: str) -> str:
            """Look up record by id."""
            return id

        class AgentNode:
            def __init__(self, tools):
                class FakeKwargs:
                    kwargs = {"tools": tools}
                self.llm = FakeKwargs()

            def __call__(self, state):
                return state

        result = LangGraphDetector(_compile_graph(AgentNode([lookup])))()
        assert len(result.tool_definitions) == 1
        assert result.tool_definitions[0]["name"] == "lookup"

    def test_tools_extracted_from_subgraph(self):
        """Inner subgraph has a 'tools' node — outer graph extracts them."""
        from langchain_core.tools import tool
        from langgraph.prebuilt import ToolNode

        @tool
        def inner_tool(x: int) -> int:
            """Inner subgraph tool."""
            return x

        inner_g = StateGraph(State)
        inner_g.add_node("agent", lambda state: state)
        inner_g.add_node("tools", ToolNode([inner_tool]))
        inner_g.set_entry_point("agent")
        inner_g.add_edge("agent", "tools")
        inner_g.add_edge("tools", END)
        inner_app = inner_g.compile()

        outer_g = StateGraph(State)
        outer_g.add_node("subagent", inner_app)
        outer_g.set_entry_point("subagent")
        outer_g.add_edge("subagent", END)
        outer_app = outer_g.compile()

        result = LangGraphDetector(outer_app)()
        assert len(result.tool_definitions) == 1
        assert result.tool_definitions[0]["name"] == "inner_tool"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_langgraph_detector.py::TestToolExtractionGaps -v
```

Expected: all FAIL.

- [ ] **Step 3: Scan all nodes for `ToolNode` pattern in `__call__`**

In `src/agentsentinel/core/agents/intake/detectors/langgraph.py`, replace the tool extraction block in `__call__` (currently lines 302-314):

```python
# ── Tool definitions ─────────────────────────────────────────────────
# Scan all non-skip nodes for ToolNode (not just nodes named 'tools')
tools_node = None
for name, node in nodes.items():
    if name in SKIP_NODES:
        continue
    candidate = getattr(node, 'bound', None) or node
    for attr in ('tools_by_name', '_tools_by_name'):
        if isinstance(getattr(candidate, attr, None), dict):
            tools_node = node
            break
    # Also check .data attribute (some LangGraph versions)
    data = getattr(candidate, 'data', None)
    if data is not None:
        for attr in ('tools_by_name', '_tools_by_name'):
            if isinstance(getattr(data, attr, None), dict):
                tools_node = node
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
    # Try bind_tools() pattern (tools in model.bind_tools closure or instance)
    bound_tools = self._extract_bound_tools(nodes)
    if bound_tools:
        result.tool_definitions = bound_tools

# Subgraph fallback — if still no tools, recurse into subgraph nodes
if not result.tool_definitions:
    for name, node in nodes.items():
        if name in SKIP_NODES:
            continue
        bound = getattr(node, 'bound', None) or node
        fn = getattr(bound, 'afunc', None) or getattr(bound, 'func', None) or bound
        fn = _unwrap_fn(fn)
        if hasattr(fn, 'nodes'):
            sub = LangGraphDetector(fn)
            sub_result = sub()
            if sub_result.tool_definitions:
                result.tool_definitions = sub_result.tool_definitions
                break
```

- [ ] **Step 4: Extend `_extract_bound_tools` to check instance attributes**

In `src/agentsentinel/core/agents/intake/detectors/langgraph.py`, in `_extract_bound_tools`, add after the closure scan block (after `if converted: return converted`):

```python
# Also check instance attributes of class-based callable nodes
fn_obj = None
if not inspect.isfunction(fn) and not inspect.ismethod(fn) and callable(fn):
    fn_obj = fn
self_obj = getattr(fn, '__self__', None)
if self_obj is not None:
    fn_obj = self_obj

if fn_obj is not None:
    for attr_name, attr_val in vars(fn_obj).items():
        if attr_name.startswith('__'):
            continue
        kwargs = getattr(attr_val, 'kwargs', {}) or {}
        raw_tools = kwargs.get('tools') if isinstance(kwargs, dict) else None
        if not isinstance(raw_tools, (list, tuple)) or not raw_tools:
            continue
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
        if converted:
            return converted
```

- [ ] **Step 5: Run all tool extraction gap tests**

```bash
pytest tests/test_langgraph_detector.py::TestToolExtractionGaps -v
```

Expected: all PASS.

- [ ] **Step 6: Run full suite**

```bash
pytest tests/test_langgraph_detector.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentsentinel/core/agents/intake/detectors/langgraph.py \
        tests/test_langgraph_detector.py
git commit -m "feat: extract tools from any ToolNode name, instance bind_tools attrs, subgraph tools"
```

---

## Task 6: `AgentIntake` Logic Fixes

Fix: tool merge vs replace, `source_object` never set, add `source_code` to `AgentProfile`.

**Files:**
- Modify: `src/agentsentinel/core/agents/intake/agent_intake.py`
- Modify: `src/agentsentinel/models/agent.py`
- Modify: `tests/test_langgraph_detector.py`

- [ ] **Step 1: Add `source_code` field to `AgentProfile`**

In `src/agentsentinel/models/agent.py`, add to `AgentProfile`:

```python
class AgentProfile(BaseModel):
    domain: str = ""
    system_prompt: str = ""
    tool_definitions: List[dict] = Field(default_factory=list)
    framework: Any = "unknown"
    warnings: list[str] = Field(default_factory=list)
    source_object: Optional[Any] = None
    source_code: Optional[str] = None   # populated by FilePathDetector

    model_config = {"arbitrary_types_allowed": True}
```

- [ ] **Step 2: Write failing tests for tool merge and source_object**

Add to `tests/test_langgraph_detector.py` inside `TestAgentIntake`:

```python
def test_user_tools_merge_with_detected(self):
    """User-provided tools merge with detected — user wins on name collision."""
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode

    @tool
    def detected_tool(x: int) -> int:
        """Auto-detected tool."""
        return x

    def node(state): return state
    app = _compile_graph(node, tools_node=ToolNode([detected_tool]))

    # User provides a different tool AND an override for the detected one
    override = AgentProfile(tool_definitions=[
        {"name": "user_only_tool", "description": "only from user"},
        {"name": "detected_tool", "description": "user override description"},
    ])
    profile = AgentIntake().extract_profile(app, agent_profile=override)
    names = {t["name"] for t in profile.tool_definitions}
    # Both detected and user tools present
    assert "detected_tool" in names
    assert "user_only_tool" in names
    # User description wins for collision
    detected = next(t for t in profile.tool_definitions if t["name"] == "detected_tool")
    assert detected["description"] == "user override description"

def test_source_object_populated(self):
    """source_object on AgentProfile must reference the live agent."""
    def node(state): return state
    app = _compile_graph(node)
    profile = AgentIntake().extract_profile(app)
    assert profile.source_object is app
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
pytest tests/test_langgraph_detector.py::TestAgentIntake::test_user_tools_merge_with_detected \
       tests/test_langgraph_detector.py::TestAgentIntake::test_source_object_populated -v
```

Expected: FAIL.

- [ ] **Step 4: Fix `LangGraphDetector.__call__` to set `source_object`**

In `src/agentsentinel/core/agents/intake/detectors/langgraph.py`, at the end of `__call__` before `return result`:

```python
result.source_object = self.agent
return result
```

- [ ] **Step 5: Fix `agent_intake.py` tool merge logic**

In `src/agentsentinel/core/agents/intake/agent_intake.py`, replace lines 32-33:

```python
if agent_profile and agent_profile.tool_definitions:
    user_by_name = {t.get('name'): t for t in agent_profile.tool_definitions}
    detected_by_name = {t.get('name'): t for t in result.tool_definitions}
    # User overrides detected on name collision; both sets merged
    merged = {**detected_by_name, **user_by_name}
    result.tool_definitions = list(merged.values())
```

- [ ] **Step 6: Also fix prompt mismatch log — downgrade from error to warning**

In `src/agentsentinel/core/agents/intake/agent_intake.py`, change line 23 from `logger.error` to `logger.warning`:

```python
logger.warning(
    "System prompt mismatch: detected prompt differs from user-provided prompt.\n"
    "Detected : %s\nProvided : %s (user-provided wins)",
    result.system_prompt,
    agent_profile.system_prompt,
)
```

- [ ] **Step 7: Update existing `test_user_tools_override_detected` — it tests old replace behavior**

In `tests/test_langgraph_detector.py`, replace `test_user_tools_override_detected`:

```python
def test_user_tools_replace_when_no_detected(self):
    """When no tools auto-detected, user-supplied tools are used as-is."""
    def node(state): return state
    app = _compile_graph(node)  # no ToolNode
    override = AgentProfile(tool_definitions=[{"name": "user_tool", "description": "override"}])
    profile = AgentIntake().extract_profile(app, agent_profile=override)
    assert len(profile.tool_definitions) == 1
    assert profile.tool_definitions[0]["name"] == "user_tool"
```

- [ ] **Step 8: Run all AgentIntake tests**

```bash
pytest tests/test_langgraph_detector.py::TestAgentIntake -v
```

Expected: all PASS.

- [ ] **Step 9: Run full suite**

```bash
pytest tests/test_langgraph_detector.py -v
```

Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add src/agentsentinel/core/agents/intake/agent_intake.py \
        src/agentsentinel/models/agent.py \
        src/agentsentinel/core/agents/intake/detectors/langgraph.py \
        tests/test_langgraph_detector.py
git commit -m "fix: merge user+detected tools, populate source_object, add source_code field to AgentProfile"
```

---

## Task 7: `FilePathDetector` — AST Scanner + Dynamic Import

**Files:**
- Create: `src/agentsentinel/core/agents/intake/detectors/filepath.py`
- Create: `tests/test_filepath_detector.py`

- [ ] **Step 1: Create test file with failing tests for `can_handle` and dynamic import path**

Create `tests/test_filepath_detector.py`:

```python
"""
Tests for FilePathDetector.
No LLM API calls. Uses tmp_path fixture to write real .py files.
"""
import textwrap
import pytest
from pathlib import Path
from agentsentinel.core.agents.intake.detectors.filepath import FilePathDetector


def write_agent(tmp_path: Path, source: str) -> Path:
    p = tmp_path / "agent.py"
    p.write_text(textwrap.dedent(source))
    return p


class TestCanHandle:
    def test_true_for_existing_py_file(self, tmp_path):
        p = write_agent(tmp_path, "x = 1")
        assert FilePathDetector(str(p)).can_handle() is True

    def test_true_for_path_object(self, tmp_path):
        p = write_agent(tmp_path, "x = 1")
        assert FilePathDetector(p).can_handle() is True

    def test_false_for_missing_file(self, tmp_path):
        assert FilePathDetector(str(tmp_path / "missing.py")).can_handle() is False

    def test_false_for_live_object(self):
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, END
        class S(TypedDict):
            x: str
        g = StateGraph(S)
        g.add_node("n", lambda s: s)
        g.set_entry_point("n")
        g.add_edge("n", END)
        app = g.compile()
        assert FilePathDetector(app).can_handle() is False

    def test_false_for_non_py_file(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text("{}")
        assert FilePathDetector(str(p)).can_handle() is False


class TestFindCompileTarget:
    def test_finds_simple_assignment(self, tmp_path):
        p = write_agent(tmp_path, """
            from langgraph.graph import StateGraph
            builder = StateGraph(dict)
            graph = builder.compile()
        """)
        detector = FilePathDetector(str(p))
        assert detector._find_compile_target(p.read_text()) == "graph"

    def test_finds_alternate_var_name(self, tmp_path):
        p = write_agent(tmp_path, """
            app = builder.compile(checkpointer=None)
        """)
        assert FilePathDetector(str(p))._find_compile_target(p.read_text()) == "app"

    def test_returns_none_when_no_compile(self, tmp_path):
        p = write_agent(tmp_path, "x = 1 + 2")
        assert FilePathDetector(str(p))._find_compile_target(p.read_text()) is None

    def test_handles_syntax_error_gracefully(self, tmp_path):
        p = tmp_path / "bad.py"
        p.write_text("def broken(")
        assert FilePathDetector(str(p))._find_compile_target(p.read_text()) is None


class TestDynamicImport:
    def test_extracts_prompt_and_tools_via_dynamic_import(self, tmp_path):
        p = write_agent(tmp_path, """
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

SYSTEM_PROMPT = "You are a dynamic-import test assistant."

class State(TypedDict):
    message: str

def call_model(state):
    _ = SYSTEM_PROMPT
    return state

builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.set_entry_point("agent")
builder.add_edge("agent", END)
graph = builder.compile()
        """)
        from agentsentinel.core.agents.intake.agent_intake import AgentIntake
        profile = AgentIntake().extract_profile(str(p))
        assert profile.framework == "langgraph"
        assert "dynamic-import test assistant" in profile.system_prompt

    def test_source_code_populated_on_result(self, tmp_path):
        p = write_agent(tmp_path, """
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
class S(TypedDict):
    x: str
builder = StateGraph(S)
builder.add_node("n", lambda s: s)
builder.set_entry_point("n")
builder.add_edge("n", END)
graph = builder.compile()
        """)
        from agentsentinel.core.agents.intake.agent_intake import AgentIntake
        profile = AgentIntake().extract_profile(str(p))
        assert profile.source_code is not None
        assert "StateGraph" in profile.source_code
```

- [ ] **Step 2: Run tests to confirm they fail (FilePathDetector doesn't exist)**

```bash
pytest tests/test_filepath_detector.py -v
```

Expected: ImportError / all FAIL.

- [ ] **Step 3: Create `FilePathDetector`**

Create `src/agentsentinel/core/agents/intake/detectors/filepath.py`:

```python
import ast
import importlib.util
import logging
from pathlib import Path
from typing import Any, Optional

from agentsentinel.models import AgentProfile

logger = logging.getLogger(__name__)


class FilePathDetector:
    def __init__(self, agent: Any):
        self.path = agent

    def can_handle(self) -> bool:
        if not isinstance(self.path, (str, Path)):
            return False
        p = Path(self.path)
        return p.exists() and p.suffix == '.py'

    def _find_compile_target(self, source: str) -> Optional[str]:
        """AST scan: find variable name that receives .compile() call at module level."""
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
        """Dynamically import module and return the named variable."""
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
        from agentsentinel.core.agents.intake.detectors.langgraph import _PROMPT_VAR_NAMES

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

        # String literal assigned to known prompt var names
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

        # @tool decorated functions
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
```

- [ ] **Step 4: Run all `FilePathDetector` tests**

```bash
pytest tests/test_filepath_detector.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentsentinel/core/agents/intake/detectors/filepath.py \
        tests/test_filepath_detector.py
git commit -m "feat: add FilePathDetector with AST scan, dynamic import, static AST fallback"
```

---

## Task 8: Static AST Fallback Coverage

Ensure static extraction covers the cases dynamic import can't reach.

**Files:**
- Modify: `tests/test_filepath_detector.py`
- Modify: `src/agentsentinel/core/agents/intake/detectors/filepath.py`

- [ ] **Step 1: Write failing tests for static fallback**

Add to `tests/test_filepath_detector.py`:

```python
class TestStaticFallback:
    """Tests for _static_extract — runs when dynamic import fails."""

    def _static(self, source: str):
        import textwrap
        from agentsentinel.core.agents.intake.detectors.filepath import FilePathDetector
        detector = FilePathDetector.__new__(FilePathDetector)
        return detector._static_extract(textwrap.dedent(source))

    def test_extracts_system_prompt_string_literal(self):
        result = self._static("""
            system_prompt = "You are a static extraction assistant."
        """)
        assert result.system_prompt == "You are a static extraction assistant."

    def test_extracts_at_tool_decorated_function(self):
        result = self._static("""
            from langchain_core.tools import tool

            @tool
            def get_weather(city: str) -> str:
                \"\"\"Get current weather for a city.\"\"\"
                return city
        """)
        assert len(result.tool_definitions) == 1
        assert result.tool_definitions[0]["name"] == "get_weather"
        assert "weather" in result.tool_definitions[0]["description"]

    def test_detects_langgraph_framework(self):
        result = self._static("from langgraph.graph import StateGraph")
        assert result.framework == "langgraph"

    def test_unknown_framework_for_plain_file(self):
        result = self._static("x = 1")
        assert result.framework == "unknown"

    def test_warns_about_static_fallback(self):
        result = self._static("x = 1")
        assert any("Static AST" in w for w in result.warnings)

    def test_source_code_preserved(self):
        source = "system_prompt = 'hello'\n"
        from agentsentinel.core.agents.intake.detectors.filepath import FilePathDetector
        detector = FilePathDetector.__new__(FilePathDetector)
        result = detector._static_extract(source)
        assert result.source_code == source

    def test_syntax_error_returns_warning(self):
        result = self._static("def broken(")
        assert any("parse" in w.lower() for w in result.warnings)
        assert result.system_prompt == ""

    def test_compile_inside_function_falls_to_static(self, tmp_path):
        """Compile inside a function — not at module level — triggers static fallback."""
        p = write_agent(tmp_path, """
system_prompt = "You are an inside-function assistant."

def build():
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    class S(TypedDict):
        x: str
    g = StateGraph(S)
    g.add_node("n", lambda s: s)
    g.set_entry_point("n")
    g.add_edge("n", END)
    graph = g.compile()  # inside function — not found by AST scan
    return graph
        """)
        from agentsentinel.core.agents.intake.detectors.filepath import FilePathDetector
        result = FilePathDetector(str(p))()
        # Falls to static — still extracts the string literal
        assert "inside-function assistant" in result.system_prompt
        assert any("Static" in w for w in result.warnings)
```

- [ ] **Step 2: Run static fallback tests**

```bash
pytest tests/test_filepath_detector.py::TestStaticFallback -v
```

Expected: all PASS (static extraction logic already written in Task 7).

If any fail, check `_static_extract` in `filepath.py` against the test case and fix.

- [ ] **Step 3: Run full `FilePathDetector` suite**

```bash
pytest tests/test_filepath_detector.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_filepath_detector.py \
        src/agentsentinel/core/agents/intake/detectors/filepath.py
git commit -m "test: full static AST fallback coverage for FilePathDetector"
```

---

## Task 9: Register `FilePathDetector` in `AgentIntake`

Wire up the new detector and verify end-to-end.

**Files:**
- Modify: `src/agentsentinel/core/agents/intake/detectors/__init__.py`
- Modify: `src/agentsentinel/core/agents/intake/agent_intake.py`
- Modify: `tests/test_filepath_detector.py`

- [ ] **Step 1: Export from `detectors/__init__.py`**

Check current contents:

```bash
cat src/agentsentinel/core/agents/intake/detectors/__init__.py
```

Add `FilePathDetector` export:

```python
from agentsentinel.core.agents.intake.detectors.langgraph import LangGraphDetector
from agentsentinel.core.agents.intake.detectors.filepath import FilePathDetector

__all__ = ["LangGraphDetector", "FilePathDetector"]
```

- [ ] **Step 2: Register in `AgentIntake._detectors`**

In `src/agentsentinel/core/agents/intake/agent_intake.py`:

```python
from agentsentinel.models import AgentProfile
from agentsentinel.core.agents.intake.detectors.langgraph import LangGraphDetector
from agentsentinel.core.agents.intake.detectors.filepath import FilePathDetector
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AgentIntake:
    def __init__(self):
        self._detectors = [
            FilePathDetector,   # check file-path first (isinstance str/Path guard)
            LangGraphDetector,
        ]
```

- [ ] **Step 3: Write end-to-end integration test**

Add to `tests/test_filepath_detector.py`:

```python
class TestAgentIntakeIntegration:
    def test_intake_accepts_file_path_string(self, tmp_path):
        p = write_agent(tmp_path, """
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

SYSTEM_PROMPT = "You are an intake file-path assistant."

class State(TypedDict):
    message: str

def call_model(state):
    _ = SYSTEM_PROMPT
    return state

builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.set_entry_point("agent")
builder.add_edge("agent", END)
graph = builder.compile()
        """)
        from agentsentinel.core.agents.intake.agent_intake import AgentIntake
        profile = AgentIntake().extract_profile(str(p))
        assert "intake file-path assistant" in profile.system_prompt
        assert profile.framework == "langgraph"
        assert profile.source_code is not None

    def test_intake_accepts_path_object(self, tmp_path):
        p = write_agent(tmp_path, """
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
class S(TypedDict):
    x: str
g = StateGraph(S)
g.add_node("n", lambda s: s)
g.set_entry_point("n")
g.add_edge("n", END)
graph = g.compile()
        """)
        from agentsentinel.core.agents.intake.agent_intake import AgentIntake
        profile = AgentIntake().extract_profile(p)  # Path object, not str
        assert profile.framework == "langgraph"

    def test_intake_unknown_type_still_falls_back(self):
        """Non-path, non-graph object → fallback warning."""
        from agentsentinel.core.agents.intake.agent_intake import AgentIntake
        profile = AgentIntake().extract_profile(42)
        assert "No compatible framework detected" in profile.warnings

    def test_live_object_not_routed_to_file_detector(self):
        """Live Pregel object still uses LangGraphDetector, not FilePathDetector."""
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, END
        from agentsentinel.core.agents.intake.agent_intake import AgentIntake

        class S(TypedDict):
            x: str

        system_prompt = "You are a live-object assistant."
        def node(state):
            _ = system_prompt
            return state

        g = StateGraph(S)
        g.add_node("agent", node)
        g.set_entry_point("agent")
        g.add_edge("agent", END)
        app = g.compile()

        profile = AgentIntake().extract_profile(app)
        assert profile.framework == "langgraph"
        assert "live-object assistant" in profile.system_prompt
```

- [ ] **Step 4: Run integration tests**

```bash
pytest tests/test_filepath_detector.py::TestAgentIntakeIntegration -v
```

Expected: all PASS.

- [ ] **Step 5: Run complete test suite**

```bash
pytest tests/ -v
```

Expected: all PASS — no regressions across `test_langgraph_detector.py` and `test_filepath_detector.py`.

- [ ] **Step 6: Final commit**

```bash
git add src/agentsentinel/core/agents/intake/detectors/__init__.py \
        src/agentsentinel/core/agents/intake/agent_intake.py \
        tests/test_filepath_detector.py
git commit -m "feat: register FilePathDetector in AgentIntake — file-path intake now supported"
```

---

## Self-Review

### Spec coverage check

| Topic from session | Task covering it |
|---|---|
| `state_modifier` singular bug | Task 1 Bug A |
| `.strip(".")` corruption | Task 1 Bug B |
| Fragile Pregel import | Task 1 Bug C |
| `PromptTemplate.template` | Task 2 |
| `model.bind(system=...)` kwargs | Task 2 |
| Function default args | Task 2 |
| Pydantic `model_fields` | Task 2 |
| `functools.partial` recursive unwrap | Task 3 |
| Inherited class attrs | Task 3 |
| `RunnableSequence.steps` (prompt) | Task 3 |
| Expanded `MODEL_NODE_NAMES` | Task 4 |
| `ToolNode` non-`'tools'` name | Task 5 |
| `bind_tools` on instance attr | Task 5 |
| Subgraph tool recursion | Task 5 |
| Tool merge vs replace | Task 6 |
| `source_object` never set | Task 6 |
| `source_code` field on `AgentProfile` | Task 6 |
| FilePathDetector: AST scan | Task 7 |
| FilePathDetector: dynamic import | Task 7 |
| FilePathDetector: static AST fallback | Task 8 |
| FilePathDetector: registration + e2e | Task 9 |

### No placeholders — all tasks contain actual code.

### Type consistency
- `AgentProfile.source_code: Optional[str]` added in Task 6 — used in Tasks 7, 8, 9. ✓
- `_unwrap_fn` defined at module level in Task 3 — used in Task 5. ✓
- `_PROMPT_VAR_NAMES` imported by `filepath.py` from `langgraph.py` in Task 7. ✓
- `FilePathDetector` exported from `__init__.py` and imported in `agent_intake.py` in Task 9. ✓
