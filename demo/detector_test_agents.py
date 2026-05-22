"""
Test agents designed to expose known weaknesses in LangGraphDetector.
Each factory function returns an agent that triggers a specific failure.

Run via: uv run tests/test_detector_cases.py
"""
from dotenv import load_dotenv
load_dotenv()

import os
from typing import Any
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, END


def _llm():
    return ChatOpenAI(
        model=os.getenv("MODEL", "stepfun/step-3.5-flash"),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    )


# ── Shared tools ─────────────────────────────────────────────────────────────

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


@tool
def search(query: str, max_results: int = 5, include_snippets: bool = True) -> str:
    """Search the web for information. Returns ranked results with optional snippets."""
    return f"Results for: {query}"


# ── Test 1: Custom node name ──────────────────────────────────────────────────
# Detector only checks MODEL_NODE_NAMES = ('model', 'agent')
# Node named 'chatbot' is never found → empty system_prompt

def make_custom_node_name_agent():
    """BREAKS: node named 'chatbot' not in MODEL_NODE_NAMES."""
    llm = _llm()
    SYSTEM = "You are a billing support agent. Only answer billing questions."

    def chatbot(state):                          # <-- not 'model' or 'agent'
        msgs = [SystemMessage(content=SYSTEM)] + state["messages"]
        return {"messages": [llm.invoke(msgs)]}

    graph = StateGraph(MessagesState)
    graph.add_node("chatbot", chatbot)
    graph.set_entry_point("chatbot")
    graph.add_edge("chatbot", END)
    return graph.compile()


# ── Test 2: Prompt under unexpected closure variable name ─────────────────────
# Detector only checks: system_prompt, system_message, prompt, state_modifiers
# Variable named 'instructions' is invisible to detector

def make_nonstandard_prompt_varname_agent():
    """BREAKS: prompt stored as 'instructions' — not in detector's checked names."""
    llm = _llm()
    instructions = "You are a tax advisor. Never give legal advice. Always cite tax code."

    def agent(state):
        msgs = [SystemMessage(content=instructions)] + state["messages"]
        return {"messages": [llm.invoke(msgs)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()


# ── Test 3: Prompt built from PromptTemplate (not a plain string) ─────────────
# Detector checks isinstance(val, str) and hasattr(val, 'content')
# A ChatPromptTemplate has neither — silently missed

def make_prompt_template_agent():
    """BREAKS: system prompt is a ChatPromptTemplate object, not str or SystemMessage."""
    from langchain_core.prompts import ChatPromptTemplate

    llm = _llm()
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a compliance officer. Policy: {policy}"),
        ("human", "{input}"),
    ])

    def agent(state):
        filled = prompt_template.format_messages(policy="ISO 27001", input=state["messages"][-1].content)
        return {"messages": [llm.invoke(filled)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()


# ── Test 4: Class-based node ─────────────────────────────────────────────────
# inspect.getclosurevars() returns {} for bound methods
# Prompt lives on self — never found

def make_class_based_node_agent():
    """BREAKS: prompt is an instance attribute, not a closure variable."""
    llm = _llm()

    class AgentNode:
        def __init__(self):
            self.system_prompt = "You are a legal document reviewer. Be precise and cite statutes."
            self.llm = llm

        def __call__(self, state):
            msgs = [SystemMessage(content=self.system_prompt)] + state["messages"]
            return {"messages": [self.llm.invoke(msgs)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", AgentNode())
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()


# ── Test 5: Tools bound to model — no tools node ─────────────────────────────
# Tools bound via bind_tools() are on the model object, no ToolNode in graph
# nodes.get("tools") returns None → zero tools extracted

def make_bound_tools_agent():
    """BREAKS: tools via model.bind_tools(), no 'tools' node in graph."""
    llm_with_tools = _llm().bind_tools([calculate, search])
    SYSTEM = "You are a research assistant. Use tools to answer questions."

    def agent(state):
        msgs = [SystemMessage(content=SYSTEM)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(msgs)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()


# ── Test 6: Multi-param tool — arg_schema typo ───────────────────────────────
# LangChain tools use args_schema, detector checks arg_schema (typo)
# Tool is found, name+description extracted, but parameters always come back {}

def make_rich_tool_agent():
    """BREAKS: tool has typed params but parameters={} due to arg_schema typo."""
    return create_agent(
        model=_llm(),
        tools=[search],          # search: query: str, max_results: int, include_snippets: bool
        system_prompt="You are a research assistant. Search thoroughly.",
    )


# ── Test 7: Multiple LLM nodes ───────────────────────────────────────────────
# Planner + executor — two different system prompts
# Neither node is named 'model' or 'agent' → both missed

def make_multi_llm_node_agent():
    """BREAKS: two LLM nodes ('planner', 'executor') — neither in MODEL_NODE_NAMES."""
    llm = _llm()
    PLANNER_PROMPT = "You are a planner. Break the task into numbered steps. Output JSON only."
    EXECUTOR_PROMPT = "You are an executor. Run each step and report results."

    def planner(state):
        msgs = [SystemMessage(content=PLANNER_PROMPT)] + state["messages"]
        return {"messages": [llm.invoke(msgs)]}

    def executor(state):
        msgs = [SystemMessage(content=EXECUTOR_PROMPT)] + state["messages"]
        return {"messages": [llm.invoke(msgs)]}

    graph = StateGraph(MessagesState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", END)
    return graph.compile()


# ── Test 8: Subgraph ──────────────────────────────────────────────────────────
# Outer graph wraps an inner compiled graph as a node
# Detector reads outer .nodes — sees 'worker' (a CompiledGraph), never recurses

def make_subgraph_agent():
    """BREAKS: agent is a compiled subgraph — detector doesn't recurse into it."""
    llm = _llm()
    INNER_PROMPT = "You are an inner specialist. Handle only financial data queries."

    def inner_node(state):
        msgs = [SystemMessage(content=INNER_PROMPT)] + state["messages"]
        return {"messages": [llm.invoke(msgs)]}

    inner_graph = StateGraph(MessagesState)
    inner_graph.add_node("specialist", inner_node)
    inner_graph.set_entry_point("specialist")
    inner_graph.add_edge("specialist", END)
    inner_compiled = inner_graph.compile()

    outer_graph = StateGraph(MessagesState)
    outer_graph.add_node("worker", inner_compiled)   # compiled subgraph as node
    outer_graph.set_entry_point("worker")
    outer_graph.add_edge("worker", END)
    return outer_graph.compile()


# ── Test 10: Module-level prompt (globals, not nonlocals) ────────────────────
# Most hand-rolled LangGraph agents define SYSTEM_PROMPT at module level
# getclosurevars().nonlocals is {} — prompt is in .globals, which detector never checks

_MODULE_PROMPT = "You are a module-level assistant. Handle customer enquiries only."

def make_module_level_prompt_agent():
    """BREAKS: SYSTEM_PROMPT at module level → in .globals, detector only checks .nonlocals."""
    llm = _llm()

    def agent(state):
        msgs = [SystemMessage(content=_MODULE_PROMPT)] + state["messages"]
        return {"messages": [llm.invoke(msgs)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()


# ── Test 11: Prompt as local variable inside node ────────────────────────────
# Local variables are never closure vars — getclosurevars() always returns {} for them
# Cannot be extracted without AST analysis or bytecode inspection

def make_local_var_prompt_agent():
    """BREAKS: prompt defined with local = '...' inside the function body — not in any closure."""
    llm = _llm()

    def agent(state):
        system_prompt = "You are a security auditor. Flag all suspicious requests."
        msgs = [SystemMessage(content=system_prompt)] + state["messages"]
        return {"messages": [llm.invoke(msgs)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()


# ── Test 12: Short prompt string (< 10 chars) ────────────────────────────────
# isinstance(val, str) path has len(val) > 10 guard — short strings silently dropped
# Note: SystemMessage path has NO length guard, so this only affects raw string closures

def make_short_prompt_agent():
    """BREAKS: prompt is a 7-char string in closure — fails len(val) > 10 guard."""
    llm = _llm()
    prompt = "Be nice"   # 7 chars — below the 10-char threshold

    def agent(state):
        msgs = [SystemMessage(content=prompt)] + state["messages"]
        return {"messages": [llm.invoke(msgs)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()


# ── Test 13: Prompt in messages list ─────────────────────────────────────────
# Some agents store a list of BaseMessage as initial context, not a single string
# Detector has no code to walk a list and extract the SystemMessage

_INITIAL_MESSAGES = [SystemMessage(content="You are a financial analyst. Never speculate.")]

def make_messages_list_prompt_agent():
    """BREAKS: prompt is a SystemMessage inside a list — detector only checks single values."""
    llm = _llm()

    def agent(state):
        return {"messages": [llm.invoke(_INITIAL_MESSAGES + state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()


# ── Test 9: can_handle false positive ────────────────────────────────────────
# can_handle only checks hasattr(agent, 'nodes')
# Any object with .nodes passes — not guaranteed to be a LangGraph graph

def make_false_positive_object():
    """BREAKS: object has .nodes but is not a LangGraph graph — can_handle() = True."""
    class FakeAgent:
        nodes = {"model": "not_a_real_node", "tools": "also_fake"}
        def invoke(self, x):
            return x

    return FakeAgent()


# ── Registry ─────────────────────────────────────────────────────────────────

TEST_CASES: list[dict[str, Any]] = [
    {
        "id": 1,
        "name": "Custom node name ('chatbot')",
        "factory": make_custom_node_name_agent,
        "expects_prompt": True,
        "expects_tools": False,
        "known_failure": "MODEL_NODE_NAMES = ('model','agent') — 'chatbot' missed",
    },
    {
        "id": 2,
        "name": "Non-standard prompt variable name",
        "factory": make_nonstandard_prompt_varname_agent,
        "expects_prompt": True,
        "expects_tools": False,
        "known_failure": "prompt stored as 'instructions', detector only checks known names",
    },
    {
        "id": 3,
        "name": "ChatPromptTemplate (not str/SystemMessage)",
        "factory": make_prompt_template_agent,
        "expects_prompt": True,
        "expects_tools": False,
        "known_failure": "PromptTemplate fails isinstance(str) and hasattr(content) checks",
    },
    {
        "id": 4,
        "name": "Class-based node",
        "factory": make_class_based_node_agent,
        "expects_prompt": True,
        "expects_tools": False,
        "known_failure": "getclosurevars() returns {} for bound methods — prompt on self",
    },
    {
        "id": 5,
        "name": "Tools via model.bind_tools() — no tools node",
        "factory": make_bound_tools_agent,
        "expects_prompt": True,
        "expects_tools": True,
        "known_failure": "nodes.get('tools') is None — bind_tools pattern not handled",
    },
    {
        "id": 6,
        "name": "Multi-param tool (arg_schema typo)",
        "factory": make_rich_tool_agent,
        "expects_prompt": True,
        "expects_tools": True,
        "known_failure": "getattr(tool, 'arg_schema') misses — correct attr is 'args_schema'",
    },
    {
        "id": 7,
        "name": "Multiple LLM nodes (planner + executor)",
        "factory": make_multi_llm_node_agent,
        "expects_prompt": True,
        "expects_tools": False,
        "known_failure": "neither 'planner' nor 'executor' in MODEL_NODE_NAMES",
    },
    {
        "id": 8,
        "name": "Subgraph (nested compiled graph)",
        "factory": make_subgraph_agent,
        "expects_prompt": True,
        "expects_tools": False,
        "known_failure": "detector reads outer .nodes only — no recursion into subgraph",
    },
    {
        "id": 9,
        "name": "False positive: non-graph object with .nodes",
        "factory": make_false_positive_object,
        "expects_prompt": False,
        "expects_tools": False,
        "known_failure": "can_handle() passes for any object with .nodes attribute",
    },
    {
        "id": 10,
        "name": "Module-level prompt (in globals, not nonlocals)",
        "factory": make_module_level_prompt_agent,
        "expects_prompt": True,
        "expects_tools": False,
        "known_failure": "getclosurevars().nonlocals={} for module-level vars — must also check .globals",
    },
    {
        "id": 11,
        "name": "Prompt as local variable inside node",
        "factory": make_local_var_prompt_agent,
        "expects_prompt": True,
        "expects_tools": False,
        "known_failure": "local vars never in closure — requires AST/bytecode inspection to extract",
    },
    {
        "id": 12,
        "name": "Short prompt string (< 10 chars) in closure",
        "factory": make_short_prompt_agent,
        "expects_prompt": True,
        "expects_tools": False,
        "known_failure": "str path guards len(val) > 10 — 'Be nice' (7 chars) silently dropped",
    },
    {
        "id": 13,
        "name": "Prompt inside messages list",
        "factory": make_messages_list_prompt_agent,
        "expects_prompt": True,
        "expects_tools": False,
        "known_failure": "detector checks single values only — no list walk for SystemMessage extraction",
    },
]
