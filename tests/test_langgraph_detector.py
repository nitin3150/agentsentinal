"""
Tests for LangGraphDetector and AgentIntake.
No LLM API calls — all tests run without network access.
"""
import pytest
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from agentsentinel.core.agents.intake.detectors.langgraph import LangGraphDetector
from agentsentinel.core.agents.intake.agent_intake import AgentIntake
from agentsentinel.models import AgentProfile


class State(TypedDict):
    message: str


def _compile_graph(node_fn, node_name: str = "model", tools_node=None):
    """Build and compile a minimal single-node StateGraph."""
    g = StateGraph(State)
    g.add_node(node_name, node_fn)
    g.set_entry_point(node_name)
    if tools_node is not None:
        g.add_node("tools", tools_node)
        g.add_edge(node_name, "tools")
        g.add_edge("tools", END)
    else:
        g.add_edge(node_name, END)
    return g.compile()


# ── can_handle ────────────────────────────────────────────────────────────────

class TestCanHandle:
    def test_true_for_compiled_langgraph(self):
        def node(state): return state
        app = _compile_graph(node)
        assert LangGraphDetector(app).can_handle() is True

    def test_false_for_plain_dict(self):
        assert LangGraphDetector({}).can_handle() is False

    def test_false_for_object_with_nodes_attr(self):
        """Object that has .nodes but isn't a Pregel must not match."""
        class FakeAgent:
            nodes = {"model": None}
        assert LangGraphDetector(FakeAgent()).can_handle() is False

    def test_false_for_none(self):
        assert LangGraphDetector(None).can_handle() is False

    def test_false_for_plain_string(self):
        assert LangGraphDetector("agent").can_handle() is False


# ── system prompt — variable names ────────────────────────────────────────────

class TestPromptFromVariableName:
    def test_system_prompt_var(self):
        def make():
            system_prompt = "You are a helpful assistant."
            def node(state):
                _ = system_prompt
                return state
            return node
        result = LangGraphDetector(_compile_graph(make()))()
        assert result.system_prompt == "You are a helpful assistant."

    def test_instructions_var(self):
        """Known var name — min_len=0, even short strings extracted."""
        def make():
            instructions = "Short."
            def node(state):
                _ = instructions
                return state
            return node
        result = LangGraphDetector(_compile_graph(make()))()
        assert result.system_prompt == "Short."

    def test_system_var(self):
        def make():
            system = "You are a data analyst."
            def node(state):
                _ = system
                return state
            return node
        result = LangGraphDetector(_compile_graph(make()))()
        assert result.system_prompt == "You are a data analyst."

    def test_sys_prompt_var(self):
        def make():
            sys_prompt = "You are a support agent."
            def node(state):
                _ = sys_prompt
                return state
            return node
        result = LangGraphDetector(_compile_graph(make()))()
        assert result.system_prompt == "You are a support agent."

    def test_state_modifier_var(self):
        """state_modifier (singular) extracted even when <15 chars — known-name scan, not fallback."""
        def make():
            state_modifier = "Be concise."  # short — fallback scan misses it, known-name must catch
            def node(state):
                _ = state_modifier
                return state
            return node
        result = LangGraphDetector(_compile_graph(make()))()
        assert result.system_prompt == "Be concise."

    def test_nonstandard_var_long_string_extracted(self):
        """Unknown var name with len > 15 — picked up by fallback scan."""
        def make():
            _agent_config = "You are an enterprise customer support agent."
            def node(state):
                _ = _agent_config
                return state
            return node
        result = LangGraphDetector(_compile_graph(make()))()
        assert result.system_prompt == "You are an enterprise customer support agent."

    def test_nonstandard_var_short_string_not_extracted(self):
        """Unknown var name with len <= 15 — NOT extracted (below min_len guard)."""
        def make():
            _cfg = "Too short here."  # 15 chars, > min_len=15 is False
            def node(state):
                _ = _cfg
                return state
            return node
        result = LangGraphDetector(_compile_graph(make()))()
        assert result.system_prompt == ""

    def test_module_level_global_captured(self):
        """Variables in cv.globals (module scope) are also checked."""
        GLOBAL_PROMPT = "You are a global-scope assistant agent."

        def node(state):
            _ = GLOBAL_PROMPT
            return state

        result = LangGraphDetector(_compile_graph(node))()
        assert result.system_prompt == GLOBAL_PROMPT


# ── system prompt — value types ───────────────────────────────────────────────

class TestPromptValueTypes:
    def test_plain_string(self):
        def make():
            system_prompt = "You are a plain string assistant."
            def node(state):
                _ = system_prompt
                return state
            return node
        result = LangGraphDetector(_compile_graph(make()))()
        assert result.system_prompt == "You are a plain string assistant."

    def test_system_message(self):
        from langchain_core.messages import SystemMessage
        def make():
            sys_msg = SystemMessage(content="You are a SystemMessage assistant.")
            def node(state):
                _ = sys_msg
                return state
            return node
        result = LangGraphDetector(_compile_graph(make()))()
        assert result.system_prompt == "You are a SystemMessage assistant."

    def test_chat_prompt_template(self):
        from langchain_core.prompts import ChatPromptTemplate
        def make():
            template = ChatPromptTemplate.from_messages([
                ("system", "You are a ChatPromptTemplate assistant."),
                ("human", "{input}"),
            ])
            def node(state):
                _ = template
                return state
            return node
        result = LangGraphDetector(_compile_graph(make()))()
        assert result.system_prompt == "You are a ChatPromptTemplate assistant."

    def test_list_of_messages_with_system(self):
        from langchain_core.messages import SystemMessage, HumanMessage
        def make():
            messages = [
                SystemMessage(content="You are a messages-list assistant."),
                HumanMessage(content="Hello"),
            ]
            def node(state):
                _ = messages
                return state
            return node
        result = LangGraphDetector(_compile_graph(make()))()
        assert result.system_prompt == "You are a messages-list assistant."

    def test_class_based_node_instance_attr(self):
        """Prompt stored on __call__ class instance, not in closure."""
        class AgentNode:
            def __init__(self):
                self.system_prompt = "You are a class-based agent assistant."
            def __call__(self, state):
                return state

        result = LangGraphDetector(_compile_graph(AgentNode()))()
        assert result.system_prompt == "You are a class-based agent assistant."


# ── node name routing ─────────────────────────────────────────────────────────

class TestNodeNameRouting:
    def test_model_node_name(self):
        def make():
            system_prompt = "Agent via model node."
            def node(state):
                _ = system_prompt
                return state
            return node
        result = LangGraphDetector(_compile_graph(make(), node_name="model"))()
        assert result.system_prompt == "Agent via model node."

    def test_agent_node_name(self):
        def make():
            system_prompt = "Agent via agent node."
            def node(state):
                _ = system_prompt
                return state
            return node
        result = LangGraphDetector(_compile_graph(make(), node_name="agent"))()
        assert result.system_prompt == "Agent via agent node."

    def test_custom_node_name_fallback_scan(self):
        """No model/agent node — detector scans all non-skip nodes."""
        def make():
            system_prompt = "Agent via custom chatbot node."
            def node(state):
                _ = system_prompt
                return state
            return node
        result = LangGraphDetector(_compile_graph(make(), node_name="chatbot"))()
        assert result.system_prompt == "Agent via custom chatbot node."

    def test_no_prompt_in_any_node_adds_warning(self):
        def node(state): return state  # no prompt reference
        result = LangGraphDetector(_compile_graph(node))()
        assert result.system_prompt == ""
        assert len(result.warnings) > 0


# ── tool extraction ───────────────────────────────────────────────────────────

class TestToolExtraction:
    def _make_graph_with_tools(self, tools: list):
        from langgraph.prebuilt import ToolNode
        def node(state): return state
        return _compile_graph(node, tools_node=ToolNode(tools))

    def test_single_tool_from_tool_node(self):
        from langchain_core.tools import tool

        @tool
        def calculate(expression: str) -> str:
            """Evaluate a math expression and return the result."""
            return str(eval(expression))  # noqa: S307

        result = LangGraphDetector(self._make_graph_with_tools([calculate]))()
        assert len(result.tool_definitions) == 1
        assert result.tool_definitions[0]["name"] == "calculate"

    def test_multiple_tools_all_extracted(self):
        from langchain_core.tools import tool

        @tool
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply two integers."""
            return a * b

        result = LangGraphDetector(self._make_graph_with_tools([add, multiply]))()
        names = {t["name"] for t in result.tool_definitions}
        assert names == {"add", "multiply"}

    def test_tool_carries_description(self):
        from langchain_core.tools import tool

        @tool
        def greet(name: str) -> str:
            """Say hello to the given name."""
            return f"Hello, {name}"

        result = LangGraphDetector(self._make_graph_with_tools([greet]))()
        assert result.tool_definitions[0]["description"] == "Say hello to the given name."

    def test_no_tools_returns_empty_list(self):
        def node(state): return state
        result = LangGraphDetector(_compile_graph(node))()
        assert result.tool_definitions == []


# ── subgraph recursion ────────────────────────────────────────────────────────

class TestSubgraph:
    def test_prompt_extracted_from_compiled_subgraph_node(self):
        def make_inner():
            system_prompt = "You are a subgraph agent assistant."
            def inner_node(state):
                _ = system_prompt
                return state
            return inner_node

        inner_g = StateGraph(State)
        inner_g.add_node("model", make_inner())
        inner_g.set_entry_point("model")
        inner_g.add_edge("model", END)
        inner_app = inner_g.compile()

        outer_g = StateGraph(State)
        outer_g.add_node("subagent", inner_app)
        outer_g.set_entry_point("subagent")
        outer_g.add_edge("subagent", END)
        outer_app = outer_g.compile()

        result = LangGraphDetector(outer_app)()
        assert result.system_prompt == "You are a subgraph agent assistant."


# ── framework field ───────────────────────────────────────────────────────────

class TestFrameworkField:
    def test_framework_is_langgraph(self):
        def node(state): return state
        result = LangGraphDetector(_compile_graph(node))()
        assert result.framework == "langgraph"


# ── AgentIntake orchestration ─────────────────────────────────────────────────

class TestAgentIntake:
    def test_selects_langgraph_detector(self):
        def make():
            system_prompt = "You are an intake test assistant."
            def node(state):
                _ = system_prompt
                return state
            return node
        app = _compile_graph(make())
        profile = AgentIntake().extract_profile(app)
        assert profile.framework == "langgraph"

    def test_user_prompt_overrides_detected(self):
        """User-supplied system_prompt wins over auto-detected one."""
        def make():
            system_prompt = "Detected prompt — should be overridden."
            def node(state):
                _ = system_prompt
                return state
            return node
        app = _compile_graph(make())
        override = AgentProfile(system_prompt="User override prompt.")
        profile = AgentIntake().extract_profile(app, agent_profile=override)
        assert profile.system_prompt == "User override prompt."

    def test_user_tools_override_detected(self):
        """User-supplied tool_definitions wins over auto-detected ones."""
        from langchain_core.tools import tool
        from langgraph.prebuilt import ToolNode

        @tool
        def detected_tool(x: int) -> int:
            """Auto-detected tool."""
            return x

        def node(state): return state
        app = _compile_graph(node, tools_node=ToolNode([detected_tool]))

        override = AgentProfile(tool_definitions=[{"name": "user_tool", "description": "override"}])
        profile = AgentIntake().extract_profile(app, agent_profile=override)
        assert len(profile.tool_definitions) == 1
        assert profile.tool_definitions[0]["name"] == "user_tool"

    def test_fallback_for_unsupported_type(self):
        class RandomObject:
            pass
        profile = AgentIntake().extract_profile(RandomObject())
        assert "No compatible framework detected" in profile.warnings

    def test_fallback_preserves_provided_system_prompt(self):
        class RandomObject:
            pass
        override = AgentProfile(system_prompt="Fallback prompt.")
        profile = AgentIntake().extract_profile(RandomObject(), agent_profile=override)
        assert profile.system_prompt == "Fallback prompt."

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

