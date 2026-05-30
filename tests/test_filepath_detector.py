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
        profile = FilePathDetector(str(p))()
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
        profile = FilePathDetector(str(p))()
        assert profile.source_code is not None
        assert "StateGraph" in profile.source_code


class TestStaticFallback:
    """Tests for _static_extract — runs when dynamic import fails."""

    def _static(self, source: str):
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
        result = FilePathDetector(str(p))()
        assert "inside-function assistant" in result.system_prompt
        assert any("Static" in w for w in result.warnings)


class TestAgentIntakeEndToEnd:
    """Integration: AgentIntake routes file-path inputs through FilePathDetector."""

    def test_file_path_str_routed_to_filepath_detector(self, tmp_path):
        p = write_agent(tmp_path, """
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

SYSTEM_PROMPT = "You are an end-to-end file-path assistant."

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
        assert "end-to-end file-path assistant" in profile.system_prompt
        assert profile.source_code is not None

    def test_path_object_also_works(self, tmp_path):
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
        profile = AgentIntake().extract_profile(p)
        assert profile.framework == "langgraph"

    def test_unsupported_input_still_falls_back(self):
        from agentsentinel.core.agents.intake.agent_intake import AgentIntake
        profile = AgentIntake().extract_profile(42)
        assert "No compatible framework detected" in profile.warnings
