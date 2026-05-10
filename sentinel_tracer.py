from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("agentsentinel")

class SentinelTracer(BaseCallbackHandler):

    def on_chain_start(self, serialized, inputs, **kwargs):
        """Fires when the agent starts processing a message."""
        self.span = tracer.start_span("agent.invoke")
        self.span.set_attribute("input", str(inputs))
        print(f"[Sentinel] Agent started. Input: {inputs}")

    def on_chain_end(self, outputs, **kwargs):
        """Fires when the agent finishes."""
        self.span.set_attribute("output", str(outputs))
        self.span.end()
        print(f"[Sentinel] Agent finished. Output: {outputs}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Fires when the agent calls a tool (e.g. search, database)."""
        tool_name = serialized.get("name", "unknown_tool")
        with tracer.start_as_current_span("tool.call") as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.input", input_str)
            print(f"[Sentinel] Tool called: {tool_name} | Input: {input_str}")

    def on_tool_end(self, output, **kwargs):
        """Fires when a tool returns a result."""
        print(f"[Sentinel] Tool returned: {output}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Fires when the LLM starts generating a response."""
        print(f"[Sentinel] LLM thinking...")

    def on_llm_end(self, response, **kwargs):
        """Fires when the LLM finishes generating."""
        with tracer.start_as_current_span("llm.generate") as span:
            text = response.generations[0][0].text if response.generations else ""
            span.set_attribute("llm.output", text)
            print(f"[Sentinel] LLM response: {text[:100]}...") 