"""
Demo agent to test SentinelTracer.
Uses LangChain with OpenTelemetry tracing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

import os
from dotenv import load_dotenv
load_dotenv()

@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    weather_data = {
        "san francisco": "Sunny, 65°F",
        "new york": "Cloudy, 55°F",
        "london": "Rainy, 50°F",
    }
    return weather_data.get(location.lower(), "Unknown location")


def create_simple_llm():
    """Create a simple LLM for demo."""
    llm = ChatOpenAI(
        model=os.getenv("MODEL"),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    return llm


def main():
    """Run demo with tracing."""
    llm = create_simple_llm()

    print("\n=== Demo LLM Test ===\n")

    # Test 1: Simple prompt
    print("Test 1: Simple prompt")
    result = llm.invoke(
        "What is 2 + 2?",
    )
    print(f"Result: {result.content}\n")

    # Test 2: Chain with prompt
    print("Test 2: With prompt template")
    prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    chain = prompt | llm

    result = chain.invoke(
        {"topic": "cats"},
    )
    print(f"Result: {result.content}\n")

    # Test 3: Bind tools
    print("Test 3: With tool binding")
    llm_with_tools = llm.bind_tools([get_weather])
    result = llm_with_tools.invoke(
        "What's the weather in San Francisco?",
    )
    print(f"Result: {result.content}\n")

    print("=== Demo Complete ===")


if __name__ == "__main__":
    main()
