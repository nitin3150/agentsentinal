from dotenv import load_dotenv
load_dotenv()

import os
import sys
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)  # LangChainPendingDeprecationWarning

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
from agentsentinel.utils.logger import setup_logger

setup_logger("agentsentinel")
logger = logging.getLogger("agentsentinel.main")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentsentinel.sentinel import AgentSentinel
from demo.langchain_agent import run_agent

policies = str(Path(__file__).resolve().parent / "sample_policies.pdf")

if __name__ =="__main__":
    logger.info("Loading demo agent")
    agent = run_agent()

    try:
        GROQ_API_KEY = os.environ["GROQ_API_KEY"]
        OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
        groq_model = os.environ["GROQ_MODEL"]
        openrouter_model = os.environ["OPENROUTER_MODEL"]
    except KeyError as exc:
        raise RuntimeError(
            f"Missing required environment variable: {exc.args[0]}. "
            "Set GROQ_API_KEY, OPENROUTER_API_KEY, GROQ_MODEL, and OPENROUTER_MODEL before running main.py."
        ) from exc

    logger.info("Initializing AgentSentinel")
    sentinel = AgentSentinel(providers=[
        {"api_key": GROQ_API_KEY, "model": groq_model},
        {"api_key": OPENROUTER_API_KEY, "model": openrouter_model},
    ])

    logger.info("Running inspect stage")
    profile = sentinel.inspect(
        agent,
        source="demo/langchain_agent.py",
        domain="personal agent",
        policies=policies,
    )

    logger.info("Running optimize stage")
    result = sentinel.optimize(
        agent_profile=profile,
        policies=policies,
    )

    logger.info("Running stress-test stage")
    sentinel.stress_test(agent, profile, policies=policies)
    logger.info("Demo run complete")