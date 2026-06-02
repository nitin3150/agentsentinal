import json
import logging
import warnings

import dspy

from agentsentinel.models.agent import InspectedAgentProfile
from agentsentinel.core.agents.tester.adversarial_prompts_generator import AdversarialPromptGenerator
from agentsentinel.core.agents.tester.runner import AgentRunner
from agentsentinel.core.agents.tester.evaluator import ResponseEvaluator
from agentsentinel.core.agents.tester.report import generate_report

logger = logging.getLogger(__name__)


class TestAgent():
    def __init__(self):
        pass

    def test(self, agent, agent_profile: InspectedAgentProfile, policies: str = ""):
        if dspy.settings.lm is None:
            raise RuntimeError(
                "No LLM configured. Pass providers=[{'model': '...', 'api_key': '...'}] "
                "to AgentSentinel() or set LLM_API_KEY + LLM_MODEL env vars."
            )

        logger.info("Step 1/3: Generating adversarial prompts")
        generator = AdversarialPromptGenerator()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            prompts = generator.generate_all(agent_profile, policies)

        with open("adversarial_prompts.json", "w") as f:
            json.dump(prompts, f, indent=2)
        logger.info("Saved %d prompts to adversarial_prompts.json", len(prompts))

        logger.info("Step 2/3: Running prompts against agent")
        runner = AgentRunner()
        responses = runner.run_prompts(agent, prompts)

        with open("agent_responses.json", "w") as f:
            json.dump(responses, f, indent=2)
        logger.info("Saved %d responses to agent_responses.json", len(responses))

        logger.info("Step 3/3: Evaluating responses")
        evaluator = ResponseEvaluator()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            evaluated = evaluator.evaluate_all(
                responses,
                system_prompt=agent_profile.system_prompt,
                policy=policies,
            )

        report = generate_report(evaluated)
        s = report["summary"]
        logger.info("Audit complete — pass rate: %s%% (%d/%d) | failures: %d",
                    s["pass_rate_pct"], s["passed"], s["total"], s["failed"])
        logger.info("Report saved to audit_report.json and audit_report.md")

        return report
