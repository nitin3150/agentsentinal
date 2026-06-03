import json
import logging
import warnings
from pathlib import Path

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

    def test(
        self,
        agent,
        agent_profile: InspectedAgentProfile,
        policies: str = "",
        output_dir: str | Path | None = None,
    ):
        if dspy.settings.lm is None:
            raise RuntimeError(
                "No LLM configured. Pass providers=[{'model': '...', 'api_key': '...'}] "
                "to AgentSentinel() or set LLM_API_KEY + LLM_MODEL env vars."
            )

        out = Path(output_dir) if output_dir else None
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)

        logger.info("Step 1/3: Generating adversarial prompts")
        generator = AdversarialPromptGenerator()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            prompts = generator.generate_all(agent_profile, policies)

        if out is not None:
            prompts_path = out / "adversarial_prompts.json"
            prompts_path.write_text(json.dumps(prompts, indent=2))
            logger.info("Saved %d prompts to %s", len(prompts), prompts_path)

        if not prompts:
            logger.error("No adversarial prompts generated — stress test aborted.")
            return {"summary": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "pass_rate_pct": 0}, "by_category": {}, "failures": []}

        logger.info("Step 2/3: Running prompts against agent")
        runner = AgentRunner()
        responses = runner.run_prompts(agent, prompts)

        if out is not None:
            responses_path = out / "agent_responses.json"
            responses_path.write_text(json.dumps(responses, indent=2))
            logger.info("Saved %d responses to %s", len(responses), responses_path)

        logger.info("Step 3/3: Evaluating responses")
        evaluator = ResponseEvaluator()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            evaluated = evaluator.evaluate_all(
                responses,
                system_prompt=agent_profile.system_prompt,
                policy=policies,
            )

        report_path = str(out / "audit_report") if out is not None else None
        report = generate_report(evaluated, output_path=report_path)
        s = report["summary"]
        logger.info("Audit complete — pass rate: %s%% (%d/%d) | failures: %d",
                    s["pass_rate_pct"], s["passed"], s["total"], s["failed"])
        if out is not None:
            logger.info("Report saved to %s", out)

        return report
