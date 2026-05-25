import logging

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class AgentRunner():
    def run_prompts(self, agent, prompts: list) -> list:
        responses = []
        for i, prompt in enumerate(prompts):
            prompt_text = prompt["prompt"]
            try:
                raw = agent.invoke({"messages": [HumanMessage(content=prompt_text)]})
                messages = raw.get("messages", [])
                answer = messages[-1].content if messages else str(raw)
            except Exception as e:
                answer = f"ERROR: {e}"
                logger.warning("Prompt %s failed: %s", prompt["id"], e)

            record = {**prompt, "response": answer}
            responses.append(record)
            logger.info("[%d/%d] %s — %s", i + 1, len(prompts), prompt["category"], answer[:80])

        return responses