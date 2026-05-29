from agentsentinal.models import AgentProfile
from agentsentinal.core.agents.intake.detectors.langgraph import LangGraphDetector
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AgentIntake:
    def __init__(self):
        self._detectors=[
            LangGraphDetector
        ]

    def extract_profile(self, agent, agent_profile: Optional[AgentProfile] = None) -> AgentProfile:
        logger.info("Starting Agent Profiling....")
        for DetectorClass in self._detectors:
            detector = DetectorClass(agent)
            if detector.can_handle():
                result = detector()
                result.system_prompt = result.system_prompt.strip("\n").strip(".")
                if agent_profile and agent_profile.system_prompt:
                    if result.system_prompt and result.system_prompt != agent_profile.system_prompt:
                        logger.error(
                            "System prompt mismatch: detected prompt differs from user-provided prompt.\n"
                            "Detected : %s\nProvided : %s",
                            result.system_prompt,
                            agent_profile.system_prompt,
                        )
                    result.system_prompt = agent_profile.system_prompt
                    result.domain = agent_profile.domain

                if agent_profile and agent_profile.tool_definitions:
                    result.tool_definitions = agent_profile.tool_definitions
                
                logger.info("Profile: %s", result.to_log_str())
                return result

        return AgentProfile(
            system_prompt=agent_profile.system_prompt if agent_profile else "",
            warnings=["No compatible framework detected"],
        )