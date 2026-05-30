from agentsentinel.models import AgentProfile
from agentsentinel.core.agents.intake.detectors.langgraph import LangGraphDetector
from agentsentinel.core.agents.intake.detectors.filepath import FilePathDetector
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AgentIntake:
    _FRAMEWORK_DETECTORS = [LangGraphDetector]

    def __init__(self):
        self._detectors = [
            FilePathDetector,
            *self._FRAMEWORK_DETECTORS,
        ]

    def extract_profile(self, source_target, agent_profile: Optional[AgentProfile] = None) -> AgentProfile:
        logger.info("Starting Agent Profiling....")
        for DetectorClass in self._detectors:
            detector = (
                DetectorClass(source_target, self._FRAMEWORK_DETECTORS)
                if DetectorClass is FilePathDetector
                else DetectorClass(source_target)
            )
            if detector.can_handle():
                result = detector()
                result.system_prompt = result.system_prompt.strip("\n")
                if agent_profile:
                    if agent_profile.domain:
                        result.domain = agent_profile.domain
                    if agent_profile.system_prompt:
                        if result.system_prompt and result.system_prompt != agent_profile.system_prompt:
                            logger.warning(
                                "System prompt mismatch: detected prompt differs from user-provided prompt.\n"
                                "Detected : %s\nProvided : %s (user-provided wins)",
                                result.system_prompt,
                                agent_profile.system_prompt,
                            )

                if agent_profile and agent_profile.tool_definitions:
                    user_by_name = {t.get('name'): t for t in agent_profile.tool_definitions}
                    detected_by_name = {t.get('name'): t for t in result.tool_definitions}
                    merged = {**detected_by_name, **user_by_name}
                    result.tool_definitions = list(merged.values())
                
                logger.info("Profile: %s", result.to_log_str())
                return result

        return AgentProfile(
            system_prompt=agent_profile.system_prompt if agent_profile else "",
            warnings=["No compatible framework detected"],
        )