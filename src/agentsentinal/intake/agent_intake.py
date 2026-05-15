from agentsentinal.models import AgentProfile
from agentsentinal.intake.detectors.langgraph import LangGraphDetector

class AgentIntake:
    def __init__(self):
        self._detectors=[
            LangGraphDetector
        ]

    def extract_profile(self,agent) -> AgentProfile:
        for DetectorClass in self._detectors:
            detector = DetectorClass(agent)
            if detector.can_handle():
                return detector()

        return AgentProfile(
            system_prompt = "",
            warnings      = ["No compatible framework detected"]
        )