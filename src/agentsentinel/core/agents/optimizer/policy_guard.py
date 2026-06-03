import re
import dspy

from agentsentinel.core.agents.optimizer.signatures import CheckPolicyCompliance


class PolicyGuard:
    """
    Hard post-generation validator.
    First pass: hardcoded regex patterns (fast, no LLM).
    Second pass: LLM check against actual policy text (when policy provided).
    Never raises — violations are collected and returned, never thrown.
    """

    POLICY_VIOLATION_PENALTY = 1_000

    def __init__(self, company_policy: str = "", regulations: str = ""):
        self.company_policy = company_policy[:6000]
        self.regulations = regulations[:6000]
        self._checker = dspy.ChainOfThought(CheckPolicyCompliance)

        self._forbidden_patterns: list[tuple[str, str]] = [
            (r"reveal\s+api\s+key",
             "Must never instruct agent to reveal API keys"),
            (r"ignore\s+(all\s+)?previous\s+instructions",
             "Must not contain prompt-injection language"),
            (r"you\s+are\s+now\s+[A-Z][a-zA-Z]+",
             "Role-override language detected"),
            (r"disregard\s+(your\s+)?(rules|guidelines|instructions)",
             "Instruction-override language detected"),
        ]

    def check(self, prompt: str) -> list[str]:
        violations: list[str] = []

        lower = prompt.lower()
        for pattern, reason in self._forbidden_patterns:
            if re.search(pattern, lower):
                violations.append(reason)

        if self.company_policy:
            try:
                result = self._checker(
                    prompt=prompt[:4000],
                    policy_text=self.company_policy,
                )
                raw = (result.violations or "").strip()
                if raw.upper() != "COMPLIANT":
                    for line in raw.splitlines():
                        line = line.strip(" -•*")
                        if line:
                            violations.append(line)
            except Exception:
                pass

        return violations
