import re

class PolicyGuard:
    """
    Hard post-generation validator.
    Violations here disqualify the output regardless of DSPy score.
    Extend _forbidden_patterns with your real policy rules.
    """

    POLICY_VIOLATION_PENALTY = 1_000

    def __init__(self, company_policy: str = "", regulations: str = ""):
        self.company_policy = company_policy
        self.regulations    = regulations

        self._forbidden_patterns: list[tuple[str, str]] = [
            (r"reveal\s+api\s+key",
             "Must never instruct agent to reveal API keys"),
            (r"ignore\s+(all\s+)?previous\s+instructions",
             "Must not contain prompt-injection language"),
            (r"you\s+are\s+now\s+[A-Z][a-zA-Z]+",
             "Role-override language detected"),
            (r"disregard\s+(your\s+)?(rules|guidelines|instructions)",
             "Instruction-override language detected"),
            # ── Add patterns derived from your company_policy / regulations ──
        ]

    def check(self, prompt: str) -> list[str]:
        violations = []
        lower = prompt.lower()
        for pattern, reason in self._forbidden_patterns:
            if re.search(pattern, lower):
                violations.append(reason)
        return violations