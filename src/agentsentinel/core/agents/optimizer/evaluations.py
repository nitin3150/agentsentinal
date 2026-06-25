import os

from agentsentinel.core.agents.optimizer.policy_guard import PolicyGuard
import asyncio
import logging
from agentsentinel.models.agent import AgentProfile, InspectedAgentProfile
import dspy
from typing import Any
from agentsentinel.models.prompt import OptimizedResult
from agentsentinel.core.agents.optimizer.prompt_optimizer import PromptOptimizer

logger = logging.getLogger("agentsentinel.improver.evaluations")

class ImprovementMetric:
    """
    Scores the improvement by re-running InspectorAgent on the output.
    Used by DSPy optimisers (BootstrapFewShot, MIPRO, etc.).

    InspectorAgent.inspect() is async; we bridge it with asyncio.run()
    so this metric stays synchronous and compatible with DSPy's compile loop.
    """

    def __init__(self, inspector, company_policy: str = "", regulations: str = ""):
        self.inspector    = inspector
        self.policy_guard = PolicyGuard(company_policy, regulations)

    def __call__(
        self,
        example,     # dspy.Example — see build_optimized_improver for required fields
        prediction,  # OptimizedResult
        trace=None,
    ) -> float:

        # Hard reject on policy violations
        if prediction.policy_violations:
            return -PolicyGuard.POLICY_VIOLATION_PENALTY

        # Re-inspect the improved prompt with the real InspectorAgent
        try:
            improved_profile: InspectedAgentProfile = asyncio.run(
                self.inspector.inspect(
                    AgentProfile(
                        system_prompt=prediction.improved_prompt,
                        tool_definitions=prediction.improved_tool_definitions,
                        framework=example.agent_profile.framework,
                    )
                )
            )
        except Exception as exc:
            logger.error("ImprovementMetric inspector error: %s", exc)
            return -50.0

        original: InspectedAgentProfile = example.agent_profile
        score = 0.0

        # ── Baseline score delta (0–100 scale) ───────────────────────────────
        score += improved_profile.estimated_baseline_score - original.estimated_baseline_score

        # ── Risk flag reduction (10 pts per flag resolved) ───────────────────
        score += (len(original.risk_flags) - len(improved_profile.risk_flags)) * 10

        # ── Sub-score improvements ────────────────────────────────────────────
        score += (improved_profile.persona_clarity_score  - original.persona_clarity_score)  * 2
        score += (improved_profile.scope_definition_score - original.scope_definition_score) * 2
        score += (improved_profile.tone_consistency_score - original.tone_consistency_score) * 1
        score += (improved_profile.avg_tool_quality       - original.avg_tool_quality)       * 2

        # ── Bonus for newly fixed structural properties ───────────────────────
        if improved_profile.output_format_defined and not original.output_format_defined:
            score += 5
        if improved_profile.constraint_count > original.constraint_count:
            score += min(improved_profile.constraint_count - original.constraint_count, 5) * 2

        # ── Penalise newly introduced risk flags (regressions) ────────────────
        original_categories = {f.category for f in original.risk_flags}
        new_flags = [
            f for f in improved_profile.risk_flags
            if f.category not in original_categories
        ]
        score -= len(new_flags) * 20

        # ── Penalise prompt bloat (> 3× original length is suspicious) ────────
        length_ratio = len(prediction.improved_prompt) / max(len(example.original_prompt), 1)
        if length_ratio > 3.0:
            score -= (length_ratio - 3.0) * 5

        return score


# ─────────────────────────────────────────────────────────────────────────────
# 6.  OPTIMISER SETUP
# ─────────────────────────────────────────────────────────────────────────────

def build_optimized_improver(
    inspector,
    trainset:        list[dspy.Example],
    company_policy:  str = "",
    regulations:     str = "",
    lm_model_string: str | None = None,
) -> PromptOptimizer:
    """
    Compiles a PromptOptimizer optimised for your Inspector, policy,
    and regulations via DSPy's BootstrapFewShot.

    Each trainset entry must be created like this:

        dspy.Example(
            original_prompt  = agent_system_prompt,    # str
            tool_definitions = agent_tool_definitions, # list[dict]
            agent_profile    = profile,                # AgentProfile from inspector.inspect()
        ).with_inputs("original_prompt", "tool_definitions", "agent_profile")

    5–10 real broken agent prompts is usually enough to get meaningful
    optimisation.

    ``lm_model_string`` is provider-agnostic — pass any LiteLLM-routed model
    (``"openai/gpt-4o"``, ``"anthropic/claude-3-5-sonnet"``, ``"groq/llama-3.3-70b-versatile"``,
    ``"vertex_ai/gemini-2.0-flash"``, etc.). When omitted, the model + key +
    base URL are inherited from the same env vars the runtime uses
    (``LLM_MODEL`` → ``GROQ_MODEL`` → ``OPENROUTER_MODEL`` → ``NVIDIA_MODEL``,
    with matching ``*_API_KEY`` / ``*_BASE_URL`` fallbacks), so DSPy
    compile/training honours the same configuration as inspect/optimize.
    """

    if lm_model_string is not None:
        lm = dspy.LM(lm_model_string)
    else:
        model = (
            os.getenv("LLM_MODEL")
            or os.getenv("GROQ_MODEL")
            or os.getenv("OPENROUTER_MODEL")
            or os.getenv("NVIDIA_MODEL")
            or ""
        )
        if not model:
            raise ValueError(
                "build_optimized_improver needs an LLM model. "
                "Pass lm_model_string=... or set LLM_MODEL "
                "(+ matching LLM_API_KEY / LLM_BASE_URL) in the environment."
            )
        api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or ""
        )
        api_base = (
            os.getenv("LLM_BASE_URL")
            or os.getenv("NVIDIA_BASE_URL")
            or os.getenv("GROQ_BASE_URL")
            or os.getenv("OPENROUTER_BASE_URL")
            or None
        )
        lm_kwargs: dict = {"num_retries": 3}
        if api_key:
            lm_kwargs["api_key"] = api_key
        if api_base:
            lm_kwargs["api_base"] = api_base
        lm = dspy.LM(model, **lm_kwargs)
    dspy.configure(lm=lm)

    metric = ImprovementMetric(
        inspector      = inspector,
        company_policy = company_policy,
        regulations    = regulations,
    )

    _bound_policy = company_policy
    _bound_regs   = regulations

    # Bind policy context so the optimiser doesn't need to pass it per-call
    class BoundImprover(PromptOptimizer):
        async def forward(
            self,
            agent_profile:    InspectedAgentProfile,
            policies:         str = "",
            regulations:      str = "",
        ) -> OptimizedResult:
            return await super().forward(
                agent_profile = agent_profile,
                policies      = policies or _bound_policy,
                regulations   = regulations or _bound_regs,
            )
    optimizer = dspy.BootstrapFewShot(
        metric                 = metric,
        max_bootstrapped_demos = 4,
        max_labeled_demos      = 2,
    )
    compiled: Any = optimizer.compile(BoundImprover(), trainset=trainset)
    return compiled