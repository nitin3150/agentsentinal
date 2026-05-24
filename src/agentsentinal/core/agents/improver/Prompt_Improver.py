"""
prompt_improver.py
──────────────────
DSPy-based module that consumes an AgentProfile from AgentSentinal's
InspectorAgent and automatically rewrites the agent's system prompt
to resolve every risk flag, while respecting company policy and
government regulations.

Usage:
    import asyncio
    import dspy
    from prompt_improver import PromptImprover, build_optimized_improver
    from agentsentinal.core.agents.inspector import InspectorAgent

    dspy.configure(lm=dspy.LM("openai/gpt-4o"))

    inspector = InspectorAgent(semantic_enabled=False)
    profile   = asyncio.run(inspector.inspect(
        agent_id="my-agent",
        system_prompt=my_prompt,
        tool_definitions=my_tools,
    ))

    improver = PromptImprover()
    result   = improver(
        original_prompt  = my_prompt,
        tool_definitions = my_tools,
        agent_profile    = profile,
        company_policy   = policy_text,
        regulations      = regulations_text,
    )

    print(result.improved_prompt)
    print(result.improved_tool_definitions)   # drop-in replacement for your tools list
    print(result.change_log)
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from agentsentinal.models.agent import InspectedAgentProfile

import asyncio
import json

import dspy

from agentsentinal.models.agent import (
    RiskCategory,
    ToolProfile,
)

from agentsentinal.core.agents.improver.signatures import (
    FixAmbiguousInstructions,
    FixConstraintsMissing,
    FixHallucinationProne,
    FixInjectionVulnerable,
    FixMemoryRisk,
    FixPersonaDrift,
    FixPolicyViolation,
    FixScopeOverflow,
    FixToolQuality,
    MergePromptSections,
)

from agentsentinal.core.agents.improver.policy_guard import PolicyGuard
from agentsentinal.models.prompt import ImprovementResult
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _has_flag(profile: InspectedAgentProfile, category: RiskCategory) -> bool:
    return any(f.category == category for f in profile.risk_flags)


def _low_quality_tools(profile: InspectedAgentProfile) -> list[ToolProfile]:
    return [t for t in profile.tool_profiles if t.quality_score < 7]


def _fmt_result(result: "ImprovementResult") -> str:
    violations = result.policy_violations or ["none"]
    changes = "\n".join(f"    {i+1}. {c}" for i, c in enumerate(result.change_log)) or "    (none)"
    tools = "\n".join(
        f"    • {t.get('name', '?')} — {t.get('description', '')[:80]}"
        for t in result.improved_tool_definitions
    ) or "    (none)"
    return (
        f"\n  --- Improved Prompt ---"
        f"\n  {result.improved_prompt}"
        f"\n  --- Tools ({len(result.improved_tool_definitions)}) ---\n{tools}"
        f"\n  --- Change Log ({len(result.change_log)}) ---\n{changes}"
        f"\n  --- Policy Violations ---"
        f"\n  {', '.join(violations)}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 3.  MAIN MODULE
# ─────────────────────────────────────────────────────────────────────────────

def _safe_call(fn, label: str, change_log: list[str], **kwargs):
    """
    Calls a DSPy ChainOfThought module and returns the result.
    If the model returns None text (common with reasoning models like
    Nvidia Nemotron that separate reasoning_content from text), or if
    DSPy fails to parse the response, returns None instead of crashing.
    The caller keeps the prompt unchanged and logs the skip.
    """
    # DSPy ChainOfThought adds these fields internally — they can be None on
    # reasoning models without indicating a failed call.
    _DSPY_INTERNAL = frozenset({"reasoning", "rationale"})

    try:
        result = fn(**kwargs)

        for field_name, field_value in vars(result).items():
            if field_name.startswith("_") or field_name in _DSPY_INTERNAL:
                continue
            if field_value is None:
                change_log.append(
                    f"[{label}] ⚠ Skipped — model returned None for field "
                    f"'{field_name}'. This is a known limitation of reasoning "
                    f"models (e.g. Nemotron) that return reasoning_content "
                    f"instead of text. The original prompt was kept for this step."
                )
                return None

        return result

    except Exception as exc:
        change_log.append(
            f"[{label}] ⚠ Skipped — model response could not be parsed "
            f"({type(exc).__name__}: {exc}). "
            f"This is a known limitation of reasoning models (e.g. Nemotron) "
            f"that return reasoning_content instead of text. "
            f"The original prompt was kept for this step."
        )
        return None

class PromptImprover(dspy.Module):
    """
    Orchestrates all fix signatures.
    Only applies a fix when the Inspector has flagged that risk category.
    Fixes are applied sequentially so each step builds on the previous one.

    Each fix is wrapped in _safe_call so that reasoning models (e.g. Nvidia
    Nemotron) that return None text don't crash the pipeline — they log a
    warning and keep the prompt unchanged for that step.
    """

    def __init__(self):
        super().__init__()
        self.fix_constraints   = dspy.ChainOfThought(FixConstraintsMissing)
        self.fix_ambiguity     = dspy.ChainOfThought(FixAmbiguousInstructions)
        self.fix_scope         = dspy.ChainOfThought(FixScopeOverflow)
        self.fix_hallucination = dspy.ChainOfThought(FixHallucinationProne)
        self.fix_injection     = dspy.ChainOfThought(FixInjectionVulnerable)
        self.fix_persona       = dspy.ChainOfThought(FixPersonaDrift)
        self.fix_memory        = dspy.ChainOfThought(FixMemoryRisk)
        self.fix_policy        = dspy.ChainOfThought(FixPolicyViolation)
        self.fix_tool          = dspy.ChainOfThought(FixToolQuality)
        self.merge_prompts     = dspy.ChainOfThought(MergePromptSections)
    

    def __call__(self, *args, **kwargs):  # type: ignore[override]
        coro = self.forward(*args, **kwargs)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    async def forward(
        self,
        agent_profile:    InspectedAgentProfile,
        policies:   str = "",
        regulations:      str = "",
    ) -> ImprovementResult:

        tool_definitions = agent_profile.tool_definitions or []
        prompt     = agent_profile.system_prompt
        change_log: list[str] = []
        flags = [f.category.value for f in agent_profile.risk_flags]

        logger.info("Improvement started — agent: %s | risk flags: %s",
                    agent_profile.agent_id or "(unnamed)", flags or "(none)")

        # ── Sequential: injection + persona first (foundation) ────────────────

        if _has_flag(agent_profile, RiskCategory.INJECTION_VULNERABLE):
            logger.info("Fixing INJECTION_VULNERABLE...")
            r = _safe_call(
                self.fix_injection, "INJECTION_VULNERABLE", change_log,
                original_prompt   = prompt,
                injection_surface = agent_profile.injection_surface.value,
            )
            if r:
                prompt = r.improved_prompt
                change_log.append(f"[INJECTION_VULNERABLE] {r.defences_added}")
                logger.info("INJECTION_VULNERABLE fixed.")
            else:
                logger.warning("INJECTION_VULNERABLE skipped — see change log.")

        if _has_flag(agent_profile, RiskCategory.PERSONA_DRIFT):
            logger.info("Fixing PERSONA_DRIFT...")
            r = _safe_call(
                self.fix_persona, "PERSONA_DRIFT", change_log,
                original_prompt = prompt,
            )
            if r:
                prompt = r.improved_prompt
                change_log.append(f"[PERSONA_DRIFT] {r.persona_section}")
                logger.info("PERSONA_DRIFT fixed.")
            else:
                logger.warning("PERSONA_DRIFT skipped — see change log.")

        # ── Parallel fixes ────────────────────────────────────────────────────

        async def run_fix(fix_fn, label, **kwargs):
            """Run a DSPy fix in a thread. Returns (label, result, task_log) with isolated log."""
            task_log: list[str] = []
            result = await asyncio.to_thread(_safe_call, fix_fn, label, task_log, **kwargs)
            return label, result, task_log

        parallel_tasks = []

        if _has_flag(agent_profile, RiskCategory.CONSTRAINT_MISSING):
            parallel_tasks.append(run_fix(
                self.fix_constraints, "CONSTRAINT_MISSING",
                original_prompt = prompt,
                company_policy  = policies,
                regulations     = regulations,
            ))

        if _has_flag(agent_profile, RiskCategory.AMBIGUOUS_INSTRUCTIONS):
            parallel_tasks.append(run_fix(
                self.fix_ambiguity, "AMBIGUOUS_INSTRUCTIONS",
                original_prompt   = prompt,
                ambiguous_phrases = agent_profile.ambiguous_phrases,
            ))

        if _has_flag(agent_profile, RiskCategory.SCOPE_OVERFLOW):
            parallel_tasks.append(run_fix(
                self.fix_scope, "SCOPE_OVERFLOW",
                original_prompt = prompt,
                company_policy  = policies,
            ))

        if _has_flag(agent_profile, RiskCategory.HALLUCINATION_PRONE):
            parallel_tasks.append(run_fix(
                self.fix_hallucination, "HALLUCINATION_PRONE",
                original_prompt = prompt,
            ))

        if parallel_tasks:
            logger.info("Running %d parallel fix(es): %s",
                        len(parallel_tasks),
                        [t.__name__ if hasattr(t, '__name__') else str(i)
                         for i, t in enumerate(parallel_tasks)])
            logger.info("Running parallel fixes...")

        parallel_results = await asyncio.gather(*parallel_tasks)

        partial_prompts = []
        for label, r, task_log in parallel_results:
            change_log.extend(task_log)
            if r:
                partial_prompts.append(r.improved_prompt)
                change_log.append(f"[{label}] Applied.")
                logger.info("%s fixed.", label)
            else:
                logger.warning("%s skipped — see change log.", label)

        if partial_prompts:
            logger.info("Merging %d parallel fix(es)...", len(partial_prompts))
            prompt = await self._merge_prompts(
                original_prompt = prompt,
                partial_prompts = partial_prompts,
                change_log      = change_log,
            )

        if _has_flag(agent_profile, RiskCategory.MEMORY_RISK):
            logger.info("Fixing MEMORY_RISK...")
            r = _safe_call(
                self.fix_memory, "MEMORY_RISK", change_log,
                original_prompt = prompt,
                memory_risks    = agent_profile.memory_risks,
                company_policy  = policies,
                regulations     = regulations,
            )
            if r:
                prompt = r.improved_prompt
                change_log.append(f"[MEMORY_RISK] {r.added_rules}")
                logger.info("MEMORY_RISK fixed.")
            else:
                logger.warning("MEMORY_RISK skipped — see change log.")

        # ── Policy violation fix ──────────────────────────────────────────────

        if _has_flag(agent_profile, RiskCategory.POLICY_VIOLATION) and policies:
            logger.info("Fixing POLICY_VIOLATION...")
            policy_violations = [
                f.description
                for f in agent_profile.risk_flags
                if f.category == RiskCategory.POLICY_VIOLATION
            ]
            r = _safe_call(
                self.fix_policy, "POLICY_VIOLATION", change_log,
                original_prompt=prompt,
                policy_text=policies[:6000],
                violations=policy_violations,
            )
            if r:
                prompt = r.improved_prompt
                change_log.append(f"[POLICY_VIOLATION] {r.changes_made}")
                logger.info("POLICY_VIOLATION fixed.")
            else:
                logger.warning("POLICY_VIOLATION skipped — see change log.")

        # ── Tool fixes ────────────────────────────────────────────────────────

        logger.info("Fixing tool definitions...")
        improved_tools = await self._fix_tools_parallel(
            tool_definitions = tool_definitions,
            agent_profile    = agent_profile,
            change_log       = change_log,
        )

        # ── Policy guard ──────────────────────────────────────────────────────

        logger.info("Running policy guard check...")
        guard      = PolicyGuard(company_policy=policies, regulations=regulations)
        violations = await asyncio.to_thread(guard.check, prompt)
        if violations:
            logger.warning("Policy violations detected: %s", violations)
        else:
            logger.info("Policy guard passed.")

        result = ImprovementResult(
            improved_prompt           = prompt,
            improved_tool_definitions = improved_tools,
            change_log                = change_log,
            policy_violations         = violations,
        )
        logger.info("Improvement complete: %s", _fmt_result(result))
        return result

    async def _merge_prompts(
            self,
            original_prompt: str,
            partial_prompts: list[str],
            change_log: list[str],
    ) -> str:
        """
        Merge multiple independently-fixed prompts into one coherent system prompt.
        Each partial_prompt fixed a different risk in isolation - the merger
        reconciles them without duplicating content or creating contradictions.
        """

        result = await asyncio.to_thread(
            _safe_call,
            self.merge_prompts, "MERGE", change_log,
            original_prompt=original_prompt,
            partial_prompts=partial_prompts,
        )
        if result:
            change_log.append("[MERGE] Parallel fixes merged successfully.")
            return result.merged_prompt
        change_log.append(
            f"[MERGE] Failed — all {len(partial_prompts)} parallel fix(es) abandoned. "
            "Original prompt (with sequential fixes) retained."
        )
        return original_prompt

    async def _fix_tools_parallel(
        self,
        tool_definitions: list[dict],
        agent_profile:    InspectedAgentProfile,
        change_log:       list[str],
    ) -> list[dict]:
        """
        Fix all low-quality tools concurrently.
        """
        if not _has_flag(agent_profile, RiskCategory.TOOL_QUALITY_LOW):
            logger.info("Tool quality OK — no tool fixes needed.")
            return list(tool_definitions)
        
        _low_quality = {t.name: t for t in _low_quality_tools(agent_profile)}

        async def fix_one_tool(tool_def: dict) -> dict:
            name = tool_def.get("name", "")
            if name not in _low_quality:
                return tool_def
            
            tool_profile = _low_quality[name]
            r = await asyncio.to_thread(
                _safe_call,
                self.fix_tool, f"TOOL_QUALITY_LOW:{name}", change_log,
                tool_name=name,
                original_description=tool_def.get("description", ""),
                missing_fields=tool_profile.missing_fields,
            )
            
            if r:
                logger.info("Tool '%s' rewritten (was %d/10).", name, tool_profile.quality_score)
                change_log.append(
                    f"[TOOL_QUALITY_LOW] '{name}' rewritten "
                    f"(was {tool_profile.quality_score}/10)."
                )
                try:
                    existing_params = dict(tool_def.get("parameters") or {})
                    existing_params["properties"] = json.loads(r.improved_parameters)
                    return {
                        **tool_def,
                        "description": r.improved_description,
                        "parameters": existing_params,
                    }
                except json.JSONDecodeError:
                    change_log.append(
                        f"[TOOL_QUALITY_LOW] '{name}' parameter JSON invalid, keeping original parameters."
                    )
                    return {
                        **tool_def,
                        "description": r.improved_description,
                    }
            return tool_def
        return list(await asyncio.gather(*[fix_one_tool(t) for t in tool_definitions]))
