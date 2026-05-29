"""
prompt_improver.py
──────────────────
DSPy-based module that consumes an AgentProfile from AgentSentinel's
InspectorAgent and automatically rewrites the agent's system prompt
to resolve every risk flag, while respecting company policy and
government regulations.

Usage:
    import asyncio
    import dspy
    from prompt_improver import PromptImprover, build_optimized_improver
    from agentsentinel.core.agents.inspector import InspectorAgent

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
from agentsentinel.models.agent import InspectedAgentProfile

import asyncio
import json

import dspy

from agentsentinel.models.agent import (
    RiskCategory,
    ToolProfile,
)

from agentsentinel.core.agents.improver.signatures import (
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

from agentsentinel.core.agents.improver.policy_guard import PolicyGuard
from agentsentinel.models.prompt import ImprovementResult, ChangeLogEntry
import logging
import difflib

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
    changes = "\n".join(
        f"    {i+1}. [{e.field}] {e.reason}"
        for i, e in enumerate(result.change_log)
    ) or "    (none)"
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

def _safe_call(fn, label: str, change_log: list[ChangeLogEntry], **kwargs):
    """
    Calls a DSPy ChainOfThought module and returns the result.
    If the model returns None text (common with reasoning models like
    Nvidia Nemotron that separate reasoning_content from text), or if
    DSPy fails to parse the response, returns None instead of crashing.
    The caller keeps the prompt unchanged and logs the skip.
    """
    _DSPY_INTERNAL = frozenset({"reasoning", "rationale"})

    try:
        result = fn(**kwargs)

        for field_name, field_value in vars(result).items():
            if field_name.startswith("_") or field_name in _DSPY_INTERNAL:
                continue
            if field_value is None:
                change_log.append(ChangeLogEntry(
                    field="system_prompt",
                    before="",
                    after="",
                    reason=(
                        f"[{label}] ⚠ Skipped — model returned None for field "
                        f"'{field_name}'. Known limitation of reasoning models "
                        f"(e.g. Nemotron). Original prompt kept."
                    ),
                ))
                return None

        return result

    except Exception as exc:
        change_log.append(ChangeLogEntry(
            field="system_prompt",
            before="",
            after="",
            reason=(
                f"[{label}] ⚠ Skipped — model response could not be parsed "
                f"({type(exc).__name__}: {exc}). Original prompt kept."
            ),
        ))
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
        change_log: list[ChangeLogEntry] = []
        flags = [f.category.value for f in agent_profile.risk_flags]

        logger.info("Improvement started — agent: %s | risk flags: %s",
                    agent_profile.agent_id or "(unnamed)", flags or "(none)")

        # ── Sequential: injection + persona first (foundation) ────────────────

        if _has_flag(agent_profile, RiskCategory.INJECTION_VULNERABLE):
            logger.info("Fixing INJECTION_VULNERABLE...")
            before_prompt = prompt
            r = _safe_call(
                self.fix_injection, "INJECTION_VULNERABLE", change_log,
                original_prompt   = prompt,
                injection_surface = agent_profile.injection_surface.value,
            )
            if r:
                prompt = r.improved_prompt
                change_log.append(ChangeLogEntry(
                    field="system_prompt",
                    before=before_prompt[:300],
                    after=prompt[:300],
                    reason=f"INJECTION_VULNERABLE — {r.defences_added}",
                ))
                logger.info("INJECTION_VULNERABLE fixed.")
                logger.debug("INJECTION_VULNERABLE before: %s", before_prompt[:80])
                logger.debug("INJECTION_VULNERABLE after:  %s", prompt[:80])
            else:
                logger.warning("INJECTION_VULNERABLE skipped — see change log.")

        if _has_flag(agent_profile, RiskCategory.PERSONA_DRIFT):
            logger.info("Fixing PERSONA_DRIFT...")
            before_prompt = prompt
            r = _safe_call(
                self.fix_persona, "PERSONA_DRIFT", change_log,
                original_prompt = prompt,
            )
            if r:
                prompt = r.improved_prompt
                change_log.append(ChangeLogEntry(
                    field="system_prompt",
                    before=before_prompt[:300],
                    after=prompt[:300],
                    reason=f"PERSONA_DRIFT — {r.persona_section}",
                ))
                logger.info("PERSONA_DRIFT fixed.")
                logger.debug("PERSONA_DRIFT before: %s", before_prompt[:80])
                logger.debug("PERSONA_DRIFT after:  %s", prompt[:80])
            else:
                logger.warning("PERSONA_DRIFT skipped — see change log.")

        # ── Parallel fixes ────────────────────────────────────────────────────

        async def run_fix(fix_fn, label, **kwargs):
            """Run a DSPy fix in a thread. Returns (label, result, task_log) with isolated log."""
            task_log: list[ChangeLogEntry] = []
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

        parallel_results = await asyncio.gather(*parallel_tasks)

        partial_prompts = []
        for label, r, task_log in parallel_results:
            change_log.extend(task_log)
            if r:
                partial_prompts.append(r.improved_prompt)
                change_log.append(ChangeLogEntry(
                    field="system_prompt",
                    before=prompt[:300],
                    after=r.improved_prompt[:300],
                    reason=f"{label} — applied",
                ))
                logger.info("%s fixed.", label)
                logger.debug("%s before: %s", label, prompt[:80])
                logger.debug("%s after:  %s", label, r.improved_prompt[:80])
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
            before_prompt = prompt
            r = _safe_call(
                self.fix_memory, "MEMORY_RISK", change_log,
                original_prompt = prompt,
                memory_risks    = agent_profile.memory_risks,
                company_policy  = policies,
                regulations     = regulations,
            )
            if r:
                prompt = r.improved_prompt
                change_log.append(ChangeLogEntry(
                    field="system_prompt",
                    before=before_prompt[:300],
                    after=prompt[:300],
                    reason=f"MEMORY_RISK — {r.added_rules}",
                ))
                logger.info("MEMORY_RISK fixed.")
                logger.debug("MEMORY_RISK before: %s", before_prompt[:80])
                logger.debug("MEMORY_RISK after:  %s", prompt[:80])
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
                before_prompt = prompt
                prompt = r.improved_prompt
                change_log.append(ChangeLogEntry(
                    field="system_prompt",
                    before=before_prompt[:300],
                    after=prompt[:300],
                    reason=f"POLICY_VIOLATION — {r.changes_made}",
                ))
                logger.info("POLICY_VIOLATION fixed.")
                logger.debug("POLICY_VIOLATION before: %s", before_prompt[:80])
                logger.debug("POLICY_VIOLATION after:  %s", prompt[:80])
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

        original_lines = agent_profile.system_prompt.splitlines()
        improved_lines = prompt.splitlines()
        diff = "\n".join(difflib.unified_diff(
            original_lines,
            improved_lines,
            fromfile="original",
            tofile="improved",
            lineterm="",
        ))

        changed_lines = sum(1 for line in diff.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
        if diff:
            logger.info("Diff: %d line(s) changed", changed_lines)
            logger.debug("Diff output:\n%s", diff)
        else:
            logger.info("Diff: no changes — prompt unchanged after all fixes.")

        result = ImprovementResult(
            improved_prompt           = prompt,
            improved_tool_definitions = improved_tools,
            change_log                = change_log,
            policy_violations         = violations,
            diff                      = diff,
        )
        logger.info("Improvement complete: %s", _fmt_result(result))
        return result

    async def _merge_prompts(
            self,
            original_prompt: str,
            partial_prompts: list[str],
            change_log: list[ChangeLogEntry],
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
            change_log.append(ChangeLogEntry(
                field="system_prompt",
                before=original_prompt[:300],
                after=result.merged_prompt[:300],
                reason=f"MERGE — {len(partial_prompts)} parallel fix(es) merged",
            ))
            logger.debug("MERGE before: %s", original_prompt[:80])
            logger.debug("MERGE after:  %s", result.merged_prompt[:80])
            return result.merged_prompt
        change_log.append(ChangeLogEntry(
            field="system_prompt",
            before="",
            after="",
            reason=f"MERGE failed — all {len(partial_prompts)} parallel fix(es) abandoned, original retained",
        ))
        return original_prompt

    async def _fix_tools_parallel(
        self,
        tool_definitions: list[dict],
        agent_profile:    InspectedAgentProfile,
        change_log:       list[ChangeLogEntry],
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
                original_desc = tool_def.get("description", "")
                try:
                    existing_params = dict(tool_def.get("parameters") or {})
                    existing_params["properties"] = json.loads(r.improved_parameters)
                    change_log.append(ChangeLogEntry(
                        field=f"tool:{name}",
                        before=original_desc[:200],
                        after=r.improved_description[:200],
                        reason=f"TOOL_QUALITY_LOW — score was {tool_profile.quality_score}/10",
                    ))
                    logger.debug("tool:%s before: %s", name, original_desc[:80])
                    logger.debug("tool:%s after:  %s", name, r.improved_description[:80])
                    return {
                        **tool_def,
                        "description": r.improved_description,
                        "parameters": existing_params,
                    }
                except json.JSONDecodeError:
                    change_log.append(ChangeLogEntry(
                        field=f"tool:{name}",
                        before=original_desc[:200],
                        after=r.improved_description[:200],
                        reason=f"TOOL_QUALITY_LOW — score was {tool_profile.quality_score}/10; parameter JSON invalid, kept original",
                    ))
                    logger.debug("tool:%s before: %s", name, original_desc[:80])
                    logger.debug("tool:%s after:  %s (params kept original)", name, r.improved_description[:80])
                    return {
                        **tool_def,
                        "description": r.improved_description,
                    }
            return tool_def
        return list(await asyncio.gather(*[fix_one_tool(t) for t in tool_definitions]))
