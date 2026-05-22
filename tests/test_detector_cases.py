"""
Run all detector test cases and show what was extracted vs what was expected.

Usage: uv run tests/test_detector_cases.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentsentinal.intake.detectors.langgraph import LangGraphDetector
from demo.detector_test_agents import TEST_CASES

GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD  = "\033[1m"


def check(label: str, passed: bool) -> str:
    icon = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    return f"  [{icon}] {label}"


def run_case(case: dict) -> None:
    print(f"\n{BOLD}Test {case['id']}: {case['name']}{RESET}")
    print(f"  Known failure: {YELLOW}{case['known_failure']}{RESET}")

    try:
        agent = case["factory"]()
    except Exception as e:
        print(f"  {RED}Agent construction failed: {e}{RESET}")
        return

    detector = LangGraphDetector(agent)

    # can_handle
    can_handle = detector.can_handle()
    print(check("can_handle() = True", can_handle))

    if not can_handle:
        print(f"  {YELLOW}Skipping extraction — can_handle returned False{RESET}")
        return

    profile = detector()

    # system_prompt
    has_prompt = bool(profile.system_prompt and len(profile.system_prompt.strip()) > 5)
    if case["expects_prompt"]:
        print(check(
            f"system_prompt extracted  (got: {repr(profile.system_prompt[:60])}...)",
            has_prompt,
        ))
    else:
        print(f"  [----] system_prompt not expected for this case")

    # tool_definitions
    has_tools = bool(profile.tool_definitions)
    has_params = any(
        bool(t.get("parameters")) for t in profile.tool_definitions
    )
    if case["expects_tools"]:
        print(check(
            f"tools extracted          (got {len(profile.tool_definitions)} tool(s))",
            has_tools,
        ))
        print(check(
            f"tool parameters non-empty",
            has_params,
        ))
    else:
        print(f"  [----] tools not expected for this case")

    # warnings
    if profile.warnings:
        for w in profile.warnings:
            print(f"  {YELLOW}warn:{RESET} {w}")


def main():
    print(f"{BOLD}LangGraphDetector — Failure Test Suite{RESET}")
    print("=" * 55)

    for case in TEST_CASES:
        run_case(case)

    print(f"\n{'=' * 55}")
    print("Done. FAIL = detector produced wrong/empty output for that field.")


if __name__ == "__main__":
    main()
