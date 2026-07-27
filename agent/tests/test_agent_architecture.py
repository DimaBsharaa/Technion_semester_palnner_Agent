"""
Architectural proof: the shipped agent (react mode) makes genuine per-step
OpenAI tool-calling decisions, and its step count / tool sequence vary with
input complexity - it is not a hardcoded draft/verify/repair/compare/explain
pipeline with a fixed shape, even though a pipeline implementation
(agent_loop.run_agent_turn) also exists in this codebase for comparison.

Runs in-process (no server needed, but agent/.env must have OPENAI_API_KEY).
Cost: ~3 live model calls, roughly $0.10-0.25 total (see agent/TESTING.md
Level 1 estimate: ~$0.05/turn).
Run: cd agent && python3 tests/test_agent_architecture.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_loop import (  # noqa: E402
    build_state_from_intake,
    get_llm_trace,
    run_agent_turn,
    start_llm_trace,
)
from agent_react_loop import run_agent_turn_v2  # noqa: E402
from test_scenarios import SCENARIOS, TRACK_ID  # noqa: E402

TRIVIAL = next(s for s in SCENARIOS if s["name"] == "clean_normal_pace")
COMPLEX = next(s for s in SCENARIOS if s["name"] == "heavily_constrained_weekdays")


def _run(implementation, scenario) -> dict:
    start_llm_trace()
    state = build_state_from_intake(scenario["intake"])
    result = implementation(TRACK_ID, messages=[], cost_so_far=0.0, state_override=state)
    result["llm_steps"] = get_llm_trace()
    return result


def test_react_makes_real_tool_calls() -> list[str]:
    failures = []
    result = _run(run_agent_turn_v2, TRIVIAL)
    planning_steps = [s for s in result["llm_steps"] if s["module"] == "PlanningLoop"]
    if not planning_steps:
        failures.append("no 'PlanningLoop' steps found in react mode's llm_steps trace")
        return failures
    tool_calling_steps = [s for s in planning_steps if s["response"].get("tool_calls")]
    if not tool_calling_steps:
        failures.append(
            "no PlanningLoop step had a non-empty 'tool_calls' list - the model never made "
            "a real OpenAI tool-calling decision"
        )
    return failures


def test_pipeline_never_uses_tool_calling() -> list[str]:
    failures = []
    result = _run(run_agent_turn, TRIVIAL)
    steps_with_tool_calls = [s for s in result["llm_steps"] if s["response"].get("tool_calls")]
    if steps_with_tool_calls:
        failures.append(
            f"pipeline mode produced {len(steps_with_tool_calls)} step(s) with tool_calls - "
            "expected zero, since agent_loop.run_agent_turn never passes tools=[...]"
        )
    return failures


def test_react_step_count_and_tool_sequence_vary_with_complexity() -> list[str]:
    failures = []
    trivial_result = _run(run_agent_turn_v2, TRIVIAL)
    complex_result = _run(run_agent_turn_v2, COMPLEX)

    trivial_steps = sum(1 for s in trivial_result["llm_steps"] if s["module"] == "PlanningLoop")
    complex_steps = sum(1 for s in complex_result["llm_steps"] if s["module"] == "PlanningLoop")

    trivial_tool_seq = tuple(t["name"] for t in trivial_result["tool_log"])
    complex_tool_seq = tuple(t["name"] for t in complex_result["tool_log"])

    if trivial_steps == complex_steps and trivial_tool_seq == complex_tool_seq:
        failures.append(
            f"identical step count ({trivial_steps}) AND identical tool sequence for a trivial "
            f"vs. heavily-constrained scenario - suspicious for a model-driven loop; "
            f"trace: {trivial_tool_seq}"
        )
    return failures


def test_health_default_mode_is_react() -> list[str]:
    import main  # noqa: E402 - only inspects module-level constants, no server needed

    failures = []
    if main.AGENT_MODE_DEFAULT != "react":
        failures.append(
            f"AGENT_MODE_DEFAULT is {main.AGENT_MODE_DEFAULT!r}, expected 'react' - the shipped "
            "default must be the agent, not agent_loop's pipeline"
        )
    if main.AGENT_IMPLEMENTATIONS.get("react") is not run_agent_turn_v2:
        failures.append("main.AGENT_IMPLEMENTATIONS['react'] is not agent_react_loop.run_agent_turn_v2")
    return failures


def run_all_checks() -> int:
    checks = [
        test_health_default_mode_is_react,  # $0, run first
        test_react_makes_real_tool_calls,
        test_pipeline_never_uses_tool_calling,
        test_react_step_count_and_tool_sequence_vary_with_complexity,
    ]
    total_failures = 0
    for check in checks:
        name = check.__name__
        print(f"--- {name} ---")
        try:
            failures = check()
        except Exception as e:
            failures = [f"raised {type(e).__name__}: {e}"]
        if failures:
            total_failures += len(failures)
            for f in failures:
                print(f"  FAIL: {f}")
        else:
            print("  PASS")
    print()
    if total_failures:
        print(f"{total_failures} failure(s) - this system may not be a real agent.")
        return 1
    print("All architecture checks passed: react mode is a genuine tool-calling agent.")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_checks())
