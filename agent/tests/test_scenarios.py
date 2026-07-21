"""
Regression suite for run_agent_turn - a small set of hand-verified student
scenarios, each submitted via the same structured-intake path the frontend's
gap-fill panel uses (build_state_from_intake), so scenario setup is fully
deterministic and only the live planning calls (draft/repair/compare) are
exercised against the real model. Run directly:

    cd agent && python3 tests/test_scenarios.py

This is the regression net for the ReAct-loop rewrite: it catches the class
of bug found via live testing this session (a course flagged as a retake
that the deterministic backfill also silently marked "already passed"),
which tools.check_invariants now catches as a hard backstop regardless of
what the drafting model does or which loop implementation produced the plan.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools  # noqa: E402
from agent_loop import SESSION_BUDGET_USD_CAP, build_state_from_intake, run_agent_turn  # noqa: E402
from agent_react_loop import run_agent_turn_v2  # noqa: E402
from data_bundle import get_track  # noqa: E402

TRACK_ID = "SC00001453"  # Data & Information Engineering - has real fetched data

IMPLEMENTATIONS = {"pipeline": run_agent_turn, "react": run_agent_turn_v2}

SCENARIOS = [
    {
        "name": "clean_normal_pace",
        "intake": {
            "semester_number": 4,
            "excluded_weekdays": [2],
            "pace": "normal",
            "passed_courses": [],
            "failed_courses": [],
        },
    },
    {
        # The exact scenario that surfaced the passed/failed contradiction
        # bug: a semester-5 student failing a course that is itself a
        # prerequisite of other mandatory courses assumed already passed.
        "name": "retake_that_is_also_a_foundation",
        "intake": {
            "semester_number": 5,
            "excluded_weekdays": [],
            "pace": "normal",
            "passed_courses": [],
            "failed_courses": ["00940224"],  # Data Structures and Algorithms
        },
    },
    {
        "name": "light_pace",
        "intake": {
            "semester_number": 4,
            "excluded_weekdays": [],
            "pace": "light",
            "passed_courses": [],
            "failed_courses": [],
        },
    },
    {
        "name": "override_minimums",
        "intake": {
            "semester_number": 6,
            "excluded_weekdays": [],
            "pace": "normal",
            "passed_courses": [],
            "failed_courses": [],
            "override_minimums": True,
        },
    },
    {
        "name": "heavily_constrained_weekdays",
        "intake": {
            "semester_number": 4,
            "excluded_weekdays": [0, 1, 2, 3],  # only Wed/Thu/Fri left
            "pace": "normal",
            "passed_courses": [],
            "failed_courses": [],
        },
    },
    {
        "name": "near_graduation",
        "intake": {
            "semester_number": 8,
            "excluded_weekdays": [],
            "pace": "normal",
            "passed_courses": [],
            "failed_courses": [],
        },
    },
]


def run_scenario(scenario: dict, implementation) -> tuple[list[str], dict]:
    """Returns (failure descriptions, the raw result) - empty failures means
    the scenario passed. The raw result is returned too so the caller can
    print cost/stopped_reason for comparison between implementations."""
    failures = []
    track = get_track(TRACK_ID)
    state = build_state_from_intake(scenario["intake"])
    result = implementation(TRACK_ID, messages=[], cost_so_far=0.0, state_override=state)

    if result["cost_usd"] > SESSION_BUDGET_USD_CAP * 1.2:
        failures.append(f"cost {result['cost_usd']:.4f} well over budget cap {SESSION_BUDGET_USD_CAP}")

    if result["needs_input"] is not None:
        failures.append("unexpectedly asked for more input despite a fully-structured intake")
        return failures, result

    plan_result = result["plan_result"]
    if plan_result is None:
        failures.append("no plan_result and no needs_input - the turn produced nothing")
        return failures, result

    plan_course_numbers = [c["course_number"] for c in plan_result["courses"]]
    intake = scenario["intake"]

    violations = tools.check_invariants(
        track, plan_course_numbers, intake["passed_courses"], intake["failed_courses"]
    )
    failures.extend(violations)

    for failed_course in intake["failed_courses"]:
        if failed_course not in plan_course_numbers:
            failures.append(f"failed course {failed_course} was not retaken in the plan - violates priority rule #1")

    if not plan_course_numbers:
        failures.append("empty plan produced")

    return failures, result


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "pipeline"
    if mode not in IMPLEMENTATIONS:
        print(f"Unknown mode {mode!r}. Valid: {sorted(IMPLEMENTATIONS)}")
        return 2
    implementation = IMPLEMENTATIONS[mode]
    print(f"=== mode: {mode} ===\n")

    total_failures = 0
    for scenario in SCENARIOS:
        print(f"--- {scenario['name']} ---")
        try:
            failures, result = run_scenario(scenario, implementation)
            print(f"  cost=${result['cost_usd']:.4f} stopped_reason={result['stopped_reason']}")
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
        print(f"{total_failures} failure(s) across {len(SCENARIOS)} scenarios ({mode})")
        return 1
    print(f"All {len(SCENARIOS)} scenarios passed ({mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
