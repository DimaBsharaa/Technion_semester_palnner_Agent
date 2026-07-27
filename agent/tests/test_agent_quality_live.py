"""
Live plan-quality proof: 'the agent runs and gives good plans' per
agent/TESTING.md's acceptance contract, focused on what the six-scenario
regression suite (test_scenarios.py) does NOT cover - multi-turn revision
continuity (TESTING.md Level 3.2, exam-conflict revision).

Run test_scenarios.py react separately for the broader one-shot quality net
(~$0.20, six scenarios); this file adds one stateful follow-up scenario
(~$0.10-0.15, 2 live calls).

Run: cd agent && python3 tests/test_agent_quality_live.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools  # noqa: E402
from agent_loop import build_state_from_intake  # noqa: E402
from agent_react_loop import run_agent_turn_v2  # noqa: E402
from data_bundle import get_track  # noqa: E402
from test_scenarios import SCENARIOS, TRACK_ID  # noqa: E402

SEED_SCENARIO = next(s for s in SCENARIOS if s["name"] == "retake_that_is_also_a_foundation")


def _first_exam_date(plan_result: dict):
    for course in plan_result["courses"]:
        moed_a = course.get("exams", {}).get("moed_a") or []
        if moed_a:
            return course["course_number"], moed_a[0]["date"]
    return None


def test_exam_conflict_revision_is_minimal_and_settled() -> list[str]:
    failures = []
    track = get_track(TRACK_ID)
    state = build_state_from_intake(SEED_SCENARIO["intake"])

    first = run_agent_turn_v2(TRACK_ID, messages=[], cost_so_far=0.0, state_override=state)
    if first["plan_result"] is None:
        return ["initial turn produced no plan_result - cannot test revision"]

    target = _first_exam_date(first["plan_result"])
    if target is None:
        return ["initial plan had no course with a moed_a exam date to conflict with"]
    conflict_course, conflict_date = target
    conflict_name = track.courses[conflict_course]["name"]
    original_courses = {c["course_number"] for c in first["plan_result"]["courses"]}

    followup_text = (
        f"I have a wedding on {conflict_date} and can't sit the {conflict_name} exam that day - "
        "please fix it, keep the rest of the plan the same."
    )
    second = run_agent_turn_v2(
        TRACK_ID,
        messages=[{"role": "user", "content": followup_text}],
        cost_so_far=first["cost_usd"],
        known_context=first["known_context"],
    )

    if second["needs_input"] is not None:
        failures.append(
            f"revision turn re-asked for input ({second['needs_input']['missing_fields']}) instead "
            "of treating the message as feedback on the existing plan"
        )
    if second["plan_result"] is None:
        return failures + ["revision turn produced no plan_result"]

    new_courses = {c["course_number"] for c in second["plan_result"]["courses"]}
    if conflict_course in new_courses:
        failures.append(
            f"conflicting course {conflict_course} ({conflict_name}) is still in the revised plan "
            f"unchanged despite the reported exam-date conflict on {conflict_date}"
        )

    kept = original_courses & new_courses
    min_expected_overlap = max(1, (len(original_courses) - 1) // 2)
    if len(kept) < min_expected_overlap:
        failures.append(
            f"only {len(kept)}/{len(original_courses)} original courses kept after a single "
            f"course-swap request - looks like a from-scratch replan, not a minimal revision "
            f"(kept: {sorted(kept)}, original: {sorted(original_courses)})"
        )

    violations = tools.check_invariants(
        track,
        list(new_courses),
        SEED_SCENARIO["intake"]["passed_courses"],
        SEED_SCENARIO["intake"]["failed_courses"],
    )
    failures.extend(violations)

    return failures


def main() -> int:
    checks = [test_exam_conflict_revision_is_minimal_and_settled]
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
        print(f"{total_failures} failure(s).")
        return 1
    print("Revision-continuity check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
