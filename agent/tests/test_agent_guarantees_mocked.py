"""
Mocked (zero-API-cost) tests for the ReAct loop's correctness guarantees:
- diagnostics are pre-injected (no wasted model round-trips for them);
- the model is nudged to verify before it can deliver;
- a fallback delivers the BEST verified plan, never a worse later one;
- the delivered plan is always re-verified server-side.

    cd agent && python3 tests/test_agent_guarantees_mocked.py
"""

import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_react_loop as arl  # noqa: E402
from agent_loop import build_state_from_intake  # noqa: E402
from data_bundle import get_track  # noqa: E402

TRACK_ID = "SC00001453"
failures: list[str] = []


def check(cond: bool, desc: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}: {desc}")
    if not cond:
        failures.append(desc)


def tool_call(name, args, cid="c1"):
    return types.SimpleNamespace(
        id=cid, function=types.SimpleNamespace(name=name, arguments=json.dumps(args))
    )


def msg(tool_calls=None, content=None):
    return types.SimpleNamespace(content=content, tool_calls=tool_calls)


def base_state():
    return build_state_from_intake(
        {"semester_number": 4, "excluded_weekdays": [], "pace": "normal",
         "passed_courses": [], "failed_courses": []}
    )


def state_with_prereqs_for_strong():
    # STRONG/WEAK (below) include 00970249, which needs 00940224 + 00960570
    # as prerequisites - state this EXPLICITLY rather than relying on
    # backfill_passed_courses's auto-inference to happen to cover it, so
    # this test stays correct regardless of how that heuristic is tuned
    # (it was, and got deliberately more conservative after live testing
    # found it wrongly auto-assuming real mandatory courses were already
    # done - see tools.EXPECTED_BY_NOW_BUFFER).
    return build_state_from_intake(
        {"semester_number": 4, "excluded_weekdays": [], "pace": "normal",
         "passed_courses": ["00940224", "00960570"], "failed_courses": []}
    )


track = get_track(TRACK_ID)
# Two real, distinct course lists from this track for verify to chew on.
PLAN_A = ["00940290", "00940314", "00970447", "00960570", "00940241"]
PLAN_B = ["00940290", "00940314"]  # smaller -> should verify worse (low credits)


# --- Test 1: diagnostics pre-injected, not fetched as tool calls ---
print("--- diagnostics pre-injected ---")


def script_deliver_only(_messages):
    # Model immediately verifies then delivers PLAN_A.
    if not getattr(script_deliver_only, "verified", False):
        script_deliver_only.verified = True
        return msg([tool_call("verify_plan", {"plan_course_numbers": PLAN_A})]), 0.0
    return msg([tool_call("deliver_plan", {"course_numbers": PLAN_A, "explanation": "done"})]), 0.0


orig = arl._call_with_tools
arl._call_with_tools = script_deliver_only
try:
    res = arl.run_agent_turn_v2(TRACK_ID, [{"role": "user", "content": "plan me"}], 0.0, state_override=base_state())
    names = [t["name"] for t in res["tool_log"]]
    preinjected = [t for t in res["tool_log"] if t.get("args", {}).get("preinjected")]
    check({"assess_progress", "fetch_catalog", "query_prereq_graph"} <= set(names),
          "all three diagnostics appear in the trace")
    check(len(preinjected) == 3, "diagnostics are marked preinjected (computed in code, not model-fetched)")
    check(res["plan_result"] is not None, "produces a plan_result")
finally:
    arl._call_with_tools = orig


# --- Test 2: verify-before-deliver nudge fires when model skips verify ---
print("--- verify-before-deliver nudge ---")


def script_deliver_without_verify(_messages):
    # Always try to deliver immediately, never verify on its own.
    return msg([tool_call("deliver_plan", {"course_numbers": PLAN_A, "explanation": "done"})]), 0.0


arl._call_with_tools = script_deliver_without_verify
try:
    res = arl.run_agent_turn_v2(TRACK_ID, [{"role": "user", "content": "plan me"}], 0.0, state_override=base_state())
    names = [t["name"] for t in res["tool_log"]]
    check("verify_before_deliver_nudge" in names, "nudge fired when the model tried to deliver unverified")
    check(res["plan_result"] is not None, "still ultimately delivers a plan (nudge fires once, then proceeds)")
finally:
    arl._call_with_tools = orig


# --- Test 3: best-plan-wins on forced wrap-up (selection logic in isolation) ---
print("--- best verified plan wins a forced wrap-up ---")

# Two elective-only plans that survive the passed/failed backfill (they are
# never auto-marked passed), with verify_plan stubbed so STRONG passes and
# WEAK fails - isolating the "best wins, not most-recent" selection.
STRONG = ["03940804", "00970249", "03940582"]
WEAK = ["03940804"]

import tools as _tools  # noqa: E402

orig_verify = _tools.verify_plan
orig_inv = _tools.check_invariants


def stub_verify(track_, plan, passed_, **kw):
    if set(plan) == set(STRONG):
        return {"pass": True, "total_credits": 18.0, "workload_score": 60.0, "issues": []}
    return {"pass": False, "total_credits": 2.0, "workload_score": 10.0,
            "issues": [{"course": None, "reason": "too few credits"}]}


def script_verify_two_never_deliver(_messages):
    n = getattr(script_verify_two_never_deliver, "n", 0)
    script_verify_two_never_deliver.n = n + 1
    if n == 0:
        return msg([tool_call("verify_plan", {"plan_course_numbers": STRONG})]), 0.0
    if n == 1:
        return msg([tool_call("verify_plan", {"plan_course_numbers": WEAK})]), 0.0
    return msg([tool_call("check_invariants", {"plan_course_numbers": WEAK})]), 0.0


arl._call_with_tools = script_verify_two_never_deliver
_tools.verify_plan = stub_verify
_tools.check_invariants = lambda *a, **k: []
try:
    res = arl.run_agent_turn_v2(
        TRACK_ID, [{"role": "user", "content": "plan me"}], 0.0, state_override=state_with_prereqs_for_strong()
    )
    delivered = set(c["course_number"] for c in res["plan_result"]["courses"])
    check(delivered == set(STRONG),
          "wrap-up delivered the best (passing) plan STRONG, not the most-recent (failing) WEAK")
    check(res["plan_result"]["verify"]["pass"] is True, "delivered plan reports as passing")
    check(res["stopped_reason"] in ("step_limit_exhausted", "budget_cap"), "wrap-up path was taken")
finally:
    arl._call_with_tools = orig
    _tools.verify_plan = orig_verify
    _tools.check_invariants = orig_inv


# --- Test 3b: final safety net swaps a stubbornly-bad delivered plan ---
# Found live: issue_budget_pushback only retries twice, then accepts
# whatever the model delivers - an 8-credit plan and a 1-day exam gap both
# reached a student this way despite the delivery standard saying "never
# acceptable". This proves the fix: even after the model exhausts its
# pushback retries and DOES call deliver_plan (unlike test 3 above, where
# it never delivers at all), a genuinely better plan verified earlier this
# same turn still wins.
print("--- final safety net swaps a stubbornly-bad delivered plan ---")


def script_stubborn_deliver_weak(_messages):
    n = getattr(script_stubborn_deliver_weak, "n", 0)
    script_stubborn_deliver_weak.n = n + 1
    if n == 0:
        return msg([tool_call("verify_plan", {"plan_course_numbers": STRONG})]), 0.0
    if n == 1:
        return msg([tool_call("verify_plan", {"plan_course_numbers": WEAK})]), 0.0
    # Stubbornly keeps delivering WEAK no matter how many times
    # issue_budget_pushback sends it back.
    return msg([tool_call("deliver_plan", {"course_numbers": WEAK, "explanation": "here", "course_reasons": {}})]), 0.0


arl._call_with_tools = script_stubborn_deliver_weak
_tools.verify_plan = stub_verify
_tools.check_invariants = lambda *a, **k: []
try:
    res = arl.run_agent_turn_v2(
        TRACK_ID, [{"role": "user", "content": "plan me"}], 0.0, state_override=state_with_prereqs_for_strong()
    )
    delivered = set(c["course_number"] for c in res["plan_result"]["courses"])
    names = [t["name"] for t in res["tool_log"]]
    check(names.count("issue_budget_pushback") == 2, "pushback exhausts its 2-attempt budget")
    check(res["stopped_reason"] == "delivered", "model DID call deliver_plan (unlike test 3's never-delivers case)")
    check(delivered == set(STRONG), "final safety net swapped WEAK for the better-verified STRONG plan anyway")
    check(res["plan_result"]["verify"]["pass"] is True, "delivered plan reports as passing")
    check("final_safety_swap" in names, "the swap is visible in the trace")
finally:
    arl._call_with_tools = orig
    _tools.verify_plan = orig_verify
    _tools.check_invariants = orig_inv


# --- Test 4: sport-padding pushback ---
print("--- sport-padding pushback ---")

SPORTY = ["03940804", "03940802", "03940805", "00970249"]  # 3 PE courses + 1 real
sport_attempts = {"n": 0}


def script_deliver_sporty(_messages):
    sport_attempts["n"] += 1
    tc = tool_call("deliver_plan", {"course_numbers": SPORTY, "explanation": "padded"}, cid=f"s{sport_attempts['n']}")
    return msg([tc]), 0.0


arl._call_with_tools = script_deliver_sporty
try:
    res = arl.run_agent_turn_v2(TRACK_ID, [{"role": "user", "content": "plan me"}], 0.0, state_override=base_state())
    names = [t["name"] for t in res["tool_log"]]
    check("sport_padding_pushback" in names, "pushback fired on a plan with 3 sport courses")
    check(names.count("sport_padding_pushback") == 1, "sport pushback fires exactly once")
    check(res["stopped_reason"] == "delivered", "delivery still completes after the one-shot pushback")
finally:
    arl._call_with_tools = orig


print()
if failures:
    print(f"{len(failures)} failure(s)")
    sys.exit(1)
print("All mocked agent-guarantee checks passed (zero API calls)")
