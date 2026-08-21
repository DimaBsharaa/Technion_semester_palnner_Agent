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


def semester1_state(**overrides):
    intake = {"semester_number": 1, "excluded_weekdays": [], "pace": "normal",
              "passed_courses": [], "failed_courses": []}
    intake.update(overrides)
    return build_state_from_intake(intake)


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
    # >= 3, not == 3: near_locked_mandatory_courses (added after a live bug
    # where a near-miss mandatory course like Data Structures was silently
    # invisible) only appears in the trace when this fixture's state
    # actually has one - additive, not a fixed count.
    check(len(preinjected) >= 3, "diagnostics are marked preinjected (computed in code, not model-fetched)")
    check(res["plan_result"] is not None, "produces a plan_result")
    # near_locked_mandatory_courses: a near-miss mandatory course (real
    # unmet-prereq data, not a fixture artifact) is proactively surfaced to
    # the model, unconditionally - not only when the student asks about it
    # by name. base_state() (semester 4, nothing passed) genuinely has
    # several of these.
    near_locked_entry = next((t for t in res["tool_log"] if t["name"] == "near_locked_mandatory_courses"), None)
    check(near_locked_entry is not None, "near-locked mandatory courses surfaced proactively, unconditionally")
    check(
        near_locked_entry is not None
        and any(e["course_number"] == "00940290" and e["still_needs"] for e in near_locked_entry["result"]),
        "each entry names the specific course(s) still blocking it",
    )
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
# WEAK fails - isolating the "best wins, not most-recent" selection. STRONG
# includes 3 real mandatory courses (00940241/00940314/00940424) that are
# BOTH unlocked AND not already auto-backfilled as passed for this state's
# semester_number=4 (verified against backfill_passed_courses's real
# output, not just prereq depth - a shallower course can still get
# auto-marked passed by the EXPECTED_BY_NOW_BUFFER heuristic, or now by
# real official_semester diagram data) and don't collide on the schedule -
# so STRONG already clears tools.DEFAULT_MIN_MANDATORY_COURSES on its own.
# Otherwise the mandatory-course top-up backstop (added after a live bug:
# filler electives delivered instead of available mandatory courses) would
# modify this plan, breaking this test's exact-match assertions, which are
# about best-plan SELECTION, not that specific backstop. (Revised once
# already, after adding official_semester data made two of the original
# three - 00940241, 00940424 - correctly auto-backfill as already-passed
# for this state.)
STRONG = ["03940804", "00970249", "03940582", "00940314", "00960210", "00960275"]
WEAK = ["03940804"]

import tools as _tools  # noqa: E402

orig_verify = _tools.verify_plan
orig_inv = _tools.check_invariants
orig_choose_sections = _tools.choose_sections


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


# --- Test 3c: final safety net's course-count summary never goes stale ---
# Found live: the swap explanation embeds "(N course(s), M credits)" at the
# moment it's built, but a LATER backstop (locked-course/overlap/live-check)
# can still drop a course from final_course_numbers afterward - the printed
# N then no longer matches the actually-delivered course list. Fixed with a
# deferred placeholder substitution done after every backstop has run.
print("--- final safety net's course-count summary stays accurate after a later drop ---")

orig_prereq_unmet = _tools.prereq_unmet_in


def stub_prereq_unmet_locks_one(track_, plan, passed_, failed_):
    # Forces the HARD locked-course backstop to fire AFTER final_safety_swap
    # already built its explanation with the pre-drop count.
    if set(plan) == set(STRONG) and "03940582" in plan:
        return {"03940582"}
    return set()


script_stubborn_deliver_weak.n = 0  # reset the shared counter from Test 3b's run above
arl._call_with_tools = script_stubborn_deliver_weak
_tools.verify_plan = stub_verify
_tools.check_invariants = lambda *a, **k: []
_tools.prereq_unmet_in = stub_prereq_unmet_locks_one
try:
    res = arl.run_agent_turn_v2(
        TRACK_ID, [{"role": "user", "content": "plan me"}], 0.0, state_override=state_with_prereqs_for_strong()
    )
    delivered = [c["course_number"] for c in res["plan_result"]["courses"]]
    explanation = res["plan_result"]["explanation"]
    check("final_safety_swap" in [t["name"] for t in res["tool_log"]], "swap still fires")
    check("locked_enforced" in [t["name"] for t in res["tool_log"]], "the later locked-course backstop also fires")
    check(f"({len(delivered)} course(s)" in explanation,
          "the printed course count matches what was ACTUALLY delivered, not the pre-drop count")
    check("{COURSE_COUNT}" not in explanation and "{TOTAL_CREDITS}" not in explanation,
          "no unsubstituted placeholder leaks into the student-facing explanation")
finally:
    arl._call_with_tools = orig
    _tools.verify_plan = orig_verify
    _tools.check_invariants = orig_inv
    _tools.prereq_unmet_in = orig_prereq_unmet


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


# --- Test 5: student proactively requests a retake, with no prior proposal ---
# Found live: a student who opens with "I need to improve/retake course X"
# was silently ignored - the only retake path wired up was the agent
# PROPOSING first, then the student accepting on a LATER turn. A student
# bringing it up themselves, turn one, had nowhere to go. This checks the
# other half: state["requested_retake_course"] (set by extraction, or here
# directly on the state dict since state_override skips extraction) flows
# into the same approved_retake_course exemption, with no round-trip needed.
print("--- student proactively requests a retake (no prior agent proposal) ---")

RETAKE_PLAN = ["03940804", "00970249"]  # 03940804 is the course being retaken


def script_deliver_retake_plan(_messages):
    if not getattr(script_deliver_retake_plan, "verified", False):
        script_deliver_retake_plan.verified = True
        return msg([tool_call("verify_plan", {"plan_course_numbers": RETAKE_PLAN})]), 0.0
    return msg([tool_call("deliver_plan", {"course_numbers": RETAKE_PLAN, "explanation": "includes your retake"})]), 0.0


self_retake_state = state_with_prereqs_for_strong()
self_retake_state["passed_courses"] = sorted(set(self_retake_state["passed_courses"]) | {"03940804"})
self_retake_state["requested_retake_course"] = "03940804"

arl._call_with_tools = script_deliver_retake_plan
try:
    res = arl.run_agent_turn_v2(TRACK_ID, [{"role": "user", "content": "I need to retake this course"}], 0.0, state_override=self_retake_state)
    names = [t["name"] for t in res["tool_log"]]
    delivered = [c["course_number"] for c in res["plan_result"]["courses"]]
    check("grade_retake_self_requested" in names, "self-requested retake recognized without a prior proposal")
    check("03940804" in delivered, "the already-passed course the student asked to retake IS included")
    issues = res["plan_result"]["verify"].get("issues", [])
    check(not any("passed" in i.get("reason", "").lower() for i in issues),
          "no 'already passed' violation raised for the one exempted course")
finally:
    arl._call_with_tools = orig
    script_deliver_retake_plan.verified = False


# --- Test 6: mandatory-course top-up backstop ---
# Found live: a plan can pass credit/workload checks while carrying almost
# no real degree progress - the model delivered 2 filler electives
# (orchestra, an entrepreneurship elective) instead of two ALREADY-UNLOCKED
# real mandatory courses, with verify_plan's own "only 1 mandatory
# course(s), expected at least 3" issue printed right in the explanation
# and delivered anyway. This checks the backstop directly: an elective-only
# delivery gets topped up with genuinely available mandatory courses before
# it ever reaches the student.
print("--- mandatory-course top-up backstop ---")

FILLER_ONLY = ["03940804"]  # a single no-exam elective, no real degree progress


def script_deliver_filler_only(_messages):
    if not getattr(script_deliver_filler_only, "verified", False):
        script_deliver_filler_only.verified = True
        return msg([tool_call("verify_plan", {"plan_course_numbers": FILLER_ONLY})]), 0.0
    return msg([tool_call("deliver_plan", {"course_numbers": FILLER_ONLY, "explanation": "light semester"})]), 0.0


def stub_verify_always_pass(track_, plan, passed_, **kw):
    return {"pass": True, "total_credits": 18.0, "workload_score": 50.0, "issues": []}


arl._call_with_tools = script_deliver_filler_only
_tools.verify_plan = stub_verify_always_pass
_tools.check_invariants = lambda *a, **k: []
try:
    res = arl.run_agent_turn_v2(TRACK_ID, [{"role": "user", "content": "plan me"}], 0.0, state_override=base_state())
    names = [t["name"] for t in res["tool_log"]]
    delivered = [c["course_number"] for c in res["plan_result"]["courses"]]
    mandatory_delivered = [c for c in delivered if c in track.mandatory_course_numbers]
    check("mandatory_topup_enforced" in names, "top-up backstop fired on an elective-only delivery")
    check(len(mandatory_delivered) >= 3, "the delivered plan now carries real mandatory courses, not just filler")
    check("03940804" in delivered, "the original elective is kept, not discarded, alongside the added courses")
finally:
    arl._call_with_tools = orig
    _tools.verify_plan = orig_verify
    _tools.check_invariants = orig_inv
    script_deliver_filler_only.verified = False


# --- Test 7: explicit "add this course" directive is enforced ---
# Found live: a student named a specific, real, unlocked mandatory course
# by name and asked for it TWICE across two revision turns - it never made
# it into the plan. Symmetric to the existing "replace X" removal
# enforcement: a student's explicit ADD directive must be a hard
# constraint too, not something the model can silently decline (e.g.
# because CheeseFork difficulty data made it look unappealing).
print("--- explicit add-course directive is enforced ---")

ADD_TARGET = "00960210"  # unlocked given state_with_prereqs_for_strong()'s passed set
NO_ADD_PLAN = ["03940804"]


def script_deliver_without_requested_add(_messages):
    n = getattr(script_deliver_without_requested_add, "n", 0)
    script_deliver_without_requested_add.n = n + 1
    if n == 0:
        return msg([tool_call("verify_plan", {"plan_course_numbers": NO_ADD_PLAN})]), 0.0
    # Stubbornly keeps delivering without the requested course, no matter
    # how many times the add-ignored pushback sends it back.
    return msg([tool_call("deliver_plan", {"course_numbers": NO_ADD_PLAN, "explanation": "here"})]), 0.0


add_state = state_with_prereqs_for_strong()
add_state["requested_add_courses"] = [ADD_TARGET]

arl._call_with_tools = script_deliver_without_requested_add
_tools.verify_plan = stub_verify_always_pass
_tools.check_invariants = lambda *a, **k: []
try:
    res = arl.run_agent_turn_v2(TRACK_ID, [{"role": "user", "content": f"add {ADD_TARGET} please"}], 0.0, state_override=add_state)
    names = [t["name"] for t in res["tool_log"]]
    delivered = [c["course_number"] for c in res["plan_result"]["courses"]]
    check("add_ignored_pushback" in names, "pushback fired when the requested course was missing")
    check("add_enforced" in names, "add enforced in code after the model ignored the pushback")
    check(ADD_TARGET in delivered, "the explicitly requested course IS in the final delivered plan")
finally:
    arl._call_with_tools = orig
    _tools.verify_plan = orig_verify
    _tools.check_invariants = orig_inv
    script_deliver_without_requested_add.n = 0


# --- Test 8: a requested-but-genuinely-locked course gets explained, not silently dropped ---
# Found live: the agent either went silent about a locked course the
# student explicitly asked for, or worse, cited unrelated CheeseFork
# difficulty as if that were the reason. A course that's ACTUALLY unlocked
# gets force-added (Test 7); a course that's GENUINELY locked cannot be -
# but the student must be told exactly what's still missing, by name, and
# that guarantee must survive even if a later backstop (e.g.
# final_safety_swap) replaces the model's own explanation entirely.
print("--- requested-but-locked course is explained by name, not silently dropped ---")

LOCKED_TARGET = "00940290"  # genuinely locked for base_state() - needs 00960224 first


def script_deliver_ignoring_locked_request(_messages):
    return msg([tool_call("deliver_plan", {"course_numbers": ["03940804"], "explanation": "here you go"})]), 0.0


locked_state = base_state()
locked_state["requested_add_courses"] = [LOCKED_TARGET]

arl._call_with_tools = script_deliver_ignoring_locked_request
try:
    res = arl.run_agent_turn_v2(TRACK_ID, [{"role": "user", "content": f"add {LOCKED_TARGET} please"}], 0.0, state_override=locked_state)
    names = [t["name"] for t in res["tool_log"]]
    adds_entry = next((t for t in res["tool_log"] if t["name"] == "requested_adds"), None)
    explanation = res["plan_result"]["explanation"]
    check(adds_entry is not None and LOCKED_TARGET in adds_entry["result"]["locked"],
          "the locked course is recognized and logged, not treated as unlocked")
    check(LOCKED_TARGET not in [c["course_number"] for c in res["plan_result"]["courses"]],
          "a genuinely locked course is never force-added")
    check("מעבדה באיסוף וניהול נתונים" in explanation, "the locked course is named in the explanation, not silently dropped")
    check("00960224" in explanation or track.courses["00960224"]["name"] in explanation,
          "the SPECIFIC missing prerequisite is named, not a vague 'prerequisites not met'")
finally:
    arl._call_with_tools = orig


# --- Test 9: credit-floor top-up backstop ---
# Found live: honoring an explicit "add X" request by dropping enough
# OTHER courses to fall from 17.5 to 13 credits, well under the floor -
# the model's own over-trimming, not any single backstop's fault. This
# checks the guarantee directly: a plan that ends up under the credit
# floor gets topped back up from the real available shortlist before
# delivery, regardless of how it got there.
print("--- credit-floor top-up backstop ---")

LIGHT_PLAN = ["03940804"]  # one no-exam elective, ~2 credits - well under the floor

orig_min_mandatory2 = _tools.DEFAULT_MIN_MANDATORY_COURSES
_tools.DEFAULT_MIN_MANDATORY_COURSES = 0  # isolate the credit floor from the mandatory-count floor


def script_deliver_light_plan(_messages):
    if not getattr(script_deliver_light_plan, "verified", False):
        script_deliver_light_plan.verified = True
        return msg([tool_call("verify_plan", {"plan_course_numbers": LIGHT_PLAN})]), 0.0
    return msg([tool_call("deliver_plan", {"course_numbers": LIGHT_PLAN, "explanation": "light plan"})]), 0.0


def stub_verify_pass_any_credits(track_, plan, passed_, **kw):
    pts = sum(float(track.courses[c].get("points") or 0) for c in plan if c in track.courses)
    return {"pass": True, "total_credits": pts, "workload_score": 40.0, "issues": []}


arl._call_with_tools = script_deliver_light_plan
_tools.verify_plan = stub_verify_pass_any_credits
_tools.check_invariants = lambda *a, **k: []
try:
    res = arl.run_agent_turn_v2(TRACK_ID, [{"role": "user", "content": "plan me"}], 0.0, state_override=base_state())
    names = [t["name"] for t in res["tool_log"]]
    delivered = [c["course_number"] for c in res["plan_result"]["courses"]]
    total_pts = sum(float(track.courses[c].get("points") or 0) for c in delivered if c in track.courses)
    check("credit_floor_topup_enforced" in names, "credit-floor backstop fired on a light delivery")
    check(total_pts >= _tools.DEFAULT_MIN_CREDITS, f"final credits ({total_pts}) reach the floor ({_tools.DEFAULT_MIN_CREDITS})")
    check("03940804" in delivered, "the original elective is kept, not discarded, alongside the top-up")
finally:
    arl._call_with_tools = orig
    _tools.verify_plan = orig_verify
    _tools.check_invariants = orig_inv
    _tools.DEFAULT_MIN_MANDATORY_COURSES = orig_min_mandatory2
    script_deliver_light_plan.verified = False


# --- Test 10: an approved grade-improvement retake is hard-enforced, not just nudged ---
# Found live: extraction correctly resolved a self-requested retake and set
# state["approved_retake_course"], the one-shot nudge fired, and the model
# STILL delivered without it on the second attempt - the exact same
# reliability gap every other explicit student directive already got a
# hard backstop for tonight (requested_add_courses, credit floor,
# mandatory count). This checks the second-violation enforcement directly.
print("--- approved grade-improvement retake is hard-enforced after the nudge ---")

RETAKE_TARGET = "00960570"  # already in state_with_prereqs_for_strong()'s passed_courses
NO_RETAKE_PLAN = ["03940804"]


def script_stubbornly_omit_approved_retake(_messages):
    n = getattr(script_stubbornly_omit_approved_retake, "n", 0)
    script_stubbornly_omit_approved_retake.n = n + 1
    if n == 0:
        return msg([tool_call("verify_plan", {"plan_course_numbers": NO_RETAKE_PLAN})]), 0.0
    # Never includes the approved retake, no matter how many times pushed.
    return msg([tool_call("deliver_plan", {"course_numbers": NO_RETAKE_PLAN, "explanation": "here"})]), 0.0


retake_state = state_with_prereqs_for_strong()
# state["approved_retake_course"] is recomputed unconditionally at the top
# of run_agent_turn_v2 (never trusts a bare state_override value - see its
# own comment about not trusting a hallucinated approval) - set it via
# requested_retake_course instead, the real mechanism, same as the
# self-requested-retake test above.
retake_state["requested_retake_course"] = RETAKE_TARGET

arl._call_with_tools = script_stubbornly_omit_approved_retake
_tools.verify_plan = stub_verify_pass_any_credits
_tools.check_invariants = lambda *a, **k: []
try:
    res = arl.run_agent_turn_v2(TRACK_ID, [{"role": "user", "content": "plan me"}], 0.0, state_override=retake_state)
    names = [t["name"] for t in res["tool_log"]]
    delivered = [c["course_number"] for c in res["plan_result"]["courses"]]
    check("approved_retake_missing_nudge" in names, "nudge fired on the first violation")
    check("approved_retake_enforced" in names, "enforced in code after the model ignored the nudge too")
    check(RETAKE_TARGET in delivered, "the approved retake IS in the final delivered plan")
    retake_row = next((c for c in res["plan_result"]["courses"] if c["course_number"] == RETAKE_TARGET), None)
    check(retake_row is not None and retake_row["is_retake"], "the RETAKE badge shows for a grade-improvement retake too")
finally:
    arl._call_with_tools = orig
    _tools.verify_plan = orig_verify
    _tools.check_invariants = orig_inv
    script_stubbornly_omit_approved_retake.n = 0


# --- Test 11: an explicit removal is never silently undone by the top-up backstops ---
# Found live: a student said "I already passed algebra, swap it out" - the
# removal was correctly recognized, but the mandatory-course top-up
# backstop (or the credit-floor one, same bug) immediately re-added the
# EXACT SAME course right back, since neither excluded requested_removals
# from its own candidate search - the system didn't yet know an equivalent
# was passed, so from its own (wrong) perspective the course still looked
# like "the missing mandatory requirement." An explicit removal must
# outrank a heuristic's guess about what's still needed, even if that
# means an honestly-disclosed open issue instead.
print("--- explicit removal is never silently undone by mandatory/credit top-up ---")

REMOVE_TARGET = "00940314"  # real, unlocked mandatory course given this state
NO_TARGET_PLAN = ["00940224", "00960570"]  # already-passed courses only - forces a real shortfall


def script_deliver_without_removed_course(_messages):
    if not getattr(script_deliver_without_removed_course, "verified", False):
        script_deliver_without_removed_course.verified = True
        return msg([tool_call("verify_plan", {"plan_course_numbers": NO_TARGET_PLAN})]), 0.0
    return msg([tool_call("deliver_plan", {"course_numbers": NO_TARGET_PLAN, "explanation": "removed as asked"})]), 0.0


removal_state = state_with_prereqs_for_strong()
removal_state["remove_courses"] = [REMOVE_TARGET]

arl._call_with_tools = script_deliver_without_removed_course
_tools.verify_plan = stub_verify_pass_any_credits
_tools.check_invariants = lambda *a, **k: []
try:
    res = arl.run_agent_turn_v2(
        TRACK_ID,
        [{"role": "user", "content": "I already passed this, swap it out"}],
        0.0,
        state_override=removal_state,
        known_context={"previous_plan": [REMOVE_TARGET, "00940224", "00960570"]},
    )
    names = [t["name"] for t in res["tool_log"]]
    delivered = [c["course_number"] for c in res["plan_result"]["courses"]]
    check("requested_removals" in names, "the removal request was recognized")
    check(
        REMOVE_TARGET not in delivered,
        "the explicitly removed course is NOT silently re-added by mandatory/credit top-up",
    )
finally:
    arl._call_with_tools = orig
    _tools.verify_plan = orig_verify
    _tools.check_invariants = orig_inv
    script_deliver_without_removed_course.verified = False


# --- Test 12: workload formula doesn't flag a normal mandatory-heavy semester ---
# Found live: the OLD formula scored a completely ordinary, real
# mandatory-heavy semester (5 real courses, 20 credits, avg CheeseFork
# difficulty 3.86 - typical for Technion engineering core courses) at
# 90.9, always over DEFAULT_WORKLOAD_CAP=80. That meant nearly every
# legitimate mandatory-heavy semester tripped "workload exceeds cap" by
# default, which pressured the model to drop courses to get under it - and
# since dropping the HARDEST course cuts the score fastest, it kept
# dropping mandatory courses specifically (the harder ones) instead of
# light electives. Recalibrated around an empirically-measured difficulty
# baseline (_DIFFICULTY_BASELINE) instead of scaling from 0.
print("--- workload formula doesn't over-flag a normal mandatory-heavy semester ---")

TYPICAL_MANDATORY_HEAVY = ["00940224", "00960210", "00940219", "00960275", "01040166"]  # 20 credits, avg diff ~4.0
EXCESSIVE_STACK = ["00950296", "01040032", "01040031", "00940224", "00960210", "00940219"]  # 25.5 credits, avg diff ~4.2

normal_score = _tools._workload_score(track, TYPICAL_MANDATORY_HEAVY)
excessive_score = _tools._workload_score(track, EXCESSIVE_STACK)
check(
    normal_score <= _tools.DEFAULT_WORKLOAD_CAP,
    f"a typical real mandatory-heavy semester ({normal_score}) stays under the cap ({_tools.DEFAULT_WORKLOAD_CAP})",
)
check(
    excessive_score > _tools.DEFAULT_WORKLOAD_CAP,
    f"a genuinely excessive stack ({excessive_score}) still correctly exceeds the cap",
)


# --- Test 13: a requested-but-unavailable course is explained, never force-added ---
# Found live: add_enforced had no guard at all against blindly forcing a
# course that isn't even offered this semester into the plan - unlike the
# retake-approval enforcement, which already refuses to do that. There's no
# section to register for either way, so this must never be "forced in no
# matter what" the way a schedule/room trade-off can be (Test 14) - it can
# only ever be explained.
print("--- requested-but-unavailable course is explained, never force-added ---")

UNAVAILABLE_TARGET = "03940582"  # תזמורת - confirmed not-offered across all 3 tracks


def script_deliver_ignoring_unavailable_request(_messages):
    return msg([tool_call("deliver_plan", {"course_numbers": ["03940804"], "explanation": "here you go"})]), 0.0


unavailable_state = state_with_prereqs_for_strong()
unavailable_state["requested_add_courses"] = [UNAVAILABLE_TARGET]

arl._call_with_tools = script_deliver_ignoring_unavailable_request
try:
    res = arl.run_agent_turn_v2(
        TRACK_ID, [{"role": "user", "content": f"add {UNAVAILABLE_TARGET} please"}], 0.0, state_override=unavailable_state
    )
    names = [t["name"] for t in res["tool_log"]]
    explanation = res["plan_result"]["explanation"]
    adds_entry = next((t for t in res["tool_log"] if t["name"] == "requested_adds"), None)
    check(
        adds_entry is not None and UNAVAILABLE_TARGET in adds_entry["result"]["unavailable"],
        "the unavailable course is recognized and logged, not treated as forceable",
    )
    check("add_enforced" not in names, "add_enforced never even attempts an unavailable course")
    check(
        UNAVAILABLE_TARGET not in [c["course_number"] for c in res["plan_result"]["courses"]],
        "an unavailable course is never force-added, no matter what",
    )
    check(track.courses[UNAVAILABLE_TARGET]["name"] in explanation, "the unavailable course is named in the explanation")
    check(
        res["known_context"].get("pending_forced_add") is None,
        "no override is ever offered for a course that genuinely isn't offered",
    )
finally:
    arl._call_with_tools = orig


# --- Test 13b: a requested add that doesn't exist in the catalog at all is explained, never silently dropped ---
# Found via independent review: a typo'd/made-up course number fell through
# requested_add_raw's "if c in track.courses" filter with zero trace
# anywhere downstream - unlike the unavailable/locked/already-passed cases
# just above, which all get a guaranteed explanation.
print("--- requested course number not in the catalog at all is explained, never silently dropped ---")

UNKNOWN_TARGET = "00000000"  # not a real course number in any track


def script_deliver_ignoring_unknown_request(_messages):
    return msg([tool_call("deliver_plan", {"course_numbers": ["03940804"], "explanation": "here you go"})]), 0.0


unknown_state = state_with_prereqs_for_strong()
unknown_state["requested_add_courses"] = [UNKNOWN_TARGET]

arl._call_with_tools = script_deliver_ignoring_unknown_request
try:
    res = arl.run_agent_turn_v2(
        TRACK_ID, [{"role": "user", "content": f"add {UNKNOWN_TARGET} please"}], 0.0, state_override=unknown_state
    )
    explanation = res["plan_result"]["explanation"]
    adds_entry = next((t for t in res["tool_log"] if t["name"] == "requested_adds"), None)
    check(
        adds_entry is not None and UNKNOWN_TARGET in adds_entry["result"]["unknown"],
        "the unknown course number is recognized and logged, not silently dropped",
    )
    check(UNKNOWN_TARGET in explanation, "the unknown course number is named in the explanation")
    check(
        UNKNOWN_TARGET not in [c["course_number"] for c in res["plan_result"]["courses"]],
        "an unknown course number is never in the delivered plan",
    )
finally:
    arl._call_with_tools = orig


# --- Test 14: a requested add that collides is negotiated, then forced in on confirmation ---
# Found live: a student asked to add a specific course FIVE TIMES across
# revision turns and it just kept silently vanishing - add_enforced forced
# it in blindly (no collision awareness), then the separate, real
# collision backstop quietly dropped it again with a generic note that
# never acknowledged it was an explicit request. Unlike Test 13's hard
# block, a schedule collision is the STUDENT's trade-off to accept, not
# Python's or the model's to silently decide - this checks both halves:
# the negotiation offer, and the confirmed override actually landing.
print("--- explicit add that collides is negotiated, then forced in on confirmation ---")

ADD_TARGET2 = "03940800"  # non-mandatory, offered, unlocked - real points (1.0) < REMAIN_COURSE's (2.5)
REMAIN_COURSE = "00960226"  # survives add_enforced's cheapest-first drop, and the collision comparison
CHEAP_FILLER = "03940804"  # cheapest droppable - what add_enforced sacrifices to make room


def stub_choose_sections_conflict(track_, course_numbers, excluded_weekdays=None):
    overlaps = (
        [{"course_a": ADD_TARGET2, "course_b": REMAIN_COURSE, "time": "Mon 10:30-12:30"}]
        if ADD_TARGET2 in course_numbers and REMAIN_COURSE in course_numbers
        else []
    )
    return {"assignments": {c: [] for c in course_numbers}, "overlaps": overlaps, "excluded_day_hits": []}


def script_deliver_without_colliding_add(_messages):
    n = getattr(script_deliver_without_colliding_add, "n", 0)
    script_deliver_without_colliding_add.n = n + 1
    if n == 0:
        return msg([tool_call("verify_plan", {"plan_course_numbers": [REMAIN_COURSE, CHEAP_FILLER]})]), 0.0
    return msg([tool_call("deliver_plan", {"course_numbers": [REMAIN_COURSE, CHEAP_FILLER], "explanation": "here"})]), 0.0


collide_state = state_with_prereqs_for_strong()
collide_state["requested_add_courses"] = [ADD_TARGET2]

arl._call_with_tools = script_deliver_without_colliding_add
_tools.choose_sections = stub_choose_sections_conflict
try:
    res = arl.run_agent_turn_v2(
        TRACK_ID, [{"role": "user", "content": f"add {ADD_TARGET2} please"}], 0.0, state_override=collide_state
    )
    names = [t["name"] for t in res["tool_log"]]
    explanation = res["plan_result"]["explanation"]
    delivered = [c["course_number"] for c in res["plan_result"]["courses"]]
    pending = res["known_context"].get("pending_forced_add")
    check("forced_add_enforced" not in names, "not forced in without confirmation")
    check(ADD_TARGET2 not in delivered, "the colliding course is not silently included")
    check(REMAIN_COURSE in delivered, "the course it collides with is kept, not both silently dropped")
    check(
        track.courses[ADD_TARGET2]["name"] in explanation and "collide" in explanation,
        "the specific collision reason is named in the explanation, not left silent",
    )
    check(
        "anyway" in explanation.lower() or "confirm" in explanation.lower(),
        "the student is asked whether to force it in despite the trade-off",
    )
    check(
        pending is not None and pending["course_number"] == ADD_TARGET2,
        "the offer to force it in survives into the next turn's known_context",
    )

    # Turn 2: a bare "yes, add it anyway" - no course named again, exactly
    # how a real confirmation reads. requested_add_courses is deliberately
    # left empty here to prove the confirmation doesn't depend on the
    # student re-naming the course.
    confirm_state = state_with_prereqs_for_strong()
    confirm_state["confirmed_forced_add"] = ADD_TARGET2
    res2 = arl.run_agent_turn_v2(
        TRACK_ID,
        [{"role": "user", "content": "yes, add it anyway"}],
        0.0,
        state_override=confirm_state,
        known_context={"pending_forced_add": pending},
    )
    names2 = [t["name"] for t in res2["tool_log"]]
    delivered2 = [c["course_number"] for c in res2["plan_result"]["courses"]]
    explanation2 = res2["plan_result"]["explanation"]
    check("forced_add_confirmed" in names2, "the confirmation is recognized against the real pending offer")
    check("forced_add_enforced" in names2, "forced in on confirmation, despite no course name this turn")
    check(ADD_TARGET2 in delivered2, "the course IS in the final delivered plan after confirmation")
    check("confirmed" in explanation2.lower(), "the explanation acknowledges this was a confirmed override")
    check(
        any("collide" in i["reason"] or "overlap" in i["reason"] for i in res2["plan_result"]["verify"]["issues"]),
        "the resulting trade-off is disclosed as an open issue, not hidden now that it's included",
    )
finally:
    arl._call_with_tools = orig
    _tools.choose_sections = orig_choose_sections
    script_deliver_without_colliding_add.n = 0


# --- Test 15: a confirmed retake survives a LATER, unrelated turn ---
# Found live: approved_retake_course resets to None every turn by design
# (the "single-course, single-turn carve-out" check_invariants/verify_plan
# always documented it as). That meant a retake accepted in turn N had
# real protection ONLY on turn N - on turn N+1, even a totally unrelated
# revision, check_invariants saw the already-passed retake course still
# sitting in the plan, had no exemption for it anymore, flagged it as
# "plan re-includes an already-passed course," and Python's own violation
# correction silently stripped it back out - not the model dropping it,
# Python itself. Test 10 above only ever checks the turn it's approved on;
# this checks the turn AFTER, which is where the real bug lived.
print("--- a confirmed retake survives a later, unrelated revision turn ---")


def script_deliver_without_retake_unrelated_turn(_messages):
    # A plan that quietly omits the retake while handling something else -
    # never told to remove it, never re-asked about it either.
    return msg([tool_call("deliver_plan", {"course_numbers": ["03940804"], "explanation": "here"})]), 0.0


later_turn_state = state_with_prereqs_for_strong()
# Deliberately NOT setting requested_retake_course/approved_retake_course
# this turn - simulates the retake having been approved in an EARLIER
# turn, with nothing about it said again now.
later_turn_known_context = {
    "state": {
        "semester_number": 4,
        "constraints": {"excluded_weekdays": [], "pace": "normal", "notes": "", "override_minimums": False},
        "passed_courses": ["00940224", "00960570"],
        "failed_courses": [],
        "grades": {},
        "confirmed_grade_retakes": [RETAKE_TARGET],
    },
    "previous_plan": [RETAKE_TARGET, "03940804"],
    "previous_issues": [],
    "previous_explanation": "Added the retake last turn.",
    "proposed_retake": None,
    "pending_forced_add": None,
}

arl._call_with_tools = script_deliver_without_retake_unrelated_turn
try:
    res = arl.run_agent_turn_v2(
        TRACK_ID,
        [{"role": "user", "content": "can you check if this still fits my schedule"}],
        0.0,
        state_override=later_turn_state,
        known_context=later_turn_known_context,
    )
    names = [t["name"] for t in res["tool_log"]]
    delivered = [c["course_number"] for c in res["plan_result"]["courses"]]
    invariant_entry = next((t for t in res["tool_log"] if t["name"] == "check_invariants"), None)
    retake_course = next((c for c in res["plan_result"]["courses"] if c["course_number"] == RETAKE_TARGET), None)
    check(
        invariant_entry is None
        or not any(RETAKE_TARGET in v for v in invariant_entry["result"]["violations"]),
        "check_invariants never flags the confirmed retake as an illegal already-passed re-inclusion",
    )
    check(RETAKE_TARGET in delivered, "the confirmed retake IS in the plan on a later, unrelated turn")
    check("approved_retake_guaranteed" in names, "the guarantee backstop is what put it back, not the model")
    check(
        retake_course is not None and retake_course["is_retake"] is True,
        "the RETAKE badge still shows on a later turn, not just the turn it was approved on",
    )
    check(
        sorted(res["known_context"]["state"]["confirmed_grade_retakes"]) == [RETAKE_TARGET],
        "the confirmation keeps persisting forward into the NEXT known_context too",
    )
finally:
    arl._call_with_tools = orig


# --- Test 16: a requested add that's already passed is explained, not silently dropped ---
# Found live: a student asked to "add" a course by plain name (no "retake"
# language) that she'd already passed. requested_add_courses correctly
# excludes an already-passed course (it's not a new course to take), but
# nothing downstream ever explained WHY - the request just vanished with
# zero trace anywhere, exactly the "he's not replying in chat" complaint.
print("--- requested add that's already passed is explained, not silently dropped ---")

ALREADY_PASSED_TARGET = "00960570"  # in state_with_prereqs_for_strong()'s passed_courses


def script_deliver_ignoring_already_passed_request(_messages):
    return msg([tool_call("deliver_plan", {"course_numbers": ["03940804"], "explanation": "here you go"})]), 0.0


already_passed_state = state_with_prereqs_for_strong()
already_passed_state["requested_add_courses"] = [ALREADY_PASSED_TARGET]

arl._call_with_tools = script_deliver_ignoring_already_passed_request
try:
    res = arl.run_agent_turn_v2(
        TRACK_ID, [{"role": "user", "content": f"add {ALREADY_PASSED_TARGET} please"}], 0.0, state_override=already_passed_state
    )
    names = [t["name"] for t in res["tool_log"]]
    explanation = res["plan_result"]["explanation"]
    adds_entry = next((t for t in res["tool_log"] if t["name"] == "requested_adds"), None)
    check(
        adds_entry is not None and ALREADY_PASSED_TARGET in adds_entry["result"]["already_passed"],
        "the already-passed request is recognized and logged, not silently discarded",
    )
    check(
        ALREADY_PASSED_TARGET not in [c["course_number"] for c in res["plan_result"]["courses"]],
        "an already-passed course is never added as if it were new",
    )
    check(
        track.courses[ALREADY_PASSED_TARGET]["name"] in explanation and "already passed" in explanation,
        "the reason is named in the explanation, not left silent",
    )
    check(
        "retake" in explanation.lower(),
        "a concrete next step (asking for a retake) is offered instead of a dead end",
    )
finally:
    arl._call_with_tools = orig


# --- Test 17: semester 1 defaults to exactly the required courses, nothing more ---
# Explicit student directive: a brand-new student's first semester gets
# ALL of the track's real semester-1 mandatory courses and nothing else by
# default - no sport, no elective padding - unless she explicitly asks for
# something. Real data check that made this safe to force unconditionally:
# this track's semester-1 mandatory set is 5 courses / 22.0 credits on its
# own, comfortably inside the normal floor/ceiling, so forcing completeness
# never fights the credit range.
print("--- semester 1 defaults to exactly the required courses, nothing more ---")

S1_MANDATORY = sorted(c for c in track.mandatory_course_numbers if track.official_semester.get(c) == 1)
S1_SPORT = "03940804"


def script_deliver_partial_s1_with_sport(_messages):
    # Only 3 of the 5 real requirements, plus an unrequested sport course.
    partial = S1_MANDATORY[:3] + [S1_SPORT]
    return msg([tool_call("deliver_plan", {"course_numbers": partial, "explanation": "here"})]), 0.0


s1_state = semester1_state()
arl._call_with_tools = script_deliver_partial_s1_with_sport
try:
    check(len(S1_MANDATORY) == 5, "fixture sanity: this track really has 5 semester-1 mandatory courses")
    res = arl.run_agent_turn_v2(TRACK_ID, [{"role": "user", "content": "starting my first semester, plan it"}], 0.0, state_override=s1_state)
    names = [t["name"] for t in res["tool_log"]]
    delivered = sorted(c["course_number"] for c in res["plan_result"]["courses"])
    check("semester1_completeness_enforced" in names, "the two missing real requirements are force-added")
    check("semester1_extras_stripped" in names, "the unrequested sport course is stripped")
    check(delivered == S1_MANDATORY, "the final plan is EXACTLY the 5 required courses, nothing more, nothing less")
    check(S1_SPORT not in delivered, "the sport course specifically is gone")
finally:
    arl._call_with_tools = orig

# Same stubborn delivery, but this time the student explicitly asked for a
# specific elective - the extras-stripping half must back off (the
# completeness-forcing half still applies), while completely different
# explicit requests (a lighter pace) skip forcing completeness entirely.
EXPLICIT_ELECTIVE = "00960226"  # a real elective in this track's catalog, unrelated to S1_MANDATORY

s1_state_with_ask = semester1_state()
s1_state_with_ask["requested_add_courses"] = [EXPLICIT_ELECTIVE]
arl._call_with_tools = script_deliver_partial_s1_with_sport
try:
    res = arl.run_agent_turn_v2(TRACK_ID, [{"role": "user", "content": f"starting my first semester, plan it, and please add {EXPLICIT_ELECTIVE}"}], 0.0, state_override=s1_state_with_ask)
    delivered = [c["course_number"] for c in res["plan_result"]["courses"]]
    check(
        all(c in delivered for c in S1_MANDATORY),
        "completeness still applies when the student explicitly asked for something extra",
    )
    check(
        EXPLICIT_ELECTIVE in delivered,
        "an explicitly requested course is never stripped by the extras rule",
    )
finally:
    arl._call_with_tools = orig

s1_state_light = semester1_state(override_minimums=True)
arl._call_with_tools = script_deliver_partial_s1_with_sport
try:
    res = arl.run_agent_turn_v2(TRACK_ID, [{"role": "user", "content": "starting my first semester, keep it as light as possible please"}], 0.0, state_override=s1_state_light)
    names = [t["name"] for t in res["tool_log"]]
    check(
        "semester1_completeness_enforced" not in names,
        "an explicit lighter-load request is respected, not overridden by the completeness guarantee",
    )
finally:
    arl._call_with_tools = orig


print()
if failures:
    print(f"{len(failures)} failure(s)")
    sys.exit(1)
print("All mocked agent-guarantee checks passed (zero API calls)")
