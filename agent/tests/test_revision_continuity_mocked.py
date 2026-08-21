"""
Mocked (zero-API-cost) tests for the revision-continuity guarantees, added
after a live failure: a student pointed out an exam conflict in their
delivered plan, got re-asked which courses they'd passed, and then received
a from-scratch plan that ignored the complaint.

Guarantee 1: once a plan exists, a revision turn can NEVER bounce back to
the intake buttons, even if extraction comes back degraded/unsure -
merge_known_state (code, not prompt) restores settled facts and forces
planning to proceed.

Guarantee 2: the planning loop sees the recent conversation verbatim -
including the complaint - not just the last message (which after a form
submit is the form's summary, not the complaint).

    cd agent && python3 tests/test_revision_continuity_mocked.py
"""

import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_loop  # noqa: E402
import agent_react_loop as arl  # noqa: E402
from agent_loop import merge_known_state  # noqa: E402

TRACK_ID = "SC00001453"
failures: list[str] = []


def check(cond: bool, desc: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}: {desc}")
    if not cond:
        failures.append(desc)


KNOWN_STATE = {
    "semester_number": 5,
    "constraints": {"excluded_weekdays": [0], "pace": "normal", "notes": "", "override_minimums": False},
    "passed_courses": ["00940219", "00940290", "00940314"],
    "failed_courses": ["00940224"],
}
PREVIOUS_PLAN = ["00940224", "00960425", "03240697"]


# --- Part 1: merge_known_state unit behavior ---
print("--- merge_known_state: degraded extraction cannot un-know facts ---")

degraded = {
    "semester_number": 0,
    # None = "this turn didn't mention days" (extraction's new semantics);
    # [] would mean "the student explicitly cleared the restriction".
    "constraints": {"excluded_weekdays": None, "pace": None, "notes": "exam conflict on 23.2", "override_minimums": False},
    "passed_courses": [],
    "failed_courses": [],
    "ready_to_plan": False,
    "clarifying_question": "Which courses have you passed?",
    "missing_fields": ["courses"],
}
merged = merge_known_state(degraded, KNOWN_STATE)
check(merged["ready_to_plan"] is True, "revision turn is forced ready_to_plan")
check(merged["missing_fields"] == [], "missing_fields cleared")
check(merged["semester_number"] == 5, "semester restored from known state")
check(merged["constraints"]["excluded_weekdays"] == [0], "excluded weekdays restored when unmentioned (None)")

cleared = dict(degraded)
cleared["constraints"] = dict(degraded["constraints"], excluded_weekdays=[])
merged_cleared = merge_known_state(cleared, KNOWN_STATE)
check(merged_cleared["constraints"]["excluded_weekdays"] == [],
      "an explicitly-cleared restriction ([]) is respected, not restored - constraint relaxing works")
check(merged["passed_courses"] == sorted(KNOWN_STATE["passed_courses"]), "passed courses restored")
check(merged["failed_courses"] == ["00940224"], "failed courses restored")
check(merged["constraints"]["notes"] == "exam conflict on 23.2", "new info from this turn kept")

print("--- merge_known_state: newest explicit claims win ---")
update = dict(degraded)
update["passed_courses"] = ["00940224"]  # student now says they passed the old failure
update["failed_courses"] = ["00940290"]  # ...and actually failed something else
merged2 = merge_known_state(update, KNOWN_STATE)
check("00940224" in merged2["passed_courses"] and "00940224" not in merged2["failed_courses"],
      "a now-passed course leaves failed")
check("00940290" in merged2["failed_courses"] and "00940290" not in merged2["passed_courses"],
      "a now-failed course leaves passed")


# --- Part 2: end-to-end - degraded extraction on a revision turn still plans ---
print("--- react loop: revision turn never bounces to intake ---")

captured_prompts: list[list] = []


def fake_extract(track, messages, known_state=None, proposed_retake=None, pending_forced_add=None):
    return dict(degraded), 0.0


def fake_call_with_tools(react_messages):
    captured_prompts.append(list(react_messages))
    plan = ["00940224", "00960425", "03940902"]
    tc = types.SimpleNamespace(
        id="c1",
        function=types.SimpleNamespace(
            name="deliver_plan",
            arguments=json.dumps({"course_numbers": plan, "explanation": "swapped the clashing course"}),
        ),
    )
    return types.SimpleNamespace(content=None, tool_calls=[tc]), 0.0


orig_extract, orig_call = arl._extract_student_state, arl._call_with_tools
arl._extract_student_state = fake_extract
arl._call_with_tools = fake_call_with_tools
try:
    conversation = [
        {"role": "user", "content": "Semester 5, failed data structures, no Sundays."},
        {"role": "assistant", "content": "Here is your plan..."},
        {"role": "user", "content": "There are two exams with no gap between them, please fix that."},
    ]
    res = arl.run_agent_turn_v2(
        TRACK_ID, conversation, 0.0,
        known_context={"state": KNOWN_STATE, "previous_plan": PREVIOUS_PLAN},
    )
    check(res["needs_input"] is None, "no needs_input on a revision turn despite degraded extraction")
    check(res["plan_result"] is not None, "planning proceeded and delivered")
    check("merge_known_state" in [t["name"] for t in res["tool_log"]], "merge step visible in trace")

    # Guarantee 2: the planner's messages include the complaint verbatim.
    sent = captured_prompts[0]
    all_text = "\n".join(m.get("content") or "" for m in sent if isinstance(m.get("content"), str))
    check("two exams with no gap" in all_text, "planner sees the student's complaint verbatim")
    check("REVISION" in sent[0]["content"], "revision framing present in system prompt")
    check("00960425" in sent[0]["content"], "previously delivered plan listed in system prompt")
finally:
    arl._extract_student_state = orig_extract
    arl._call_with_tools = orig_call


# --- Part 2b: persistent-issue pushback on revision turns ---
print("--- revision that carries over the complained-about issue is pushed back ---")

import tools as _tools  # noqa: E402

CARRIED = "exam only 2 day(s) after 00940842's exam (minimum 3)"
PLAN = ["00940224", "00960425", "03940902"]

orig_verify = _tools.verify_plan


def stub_verify_carries_issue(track_, plan, passed_, **kw):
    return {"pass": False, "total_credits": 18.0, "workload_score": 60.0,
            "issues": [{"course": "00940842", "reason": CARRIED}]}


deliver_attempts = {"n": 0}


def script_always_deliver(_messages):
    deliver_attempts["n"] += 1
    tc = types.SimpleNamespace(
        id=f"c{deliver_attempts['n']}",
        function=types.SimpleNamespace(
            name="deliver_plan",
            arguments=json.dumps({"course_numbers": PLAN, "explanation": "here"}),
        ),
    )
    return types.SimpleNamespace(content=None, tool_calls=[tc]), 0.0


arl._extract_student_state = fake_extract
arl._call_with_tools = script_always_deliver
_tools.verify_plan = stub_verify_carries_issue
try:
    res = arl.run_agent_turn_v2(
        TRACK_ID,
        [{"role": "user", "content": "two exams too close, fix it"}],
        0.0,
        known_context={"state": KNOWN_STATE, "previous_plan": PREVIOUS_PLAN, "previous_issues": [CARRIED]},
    )
    names = [t["name"] for t in res["tool_log"]]
    check("persistent_issue_pushback" in names, "pushback fired when the carried-over issue survived the revision")
    check(names.count("persistent_issue_pushback") == 1, "pushback fires exactly once (no infinite loop)")
    check(res["stopped_reason"] == "delivered", "second delivery stands after the one-shot pushback")
    check(deliver_attempts["n"] >= 2, "model was actually sent back and tried again")
finally:
    arl._extract_student_state = orig_extract
    arl._call_with_tools = orig_call
    _tools.verify_plan = orig_verify


# --- Part 2c: wrap-up on a revision turn stays close to the previous plan ---
print("--- forced wrap-up prefers the student's own plan over a stranger plan ---")

STRANGER = ["00940591", "00960414", "00960578"]  # disjoint from PREVIOUS_PLAN


def stub_verify_equal(track_, plan, passed_, **kw):
    # Every plan fails identically - only the overlap component can differ.
    return {"pass": False, "total_credits": 15.0, "workload_score": 50.0,
            "issues": [{"course": None, "reason": "some issue"}]}


def script_verify_stranger_never_deliver(_messages):
    n = getattr(script_verify_stranger_never_deliver, "n", 0)
    script_verify_stranger_never_deliver.n = n + 1
    if n == 0:
        tc = types.SimpleNamespace(id="v1", function=types.SimpleNamespace(
            name="verify_plan", arguments=json.dumps({"plan_course_numbers": STRANGER})))
        return types.SimpleNamespace(content=None, tool_calls=[tc]), 0.0
    tc = types.SimpleNamespace(id=f"x{n}", function=types.SimpleNamespace(
        name="check_invariants", arguments=json.dumps({"plan_course_numbers": STRANGER})))
    return types.SimpleNamespace(content=None, tool_calls=[tc]), 0.0


script_verify_stranger_never_deliver.n = 0
arl._extract_student_state = fake_extract
arl._call_with_tools = script_verify_stranger_never_deliver
_tools.verify_plan = stub_verify_equal
# PREVIOUS_PLAN only carries 1 real mandatory course (the retake) and 8.5
# credits - fine for this test's actual purpose (does wrap-up prefer the
# student's OWN seed over a stranger plan?), but the mandatory-course and
# credit-floor top-up backstops (agent_react_loop.py, both added after
# live bugs) would otherwise append more real courses on top of the exact
# seed being checked for here. Lowering both floors to what PREVIOUS_PLAN
# already has keeps this test isolated to the SEED-preference behavior
# it's actually about.
orig_min_mandatory = _tools.DEFAULT_MIN_MANDATORY_COURSES
orig_min_credits = _tools.DEFAULT_MIN_CREDITS
_tools.DEFAULT_MIN_MANDATORY_COURSES = 1
_tools.DEFAULT_MIN_CREDITS = 8
try:
    res = arl.run_agent_turn_v2(
        TRACK_ID,
        [{"role": "user", "content": "swap one course please"}],
        0.0,
        known_context={"state": KNOWN_STATE, "previous_plan": PREVIOUS_PLAN, "previous_issues": []},
    )
    delivered = set(c["course_number"] for c in res["plan_result"]["courses"])
    seeded = [t for t in res["tool_log"] if t["name"] == "verify_plan" and t.get("args", {}).get("seeded")]
    check(len(seeded) == 1, "previous plan was seed-verified at loop start")
    expected_seed = {c for c in PREVIOUS_PLAN if c not in KNOWN_STATE["passed_courses"]}
    check(delivered == expected_seed,
          "wrap-up delivered the student's own plan, not the stranger plan the model wandered to")
    check("kept" in res["plan_result"]["explanation"], "wrap-up explanation is revision-aware (mentions kept courses)")
    check("tool-call budget" not in res["plan_result"]["explanation"], "no internal jargon in student-facing text")
finally:
    arl._extract_student_state = orig_extract
    arl._call_with_tools = orig_call
    _tools.verify_plan = orig_verify
    _tools.DEFAULT_MIN_MANDATORY_COURSES = orig_min_mandatory
    _tools.DEFAULT_MIN_CREDITS = orig_min_credits


# --- Part 2d: a verified swap candidate beats the unchanged seed at wrap-up ---
print("--- swap candidate (requested change) beats the unchanged plan ---")

# PREVIOUS_PLAN with exactly one course swapped out (00960425 -> 03940902).
SWAPPED = [c for c in PREVIOUS_PLAN if c != "00960425"] + ["03940902"]


def script_verify_swap_never_deliver(_messages):
    n = getattr(script_verify_swap_never_deliver, "n", 0)
    script_verify_swap_never_deliver.n = n + 1
    if n == 0:
        tc = types.SimpleNamespace(id="v1", function=types.SimpleNamespace(
            name="verify_plan", arguments=json.dumps({"plan_course_numbers": SWAPPED})))
        return types.SimpleNamespace(content=None, tool_calls=[tc]), 0.0
    tc = types.SimpleNamespace(id=f"x{n}", function=types.SimpleNamespace(
        name="check_invariants", arguments=json.dumps({"plan_course_numbers": SWAPPED})))
    return types.SimpleNamespace(content=None, tool_calls=[tc]), 0.0


script_verify_swap_never_deliver.n = 0
arl._extract_student_state = fake_extract
arl._call_with_tools = script_verify_swap_never_deliver
_tools.verify_plan = stub_verify_equal
try:
    res = arl.run_agent_turn_v2(
        TRACK_ID,
        [{"role": "user", "content": "replace one course please"}],
        0.0,
        known_context={"state": KNOWN_STATE, "previous_plan": PREVIOUS_PLAN, "previous_issues": []},
    )
    delivered = set(c["course_number"] for c in res["plan_result"]["courses"])
    check("00960425" not in delivered and "03940902" in delivered,
          "wrap-up delivered the swap candidate, not the unchanged seed")
finally:
    arl._extract_student_state = orig_extract
    arl._call_with_tools = orig_call
    _tools.verify_plan = orig_verify


# --- Part 2e: explicit removal request is ENFORCED, not suggested ---
print("--- 'replace X' is enforced in code when the model ignores it ---")

REMOVE_TARGET = "00960425"  # in PREVIOUS_PLAN
STUBBORN = list(PREVIOUS_PLAN)  # model keeps delivering the unchanged plan, target included


def extract_with_removal(track, messages, known_state=None, proposed_retake=None, pending_forced_add=None):
    st = dict(degraded)
    st["remove_courses"] = [REMOVE_TARGET]
    return st, 0.0


stubborn_attempts = {"n": 0}


def script_stubborn_deliver(_messages):
    stubborn_attempts["n"] += 1
    tc = types.SimpleNamespace(id=f"d{stubborn_attempts['n']}", function=types.SimpleNamespace(
        name="deliver_plan",
        arguments=json.dumps({"course_numbers": STUBBORN, "explanation": "unchanged"})))
    return types.SimpleNamespace(content=None, tool_calls=[tc]), 0.0


arl._extract_student_state = extract_with_removal
arl._call_with_tools = script_stubborn_deliver
_tools.verify_plan = stub_verify_equal
try:
    res = arl.run_agent_turn_v2(
        TRACK_ID,
        [{"role": "user", "content": f"replace {REMOVE_TARGET} please"}],
        0.0,
        known_context={"state": KNOWN_STATE, "previous_plan": PREVIOUS_PLAN, "previous_issues": []},
    )
    names = [t["name"] for t in res["tool_log"]]
    delivered = set(c["course_number"] for c in res["plan_result"]["courses"])
    check("requested_removals" in names, "removal request recognized and logged")
    check("removal_ignored_pushback" in names, "model pushed back once for ignoring the removal")
    check("removal_enforced" in names, "Python enforced the removal after the second violation")
    check(REMOVE_TARGET not in delivered, "the course the student asked to remove is NOT in the delivered plan")
finally:
    arl._extract_student_state = orig_extract
    arl._call_with_tools = orig_call
    _tools.verify_plan = orig_verify


# --- Part 2f: dropping the retake without being asked is pushed back ---
print("--- dropping the required retake uninvited triggers the one-shot pushback ---")

RETAKE = "00940224"  # KNOWN_STATE's failed course, present in PREVIOUS_PLAN
PLAN_NO_RETAKE = ["00960425", "03940902"]  # model "resolves" the conflict by dropping the retake


def stub_verify_pass(track_, plan, passed_, **kw):
    # Clean verify so the underload/issue-budget guards stay quiet and this
    # test isolates the retake guard specifically.
    return {"pass": True, "total_credits": 18.0, "workload_score": 60.0, "issues": []}


drop_attempts = {"n": 0}


def script_drop_retake(_messages):
    drop_attempts["n"] += 1
    tc = types.SimpleNamespace(id=f"r{drop_attempts['n']}", function=types.SimpleNamespace(
        name="deliver_plan",
        arguments=json.dumps({"course_numbers": PLAN_NO_RETAKE, "explanation": "moved the exam conflict away"})))
    return types.SimpleNamespace(content=None, tool_calls=[tc]), 0.0


arl._extract_student_state = fake_extract
arl._call_with_tools = script_drop_retake
_tools.verify_plan = stub_verify_pass
try:
    res = arl.run_agent_turn_v2(
        TRACK_ID,
        [{"role": "user", "content": "I have a wedding on the data structures exam date, fix it"}],
        0.0,
        known_context={"state": KNOWN_STATE, "previous_plan": PREVIOUS_PLAN, "previous_issues": []},
    )
    names = [t["name"] for t in res["tool_log"]]
    check("retake_dropped_pushback" in names, "pushback fired when the uninvited drop was delivered")
    check(names.count("retake_dropped_pushback") == 1, "retake pushback fires exactly once (no infinite loop)")
    check(res["stopped_reason"] == "delivered", "second delivery stands after the one-shot pushback")
    check(drop_attempts["n"] >= 2, "model was actually sent back and tried again")
finally:
    arl._extract_student_state = orig_extract
    arl._call_with_tools = orig_call
    _tools.verify_plan = orig_verify


# --- Part 2g: an EXPLICIT removal request for the retake is respected ---
print("--- explicitly requested retake removal does NOT trigger the pushback ---")


def extract_remove_retake(track, messages, known_state=None, proposed_retake=None, pending_forced_add=None):
    st = dict(degraded)
    st["remove_courses"] = [RETAKE]
    return st, 0.0


drop_attempts_2 = {"n": 0}


def script_honor_removal(_messages):
    drop_attempts_2["n"] += 1
    tc = types.SimpleNamespace(id=f"h{drop_attempts_2['n']}", function=types.SimpleNamespace(
        name="deliver_plan",
        arguments=json.dumps({"course_numbers": PLAN_NO_RETAKE, "explanation": "removed as you asked"})))
    return types.SimpleNamespace(content=None, tool_calls=[tc]), 0.0


arl._extract_student_state = extract_remove_retake
arl._call_with_tools = script_honor_removal
_tools.verify_plan = stub_verify_pass
try:
    res = arl.run_agent_turn_v2(
        TRACK_ID,
        [{"role": "user", "content": f"please remove {RETAKE} from my plan"}],
        0.0,
        known_context={"state": KNOWN_STATE, "previous_plan": PREVIOUS_PLAN, "previous_issues": []},
    )
    names = [t["name"] for t in res["tool_log"]]
    check("retake_dropped_pushback" not in names, "no retake pushback when the student explicitly asked to remove it")
    check(res["stopped_reason"] == "delivered", "explicit-removal delivery goes through normally")
finally:
    arl._extract_student_state = orig_extract
    arl._call_with_tools = orig_call
    _tools.verify_plan = orig_verify


# --- Part 2f: a pure closing remark never touches the plan ---
# Found live: "Looks perfect, thank you!" - a message with nothing to act
# on - was falling through to a normal revision turn and running the ENTIRE
# planning loop, non-deterministically churning a plan the student already
# accepted (once dropping a real course). intent="acknowledgment" should
# short-circuit before any tool call at all.
print("--- pure acknowledgment ('looks perfect, thanks!') never re-plans, never touches the plan ---")


def fake_extract_acknowledgment(track, messages, known_state=None, proposed_retake=None, pending_forced_add=None):
    d = dict(degraded)
    d["intent"] = "acknowledgment"
    return d, 0.0


def call_with_tools_must_not_fire(_messages):
    raise AssertionError("the planning loop must never run on a pure acknowledgment turn")


arl._extract_student_state = fake_extract_acknowledgment
arl._call_with_tools = call_with_tools_must_not_fire
try:
    res = arl.run_agent_turn_v2(
        TRACK_ID,
        [{"role": "user", "content": "Looks perfect, thank you!"}],
        0.0,
        known_context={"state": KNOWN_STATE, "previous_plan": PREVIOUS_PLAN, "previous_issues": []},
    )
    check(res["plan_result"] is None, "no new plan_result - the frontend keeps whatever's already on the desk")
    check(res["stopped_reason"] == "acknowledged", "stopped_reason marks this as a pure acknowledgment")
    check("deliver_plan" not in json.dumps(res["tool_log"]), "no planning tool ever ran")
    check(res["known_context"]["previous_plan"] == PREVIOUS_PLAN, "previous_plan carried through completely untouched")
    last = res["messages"][-1]
    check(last["role"] == "assistant" and len(last["content"]) > 0, "a short acknowledgment reply is still sent")
finally:
    arl._extract_student_state = orig_extract
    arl._call_with_tools = orig_call


# --- Part 3: without a previous plan, gap-fill still works normally ---
print("--- first turn: gap-fill unaffected ---")
arl._extract_student_state = fake_extract
try:
    res_first = arl.run_agent_turn_v2(TRACK_ID, [{"role": "user", "content": "hi"}], 0.0, known_context=None)
    check(res_first["needs_input"] is not None, "first-turn degraded extraction still asks via buttons")
finally:
    arl._extract_student_state = orig_extract


print()
if failures:
    print(f"{len(failures)} failure(s)")
    sys.exit(1)
print("All revision-continuity checks passed (zero API calls)")
