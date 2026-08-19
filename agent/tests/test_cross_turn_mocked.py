"""
Mocked (zero-API-cost) test of the cross-turn revision flow: every LLM call
is stubbed, so this validates the plumbing - known_context in, revision
prompt built, known_context out - without spending a cent. The live
behavioral validation (does the model actually revise minimally?) is a
separate, deferred task gated on the budget reset.

    cd agent && python3 tests/test_cross_turn_mocked.py
"""

import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_loop  # noqa: E402
import agent_react_loop  # noqa: E402
from data_bundle import get_track  # noqa: E402

TRACK_ID = "SC00001453"

failures: list[str] = []


def check(condition: bool, description: str) -> None:
    if condition:
        print(f"  PASS: {description}")
    else:
        failures.append(description)
        print(f"  FAIL: {description}")


# --- Part 1: extraction receives and is told to preserve known state ---

print("--- extraction: known_state carried into the prompt ---")

captured_prompts: list[str] = []


def fake_call(messages, json_mode, module="LLM"):
    captured_prompts.append(messages[0]["content"])
    return (
        json.dumps(
            {
                "semester_number": 5,
                "constraints": {
                    "excluded_weekdays": [],
                    "pace": "normal",
                    "notes": "wedding on 23.2, exam that day",
                    "override_minimums": False,
                },
                "passed_courses": ["00940219"],
                "failed_courses": ["00940224"],
                "ready_to_plan": True,
                "clarifying_question": None,
                "missing_fields": [],
            }
        ),
        0.0,
    )


original_call = agent_loop._call
agent_loop._call = fake_call
try:
    track = get_track(TRACK_ID)
    known_state = {
        "semester_number": 5,
        "constraints": {"excluded_weekdays": [], "pace": "normal", "notes": "", "override_minimums": False},
        "passed_courses": ["00940219"],
        "failed_courses": ["00940224"],
    }
    state, _ = agent_loop._extract_student_state(
        track,
        [{"role": "user", "content": "I have a wedding on 23.2 and there's an exam that day"}],
        known_state=known_state,
    )
    prompt = captured_prompts[-1]
    check("Already established from an earlier turn" in prompt, "prompt contains the carry-forward preamble")
    check("00940224" in prompt, "prompt carries the known failed course forward")
    check("do not ask about anything already settled" in prompt.lower(), "prompt forbids re-asking settled facts")

    captured_prompts.clear()
    agent_loop._extract_student_state(track, [{"role": "user", "content": "hi"}], known_state=None)
    check(
        "Already established" not in captured_prompts[-1],
        "no carry-forward preamble on a first turn (known_state=None)",
    )
finally:
    agent_loop._call = original_call


# --- Part 2: react loop treats known_context as a revision and echoes it back ---

print("--- react loop: revision prompt + known_context round-trip ---")


def make_deliver_message(course_numbers, explanation):
    tool_call = types.SimpleNamespace(
        id="call_1",
        function=types.SimpleNamespace(
            name="deliver_plan",
            arguments=json.dumps({"course_numbers": course_numbers, "explanation": explanation}),
        ),
    )
    return types.SimpleNamespace(content=None, tool_calls=[tool_call])


captured_react_prompts: list[str] = []
# A verifiable, realistic plan: the retake + real courses from the track.
# Includes enough real mandatory courses (00940224 the retake, plus
# 00960275/00960411, both unlocked given this state) to clear
# tools.DEFAULT_MIN_MANDATORY_COURSES on its own - otherwise the
# mandatory-course top-up backstop in agent_react_loop.py (added after a
# live bug where filler electives got delivered instead of available
# mandatory courses) would append more courses here, breaking this test's
# exact-match assertions below, which are about the known_context plumbing,
# not that specific backstop.
DELIVERED = ["00940224", "00960425", "03940902", "00960275", "00960411"]


def fake_call_with_tools(react_messages):
    captured_react_prompts.append(react_messages[0]["content"])
    return make_deliver_message(DELIVERED, "Swapped only the conflicting course; everything else kept."), 0.0


original_call_with_tools = agent_react_loop._call_with_tools
agent_react_loop._call_with_tools = fake_call_with_tools
try:
    known_context = {
        "state": known_state,
        "previous_plan": ["00940224", "00960425", "00970249"],
    }
    state_override = agent_loop.build_state_from_intake(
        {
            "semester_number": 5,
            "excluded_weekdays": [],
            "pace": "normal",
            "passed_courses": ["00940219"],
            "failed_courses": ["00940224"],
        }
    )
    result = agent_react_loop.run_agent_turn_v2(
        TRACK_ID,
        messages=[{"role": "user", "content": "I have a wedding on 23.2 and there's an exam that day"}],
        cost_so_far=0.0,
        state_override=state_override,
        known_context=known_context,
    )
    prompt = captured_react_prompts[-1]
    check("REVISION, not a fresh request" in prompt, "system prompt flags the turn as a revision")
    check("00970249" in prompt, "system prompt lists the previously delivered plan's courses")
    check("not a completely different plan" in prompt, "system prompt demands minimal change")

    check(result["known_context"] is not None, "response carries known_context")
    check(
        result["known_context"]["previous_plan"] == DELIVERED,
        "known_context.previous_plan is the newly delivered plan",
    )
    check(
        result["known_context"]["state"]["failed_courses"] == ["00940224"],
        "known_context.state preserves failed_courses",
    )
    check(result["plan_result"] is not None, "plan_result present")

    # First-turn behavior: no known_context -> no revision note.
    captured_react_prompts.clear()
    result_first = agent_react_loop.run_agent_turn_v2(
        TRACK_ID,
        messages=[{"role": "user", "content": "starting semester 5, failed data structures"}],
        cost_so_far=0.0,
        state_override=state_override,
        known_context=None,
    )
    check("REVISION" not in captured_react_prompts[-1], "no revision note on a first turn")
    check(result_first["known_context"]["previous_plan"] == DELIVERED, "first turn still returns known_context")
finally:
    agent_react_loop._call_with_tools = original_call_with_tools


print()
if failures:
    print(f"{len(failures)} failure(s)")
    sys.exit(1)
print("All mocked cross-turn checks passed (zero API calls)")
