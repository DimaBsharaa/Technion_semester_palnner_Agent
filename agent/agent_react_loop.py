"""
Stage 1 of the pipeline -> agent migration: a real, bounded, tool-calling
agent loop, replacing agent_loop.py's hardcoded draft -> verify -> repair ->
compare -> explain sequence with one loop where the MODEL decides which
tool to call and when it's done - genuine OpenAI tool-calling
(`tools=[...]`), not JSON-mode structured output at fixed Python-decided
points.

Extraction (was there enough info to plan at all?) stays exactly as it is
in agent_loop.py - that was never the part anyone objected to; it already
makes two real model-driven decisions (ready_to_plan, missing_fields) and
reuses proven, tested logic. This module only takes over once that's
settled: given a fully-resolved student state, how should the actual
semester plan get built?

The property that actually fixed the original free-form tool-calling
design's reliability problems is preserved and, if anything, strengthened
here, not removed:

1. Nothing the model does in this loop is visible to the student until it
   calls the one terminal tool, deliver_plan. It can loop, retry, call the
   same tool twice - none of that reaches a chat turn. The failure mode
   that killed the original design (asking permission mid-conversation,
   rambling) structurally cannot happen here, because there is no
   student-facing turn inside the loop, exactly like the old repair loop.
2. Python still owns the ceiling: MAX_STEPS and SESSION_BUDGET_USD_CAP are
   hard limits. If the model never calls deliver_plan, Python forces a
   wrap-up using the last plan it verified (or an honest "ran out of
   budget" note if it verified nothing).
3. Python still owns final correctness: deliver_plan's course list is
   always re-run through tools.verify_plan and tools.check_invariants
   here, regardless of what the model claims about its own plan - the
   model's own bookkeeping is never trusted for hard constraints.

Enabled via AGENT_MODE=react in main.py; agent_loop.py's pipeline is
untouched and remains the default until this is validated against
tests/test_scenarios.py and live adversarial testing (see the migration
plan).
"""

import json
import os

import tools
from agent_loop import (
    BASE_URL,
    MODEL,
    PRICE_INPUT_PER_1M,
    PRICE_OUTPUT_PER_1M,
    _catalog_with_meta_text,
    _extract_student_state,
    _get_client,
    _normalize_state,
    answer_course_question,
    backfill_passed_courses,
    merge_known_state,
    resolve_verify_kwargs,
)
from data_bundle import Track, get_track
from tool_schemas import REACT_TOOL_SCHEMAS, TERMINAL_TOOL_NAME, build_dispatch

# With the always-needed diagnostics pre-injected (see PREINJECTED_TOOL_NAMES),
# a full turn is now typically draft -> verify -> maybe repair -> maybe
# compare -> deliver, so fewer steps are needed. A tighter ceiling both caps
# worst-case spend and forces the model to converge instead of dithering.
MAX_STEPS = int(os.environ.get("REACT_MAX_STEPS", "16"))
SESSION_BUDGET_USD_CAP = float(os.environ.get("SESSION_BUDGET_USD_CAP", "0.50"))


def _call_with_tools(messages: list[dict]) -> tuple[object, float]:
    response = _get_client().chat.completions.create(
        model=MODEL, messages=messages, tools=REACT_TOOL_SCHEMAS, tool_choice="auto"
    )
    usage = response.usage
    cost = (usage.prompt_tokens / 1_000_000) * PRICE_INPUT_PER_1M + (
        usage.completion_tokens / 1_000_000
    ) * PRICE_OUTPUT_PER_1M
    return response.choices[0].message, cost


def _available_courses_text(track: Track, passed: list[str]) -> str:
    """The courses the student can ACTUALLY take next semester: offered then,
    prerequisites already satisfied, and not already passed. The model was
    otherwise handed the whole catalog (including locked/future courses) and
    had to infer eligibility itself - and got it wrong, padding credits with
    filler when it couldn't tell what was really takeable. Handing it this
    shortlist directly is the difference between a real 18-credit plan and a
    stack of one-point sport courses."""
    passed_set = set(passed)
    rows = []
    for num, c in sorted(track.courses.items()):
        if num in passed_set or not c.get("offered_next_semester"):
            continue
        if not tools._prereq_satisfied(c["prerequisites"], passed_set):
            continue
        cf = c["cheesefork"]
        load = f"{cf['avg_difficulty']}/5 diff" if cf["avg_difficulty"] is not None else "no reviews"
        sat = f", {cf['avg_general']}/5 liked" if cf["avg_general"] is not None else ""
        exams = c["exams"].get("moed_a", [])
        exam = f"exam {exams[0]['date']}" if exams else "no exam"
        tag = "MANDATORY" if num in track.mandatory_course_numbers else "elective"
        rows.append(f"{num} — {c['name']} ({c['points']} pts, {load}{sat}, {exam}, {tag})")
    if not rows:
        return "(none - every remaining course is either not offered next semester or has unmet prerequisites)"
    return "\n".join(rows)


def _sport_courses_in(track: Track, course_numbers: list[str]) -> list[str]:
    """Sport/PE course numbers in a plan, identified by name - used by the
    deterministic anti-padding guard, since 'at most one sport course' in
    the prompt alone was observed being ignored live."""
    sport = []
    for c in course_numbers:
        name = track.courses.get(c, {}).get("name", "")
        if "גופני" in name or "ספורט" in name:
            sport.append(c)
    return sport


def _plan_score(
    verify_result: dict | None,
    plan: list[str] | None = None,
    previous_plan: list[str] | None = None,
) -> tuple:
    """Higher is better. A passing plan always beats a failing one; then -
    on revision turns - a plan that keeps more of the previously delivered
    courses beats a stranger plan (a student who asked to swap ONE course
    must never get back an unrelated plan just because it scored well);
    then fewer open issues, then a fuller credit load. Used to keep the
    BEST plan verified this turn - not merely the most recent - so a
    fallback never hands the student a worse or unrecognizable plan."""
    if not verify_result:
        return (-1, 0, 0, 0)
    overlap = len(set(plan or []) & set(previous_plan or []))
    return (
        1 if verify_result.get("pass") else 0,
        overlap,
        -len(verify_result.get("issues", [])),
        verify_result.get("total_credits", 0),
    )


def _system_prompt(
    track: Track,
    state: dict,
    semester_number: int,
    progress: dict,
    catalog: dict,
    prereq: dict,
    available_text: str,
    previous_plan: list[str] | None = None,
    previous_issues: set[str] | None = None,
    requested_removals: list[str] | None = None,
) -> dict:
    revision_note = ""
    if previous_plan:
        plan_names = [
            f"{c} ({track.courses[c]['name']})" if c in track.courses else c for c in previous_plan
        ]
        removals_note = ""
        if requested_removals:
            removal_names = [
                f"{c} ({track.courses[c]['name']})" if c in track.courses else c for c in requested_removals
            ]
            removals_note = (
                "\nHARD REQUIREMENT: the student explicitly asked to REMOVE these course(s) - "
                "the delivered plan MUST NOT contain them, no exceptions:\n"
                + "\n".join("- " + n for n in removal_names)
                + "\nReplace them with suitable alternatives from the available-courses shortlist."
            )
        issues_note = ""
        if previous_issues:
            issues_note = (
                "\nThat plan had these open issues when it was delivered:\n"
                + "\n".join("- " + i for i in sorted(previous_issues))
                + "\nIf the student's feedback is about one of these, RESOLVING it is the "
                "whole point of this turn - swap out the involved course(s); carrying the "
                "issue over unchanged is a failed revision."
            )
        revision_note = f"""

IMPORTANT - this is a REVISION, not a fresh request. The student already \
received this plan in an earlier turn:
{chr(10).join("- " + n for n in plan_names)}
{removals_note}
{issues_note}
Their latest message is feedback on it. Start from that exact plan: keep \
every course that still fits the updated constraints, and change ONLY what \
the new information actually breaks - but "keep the rest" NEVER means \
keeping the specific problem the student complained about. A student who \
reports one exam conflict expects the offending course swapped, not a \
completely different plan - call verify_plan on your minimally-changed \
version first, before considering anything more drastic."""

    return {
        "role": "system",
        "content": f"""\
You are an experienced Technion academic advisor building a {track.target_semester} \
course plan for a "{track.name}" student, with real tools to check your own \
work before committing to an answer.
{revision_note}

Hard rule, never negotiable: never include a course from the student's \
passed list below in the plan - it's already completed, retaking it would \
waste a slot. Only a course from the failed/outstanding list may reappear, \
and only as a deliberate retake. check_invariants will catch this if you \
miss it, but you shouldn't rely on that.

This is how that advisor actually thinks, in priority order:

1. Retaking a failed course with a high gateway_score (see the \
prerequisite-graph analysis below) happens now, regardless of its \
difficulty - this is not up for negotiation with the student's stated \
pace. A blocked degree costs more than one hard semester.
2. The student's pace preference shapes everything BUILT AROUND that \
retake, not whether it happens: pair a heavy mandatory retake with \
genuinely light electives or "no exam" courses (sport/PE, choir - look for \
"no exam" in the catalog below); if the mandatory side of the plan is \
already light, there's more room for a substantive elective instead of \
just the easiest option available. Look at each candidate's difficulty and \
exam presence in the catalog to make this trade-off directly, not just to \
react to the credit total.
3. Exam-date clustering is the single most common real-advisor mistake to \
avoid: look at the exam date printed for each candidate course and \
actively avoid putting two exams within a few days of each other - don't \
wait for verify_plan to catch this after the fact, route around it while \
choosing.
4. Unless the student's constraints say override_minimums: true, include \
at least {tools.DEFAULT_MIN_MANDATORY_COURSES} mandatory courses if that \
many remain. Target {tools.DEFAULT_MIN_CREDITS} credit points total - a \
bit above ({tools.DEFAULT_MAX_CREDITS} max) is perfectly fine, don't trim \
a good plan just to land exactly on {tools.DEFAULT_MIN_CREDITS}. If the \
student's pace is "light", the target drops to \
{tools.LIGHT_PACE_MIN_CREDITS}-{tools.LIGHT_PACE_MAX_CREDITS} instead - \
still a real course load, just the lighter end of one. Use lighter \
electives or no-exam courses to round out credits rather than leaving the \
semester underloaded.
5. When choosing which elective fills a credit gap, prefer one with a \
decent satisfaction rating (shown in the catalog) over a similarly-light \
one students clearly dislike (roughly below 2.5/5) - light workload \
doesn't excuse picking something students regret taking, when an equally \
light alternative with a better rating is available.
6. simulate_future_impact's bottleneck_warnings are advisory, not a hard \
rule: a course you're leaving for a later semester that would block \
several others (gateway_score 2+) is worth weighing against everything \
else above, but don't force it into this plan if doing so breaks a \
higher-priority rule - the retake (rule 1) and exam-spacing (rule 3) still \
come first.
7. If you use compare_plans to weigh two real alternatives, don't just \
pick whichever has fewer verify_plan issues - consider what actually \
serves this specific student better given their stated situation (pace, \
constraints, how close they are to falling behind).

Student's semester: {semester_number}
Student's passed courses: {state["passed_courses"]}
Student's failed/outstanding required courses: {state["failed_courses"]}
Student's constraints: {json.dumps(state["constraints"], ensure_ascii=False)}

Full course catalog (number - name (points, difficulty, exam date, \
mandatory/elective status)):
{_catalog_with_meta_text(track.otjid)}

Diagnostics already computed for you (you do NOT need to fetch these):

Progress vs. a typical student at this point in the track:
{json.dumps(progress, ensure_ascii=False)}

Degree requirement audit (per category, what's left):
{json.dumps(catalog, ensure_ascii=False)}

Prerequisite-graph analysis of the student's failed/outstanding courses \
(what each blocks, gateway scores):
{json.dumps(prereq, ensure_ascii=False)}

COURSES THE STUDENT CAN ACTUALLY TAKE NEXT SEMESTER (offered then, \
prerequisites already met, not yet passed) - build the plan from THIS \
shortlist; anything not here is either locked or not offered, so don't \
include it:
{available_text}

Build a real, substantive semester from that shortlist. Reach the credit \
target with genuine mandatory courses and worthwhile electives - do NOT \
pad the credit count with a stack of one-point sport/PE courses. At most \
ONE sport/PE course in a plan; if you still can't reach the target after \
using the real courses available, that's a genuine finding to report \
honestly, not something to paper over with filler.

Your job is the judgment, not the bookkeeping: using the diagnostics above, \
draft a candidate course list, then use your tools to check and refine it. \
Available tools - fetch_exam_dates and summarize_cheesefork (to inspect \
candidates), simulate_future_impact (planning-horizon check), \
roadmap_to_graduation (project the remaining semesters assuming your \
candidate completes - use it to justify choices by what they unlock), \
risk_report (stress-test your final candidate: fragile courses, exam \
crunches, stacked heavy courses - call it once before delivering), \
verify_plan (the critic - ALWAYS call this on your candidate before \
delivering), compare_plans (to weigh two real alternatives), \
check_invariants, search_courses (to resolve a course name). You have at \
most {MAX_STEPS} tool calls this turn - budget them; never call the same \
tool with the same arguments twice.

You MUST end the turn by calling deliver_plan exactly once - never respond \
with plain text instead. deliver_plan's course_numbers MUST be exactly the \
same list you most recently passed to verify_plan (or check_invariants) - \
never introduce a new, never-verified course at the last moment; if you \
want to change the plan, call verify_plan on the new list first. Call \
deliver_plan once you have a plan that verifies, or once you've concluded \
that no further attempt will resolve the remaining issue and this is the \
best achievable option.

Write deliver_plan's explanation as the final message to the student: 3-5 \
short sentences, not a report. The student already sees a visual breakdown \
with every course's name, credits, difficulty, satisfaction rating, and a \
review excerpt, plus an exam calendar - do NOT repeat that in prose, write \
only what the visual can't show: the reasoning behind the choices. Cover, \
only briefly: whether it verified and the credit/workload numbers in one \
clause; why this specific combination (which retake mattered and why, \
what got deferred and why - name it once, don't re-justify it course by \
course); a progress-vs-typical-student comparison ONLY if you're behind or \
slightly behind schedule and it's relevant to urgency (skip entirely if on \
track - that's noise); and, if one elective has a notably poor \
satisfaction rating (roughly below 2.5/5) and a better same-credit \
alternative existed, say so in one clause - otherwise don't comment on \
elective quality at all, the cards already show it. If verification did \
NOT pass, be completely honest about it rather than glossing over it: \
name the specific unresolved issue(s) in plain language (e.g. "the only \
section of X meets on a day you excluded, and no alternative section \
exists this semester") so the student has the whole picture, not just a \
vague "didn't verify." Say plainly that this is the best option found \
given every constraint, and that it's their call how to proceed (drop the \
constraint, accept the conflict, or take the course a different \
semester) - this is a genuine trade-off to disclose, not a \
permission-seeking question about an obvious next step. Do not ask the \
student any other question or offer to build an alternative as a pending \
option beyond disclosing the trade-off - if a real trade-off exists, \
state it as already-considered context in this same message.""",
    }


class _EmittingList(list):
    """A tool_log that notifies a callback on every append - lets the HTTP
    layer stream the agent's real steps to the browser as they happen,
    without touching a single line of loop logic."""

    def __init__(self, on_event):
        super().__init__()
        self._on_event = on_event

    def append(self, item):
        super().append(item)
        try:
            self._on_event(item)
        except Exception:
            pass  # a broken stream consumer must never break the turn itself


def run_agent_turn_v2(
    track_id: str,
    messages: list[dict],
    cost_so_far: float,
    state_override: dict | None = None,
    known_context: dict | None = None,
    on_event=None,
) -> dict:
    """Same external contract as agent_loop.run_agent_turn (messages,
    cost_usd, tool_log, stopped_reason, plan_result, needs_input) - the
    frontend and main.py cannot tell which implementation answered.

    known_context ({"state": {...}, "previous_plan": [...]}) carries
    forward what an earlier turn already resolved, so a follow-up message
    is treated as a revision of the existing plan against known facts,
    not a from-scratch re-derivation that forgets what the student was
    already shown."""
    track = get_track(track_id)
    tool_log = _EmittingList(on_event) if on_event else []
    cost = cost_so_far
    known_state = (known_context or {}).get("state")
    previous_plan = (known_context or {}).get("previous_plan")
    previous_issues = set((known_context or {}).get("previous_issues") or [])

    if state_override is not None:
        state = state_override
        tool_log.append({"name": "structured_intake", "args": {}, "result": state})
    else:
        state, c = _extract_student_state(track, messages, known_state=known_state)
        cost += c
        tool_log.append({"name": "extract_student_state", "args": {}, "result": state})
        if previous_plan and known_state:
            # A plan was already delivered this conversation, so its facts
            # are settled: enforce continuity in code (extraction refines,
            # never un-knows) and always proceed to planning - a revision
            # turn must never bounce the student back to the intake buttons,
            # which is exactly the failure observed live when a student
            # pointing out an exam conflict got re-asked what they'd passed.
            state = _normalize_state(merge_known_state(state, known_state))
            tool_log.append({"name": "merge_known_state", "args": {}, "result": state})

    # Question turns ("is X hard?") get one grounded answer from the real
    # review data - no planning loop, no gap-fill buttons, plan untouched.
    if state.get("intent") == "question" and state_override is None:
        answer, c = answer_course_question(track, state, messages)
        cost += c
        tool_log.append(
            {"name": "answer_course_question", "args": {"courses": state.get("question_courses", [])}, "result": {}}
        )
        return {
            "messages": messages + [{"role": "assistant", "content": answer}],
            "cost_usd": cost,
            "tool_log": tool_log,
            "stopped_reason": "answered_question",
            "plan_result": None,
            "needs_input": None,
            # Pass the existing context through untouched - a question
            # doesn't change the student's state or their plan.
            "known_context": known_context,
        }

    if not state.get("ready_to_plan"):
        reply = state.get("clarifying_question") or "Just need a couple more things:"
        messages = messages + [{"role": "assistant", "content": reply}]
        return {
            "messages": messages,
            "cost_usd": cost,
            "tool_log": tool_log,
            "stopped_reason": "done",
            "plan_result": None,
            "needs_input": {
                "missing_fields": state.get("missing_fields", []),
                "semester_number": state.get("semester_number"),
                "constraints": state.get("constraints"),
                "passed_courses": state.get("passed_courses"),
                "failed_courses": state.get("failed_courses"),
            },
            # Don't lose already-established facts on a not-ready turn.
            "known_context": {
                "state": {
                    "semester_number": state.get("semester_number"),
                    "constraints": state.get("constraints"),
                    "passed_courses": state.get("passed_courses"),
                    "failed_courses": state.get("failed_courses"),
                },
                "previous_plan": previous_plan,
                "previous_issues": sorted(previous_issues),
            },
        }

    semester_number = state.get("semester_number") or 0
    passed, failed = backfill_passed_courses(track, state)
    state["passed_courses"] = passed
    state["failed_courses"] = failed
    excluded_weekdays = state["constraints"].get("excluded_weekdays") or []
    verify_kwargs = resolve_verify_kwargs(state)

    dispatch = build_dispatch(track, semester_number, passed, failed, excluded_weekdays, verify_kwargs)

    # Pre-compute the always-needed deterministic diagnostics once, here in
    # code, and hand them to the model in its prompt - rather than making it
    # spend three paid round-trips fetching them. They're logged like tool
    # calls so the trace still shows they happened.
    progress = tools.assess_progress(track, semester_number, passed)
    catalog = tools.fetch_catalog(track, passed)
    prereq = tools.query_prereq_graph(track, passed, failed)
    tool_log.append({"name": "assess_progress", "args": {"preinjected": True}, "result": progress})
    tool_log.append({"name": "fetch_catalog", "args": {"preinjected": True}, "result": catalog})
    tool_log.append({"name": "query_prereq_graph", "args": {"preinjected": True}, "result": prereq})

    available_text = _available_courses_text(track, passed)

    # Courses the student EXPLICITLY asked to remove/replace this turn -
    # extraction resolves the names to numbers; Python enforces the removal
    # on whatever gets delivered, because "please replace X" coming back
    # with X still in the plan was observed live even with the request in
    # the prompt. An explicit student directive is a hard constraint, not
    # a suggestion the model may weigh.
    requested_removals = [c for c in state.get("remove_courses", []) if c in (previous_plan or [])]
    if requested_removals:
        tool_log.append(
            {"name": "requested_removals", "args": {}, "result": {"course_numbers": requested_removals}}
        )

    react_messages = [
        _system_prompt(
            track,
            state,
            semester_number,
            progress,
            catalog,
            prereq,
            available_text,
            previous_plan=previous_plan,
            previous_issues=previous_issues,
            requested_removals=requested_removals,
        )
    ]
    # The loop otherwise only sees the condensed extracted state - give it
    # the recent conversation verbatim, not just the single last message.
    # This matters concretely: after a gap-fill form submit, the "last
    # message" is the form's summary, and the student's actual complaint
    # ("two exams with no gap between them") sits one message earlier - a
    # planner that only saw the summary replanned from scratch and ignored
    # the complaint entirely (observed live).
    tail = [m for m in messages if m.get("role") in ("user", "assistant") and m.get("content")][-6:]
    if tail:
        transcript = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in tail)
        react_messages.append(
            {
                "role": "user",
                "content": (
                    "Recent conversation with the student, verbatim (oldest first - "
                    "weigh the LATEST student message most heavily; it is what this "
                    f"turn must address):\n\n{transcript}"
                ),
            }
        )
    last_verified_plan: list[str] | None = None
    last_verify_result: dict | None = None
    # The single best plan verified this turn (by _plan_score), so a fallback
    # never hands over something worse than one already found to work.
    best_verified_plan: list[str] | None = None
    best_verify_result: dict | None = None
    verified_course_sets: set[frozenset] = set()

    # Revision-turn seed: verify the PREVIOUS plan against the current
    # constraints up front (pure Python, no model call). Kept SEPARATE from
    # best-plan tracking: at wrap-up it competes with a one-course overlap
    # handicap, so a model candidate that actually made the requested change
    # (overlap n-1) beats "your plan, unchanged" (overlap n) - but a
    # stranger plan (low overlap) still loses to the seed. Without the
    # handicap the unchanged plan mathematically beats every swap, which is
    # exactly the failure observed live: "replace X" came back with X still
    # in the plan.
    seed_plan: list[str] | None = None
    seed_verify: dict | None = None
    if previous_plan:
        seed_plan = [
            c for c in previous_plan if c in track.courses and c not in passed and c not in requested_removals
        ]
        if seed_plan:
            seed_verify = tools.verify_plan(
                track, seed_plan, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
            )
            tool_log.append(
                {"name": "verify_plan", "args": {"plan_course_numbers": seed_plan, "seeded": True}, "result": seed_verify}
            )
            verified_course_sets.add(frozenset(seed_plan))
    verify_nudged = False
    underload_pushed = False
    persistent_issue_pushed = False
    sport_padding_pushed = False
    removal_pushed = False
    issues_pushed = False
    stopped_reason = "step_limit_exhausted"
    final_course_numbers: list[str] = []
    final_explanation = ""
    course_reasons: dict = {}

    for step in range(MAX_STEPS):
        # Per-TURN spend cap, not per-conversation: cost_so_far is the whole
        # conversation's history, and capping against the running total made
        # every conversation degrade into budget-capped wrap-ups by turn
        # 3-4 (found during test-plan execution). Each turn gets the same
        # fresh allowance; the dashboard hard limit remains the real
        # backstop for total spend.
        if (cost - cost_so_far) >= SESSION_BUDGET_USD_CAP:
            stopped_reason = "budget_cap"
            break

        # Two steps before the ceiling, tell the model to close out NOW
        # with its best verified candidate - a self-chosen delivery with a
        # real explanation always beats drifting into the generic forced
        # wrap-up.
        if step == MAX_STEPS - 2:
            tool_log.append({"name": "deliver_now_nudge", "args": {"steps_left": 2}, "result": {}})
            react_messages.append(
                {
                    "role": "user",
                    "content": (
                        "You are nearly out of tool calls for this turn. Stop exploring and call "
                        "deliver_plan NOW with the best course list you have already verified."
                    ),
                }
            )

        assistant_message, c = _call_with_tools(react_messages)
        cost += c

        if not assistant_message.tool_calls:
            # The model responded with plain text instead of a tool call.
            # Per the system prompt this shouldn't happen, but nothing it
            # says here is student-facing regardless - treat it as a
            # nudge, not a leak, and give it one more chance to comply.
            react_messages.append({"role": "assistant", "content": assistant_message.content or ""})
            react_messages.append(
                {"role": "user", "content": "You must call a tool, ending with deliver_plan when done - not reply in plain text."}
            )
            continue

        react_messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in assistant_message.tool_calls
                ],
            }
        )

        delivered = False
        for tool_call in assistant_message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == TERMINAL_TOOL_NAME:
                candidate = [c for c in args.get("course_numbers", []) if c in track.courses]

                # Explicit-removal enforcement: the student BY NAME asked for
                # specific course(s) out of the plan. First violation: send
                # the model back once to do the swap properly (so a real
                # replacement gets picked). Second violation: strip the
                # course(s) in code - an explicit student directive is a
                # hard constraint, and "please replace X" returning a plan
                # that still contains X was observed live.
                ignored_removals = [c for c in candidate if c in requested_removals]
                if ignored_removals:
                    if not removal_pushed and step < MAX_STEPS - 1:
                        removal_pushed = True
                        removal_names = ", ".join(
                            f"{c} ({track.courses[c]['name']})" for c in ignored_removals if c in track.courses
                        )
                        tool_log.append(
                            {"name": "removal_ignored_pushback", "args": {}, "result": {"still_in_plan": ignored_removals}}
                        )
                        react_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": (
                                    f"Not delivered: the student explicitly asked to REMOVE {removal_names}, "
                                    "but your plan still contains it. Remove it, pick a suitable replacement "
                                    "from the available-courses shortlist, verify the new list, and deliver."
                                ),
                            }
                        )
                        continue
                    # Model ignored the directive twice - enforce in code.
                    candidate = [c for c in candidate if c not in requested_removals]
                    tool_log.append(
                        {"name": "removal_enforced", "args": {}, "result": {"stripped": ignored_removals}}
                    )

                # Verify-before-deliver nudge: if the model tries to deliver
                # without ever having run verify_plan this turn, send it back
                # once to actually check its work. This costs one extra step
                # but is the difference between a real critic loop and a
                # single-shot guess. Guarded so it can only fire once (never
                # an infinite loop) and skipped entirely once the budget/step
                # ceiling is close, where correctness is handled server-side
                # anyway.
                if not verified_course_sets and not verify_nudged and step < MAX_STEPS - 2:
                    verify_nudged = True
                    tool_log.append({"name": "verify_before_deliver_nudge", "args": {}, "result": {}})
                    react_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Not delivered: you must call verify_plan on this exact course list and address any issues before delivering. Call verify_plan now.",
                        }
                    )
                    # continue (not break) so any sibling tool calls in this
                    # same message still get their required tool responses.
                    continue

                final_explanation = args.get("explanation") or ""
                course_reasons = args.get("course_reasons") or {}
                # Audit note only, not a safety gate - verify_plan and
                # check_invariants are re-run unconditionally below
                # regardless of this, so the student is never shown a
                # falsely-"verified" plan either way. This just makes it
                # visible in tool_log when the model delivers a course list
                # it never actually ran through verify_plan itself.
                if frozenset(candidate) not in verified_course_sets:
                    tool_log.append(
                        {
                            "name": "audit_note",
                            "args": {},
                            "result": {"note": "delivered plan does not match any course set the model itself ran through verify_plan this turn"},
                        }
                    )
                verify_result = tools.verify_plan(
                    track, candidate, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
                )
                violations = tools.check_invariants(track, candidate, passed, failed)
                if violations:
                    tool_log.append({"name": "check_invariants", "args": {}, "result": {"violations": violations}})
                    seen: set[str] = set()
                    candidate = [c for c in candidate if c not in passed and c in track.courses and not (c in seen or seen.add(c))]
                    verify_result = tools.verify_plan(
                        track, candidate, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
                    )
                    # The model's own explanation was written for the plan
                    # BEFORE this correction - it can describe issues that
                    # no longer exist in the corrected list (e.g. narrating
                    # exam-clustering from a candidate that's since been
                    # stripped down). Once Python has changed what's
                    # actually being delivered, only a freshly-generated,
                    # deterministic account of the corrected list's real
                    # issues is trustworthy - same principle as the old
                    # pipeline never trusting stale reasoning.
                    final_explanation = (
                        "The plan you were about to see needed a correction: "
                        + "; ".join(violations)
                        + f". After fixing that, it now has {verify_result['total_credits']} credits "
                        + ("and verifies cleanly." if verify_result["pass"] else "but still has open issues: "
                           + "; ".join(i["reason"] for i in verify_result["issues"]) + ".")
                    )
                # The delivered plan counts toward best-tracking too, so the
                # empty-plan fallback below can reach for it if needed.
                if candidate and _plan_score(verify_result, candidate, previous_plan) > _plan_score(
                    best_verify_result, best_verified_plan, previous_plan
                ):
                    best_verified_plan = candidate
                    best_verify_result = verify_result

                # Underload pushback: don't let the model hand over a
                # half-empty semester. If the plan is clearly below the
                # credit floor (not just a hair under) and there's still
                # room to iterate, send it back ONCE to add courses rather
                # than accept a punt. Fires at most once; skipped when the
                # student opted out of minimums (min_credits 0) or near the
                # step ceiling where correctness is handled anyway.
                min_credits = verify_kwargs.get("min_credits", tools.DEFAULT_MIN_CREDITS)
                if (
                    not underload_pushed
                    and step < MAX_STEPS - 1
                    and candidate
                    and min_credits > 0
                    and verify_result.get("total_credits", 0) < min_credits
                ):
                    underload_pushed = True
                    tool_log.append(
                        {"name": "underload_pushback", "args": {}, "result": {"credits": verify_result.get("total_credits")}}
                    )
                    react_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": (
                                f"Not delivered: this plan is only {verify_result.get('total_credits')} credits, "
                                f"well below the {min_credits}-credit target. Add more courses that are offered next "
                                "semester and whose prerequisites are met (use the requirement audit's remaining "
                                "courses and the catalog), then call verify_plan on the fuller list and deliver that. "
                                "Only deliver an underloaded plan if you have genuinely exhausted the available "
                                "courses under the student's constraints."
                            ),
                        }
                    )
                    continue

                # Sport-padding pushback: "at most one sport/PE course" in
                # the prompt was observed being ignored live (a delivered
                # plan carried FOUR one-point sport courses as credit
                # filler). Enforce it in code like every other rule that
                # must actually hold: one shot, then the delivery stands.
                sport = _sport_courses_in(track, candidate)
                if len(sport) > 1 and not sport_padding_pushed and step < MAX_STEPS - 1:
                    sport_padding_pushed = True
                    tool_log.append(
                        {"name": "sport_padding_pushback", "args": {}, "result": {"sport_courses": sport}}
                    )
                    react_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": (
                                f"Not delivered: this plan contains {len(sport)} sport/PE courses "
                                f"({', '.join(sport)}) - the limit is ONE. Replace the extras with real "
                                "courses from the available-courses shortlist (substantive electives or "
                                "remaining requirements), verify the new list, and deliver that."
                            ),
                        }
                    )
                    continue

                # Fixable-issues pushback (first turns): the happy-path test
                # delivered a failing 21-credit plan with prereq-blocked
                # courses and clashing exams while NO guard fired - the
                # issue-carryover guard below only covers revisions. On a
                # first turn with a completely unconstrained student, a
                # failing verify is almost always fixable - send the model
                # back ONCE with the concrete issues before accepting.
                if (
                    verify_result.get("issues")
                    and not previous_plan
                    and not issues_pushed
                    and step < MAX_STEPS - 1
                ):
                    issues_pushed = True
                    tool_log.append(
                        {"name": "fixable_issues_pushback", "args": {},
                         "result": {"issues": [i["reason"] for i in verify_result["issues"]]}}
                    )
                    react_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": (
                                "Not delivered: this plan has fixable issues: "
                                + "; ".join(i["reason"] for i in verify_result["issues"])
                                + ". Swap or drop the specific courses involved (use the "
                                "available-courses shortlist - it only contains courses whose "
                                "prerequisites are already met), verify the corrected list, and "
                                "deliver that. Only re-deliver with an issue intact if it is "
                                "genuinely unavoidable - then say so explicitly."
                            ),
                        }
                    )
                    continue

                # Persistent-issue pushback (revision turns only): the
                # student came back BECAUSE the previous plan had problems.
                # If the "revised" plan still carries an issue verbatim
                # identical to one the previous plan had, the revision
                # didn't do its one job - send it back ONCE with the exact
                # carried-over issues named, demanding a swap of the
                # involved course(s). One shot only: if the model concludes
                # after a real attempt that the issue is genuinely
                # unavoidable, the second delivery stands (with the honest
                # disclosure the explanation rules already require).
                persistent = [
                    i["reason"] for i in verify_result.get("issues", []) if i["reason"] in previous_issues
                ]
                if (
                    previous_plan
                    and persistent
                    and not persistent_issue_pushed
                    and step < MAX_STEPS - 1
                ):
                    persistent_issue_pushed = True
                    tool_log.append(
                        {"name": "persistent_issue_pushback", "args": {}, "result": {"carried_over": persistent}}
                    )
                    react_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": (
                                "Not delivered: the student asked you to FIX the previous plan, but this "
                                "revision still carries these exact unresolved issue(s) from it: "
                                + "; ".join(persistent)
                                + ". 'Keep the rest of the plan' never means keeping the specific problem "
                                "the student complained about. Swap out the course(s) involved in the "
                                "carried-over issue(s) for alternatives from the available-courses "
                                "shortlist, verify the new list, and deliver that. Only re-deliver with "
                                "the issue intact if no alternative in the shortlist can resolve it - and "
                                "then say so explicitly in the explanation."
                            ),
                        }
                    )
                    continue

                tool_log.append(
                    {"name": "deliver_plan", "args": {"course_numbers": candidate}, "result": verify_result}
                )
                final_course_numbers = candidate
                last_verify_result = verify_result
                stopped_reason = "delivered"
                delivered = True
                break

            try:
                result = dispatch[name](args)
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
            tool_log.append({"name": name, "args": args, "result": result})
            react_messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result, ensure_ascii=False)}
            )

            if name == "verify_plan" and not result.get("error"):
                last_verified_plan = args.get("plan_course_numbers", [])
                last_verify_result = result
                verified_course_sets.add(frozenset(last_verified_plan))
                if _plan_score(result, last_verified_plan, previous_plan) > _plan_score(
                    best_verify_result, best_verified_plan, previous_plan
                ):
                    best_verified_plan = last_verified_plan
                    best_verify_result = result

        if delivered:
            break
    else:
        stopped_reason = "step_limit_exhausted"

    if stopped_reason != "delivered":
        # The model never self-terminated (ran out of steps or budget) -
        # Python forces a wrap-up exactly like the old pipeline's
        # attempts-exhausted case: never a silent failure, always
        # something honest reaches the student. Use the BEST plan verified
        # this turn, not merely the most recent one - and on revision
        # turns, weigh the seed (the student's unchanged plan) with a
        # one-course overlap handicap so a candidate that actually made
        # the requested change beats "no change", while a stranger plan
        # still loses to the seed.
        final_course_numbers = [
            c for c in (best_verified_plan or last_verified_plan or []) if c not in requested_removals
        ]
        if seed_plan and seed_verify:
            model_score = _plan_score(best_verify_result, best_verified_plan, previous_plan)
            passing, overlap, neg_issues, credits = _plan_score(seed_verify, seed_plan, previous_plan)
            seed_score = (passing, overlap - 1.5, neg_issues, credits)
            if not final_course_numbers or seed_score > model_score:
                final_course_numbers = seed_plan
                tool_log.append(
                    {"name": "wrapup_kept_seed", "args": {}, "result": {"reason": "no model candidate beat the student's own plan"}}
                )
        violations = tools.check_invariants(track, final_course_numbers, passed, failed)
        if violations:
            seen = set()
            final_course_numbers = [
                c for c in final_course_numbers if c not in passed and c in track.courses and not (c in seen or seen.add(c))
            ]
        last_verify_result = tools.verify_plan(
            track, final_course_numbers, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
        )
        # Student-facing text: no internal jargon ("tool-call budget"), and
        # on a revision turn, describe the change relative to the plan the
        # student already has - kept / swapped out / added - which is the
        # question they actually care about.
        issues = last_verify_result.get("issues", []) if last_verify_result else []
        issues_sentence = (
            " Everything checks out." if not issues
            else " Still open: " + "; ".join(i["reason"] for i in issues) + "."
        )
        if previous_plan:
            prev_set, new_set = set(previous_plan), set(final_course_numbers)
            def names_of(nums):
                return ", ".join(track.courses[c]["name"] for c in nums if c in track.courses) or "none"
            kept = sorted(prev_set & new_set)
            removed = sorted(prev_set - new_set)
            added = sorted(new_set - prev_set)
            final_explanation = (
                f"Here's your updated plan. I kept {len(kept)} of your {len(previous_plan)} courses"
                + (f", removed {names_of(removed)}" if removed else "")
                + (f", and added {names_of(added)}" if added else "")
                + f" — {last_verify_result.get('total_credits', 0)} credits total."
                + issues_sentence
            )
        else:
            final_explanation = (
                f"This is the best plan I found ({len(final_course_numbers)} courses, "
                f"{last_verify_result.get('total_credits', 0)} credits) - I couldn't fully satisfy "
                "every constraint within this round." + issues_sentence
            )
        tool_log.append(
            {"name": "forced_wrapup", "args": {"reason": stopped_reason}, "result": last_verify_result}
        )

    # An empty plan should never be silently handed to the student - if the
    # delivered/corrected list came out empty but the model verified a real
    # non-empty candidate earlier in the same turn, fall back to the BEST
    # one rather than present nothing.
    if not final_course_numbers and (best_verified_plan or last_verified_plan):
        source = best_verified_plan or last_verified_plan
        fallback = [c for c in source if c not in passed and c in track.courses and c not in requested_removals]
        if fallback:
            final_course_numbers = fallback
            last_verify_result = tools.verify_plan(
                track, final_course_numbers, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
            )
            final_explanation = (
                "The plan being delivered came out empty after corrections, so this falls back to the last "
                f"candidate that was actually checked ({len(final_course_numbers)} course(s)); "
                "see the verification issues below for what's still unresolved."
            )
            tool_log.append({"name": "empty_plan_fallback", "args": {}, "result": last_verify_result})

    if not final_course_numbers:
        final_explanation = (
            "No viable plan could be constructed under the given constraints even as a fallback - "
            "this genuinely needs a constraint relaxed (a different weekday exclusion, pace, or override) "
            "before a semester plan can be built."
        )

    verify_result = last_verify_result or {"pass": False, "total_credits": 0, "workload_score": 0, "issues": []}
    exam_dates = tools.fetch_exam_dates(track, final_course_numbers) if final_course_numbers else {}
    cheesefork = tools.summarize_cheesefork(track, final_course_numbers) if final_course_numbers else {}
    progress = tools.assess_progress(track, semester_number, passed)
    # Computed deterministically for the UI regardless of whether the model
    # chose to consult these tools itself mid-loop.
    roadmap = tools.roadmap_to_graduation(track, passed, failed, plan_course_numbers=final_course_numbers)
    risks = tools.risk_report(track, final_course_numbers, passed, failed) if final_course_numbers else {"risks": [], "top_risk": None}

    messages = messages + [{"role": "assistant", "content": final_explanation}]

    plan_result = {
        "track_id": track.otjid,
        "track_name": track.name,
        "target_semester": track.target_semester,
        "constraints": state["constraints"],
        "verify": verify_result,
        "progress": progress,
        "explanation": final_explanation,
        "roadmap": roadmap,
        "risk_report": risks,
        "courses": [
            {
                "course_number": c,
                "name": track.courses[c]["name"],
                "points": track.courses[c]["points"],
                "is_retake": c in failed,
                "is_mandatory": c in track.mandatory_course_numbers,
                "schedule": track.courses[c]["schedule"],
                # One coherent registration group for the weekly timetable
                # view, honoring the student's excluded days.
                "timetable_slots": tools.pick_section(track.courses[c], excluded_weekdays),
                "why": course_reasons.get(c, ""),
                "exams": exam_dates.get(c, {}),
                "cheesefork": cheesefork.get(c, {}),
            }
            for c in final_course_numbers
        ],
    }

    return {
        "messages": messages,
        "cost_usd": cost,
        "tool_log": tool_log,
        "stopped_reason": stopped_reason,
        "plan_result": plan_result,
        "needs_input": None,
        # Echoed back by the frontend on the next request, so a follow-up
        # message is treated as feedback on THIS plan for THIS student -
        # not a from-scratch re-derivation (the bug this fixes: a student
        # mentioned one exam conflict and got a totally new plan, after
        # being re-asked things they'd already answered).
        "known_context": {
            "state": {
                "semester_number": semester_number,
                "constraints": state["constraints"],
                "passed_courses": passed,
                "failed_courses": failed,
            },
            "previous_plan": final_course_numbers,
            # The delivered plan's open verify issues, so the NEXT turn can
            # deterministically detect "the student complained, and the
            # revised plan still has the exact same problem" and push back.
            "previous_issues": [i["reason"] for i in verify_result.get("issues", [])],
        },
    }
