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

import live_offering_check
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
    record_llm_step,
    resolve_verify_kwargs,
)
from data_bundle import Track, get_track
from tool_schemas import REACT_TOOL_SCHEMAS, TERMINAL_TOOL_NAME, build_dispatch

# With the always-needed diagnostics pre-injected (see PREINJECTED_TOOL_NAMES),
# a full turn is now typically draft -> verify -> maybe repair -> maybe
# compare -> deliver, so fewer steps are needed. A tighter ceiling both caps
# worst-case spend and forces the model to converge instead of dithering.
MAX_STEPS = int(os.environ.get("REACT_MAX_STEPS", "18"))
SESSION_BUDGET_USD_CAP = float(os.environ.get("SESSION_BUDGET_USD_CAP", "0.50"))


def _call_with_tools(messages: list[dict]) -> tuple[object, float]:
    response = _get_client().chat.completions.create(
        model=MODEL, messages=messages, tools=REACT_TOOL_SCHEMAS, tool_choice="auto"
    )
    usage = response.usage
    cost = (usage.prompt_tokens / 1_000_000) * PRICE_INPUT_PER_1M + (
        usage.completion_tokens / 1_000_000
    ) * PRICE_OUTPUT_PER_1M
    message = response.choices[0].message
    record_llm_step(
        "PlanningLoop",
        messages,
        {
            "text": message.content,
            "tool_calls": [
                {"tool": tc.function.name, "arguments": tc.function.arguments}
                for tc in (message.tool_calls or [])
            ],
        },
    )
    return message, cost


def _available_courses_text(track: Track, passed: list[str], excluded_weekdays: list[int] | None = None) -> str:
    """The courses the student can ACTUALLY take next semester: offered then,
    prerequisites already satisfied, and not already passed. The model was
    otherwise handed the whole catalog (including locked/future courses) and
    had to infer eligibility itself - and got it wrong, padding credits with
    filler when it couldn't tell what was really takeable.

    Courses whose EVERY section hits one of the student's excluded days get
    an explicit DAY-BLOCKED tag: the model must not pick them (a delivered
    plan is clash-free), but it uses them for the 'freeing Monday would
    unlock X and Y' disclosure the advisor methodology requires."""
    passed_set = set(passed)
    excluded = list(excluded_weekdays or [])
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
        if num in track.mandatory_course_numbers:
            tag = "MANDATORY"
        else:
            group = next((g for g in track.mandatory_choice_groups if num in g["options"]), None)
            if group is None:
                tag = "elective"
            elif set(group["options"]) & passed_set:
                tag = "elective (its required group is already satisfied)"
            else:
                tag = "MANDATORY-CHOICE: required, take exactly one of [" + group["label"] + "]"
        day_note = ""
        if excluded and c.get("schedule"):
            best = tools.pick_section(c, excluded)
            hit_days = sorted({s["weekday"] for s in best if s.get("weekday") in excluded})
            if hit_days:
                day_note = f" [DAY-BLOCKED: only meets on excluded day(s) {hit_days} - do NOT include; cite in the relax-note]"
        rows.append(f"{num} — {c['name']} ({c['points']} pts, {load}{sat}, {exam}, {tag}){day_note}")
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
    requested_adds: list[str] | None = None,
    requested_adds_locked: set[str] | None = None,
    previous_explanation: str | None = None,
    grade_candidates: list[dict] | None = None,
    approved_retake_course: str | None = None,
) -> dict:
    requested_adds_note = ""
    if requested_adds:
        add_names = [f"{c} ({track.courses[c]['name']})" for c in requested_adds]
        requested_adds_note = (
            "\nHARD REQUIREMENT: the student explicitly asked to ADD these course(s) by "
            "name - the delivered plan MUST include them:\n"
            + "\n".join("- " + n for n in add_names)
            + "\nA course being hard, poorly reviewed, or workload-heavy is NEVER a reason "
            "to leave it out silently - CheeseFork/difficulty data is color for the "
            "explanation (mention the tradeoff honestly), never a veto over an explicit "
            "student directive. If including it forces something else out to fit "
            "credits/workload, drop the least valuable elective, not this course."
        )
    if requested_adds_locked:
        passed_set = set(state.get("passed_courses") or [])
        locked_lines = []
        for c in sorted(requested_adds_locked):
            if c not in track.courses:
                continue
            still_needed = tools.missing_prereq_courses(track.courses[c]["prerequisites"], passed_set)
            needed_names = [
                f"{n} ({track.courses[n]['name']})" if n in track.courses else n for n in still_needed
            ]
            needed_text = " + ".join(needed_names) if needed_names else "an unmet prerequisite"
            locked_lines.append(f"- {c} ({track.courses[c]['name']}) - still needs: {needed_text}")
        requested_adds_note += (
            "\nThe student also asked for these, but prerequisites genuinely aren't met yet "
            "- they cannot be registered for this semester no matter what. Tell the student "
            "EXACTLY what's still needed (named below), don't just say \"prerequisites aren't "
            "met\" with no specifics:\n" + "\n".join(locked_lines)
        )
    grade_suggestions_block = ""
    if grade_candidates:
        grade_suggestions_block = f"""
Grade-improvement retake candidates - RAW DATA ONLY, your judgment call, not \
a suggestion I'm making for you. Each entry shows the student's grade in a \
PASSED course against TWO baselines: this course's own historical average \
(technion-histograms), and the student's OWN average across every grade \
known about them. A gap against one baseline but not the other is a weaker \
signal than a gap against both - weigh them together, not as a single number:
{json.dumps(grade_candidates, ensure_ascii=False)}
If one looks genuinely promising, call summarize_cheesefork on it first to \
check what other students actually said (a course that's brutally curved or \
notoriously easy to improve on is a much stronger signal than the numbers \
alone) - this is real multi-step reasoning, not a lookup. Only THEN, if \
still genuinely worth it, you may PROPOSE (never include) it: fill in \
deliver_plan's proposed_retake field with that one course, and end your \
explanation with a short, direct question asking whether to add it next \
time - this is the ONE exception to "never ask a question" elsewhere in \
this prompt. The proposed course must NOT appear in course_numbers this \
turn - proposing and delivering are different things.
NEVER claim or imply guaranteed eligibility - Technion's actual \
grade-improvement (שיפור ציון) policy has its own eligibility rules (which \
courses qualify, attempt limits, time limits) that are not verified here; \
if you propose anything, make clear it's worth confirming with the \
registrar, not a rule you applied. Most turns nothing here is worth \
proposing - don't force it, and never propose more than one course at once.
"""
    approved_retake_note = ""
    if approved_retake_course:
        approved_retake_name = (
            track.courses[approved_retake_course]["name"]
            if approved_retake_course in track.courses
            else approved_retake_course
        )
        approved_retake_note = (
            f" The ONE exception this turn: the student just explicitly approved "
            f"retaking {approved_retake_name} ({approved_retake_course}) - already "
            f"passed - for grade improvement. You MAY include this specific course "
            f"in the plan as a deliberate additional course. Do not add any OTHER "
            f"passed course without the same explicit approval."
        )

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
        explanation_note = ""
        if previous_explanation:
            explanation_note = (
                f'\nWhat you told the student when you delivered that plan: "{previous_explanation}"\n'
                "If they ask what you recommended last time, or why, answer directly from "
                "this - no tool call needed."
            )
        revision_note = f"""

IMPORTANT - this is a REVISION, not a fresh request. The student already \
received this plan in an earlier turn:
{chr(10).join("- " + n for n in plan_names)}
{removals_note}
{issues_note}
{explanation_note}
Their latest message is feedback on it. Start from that exact plan: keep \
every course that still fits the updated constraints, and change ONLY what \
the new information actually breaks - but "keep the rest" NEVER means \
keeping the specific problem the student complained about. A student who \
reports one exam conflict expects the offending course swapped, not a \
completely different plan - call verify_plan on your minimally-changed \
version first, before considering anything more drastic. One hard limit on \
revisions: a required RETAKE is never dropped to satisfy feedback the \
student didn't aim at it - if the complaint (e.g. an exam-date conflict) \
lands on the retake itself and no alternative section or date exists, keep \
the retake and disclose the conflict as unavoidable; only an explicit \
"remove <that course>" from the student justifies dropping it."""

    return {
        "role": "system",
        "content": f"""\
You are an experienced Technion academic advisor building a {track.target_semester} \
course plan for a "{track.name}" student, with real tools to check your own \
work before committing to an answer.
{revision_note}
{requested_adds_note}

Hard rule, never negotiable: never include a course from the student's \
passed list below in the plan - it's already completed, retaking it would \
waste a slot. Only a course from the failed/outstanding list may reappear, \
and only as a deliberate retake. check_invariants will catch this if you \
miss it, but you shouldn't rely on that.{approved_retake_note}

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
choosing. Before delivering, run exam_study_planner on your candidate: it \
shows the REAL study days between every exam including moed B (which \
verify_plan does not enforce), and even names the single swap that most \
relieves the worst crunch. A plan where the student has 0-1 days to study \
between exams is a bad plan even if it technically verifies.
3b. The weekly schedule must be livable: run weekly_schedule_analyzer on \
your candidate before delivering - overlapping class times are a hard \
defect (verify_plan flags irreducible ones), and a week crammed into \
back-to-back marathon days is worth fixing when an alternative exists.
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

Diagnostics already computed for you (you do NOT need to fetch these):

Progress vs. a typical student at this point in the track:
{json.dumps(progress, ensure_ascii=False)}

Degree requirement audit (per category, what's left):
{json.dumps(catalog, ensure_ascii=False)}

Prerequisite-graph analysis of the student's failed/outstanding courses \
(what each blocks, gateway scores):
{json.dumps(prereq, ensure_ascii=False)}
{grade_suggestions_block}
THE COURSE MENU - the ONLY courses the student can take next semester \
(offered then, prerequisites already met, not yet passed). Every course \
number in your plan MUST come from this list - there is no other valid \
source; a number not listed here is locked, not offered, or already done, \
and will be rejected:
{available_text}

Build a real, substantive semester from that shortlist. Reach the credit \
target with genuine mandatory courses and worthwhile electives - do NOT \
pad the credit count with a stack of one-point sport/PE courses. At most \
ONE sport/PE course in a plan; if you still can't reach the target after \
using the real courses available, that's a genuine finding to report \
honestly, not something to paper over with filler.

BUILD THE PLAN THE WAY THE ADVISOR DOES, in this exact order:
Step 1 - the mandatory skeleton: place this semester's required courses \
and retakes that fit the student's constraints. These are the anchors. \
"Constraints" means schedule/day/prerequisite constraints ONLY - a \
mandatory or explicitly-requested course's difficulty, workload, or \
CheeseFork satisfaction score is NEVER grounds to leave it out. Mention \
that color honestly in the explanation; don't let it silently override \
degree progress.
Step 2 - substitution, when a required course cannot fit (day-blocked or \
exam clash): replace it like-for-like with the nearest-topic elective \
from the same degree category, OR pull a NEXT-semester mandatory course \
forward if its prerequisites are already met - keep real degree progress, \
never pad with filler.
Step 3 - fill the remaining credits with well-liked electives that fit \
the week and the exam calendar.

DELIVERY STANDARD - the advisor's bar for what may reach the student:
- A delivered plan carries AT MOST 2 open issues, and only ones the \
student's own constraints truly force. "Lazy" issues are NEVER acceptable \
- low credits, too few mandatory courses, exam clashes, class-time \
overlaps, or including a DAY-BLOCKED course - all of those are fixable by \
substitution, so fix them before delivering.
- Never include a course tagged DAY-BLOCKED in the shortlist. Deliver \
clash-free, and in your explanation name what relaxing ONE specific day \
would unlock (e.g. "freeing Monday would add X and Y and reach 18 \
credits") using the DAY-BLOCKED tags - the student decides, with the \
whole picture, whether their day off is worth it.

Your job is the judgment, not the bookkeeping: using the diagnostics above, \
draft a candidate course list, then use your tools to check and refine it. \
Available tools - fetch_exam_dates and summarize_cheesefork (to inspect \
candidates), simulate_future_impact (planning-horizon check), \
roadmap_to_graduation (project the remaining semesters assuming your \
candidate completes - use it to justify choices by what they unlock), \
risk_report (stress-test your final candidate: fragile courses, exam \
crunches, stacked heavy courses), exam_study_planner (the exam period as a \
study plan, moed A + B), weekly_schedule_analyzer (what the week actually \
looks like: overlaps, days on campus, free days), verify_plan (the critic \
- ALWAYS call this on your candidate before delivering), compare_plans \
(to weigh two real alternatives), check_invariants, search_courses (to \
resolve a course name). You have at most {MAX_STEPS} tool calls this turn \
- budget them; never call the same tool with the same arguments twice.

You MUST end the turn by calling deliver_plan exactly once - never respond \
with plain text instead. deliver_plan's course_numbers MUST be exactly the \
same list you most recently passed to verify_plan (or check_invariants) - \
never introduce a new, never-verified course at the last moment; if you \
want to change the plan, call verify_plan on the new list first. Call \
deliver_plan once you have a plan that verifies, or once you've concluded \
that no further attempt will resolve the remaining issue and this is the \
best achievable option.

Write deliver_plan's explanation as the final message to the student: 3-5 \
short sentences, not a report. Write it like an advisor talking to the \
student directly - "you"/"your", plain and warm, not a dry status readout - \
while staying just as honest and specific about trade-offs. The student already sees a visual breakdown \
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
state it as already-considered context in this same message.

EXACTLY ONE exception to "never a question," and it's an atomic pair, not \
two independent choices - if you decided a grade-improvement retake is \
worth proposing this turn (see the grade-improvement section above), you \
MUST do BOTH of these together, never just one: (1) set deliver_plan's \
proposed_retake field to that course, AND (2) add one extra sentence at \
the end of the explanation asking directly whether to include it next \
time. Filling in proposed_retake with no matching question leaves the \
student unaware anything was proposed; asking the question without \
filling in proposed_retake means their "yes" next turn has nothing to \
attach to and will be lost. This is the ONE case where an extra sentence \
beyond the normal 3-5 is expected - it doesn't count against that budget.""",
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
    previous_explanation = (known_context or {}).get("previous_explanation")
    proposed_retake = (known_context or {}).get("proposed_retake")

    if state_override is not None:
        state = state_override
        tool_log.append({"name": "structured_intake", "args": {}, "result": state})
    else:
        state, c = _extract_student_state(track, messages, known_state=known_state, proposed_retake=proposed_retake)
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

    # Deterministic, code-level check - never trust the model's own claim
    # that "the student approved this" without confirming it matches a REAL
    # proposal made last turn. A hallucinated or stale approval must never
    # reach the plan-building rules below (see tools.verify_plan /
    # check_invariants's approved_retake_course parameter).
    state["approved_retake_course"] = None
    if proposed_retake and state.get("approved_grade_retake") == proposed_retake.get("course_number"):
        state["approved_retake_course"] = proposed_retake["course_number"]
        tool_log.append(
            {"name": "grade_retake_approved", "args": {}, "result": {"course_number": state["approved_retake_course"]}}
        )
    elif state.get("requested_retake_course") in state.get("passed_courses", []):
        # The student brought this up themselves this turn (no prior
        # proposal to check against) - see requested_retake_course's
        # extraction docstring for how this differs from approved_grade_retake.
        state["approved_retake_course"] = state["requested_retake_course"]
        tool_log.append(
            {
                "name": "grade_retake_self_requested",
                "args": {},
                "result": {"course_number": state["approved_retake_course"]},
            }
        )

    # Question turns ("is X hard?") get one grounded answer from the real
    # review data - no planning loop, no gap-fill buttons, plan untouched.
    if state.get("intent") == "question" and state_override is None:
        answer, c = answer_course_question(track, state, messages, known_context=known_context)
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
                    "grades": state.get("grades", {}),
                },
                "previous_plan": previous_plan,
                "previous_issues": sorted(previous_issues),
                "previous_explanation": previous_explanation,
                # Not resolved this turn (never reached the planning loop) -
                # carry the incoming proposal forward unchanged rather than
                # silently dropping it.
                "proposed_retake": proposed_retake,
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

    # Only computed (and only spends any prompt tokens) when the student has
    # given at least one grade - the common case is zero grades known, and
    # this must cost nothing when there's nothing to compare. This is raw
    # comparison data only (course average AND the student's own average
    # across known grades) - no threshold decides "worth mentioning" here
    # anymore, that judgment belongs to the model (see _system_prompt).
    grade_candidates = None
    if state.get("grades"):
        grade_candidates = tools.analyze_grade_improvement_candidates(track, passed, state["grades"])
        tool_log.append(
            {
                "name": "analyze_grade_improvement_candidates",
                "args": {"preinjected": True},
                "result": grade_candidates,
            }
        )

    available_text = _available_courses_text(track, passed, excluded_weekdays)

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

    # Courses the student EXPLICITLY asked to ADD this turn, by name -
    # found live: a student named a specific mandatory course twice across
    # two revision turns and it still never made it into the plan (the
    # model apparently weighed CheeseFork difficulty/sentiment against an
    # explicit directive, which it must never do - sentiment is color, not
    # veto power). Symmetric to requested_removals above: an explicit
    # student directive is a hard constraint the model must satisfy or
    # explain, not a suggestion it may silently decline. Applies on ANY
    # turn, not just revisions - a first message can name a course too.
    # Genuinely locked (unmet prerequisites) courses are excluded here -
    # those can't be forced in no matter what the student wants, but the
    # model is told to explain why, not just go silent.
    requested_adds = [c for c in state.get("requested_add_courses", []) if c in track.courses and c not in passed]
    requested_adds_locked = set(tools.prereq_unmet_in(track, requested_adds, passed, failed)) if requested_adds else set()
    requested_adds = [c for c in requested_adds if c not in requested_adds_locked]
    if requested_adds or requested_adds_locked:
        tool_log.append(
            {
                "name": "requested_adds",
                "args": {},
                "result": {"course_numbers": requested_adds, "locked": sorted(requested_adds_locked)},
            }
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
            requested_adds=requested_adds,
            requested_adds_locked=requested_adds_locked,
            previous_explanation=previous_explanation,
            grade_candidates=grade_candidates,
            approved_retake_course=state.get("approved_retake_course"),
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
    add_pushed = False
    retake_dropped_pushed = False
    choice_group_pushed = False
    approved_retake_pushed = False
    issues_pushed = 0
    stopped_reason = "step_limit_exhausted"
    final_course_numbers: list[str] = []
    final_explanation = ""
    course_reasons: dict = {}
    new_proposed_retake: dict | None = None

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

                # Explicit-add enforcement: symmetric to the removal case
                # above - the student BY NAME asked for specific course(s)
                # IN. Found live: a student named a real, unlocked mandatory
                # course explicitly, twice across two revision turns, and it
                # never made it into the plan - the model apparently let
                # CheeseFork difficulty/sentiment override an explicit
                # directive. First violation: send the model back once.
                # Second violation: add it in code (dropping the cheapest
                # elective to make room, matching the mandatory top-up
                # backstop's approach) - an explicit student directive is a
                # hard constraint here too.
                missing_adds = [c for c in requested_adds if c not in candidate]
                if missing_adds:
                    if not add_pushed and step < MAX_STEPS - 1:
                        add_pushed = True
                        add_names = ", ".join(
                            f"{c} ({track.courses[c]['name']})" for c in missing_adds if c in track.courses
                        )
                        tool_log.append(
                            {"name": "add_ignored_pushback", "args": {}, "result": {"still_missing": missing_adds}}
                        )
                        react_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": (
                                    f"Not delivered: the student explicitly asked to ADD {add_names}, but "
                                    "your plan doesn't include it. Difficulty/CheeseFork sentiment is never "
                                    "a reason to leave out an explicit request - include it (drop the least "
                                    "valuable elective if needed to fit), verify, and deliver."
                                ),
                            }
                        )
                        continue
                    # Model ignored the directive twice - enforce in code.
                    # Drop the cheapest non-mandatory course(s) already in
                    # the plan to make room, then force the requested
                    # course(s) in regardless of the credit ceiling - an
                    # explicit directive outranks the soft credit target.
                    droppable = sorted(
                        (c for c in candidate if c not in track.mandatory_course_numbers and c not in requested_adds),
                        key=lambda c: float(track.courses.get(c, {}).get("points") or 0),
                    )
                    for _ in missing_adds:
                        if droppable:
                            candidate = [c for c in candidate if c != droppable.pop(0)]
                    candidate = candidate + [c for c in missing_adds if c not in candidate]
                    tool_log.append({"name": "add_enforced", "args": {}, "result": {"added": missing_adds}})

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

                # The student just explicitly approved a grade-improvement
                # retake (state["approved_retake_course"]) but the model's
                # course list doesn't include it - live testing showed this
                # happening even with the exception clearly stated in the
                # system prompt (a "may include" instruction alone wasn't
                # reliably acted on). One-shot nudge, same pattern as the
                # removal/verify nudges above: give the model one real
                # chance to either include it or explicitly explain why it
                # genuinely can't (a real schedule/prereq conflict is a
                # legitimate reason - this is a nudge, not a blind force).
                approved = state.get("approved_retake_course")
                if approved and approved not in candidate and not approved_retake_pushed and step < MAX_STEPS - 2:
                    approved_retake_pushed = True
                    approved_name = track.courses[approved]["name"] if approved in track.courses else approved
                    tool_log.append(
                        {"name": "approved_retake_missing_nudge", "args": {}, "result": {"course_number": approved}}
                    )
                    react_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": (
                                f"Not delivered: the student just explicitly approved retaking "
                                f"{approved_name} ({approved}) for grade improvement, but it's not in "
                                "this course list. Add it and re-verify, unless there's a genuine "
                                "scheduling or prerequisite conflict that makes it truly impossible - "
                                "if so, deliver without it but say so plainly in the explanation."
                            ),
                        }
                    )
                    continue

                final_explanation = args.get("explanation") or ""
                course_reasons = args.get("course_reasons") or {}
                # A NEW proposal the model is making THIS turn - deliberately
                # a different variable from `proposed_retake` above (which
                # holds LAST turn's proposal, already consumed into
                # state["approved_retake_course"] earlier). This is what the
                # NEXT turn will see as its incoming proposed_retake.
                new_proposed_retake = args.get("proposed_retake") or None
                if new_proposed_retake and new_proposed_retake.get("course_number") in candidate:
                    # Contradiction guard: a course can't be both proposed
                    # and already included - if it's in the plan, there's
                    # nothing left to propose about it.
                    new_proposed_retake = None
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
                violations = tools.check_invariants(
                    track, candidate, passed, failed, approved_retake_course=state.get("approved_retake_course")
                )
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
                    correction = (
                        "Note - a correction was applied before delivery: "
                        + "; ".join(violations)
                        + f". The corrected plan has {verify_result['total_credits']} credits"
                        + ("." if verify_result["pass"] else ", with these still open: "
                           + "; ".join(i["reason"] for i in verify_result["issues"]) + ".")
                    )
                    # Keep the model's own reasoning (it may carry the
                    # relax-note and per-course rationale) - prepend the
                    # deterministic correction instead of erasing it.
                    final_explanation = (final_explanation + " " + correction).strip()
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

                # Retake-dropped pushback: a failed course the student did
                # NOT ask to remove is missing from the delivery. Priority
                # rule #1 (a failed course is retaken now) lived only in the
                # prompt, and under revision pressure ("I have a conflict on
                # that exam date") the model was observed live silently
                # dropping the retake to satisfy the request. One shot: send
                # it back with the dropped course(s) named - reinstate and
                # rebalance, or deliver anyway with the omission explained
                # honestly. Second delivery stands (the honest-disclosure
                # explanation rules already cover it). Only fires for
                # courses actually offered next semester - a retake that
                # cannot be taken at all is a legitimate omission.
                dropped_retakes = [
                    c for c in failed
                    if c not in candidate
                    and c not in requested_removals
                    and track.courses.get(c, {}).get("offered_next_semester")
                ]
                if dropped_retakes and not retake_dropped_pushed and step < MAX_STEPS - 1:
                    retake_dropped_pushed = True
                    dropped_names = ", ".join(
                        f"{c} ({track.courses[c]['name']})" for c in dropped_retakes if c in track.courses
                    )
                    tool_log.append(
                        {"name": "retake_dropped_pushback", "args": {}, "result": {"dropped": dropped_retakes}}
                    )
                    react_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": (
                                f"Not delivered: this plan drops the student's required retake {dropped_names}, "
                                "which the student never asked to remove. A failed course is retaken NOW - "
                                "priority rule #1 - and it is never dropped just to resolve an exam-date or "
                                "schedule complaint; an unavoidable conflict on the retake itself is disclosed "
                                "honestly instead. Reinstate it, rebalance the rest if needed, verify the new "
                                "list and deliver that. Only if the retake genuinely cannot be taken this "
                                "semester may you deliver without it - and then your explanation must say "
                                "exactly why, in plain language."
                            ),
                        }
                    )
                    continue

                # Choice-group pushback: the model tried to deliver while a
                # required pick-one-variant group (calculus/algebra/physics/
                # probability) it could cover RIGHT NOW is missing. Observed
                # live even after two generic issue pushes (a first-semester
                # plan kept orchestra over physics). This guard doesn't fix
                # the plan - it hands the model the exact takeable variants
                # with their facts, once, and the model decides which fits
                # (or justifies leaving the gap honestly).
                uncovered_groups = []
                candidate_set = set(candidate)
                passed_set_now = set(passed)
                for grp in track.mandatory_choice_groups:
                    opts = set(grp["options"])
                    if opts & passed_set_now or opts & candidate_set:
                        continue
                    takeable = [
                        o for o in grp["options"]
                        if track.courses[o]["offered_next_semester"]
                        and tools._prereq_satisfied(track.courses[o]["prerequisites"], passed_set_now)
                    ]
                    if takeable:
                        uncovered_groups.append((grp, takeable))
                if uncovered_groups and not choice_group_pushed and step < MAX_STEPS - 1:
                    choice_group_pushed = True
                    fact_lines = []
                    for grp, takeable in uncovered_groups:
                        variants = ", ".join(
                            f"{o} {track.courses[o]['name']} ({track.courses[o]['points']} pts)"
                            for o in takeable
                        )
                        fact_lines.append(f"- {grp['label']}: takeable now -> {variants}")
                    tool_log.append(
                        {"name": "choice_group_pushback", "args": {},
                         "result": {"uncovered": [g["label"] for g, _ in uncovered_groups]}}
                    )
                    react_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": (
                                "Not delivered: this plan skips required course group(s) the student "
                                "can take RIGHT NOW - these are gateway requirements that block later "
                                "courses, and postponing them compounds:\n"
                                + "\n".join(fact_lines)
                                + "\nPick ONE variant per group (weigh difficulty/reviews/schedule "
                                "fit like any other decision), rebalance the rest of the plan if "
                                "credits demand it, verify the corrected list and deliver. If you "
                                "conclude a group genuinely cannot fit this semester, deliver anyway "
                                "but say exactly why in the explanation."
                            ),
                        }
                    )
                    continue

                # Issue-budget guard (the advisor's delivery standard): a
                # plan may reach the student with AT MOST 2 open issues,
                # and NONE of them "lazy" (credits, mandatory count, exam
                # clashes, overlaps - all fixable by substitution). Only
                # genuinely forced issues (a day conflict the student's own
                # constraints create) may remain. Fires up to twice, on any
                # turn - a 5-issue delivery was observed live and is
                # exactly what this exists to prevent.
                issues_list = verify_result.get("issues", [])
                lazy_issues = [
                    i["reason"] for i in issues_list if "excluded weekday" not in i["reason"]
                ]
                if (
                    (lazy_issues or len(issues_list) > 2)
                    and issues_pushed < 2
                    and step < MAX_STEPS - 1
                ):
                    issues_pushed += 1
                    tool_log.append(
                        {"name": "issue_budget_pushback", "args": {"attempt": issues_pushed},
                         "result": {"issues": [i["reason"] for i in issues_list]}}
                    )
                    react_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": (
                                "Not delivered: this plan has "
                                + str(len(issues_list))
                                + " open issue(s), including fixable ones: "
                                + "; ".join(lazy_issues or [i["reason"] for i in issues_list])
                                + ". The delivery standard is AT MOST 2 issues, none of them "
                                "fixable-by-substitution. Follow the build order: keep the "
                                "mandatory skeleton, substitute like-for-like from the same "
                                "category or pull a next-semester mandatory forward, fill with "
                                "well-liked electives that fit. Verify the corrected list and "
                                "deliver that."
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
        violations = tools.check_invariants(
            track, final_course_numbers, passed, failed, approved_retake_course=state.get("approved_retake_course")
        )
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

    # Final safety net, regardless of which path got here: issue_budget_pushback
    # only retries twice, then accepts whatever the model delivers even if the
    # delivery standard ("never acceptable" - low credits, exam clashes,
    # overlaps) is still violated. Found live: an 8-credit plan and a 1-day
    # exam gap both reached a student this way. If a genuinely better
    # candidate was verified earlier THIS SAME turn, prefer it over knowingly
    # accepting a worse one - never invents a new plan, only picks among ones
    # already checked.
    current_issues = (last_verify_result or {}).get("issues", [])
    current_lazy = [i["reason"] for i in current_issues if "excluded weekday" not in i["reason"]]
    if (current_lazy or len(current_issues) > 2) and best_verified_plan:
        swap_candidate = [
            c for c in best_verified_plan if c not in requested_removals and c not in passed and c in track.courses
        ]
        if swap_candidate and set(swap_candidate) != set(final_course_numbers):
            swap_score = _plan_score(best_verify_result, swap_candidate, previous_plan)
            current_score = _plan_score(last_verify_result, final_course_numbers, previous_plan)
            if swap_score > current_score:
                final_course_numbers = swap_candidate
                last_verify_result = tools.verify_plan(
                    track, final_course_numbers, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
                )
                # {ISSUES_SENTENCE} deferred the same way {COURSE_COUNT}/
                # {TOTAL_CREDITS} are below - a later backstop (mandatory
                # top-up, locked-course, overlap...) can still resolve one of
                # these issues, and a sentence written NOW would then claim a
                # problem that's already fixed by the time this is shown to
                # the student (found live: "only 2 mandatory course(s))"
                # still printed after the top-up backstop had already added
                # a 3rd).
                final_explanation = (
                    f"Switched to a better-verified alternative before delivering "
                    f"({{COURSE_COUNT}} course(s), {{TOTAL_CREDITS}} credits) - the plan initially reached "
                    "was still short of the delivery standard even after correction attempts."
                    "{ISSUES_SENTENCE}"
                )
                tool_log.append({"name": "final_safety_swap", "args": {}, "result": last_verify_result})

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
                "candidate that was actually checked ({COURSE_COUNT} course(s)); "
                "see the verification issues below for what's still unresolved."
            )
            tool_log.append({"name": "empty_plan_fallback", "args": {}, "result": last_verify_result})

    if not final_course_numbers:
        final_explanation = (
            "No viable plan could be constructed under the given constraints even as a fallback - "
            "this genuinely needs a constraint relaxed (a different weekday exclusion, pace, or override) "
            "before a semester plan can be built."
        )

    # HARD locked-course backstop: a course whose prerequisites aren't met
    # cannot be registered for, period - strip it on any delivery path.
    # (Retakes are exempt: their prereqs were met the first time around.)
    if final_course_numbers:
        locked = tools.prereq_unmet_in(track, final_course_numbers, passed, failed)
        if locked:
            final_course_numbers = [c for c in final_course_numbers if c not in locked]
            last_verify_result = tools.verify_plan(
                track, final_course_numbers, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
            )
            names = ", ".join(track.courses[c]["name"] for c in locked if c in track.courses)
            final_explanation += (
                f" Note: {names} was removed - its prerequisites aren't met yet, so it can't be "
                "registered for this semester; it stays on the roadmap for when it unlocks."
            )
            tool_log.append({"name": "locked_enforced", "args": {}, "result": {"dropped": locked}})

    # HARD mandatory-course top-up backstop: found live, a plan can pass
    # credit/workload checks while making almost no real degree progress -
    # 2 filler electives (orchestra, an entrepreneurship elective) got
    # delivered instead of two ALREADY-UNLOCKED real mandatory courses,
    # with verify_plan's own "only 1 mandatory course(s), expected at
    # least 3" issue printed right in the explanation and delivered
    # anyway. The earlier safety net only ever picks the best plan the
    # model itself tried - if every attempt padded with electives instead
    # of taking available requirements, there was nothing better to swap
    # in. This swaps unlocked, not-yet-included mandatory courses in for
    # the cheapest electives currently in the plan, up to the credit
    # ceiling - mirrors verify_plan's own min_mandatory_courses logic so
    # the two can never disagree about what's actually required here.
    if final_course_numbers:
        override_minimums = state["constraints"].get("override_minimums", False)
        min_mandatory = 0 if override_minimums else tools.DEFAULT_MIN_MANDATORY_COURSES
        remaining_mandatory = track.mandatory_course_numbers - set(passed)
        effective_min = min(min_mandatory, len(remaining_mandatory))
        mandatory_in_plan = [c for c in final_course_numbers if c in track.mandatory_course_numbers and c not in passed]
        shortfall = effective_min - len(mandatory_in_plan)
        if shortfall > 0:
            candidate_pool = sorted(remaining_mandatory - set(final_course_numbers))
            still_locked = tools.prereq_unmet_in(track, candidate_pool, passed, failed) if candidate_pool else set()
            candidates = [
                c for c in candidate_pool
                if c not in still_locked and track.courses.get(c, {}).get("offered_next_semester", True)
            ]
            max_credits = verify_kwargs.get("max_credits", tools.DEFAULT_MAX_CREDITS)

            def _plan_points(nums):
                return sum(float(track.courses[c].get("points") or 0) for c in nums if c in track.courses)

            # Never drop a mandatory course already in the plan or the
            # approved retake - only genuine filler is fair game, cheapest
            # (least degree value) first.
            droppable = sorted(
                (
                    c for c in final_course_numbers
                    if c not in track.mandatory_course_numbers and c != state.get("approved_retake_course")
                ),
                key=lambda c: float(track.courses.get(c, {}).get("points") or 0),
            )
            added = []
            for candidate in candidates:
                if shortfall <= 0:
                    break
                candidate_points = float(track.courses[candidate].get("points") or 0)
                while droppable and _plan_points(final_course_numbers) + candidate_points > max_credits:
                    drop = droppable.pop(0)
                    final_course_numbers = [c for c in final_course_numbers if c != drop]
                if _plan_points(final_course_numbers) + candidate_points > max_credits:
                    continue  # no room even after dropping every droppable elective
                final_course_numbers = final_course_numbers + [candidate]
                added.append(candidate)
                shortfall -= 1
            if added:
                last_verify_result = tools.verify_plan(
                    track, final_course_numbers, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
                )
                names = ", ".join(track.courses[c]["name"] for c in added)
                final_explanation += (
                    f" Note: added {names} - genuinely available, unlocked mandatory requirement(s) the "
                    "plan was missing; the least valuable elective(s) made room for them."
                )
                tool_log.append({"name": "mandatory_topup_enforced", "args": {}, "result": {"added": added}})

    # HARD collision backstop: a delivered week must NEVER contain a real
    # class-time collision, on any path (model delivery, forced wrap-up,
    # empty-plan fallback). If the coordinated section assignment still has
    # irreducible overlaps, deterministically drop the least valuable
    # course of each colliding pair (keep retakes, then mandatory, then
    # higher credits) until the week is clean - same philosophy as removal
    # enforcement: judgment belongs to the model, but this is correctness.
    if final_course_numbers:
        dropped_for_overlap: list[str] = []
        while True:
            section_check = tools.choose_sections(track, final_course_numbers, excluded_weekdays)
            if not section_check["overlaps"]:
                break
            pair = section_check["overlaps"][0]
            a, b = pair["course_a"], pair["course_b"]

            def keep_value(c):
                course = track.courses.get(c, {})
                return (c in failed, c in track.mandatory_course_numbers, float(course.get("points") or 0))

            drop = a if keep_value(a) <= keep_value(b) else b
            final_course_numbers = [c for c in final_course_numbers if c != drop]
            dropped_for_overlap.append(drop)
        if dropped_for_overlap:
            last_verify_result = tools.verify_plan(
                track, final_course_numbers, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
            )
            names = ", ".join(track.courses[c]["name"] for c in dropped_for_overlap if c in track.courses)
            final_explanation += (
                f" Note: {names} was removed from the plan because its class times collide with another "
                "course in every available section combination - a clashing week is never delivered."
            )
            tool_log.append(
                {"name": "overlap_enforced", "args": {}, "result": {"dropped": dropped_for_overlap}}
            )

    # One last, real-time check against Technion's own live system - the
    # static data/track_*.json bundle only refreshes when a developer
    # manually reruns the pipeline, so a course closed/cancelled since then
    # would otherwise be confidently recommended anyway. Fail-open by
    # design (live_offering_check.py): a course is only ever dropped when
    # POSITIVELY confirmed gone, never because the live check itself failed.
    if final_course_numbers:
        year_str, semester_str = track.target_semester.split("_")
        live_status = live_offering_check.check_still_offered(
            int(year_str), int(semester_str), final_course_numbers
        )
        tool_log.append({"name": "live_offering_check", "args": {}, "result": live_status})
        no_longer_offered = [c for c, offered in live_status.items() if offered is False]
        if no_longer_offered:
            final_course_numbers = [c for c in final_course_numbers if c not in no_longer_offered]
            last_verify_result = tools.verify_plan(
                track, final_course_numbers, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
            )
            names = ", ".join(track.courses[c]["name"] for c in no_longer_offered if c in track.courses)
            final_explanation += (
                f" Note: {names} was removed - a live check against Technion's system just now shows "
                "it's no longer offered this semester, even though the cached course data said otherwise."
            )

    # Deterministic backstop, not left to the model to remember: live
    # testing showed the model reliably filling in deliver_plan's
    # proposed_retake WITHOUT actually asking the matching question in the
    # explanation prose (or vice versa) - a silent proposal the student
    # never sees is worse than no proposal at all, since it just look like
    # the agent's supposed to answer "yes" to nothing. If the field is set,
    # the question WILL appear, regardless of what the model wrote.
    if new_proposed_retake and new_proposed_retake.get("course_number") not in final_course_numbers:
        retake_number = new_proposed_retake.get("course_number")
        retake_name = (
            track.courses[retake_number]["name"] if retake_number in track.courses else retake_number
        )
        if retake_name not in final_explanation:
            final_explanation += (
                f" One more thing: your grade in {retake_name} was notably below the comparison "
                f"averages - want me to add a retake of it to your plan next time?"
            )
    else:
        new_proposed_retake = None  # contradicted or malformed - never carry a broken proposal forward

    # Deterministic backstop, same reasoning as the proposed-retake one
    # above: the prompt TELLS the model to explain a locked-but-requested
    # course by name, but a later backstop (final_safety_swap, mandatory
    # top-up...) can throw away the model's own prose entirely and replace
    # it with a synthetic one that never saw this instruction - found
    # exactly this live, testing the locked-course path. Guarantee the
    # explanation, don't trust it survived.
    for locked_course in sorted(requested_adds_locked or []):
        if locked_course not in track.courses:
            continue
        locked_name = track.courses[locked_course]["name"]
        if locked_name in final_explanation or locked_course in final_explanation:
            continue
        still_needed = tools.missing_prereq_courses(
            track.courses[locked_course]["prerequisites"], set(state.get("passed_courses") or [])
        )
        needed_text = (
            " + ".join(f"{n} ({track.courses[n]['name']})" if n in track.courses else n for n in still_needed)
            if still_needed
            else "a prerequisite you haven't taken yet"
        )
        final_explanation += (
            f" On {locked_name}: you asked for it, but it's not registerable yet - it still needs "
            f"{needed_text} first."
        )

    verify_result = last_verify_result or {"pass": False, "total_credits": 0, "workload_score": 0, "issues": []}
    # The safety-net explanations above are built BEFORE the locked-course/
    # overlap/live-offering backstops below can still drop a course, so any
    # course-count or credit figure they embed would go stale the moment one
    # of those backstops fires afterward (found live: "7 course(s)" printed
    # next to an explanation that had already dropped to 6). Substituted
    # here, after every backstop has had its say, against the truly final
    # numbers - a no-op replace when no placeholder was ever inserted.
    if "{COURSE_COUNT}" in final_explanation or "{TOTAL_CREDITS}" in final_explanation or "{ISSUES_SENTENCE}" in final_explanation:
        remaining_issues = verify_result.get("issues", [])
        issues_sentence = (
            " Everything checks out."
            if not remaining_issues
            else " Still open: " + "; ".join(i["reason"] for i in remaining_issues) + "."
        )
        final_explanation = (
            final_explanation.replace("{COURSE_COUNT}", str(len(final_course_numbers)))
            .replace("{TOTAL_CREDITS}", str(verify_result.get("total_credits", 0)))
            .replace("{ISSUES_SENTENCE}", issues_sentence)
        )
    exam_dates = tools.fetch_exam_dates(track, final_course_numbers) if final_course_numbers else {}
    cheesefork = tools.summarize_cheesefork(track, final_course_numbers) if final_course_numbers else {}
    progress = tools.assess_progress(track, semester_number, passed)
    # Computed deterministically for the UI regardless of whether the model
    # chose to consult these tools itself mid-loop.
    roadmap = tools.roadmap_to_graduation(track, passed, failed, plan_course_numbers=final_course_numbers)
    risks = tools.risk_report(track, final_course_numbers, passed, failed) if final_course_numbers else {"risks": [], "top_risk": None}
    # ONE coordinated section assignment for the whole plan - sections are
    # chosen to avoid excluded days AND each other, and the timetable the
    # student sees is exactly this assignment (per-course independent picks
    # previously produced overlapping blocks on the weekly grid).
    section_assignments = (
        tools.choose_sections(track, final_course_numbers, excluded_weekdays)["assignments"]
        if final_course_numbers
        else {}
    )

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
                # The coordinated section assignment - one coherent group
                # per course, chosen to avoid excluded days and each other.
                "timetable_slots": section_assignments.get(c, []),
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
                "grades": state.get("grades", {}),
            },
            "previous_plan": final_course_numbers,
            # The delivered plan's open verify issues, so the NEXT turn can
            # deterministically detect "the student complained, and the
            # revised plan still has the exact same problem" and push back.
            "previous_issues": [i["reason"] for i in verify_result.get("issues", [])],
            # The prose explanation just delivered, so a later turn (same
            # conversation, or a recalled cross-session profile) can answer
            # "what did you last recommend?" directly - see previous_explanation
            # in _system_prompt.
            "previous_explanation": plan_result.get("explanation"),
            # A grade-improvement retake proposed THIS turn (or None) - the
            # next turn's extraction checks this against the student's
            # response to resolve approved_grade_retake. Deliberately
            # replaces (never stacks with) whatever was pending before -
            # see the plan's "resolved within one turn boundary" design.
            "proposed_retake": new_proposed_retake,
        },
    }
