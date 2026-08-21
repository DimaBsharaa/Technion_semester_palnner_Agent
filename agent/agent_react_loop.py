"""
Stage 1 of the pipeline -> agent migration: a bounded, tool-calling agent
loop that replaces agent_loop.py's hardcoded draft -> verify -> repair ->
compare -> explain sequence with one loop where the MODEL decides which
tool to call and when it's done (genuine OpenAI tool-calling, not
JSON-mode structured output at fixed Python-decided points).

Extraction (is there enough info to plan?) is unchanged from agent_loop.py
and reused as-is. This module only takes over once that's settled: given a
resolved student state, how should the plan get built?

Three properties carried over from the original design, preserved here:
1. Nothing the model does mid-loop is visible to the student until it
   calls the terminal tool, deliver_plan - no student-facing turn inside
   the loop, so mid-conversation rambling/permission-asking can't happen.
2. Python owns the ceiling: MAX_STEPS and SESSION_BUDGET_USD_CAP are hard
   limits; if the model never calls deliver_plan, Python forces a wrap-up
   from the last verified plan.
3. Python owns final correctness: deliver_plan's course list is always
   re-run through tools.verify_plan and tools.check_invariants - the
   model's own bookkeeping is never trusted for hard constraints.

Enabled via AGENT_MODE=react in main.py; agent_loop.py remains the default
until validated against tests/test_scenarios.py and live testing.
"""

import json
import os
import random

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

# With diagnostics pre-injected (see PREINJECTED_TOOL_NAMES), a turn is
# typically draft -> verify -> maybe repair/compare -> deliver, so a tight
# ceiling caps worst-case spend and forces convergence over dithering.
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
    prerequisites satisfied, not already passed. Handing the model the whole
    catalog (incl. locked/future courses) made it infer eligibility itself
    and pad credits with filler when it guessed wrong.

    Courses whose EVERY section hits an excluded day get a DAY-BLOCKED tag:
    the model must not pick them (plan stays clash-free), but uses them for
    the 'freeing Monday would unlock X and Y' disclosure."""
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
    # Shuffled, not sorted by course number: a fixed order caused positional
    # bias (the model kept defaulting to the same 1-2 electives shown first).
    # Every row is still explicitly tagged MANDATORY/MANDATORY-CHOICE/
    # elective, so shuffling only changes which electives are seen first.
    random.shuffle(rows)
    return "\n".join(rows)


def _sport_courses_in(track: Track, course_numbers: list[str]) -> list[str]:
    """Sport/PE course numbers in a plan, identified by name - used by the
    deterministic anti-padding guard, since 'at most one sport course' in
    the prompt alone was not reliably followed."""
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
    """Higher is better: passing beats failing; on revision turns, more
    overlap with the previous plan beats a stranger plan that scored well
    (a swap-one-course request must never come back unrelated); then fewer
    open issues, then a fuller credit load. Used to keep the BEST plan
    verified this turn, not merely the most recent."""
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
    near_locked: list[dict] | None = None,
    previous_explanation: str | None = None,
    grade_candidates: list[dict] | None = None,
    approved_retake_course: str | None = None,
) -> dict:
    near_locked_note = ""
    if near_locked:
        lines = []
        for entry in near_locked:
            needed = " + ".join(f"{n['course_number']} ({n['name']})" for n in entry["still_needs"])
            lines.append(f"- {entry['course_number']} ({entry['name']}) - still needs: {needed}")
        near_locked_note = (
            "\nFYI, not in the menu below because they're genuinely locked - mandatory "
            "course(s) that would normally matter around now, blocked by a specific "
            "unmet prerequisite:\n"
            + "\n".join(lines)
            + "\nIf your plan doesn't cover much real degree progress, proactively name the "
            "most relevant one of these and what's blocking it in your explanation - don't "
            "make the student ask \"where's X?\" to find out it exists and why it's missing."
        )
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
                + "\nReplace them with suitable alternatives from the available-courses shortlist. If a "
                "removed course was the only currently-known way to satisfy a MANDATORY-CHOICE group "
                "(found live: the student says they already passed it, e.g. \"I already passed algebra\" "
                "- believe them even if extraction couldn't confidently attach a specific course number to "
                "that claim), do NOT put the exact same course back to cover that group - either find a "
                "genuinely different variant, or leave the requirement honestly open and say so plainly; "
                "never silently undo an explicit removal by re-adding the identical course."
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
{near_locked_note}

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
light alternative with a better rating is available. "No-exam" is a useful \
property, not a shortcut to the same 1-2 courses every time - the \
shortlist below usually has multiple decent-rated options; actually weigh \
them against each other instead of defaulting to whichever is most \
familiar.
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
- If workload score exceeds the cap, fix it by SWAPPING one heavy ELECTIVE \
for a lighter one - never by dropping courses outright until credits fall \
below the target, and NEVER by dropping a mandatory course. A mandatory \
course is never "the workload problem" to solve, no matter how heavy its \
own difficulty rating is - a real degree requirement outranks a workload \
number every time. A too-heavy week from genuinely required courses is a \
real, disclosable trade-off, not something to silently engineer away by \
cutting a requirement. A too-light one is a worse outcome than a somewhat \
heavy but real semester. When a genuinely fewer-credits plan is the right \
call, that's the student's decision (pace="light" or override_minimums), \
not a silent one you make for them by over-trimming.
- The same rule applies even when the workload score is nowhere near the \
cap: a mandatory course officially placed in the student's CURRENT target \
semester is never "deferred for a lighter pacing" or "the safer choice for \
a brand-new student" as your own judgment call - only a genuine schedule \
conflict or an unmet prerequisite is a real reason to leave one out, and \
either of those must be named as an open, disclosed trade-off, not folded \
into the explanation as if lightening the load were the point.
- Never include a course tagged DAY-BLOCKED in the shortlist. Deliver \
clash-free, and in your explanation name what relaxing ONE specific day \
would unlock (e.g. "freeing Monday would add X and Y and reach 18 \
credits") using the DAY-BLOCKED tags - the student decides, with the \
whole picture, whether their day off is worth it.
- If the plan includes an English-requirement course (name contains \
"אנגלית"/"English"), name it explicitly in the explanation and say plainly \
that it assumes the standard track placement - English level is decided \
by Technion's own placement exam and varies per student (some have a \
different level, some an exemption), which this system has no way to \
verify. Never silently bury it among the other mandatory courses as if \
it's a settled fact the same way a real degree requirement is.

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
    forward what an earlier turn resolved, so a follow-up is treated as a
    revision against known facts, not a from-scratch re-derivation."""
    track = get_track(track_id)
    tool_log = _EmittingList(on_event) if on_event else []
    cost = cost_so_far
    known_state = (known_context or {}).get("state")
    previous_plan = (known_context or {}).get("previous_plan")
    previous_issues = set((known_context or {}).get("previous_issues") or [])
    previous_explanation = (known_context or {}).get("previous_explanation")
    proposed_retake = (known_context or {}).get("proposed_retake")
    pending_forced_add = (known_context or {}).get("pending_forced_add")

    if state_override is not None:
        state = state_override
        tool_log.append({"name": "structured_intake", "args": {}, "result": state})
    else:
        state, c = _extract_student_state(
            track,
            messages,
            known_state=known_state,
            proposed_retake=proposed_retake,
            pending_forced_add=pending_forced_add,
        )
        cost += c
        tool_log.append({"name": "extract_student_state", "args": {}, "result": state})
        if previous_plan and known_state:
            # A plan was already delivered this conversation, so its facts
            # are settled: merge enforces continuity (extraction refines,
            # never un-knows) so a revision turn never bounces the student
            # back to the intake buttons for facts already established.
            state = _normalize_state(merge_known_state(state, known_state))
            tool_log.append({"name": "merge_known_state", "args": {}, "result": state})

    # Deterministic check - never trust the model's own claim that "the
    # student approved this"; it must match a REAL proposal from last turn
    # (see tools.verify_plan / check_invariants's approved_retake_course).
    state["approved_retake_course"] = None
    if proposed_retake and state.get("approved_grade_retake") == proposed_retake.get("course_number"):
        state["approved_retake_course"] = proposed_retake["course_number"]
        tool_log.append(
            {"name": "grade_retake_approved", "args": {}, "result": {"course_number": state["approved_retake_course"]}}
        )
    elif state.get("requested_retake_course") in state.get("passed_courses", []):
        # Student-initiated this turn (no prior proposal to check against) -
        # see requested_retake_course's docstring re: approved_grade_retake.
        state["approved_retake_course"] = state["requested_retake_course"]
        tool_log.append(
            {
                "name": "grade_retake_self_requested",
                "args": {},
                "result": {"course_number": state["approved_retake_course"]},
            }
        )

    # Same deterministic-confirmation pattern as approved_retake_course:
    # only trust a confirmed override when it matches a REAL proposal from
    # last turn, never the model's own say-so.
    state["forced_add_course"] = None
    if pending_forced_add and state.get("confirmed_forced_add") == pending_forced_add.get("course_number"):
        state["forced_add_course"] = pending_forced_add["course_number"]
        tool_log.append(
            {"name": "forced_add_confirmed", "args": {}, "result": {"course_number": state["forced_add_course"]}}
        )

    # Question turns ("is X hard?") get one grounded answer from real
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

    # A closing remark with nothing to act on ("looks perfect, thank you!")
    # re-delivers the same plan untouched, cheaply, instead of falling
    # through to a normal revision turn - which previously could
    # non-deterministically churn an already-accepted plan with nothing
    # for the model to actually react to. Only short-circuits when there's
    # a previous plan to re-affirm; otherwise falls through to planning.
    if state.get("intent") == "acknowledgment" and previous_plan and state_override is None:
        return {
            "messages": messages + [{"role": "assistant", "content": "Glad it works for you! Come back anytime you want to change something."}],
            "cost_usd": cost,
            "tool_log": tool_log,
            "stopped_reason": "acknowledged",
            "plan_result": None,
            "needs_input": None,
            # Untouched - an acknowledgment doesn't change the student's
            # state or the plan already on the desk.
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
                    # No fresh approval resolved this turn - carry forward what
                    # was already confirmed earlier.
                    "confirmed_grade_retakes": sorted((known_state or {}).get("confirmed_grade_retakes") or []),
                },
                "previous_plan": previous_plan,
                "previous_issues": sorted(previous_issues),
                "previous_explanation": previous_explanation,
                # Not resolved this turn - carry forward rather than drop.
                "proposed_retake": proposed_retake,
                "pending_forced_add": pending_forced_add,
            },
        }

    semester_number = state.get("semester_number") or 0
    passed, failed = backfill_passed_courses(track, state)
    state["passed_courses"] = passed
    state["failed_courses"] = failed
    # Second chance for grade_retake_self_requested: the check above ran
    # BEFORE backfill, against whatever passed_courses raw extraction
    # listed explicitly - extraction can say "passed_all_expected: true"
    # without also listing the retake course by number, leaving
    # passed_courses empty there even though requested_retake_course and a
    # real grade were set correctly. backfill_passed_courses's own
    # grade-based inference can still confirm it here.
    if (
        not state["approved_retake_course"]
        and state.get("requested_retake_course")
        and state["requested_retake_course"] in passed
    ):
        state["approved_retake_course"] = state["requested_retake_course"]
        tool_log.append(
            {
                "name": "grade_retake_self_requested",
                "args": {},
                "result": {"course_number": state["approved_retake_course"]},
            }
        )

    # Persists a grade-improvement retake beyond the single turn it was
    # approved on: approved_retake_course resets to None every turn by
    # design, and without this, verify_plan/check_invariants' exemption for
    # it would vanish on any later turn, stripping it back out of the plan
    # even when the model correctly kept it. Union, never replace - same
    # lifecycle as passed_courses/failed_courses. requested_removals
    # (resolved below) still lets the student explicitly drop it later.
    state["confirmed_grade_retakes"] = set((known_state or {}).get("confirmed_grade_retakes") or [])
    if state.get("approved_retake_course"):
        state["confirmed_grade_retakes"].add(state["approved_retake_course"])
    state["confirmed_grade_retakes"] -= set(state.get("remove_courses") or [])

    excluded_weekdays = state["constraints"].get("excluded_weekdays") or []
    verify_kwargs = resolve_verify_kwargs(state)

    dispatch = build_dispatch(track, semester_number, passed, failed, excluded_weekdays, verify_kwargs)

    # Pre-computed once in code and handed to the model in its prompt,
    # rather than making it spend paid round-trips fetching them. Logged
    # like tool calls so the trace still shows they happened.
    progress = tools.assess_progress(track, semester_number, passed)
    catalog = tools.fetch_catalog(track, passed)
    prereq = tools.query_prereq_graph(track, passed, failed)
    # Courses locked out of the model's own menu entirely (an unmet
    # prereq) would otherwise never get explained unless the student named
    # them. Surfaced unconditionally so the model can mention them proactively.
    near_locked = tools.near_locked_mandatory_courses(track, semester_number, passed, failed)
    tool_log.append({"name": "assess_progress", "args": {"preinjected": True}, "result": progress})
    tool_log.append({"name": "fetch_catalog", "args": {"preinjected": True}, "result": catalog})
    tool_log.append({"name": "query_prereq_graph", "args": {"preinjected": True}, "result": prereq})
    if near_locked:
        tool_log.append(
            {"name": "near_locked_mandatory_courses", "args": {"preinjected": True}, "result": near_locked}
        )

    # Only computed when the student has given at least one grade - free
    # when there's nothing to compare. Raw comparison data only (course
    # average and the student's own average); the model decides what's
    # worth mentioning (see _system_prompt).
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
    # extraction resolves names to numbers; Python enforces the removal on
    # whatever gets delivered (the prompt alone wasn't reliably followed).
    # An explicit student directive is a hard constraint, not a suggestion.
    requested_removals = [c for c in state.get("remove_courses", []) if c in (previous_plan or [])]
    if requested_removals:
        tool_log.append(
            {"name": "requested_removals", "args": {}, "result": {"course_numbers": requested_removals}}
        )

    # Courses the student EXPLICITLY asked to ADD this turn, by name -
    # symmetric to requested_removals above: a hard constraint, never
    # something CheeseFork sentiment gets to silently veto. Genuinely
    # locked (unmet prereq) courses are excluded - can't be forced in, but
    # must be explained. An unknown course number (typo/wrong track) is
    # split off too, for the same guaranteed explanation.
    requested_adds_unknown = {c for c in state.get("requested_add_courses", []) if c not in track.courses}
    requested_add_raw = [c for c in state.get("requested_add_courses", []) if c in track.courses]
    # "Add" (no retake language) naming a course already passed falls
    # between two mechanisms: requested_add_courses excludes already-passed
    # courses, and requested_retake_course never fires without explicit
    # retake language - so split it off here for its own guaranteed
    # explanation and a concrete next step ("retake it" routes through the
    # self-requested-retake path).
    requested_adds_already_passed = {c for c in requested_add_raw if c in passed}
    requested_adds = [c for c in requested_add_raw if c not in passed]
    requested_adds_locked = set(tools.prereq_unmet_in(track, requested_adds, passed, failed)) if requested_adds else set()
    requested_adds = [c for c in requested_adds if c not in requested_adds_locked]
    # Same reasoning as the locked split above: a course not offered this
    # semester has no section to register for, so it's split off and never
    # handed to add_enforced; the guaranteed explanation backstop below
    # tells the student plainly why.
    requested_adds_unavailable = {c for c in requested_adds if not track.courses[c].get("offered_next_semester")}
    requested_adds = [c for c in requested_adds if c not in requested_adds_unavailable]
    if requested_adds or requested_adds_locked or requested_adds_unavailable or requested_adds_already_passed or requested_adds_unknown:
        tool_log.append(
            {
                "name": "requested_adds",
                "args": {},
                "result": {
                    "course_numbers": requested_adds,
                    "locked": sorted(requested_adds_locked),
                    "unavailable": sorted(requested_adds_unavailable),
                    "already_passed": sorted(requested_adds_already_passed),
                    "unknown": sorted(requested_adds_unknown),
                },
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
            near_locked=near_locked,
            previous_explanation=previous_explanation,
            grade_candidates=grade_candidates,
            approved_retake_course=state.get("approved_retake_course"),
        )
    ]
    # The loop otherwise only sees the condensed extracted state - give it
    # the recent conversation verbatim too. Matters concretely: after a
    # gap-fill form submit, the "last message" is the form's summary, and
    # the student's actual complaint can sit one message earlier - a
    # planner that only saw the summary ignored the complaint entirely.
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

    # Revision-turn seed: verify the PREVIOUS plan against current
    # constraints up front (pure Python, no model call). Kept SEPARATE from
    # best-plan tracking: at wrap-up it competes with a one-course overlap
    # handicap, so a candidate that actually made the requested change
    # beats "your plan, unchanged", but a stranger plan still loses to the
    # seed. Without the handicap, the unchanged plan mathematically beats
    # every swap - "replace X" would come back with X still in the plan.
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
        # Per-TURN spend cap, not per-conversation: capping against the
        # running total (cost_so_far includes prior turns) made later turns
        # in a conversation degrade into budget-capped wrap-ups. Each turn
        # gets the same fresh allowance; the dashboard hard limit is the
        # real backstop for total spend.
        if (cost - cost_so_far) >= SESSION_BUDGET_USD_CAP:
            stopped_reason = "budget_cap"
            break

        # Two steps before the ceiling, tell the model to close out NOW -
        # a self-chosen delivery with a real explanation beats drifting
        # into the generic forced wrap-up.
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
            # Plain text instead of a tool call shouldn't happen per the
            # system prompt, but it's never student-facing regardless -
            # treat it as a nudge, not a leak, and give one more chance.
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
                # specific course(s) out. First violation: send the model
                # back once to do the swap properly. Second violation:
                # strip the course(s) in code - an explicit directive is a
                # hard constraint the prompt alone didn't reliably enforce.
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
                # IN, and CheeseFork difficulty/sentiment must never
                # override that. First violation: send the model back once.
                # Second violation: add it in code, dropping the cheapest
                # elective to make room (matches the mandatory top-up
                # backstop's approach).
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
                    # droppable.pop(0) MUST stay outside any comprehension
                    # condition below: inside one, it evaluates once per
                    # candidate element (not once per drop) and raises
                    # IndexError as soon as candidate outsizes droppable -
                    # a real crash this loop hit before, on a heavily-used path.
                    for _ in missing_adds:
                        if droppable:
                            drop_c = droppable.pop(0)
                            candidate = [c for c in candidate if c != drop_c]
                    candidate = candidate + [c for c in missing_adds if c not in candidate]
                    tool_log.append({"name": "add_enforced", "args": {}, "result": {"added": missing_adds}})

                # Verify-before-deliver nudge: if the model tries to deliver
                # without ever having run verify_plan this turn, send it back
                # once - the difference between a real critic loop and a
                # single-shot guess. Fires at most once, and skipped near
                # the step ceiling where correctness is handled server-side.
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
                # retake but the model's course list doesn't include it -
                # a "may include" instruction alone wasn't reliably acted
                # on. First violation: one real chance to include it or
                # explain why it genuinely can't (a real conflict is
                # legitimate). Second violation: enforce in code, same as
                # requested_adds above. Guarded on offered_next_semester
                # too: extraction can resolve a mention to a stale/
                # duplicate course number, and an unregisterable course
                # must never get force-included just because it's "approved".
                approved = state.get("approved_retake_course")
                if approved and not track.courses.get(approved, {}).get("offered_next_semester"):
                    approved = None
                if approved and approved not in candidate:
                    if not approved_retake_pushed and step < MAX_STEPS - 2:
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
                    if approved_retake_pushed:
                        droppable = sorted(
                            (c for c in candidate if c not in track.mandatory_course_numbers and c != approved),
                            key=lambda c: float(track.courses.get(c, {}).get("points") or 0),
                        )
                        if droppable:
                            candidate = [c for c in candidate if c != droppable[0]]
                        candidate = candidate + [approved]
                        tool_log.append(
                            {"name": "approved_retake_enforced", "args": {}, "result": {"course_number": approved}}
                        )

                final_explanation = args.get("explanation") or ""
                course_reasons = args.get("course_reasons") or {}
                # A NEW proposal for THIS turn - distinct from `proposed_retake`
                # above (LAST turn's, already consumed into
                # state["approved_retake_course"]). This becomes NEXT turn's
                # incoming proposed_retake.
                new_proposed_retake = args.get("proposed_retake") or None
                if new_proposed_retake and new_proposed_retake.get("course_number") in candidate:
                    # A course can't be both proposed and already included.
                    new_proposed_retake = None
                # Audit note only, not a safety gate - verify_plan and
                # check_invariants always re-run below regardless, so the
                # student is never shown a falsely-"verified" plan. This
                # just logs when the model delivered without verifying itself.
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
                    track,
                    candidate,
                    passed,
                    failed,
                    approved_retake_course=state.get("approved_retake_course"),
                    confirmed_grade_retakes=state.get("confirmed_grade_retakes"),
                )
                if violations:
                    tool_log.append({"name": "check_invariants", "args": {}, "result": {"violations": violations}})
                    seen: set[str] = set()
                    candidate = [c for c in candidate if c not in passed and c in track.courses and not (c in seen or seen.add(c))]
                    verify_result = tools.verify_plan(
                        track, candidate, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
                    )
                    # The model's own explanation was written BEFORE this
                    # correction, so it can describe issues no longer in the
                    # corrected list. Once Python has changed what's
                    # delivered, only a freshly-generated, deterministic
                    # account of the real issues is trustworthy.
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
                # half-empty semester. If clearly below the credit floor
                # with room to iterate, send it back ONCE to add courses
                # rather than accept a punt. Fires at most once; skipped
                # when the student opted out of minimums (min_credits 0).
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
                # the prompt alone wasn't reliably followed (plans have been
                # delivered with several as credit filler). Enforced in
                # code: one shot, then the delivery stands.
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

                # Retake-dropped pushback: a failed course OR a confirmed
                # grade-improvement retake the student did NOT ask to remove
                # is missing from the delivery. Priority rule #1 (a required
                # retake is retaken now) lived only in the prompt, and under
                # revision pressure the model has silently dropped the
                # retake to satisfy an unrelated request. One shot: send it
                # back with the dropped course(s) named; second delivery
                # stands (honest-disclosure rules cover it). Only fires for
                # courses actually offered next semester - an untakeable
                # retake is a legitimate omission.
                dropped_retakes = [
                    c for c in (set(failed) | (state.get("confirmed_grade_retakes") or set()))
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
                # probability) it could cover RIGHT NOW is missing - this
                # has survived even the generic issue pushes below. Doesn't
                # fix the plan; hands the model the exact takeable variants
                # once and lets it decide (or justify leaving the gap).
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
                # plan may reach the student with AT MOST 2 open issues, and
                # NONE of them "lazy" (credits, mandatory count, exam
                # clashes, overlaps - all fixable by substitution). Only
                # genuinely forced issues (e.g. a day conflict from the
                # student's own constraints) may remain. Fires up to twice.
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
                # didn't do its job - send it back ONCE with the carried-
                # over issues named, demanding a swap. One shot only: if
                # genuinely unavoidable, the second delivery stands (with
                # the honest-disclosure rules already required).
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
        # Python forces a wrap-up: never a silent failure. Uses the BEST
        # plan verified this turn, not merely the most recent one - and on
        # revision turns, weighs the seed (student's unchanged plan) with a
        # one-course overlap handicap so a candidate that actually made the
        # requested change beats "no change", while a stranger plan still
        # loses to the seed.
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
            track,
            final_course_numbers,
            passed,
            failed,
            approved_retake_course=state.get("approved_retake_course"),
            confirmed_grade_retakes=state.get("confirmed_grade_retakes"),
        )
        if violations:
            seen = set()
            final_course_numbers = [
                c for c in final_course_numbers if c not in passed and c in track.courses and not (c in seen or seen.add(c))
            ]
        last_verify_result = tools.verify_plan(
            track, final_course_numbers, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
        )
        # Student-facing text: no internal jargon, and on a revision turn,
        # describe the change relative to the student's existing plan
        # (kept / swapped out / added) - what they actually care about.
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

    # Final safety net: issue_budget_pushback only retries twice, then
    # accepts whatever the model delivers even if the delivery standard
    # (low credits, exam clashes, overlaps) is still violated. If a
    # genuinely better candidate was verified earlier THIS SAME turn,
    # prefer it - never invents a new plan, only picks among ones already
    # checked.
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
                # {ISSUES_SENTENCE}/{COURSE_COUNT}/{TOTAL_CREDITS} are
                # placeholders, deferred because a later backstop (mandatory
                # top-up, locked-course, overlap...) can still resolve one
                # of these issues - a sentence written now could claim a
                # problem already fixed by the time it's shown.
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

    # HARD approved-retake guarantee: the in-loop nudge/enforce above only
    # ever fires when the model actually calls deliver_plan - but
    # the loop can also resolve via forced wrap-up (when the model never
    # delivers) or the final safety swap, neither of which knew about
    # approved_retake_course. An explicit directive must survive EVERY
    # delivery path - checked here unconditionally, before the mandatory/
    # credit top-ups below (so their counts already include it). Covers
    # every retake ever confirmed this conversation
    # (state["confirmed_grade_retakes"]), not just one approved THIS turn,
    # so it doesn't lose protection after the turn it was confirmed on.
    approved = state.get("approved_retake_course")
    retake_targets = sorted(
        (({approved} if approved else set()) | (state.get("confirmed_grade_retakes") or set()))
        - set(requested_removals)
    )
    for retake_target in retake_targets:
        if (
            final_course_numbers
            and retake_target not in final_course_numbers
            and track.courses.get(retake_target, {}).get("offered_next_semester")
        ):
            droppable = sorted(
                (c for c in final_course_numbers if c not in track.mandatory_course_numbers and c != retake_target),
                key=lambda c: float(track.courses.get(c, {}).get("points") or 0),
            )
            if droppable:
                final_course_numbers = [c for c in final_course_numbers if c != droppable[0]]
            final_course_numbers = final_course_numbers + [retake_target]
            retake_name = track.courses[retake_target]["name"] if retake_target in track.courses else retake_target
            last_verify_result = tools.verify_plan(
                track, final_course_numbers, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
            )
            final_explanation += f" Note: added your confirmed retake of {retake_name}."
            tool_log.append(
                {"name": "approved_retake_guaranteed", "args": {}, "result": {"course_number": retake_target}}
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

    # HARD collision backstop: a delivered week must NEVER contain a real
    # class-time collision, on any path. If the coordinated section
    # assignment still has irreducible overlaps, deterministically drop the
    # least valuable course of each colliding pair (keep retakes, then
    # mandatory, then higher credits) until the week is clean.
    if final_course_numbers:
        dropped_for_overlap: list[str] = []
        while True:
            section_check = tools.choose_sections(track, final_course_numbers, excluded_weekdays)
            if not section_check["overlaps"]:
                break
            pair = section_check["overlaps"][0]
            a, b = pair["course_a"], pair["course_b"]

            confirmed_retakes = state.get("confirmed_grade_retakes") or set()

            def keep_value(c):
                course = track.courses.get(c, {})
                is_required_retake = c in failed or c in confirmed_retakes
                return (is_required_retake, c in track.mandatory_course_numbers, float(course.get("points") or 0))

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

    # One last, real-time check against Technion's own system - the static
    # data/track_*.json bundle only refreshes on a manual pipeline rerun,
    # so a course closed/cancelled since then would otherwise be
    # confidently recommended anyway. Fail-open (live_offering_check.py):
    # only dropped when POSITIVELY confirmed gone, never on check failure.
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

    # HARD mandatory-course top-up backstop: a plan can pass credit/workload
    # checks while making almost no real degree progress (filler electives
    # instead of already-unlocked mandatory courses). Swaps unlocked,
    # not-yet-included mandatory courses in for the cheapest electives, up
    # to the credit ceiling - mirrors verify_plan's own min_mandatory_courses
    # so the two never disagree. Runs LAST so a later drop can't reopen the
    # shortfall; each addition is collision-checked.
    if final_course_numbers:
        override_minimums = state["constraints"].get("override_minimums", False)
        min_mandatory = 0 if override_minimums else tools.DEFAULT_MIN_MANDATORY_COURSES
        remaining_mandatory = track.mandatory_course_numbers - set(passed)
        effective_min = min(min_mandatory, len(remaining_mandatory))
        mandatory_in_plan = [c for c in final_course_numbers if c in track.mandatory_course_numbers and c not in passed]
        shortfall = effective_min - len(mandatory_in_plan)
        if shortfall > 0:
            # Never re-add a course the student explicitly asked removed
            # this turn - the directive outranks this heuristic's guess
            # about what's still needed; an open choice-group issue then
            # honestly discloses the trade-off instead of overriding it.
            candidate_pool = sorted(remaining_mandatory - set(final_course_numbers) - set(requested_removals))
            still_locked = tools.prereq_unmet_in(track, candidate_pool, passed, failed) if candidate_pool else set()
            candidates = [
                c for c in candidate_pool
                if c not in still_locked and track.courses.get(c, {}).get("offered_next_semester", True)
            ]
            max_credits = verify_kwargs.get("max_credits", tools.DEFAULT_MAX_CREDITS)

            def _plan_points(nums):
                return sum(float(track.courses[c].get("points") or 0) for c in nums if c in track.courses)

            # Never drop a mandatory course already in the plan, or any
            # retake ever confirmed this conversation (not just one
            # approved this turn) - only genuine filler is fair game,
            # cheapest (least degree value) first.
            retake_protected = ({state.get("approved_retake_course")} if state.get("approved_retake_course") else set()) | (
                state.get("confirmed_grade_retakes") or set()
            )
            droppable = sorted(
                (
                    c for c in final_course_numbers
                    if c not in track.mandatory_course_numbers and c not in retake_protected
                ),
                key=lambda c: float(track.courses.get(c, {}).get("points") or 0),
            )
            added = []
            for candidate in candidates:
                if shortfall <= 0:
                    break
                candidate_points = float(track.courses[candidate].get("points") or 0)
                trial = [c for c in final_course_numbers]
                trial_droppable = list(droppable)
                while trial_droppable and _plan_points(trial) + candidate_points > max_credits:
                    trial.remove(trial_droppable.pop(0))
                if _plan_points(trial) + candidate_points > max_credits:
                    continue  # no room even after dropping every droppable elective
                trial = trial + [candidate]
                if tools.choose_sections(track, trial, excluded_weekdays)["overlaps"]:
                    continue  # would introduce a new collision - skip, nothing runs after this to fix it
                final_course_numbers = trial
                droppable = trial_droppable
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

    # HARD credit-floor top-up backstop: mirrors the mandatory top-up
    # above, but for the credit minimum - the model's own over-trimming
    # (e.g. dropping enough courses to satisfy an "add X" request that it
    # falls well under the floor) can happen regardless of which other
    # backstop fired, so this checks unconditionally against the current
    # total. Runs LAST, after locked-course/overlap/live-offering, so a
    # later drop can't silently reopen a shortfall this just closed. Each
    # candidate is collision-checked before being added since nothing runs
    # afterward to clean up a new overlap this might introduce.
    if final_course_numbers:
        min_credits = verify_kwargs.get("min_credits", tools.DEFAULT_MIN_CREDITS)
        current_points = sum(
            float(track.courses[c].get("points") or 0) for c in final_course_numbers if c in track.courses
        )
        if current_points < min_credits:
            already_in = set(final_course_numbers)
            # Same "never re-add an explicitly removed course" rule as the
            # mandatory top-up above.
            pool = [
                c
                for c in track.courses
                if c not in already_in
                and c not in passed
                and c not in requested_removals
                and track.courses[c].get("offered_next_semester")
            ]
            locked_pool = set(tools.prereq_unmet_in(track, pool, passed, failed)) if pool else set()
            existing_sport = len(_sport_courses_in(track, final_course_numbers))
            candidates = [
                c
                for c in pool
                if c not in locked_pool
                and not (existing_sport >= 1 and c in _sport_courses_in(track, [c]))
                and not (
                    excluded_weekdays
                    and track.courses[c].get("schedule")
                    and {s["weekday"] for s in tools.pick_section(track.courses[c], excluded_weekdays)}
                    <= set(excluded_weekdays)
                )
            ]
            # Shuffle before the (stable) sort so ties on points break
            # randomly instead of by catalog order - this backstop bypasses
            # the shuffle already applied to the model's own elective menu
            # (_available_courses_text), so without this it always added
            # the same highest-point elective. Still prioritizes higher
            # points first (fewer courses needed to close the gap).
            random.shuffle(candidates)
            candidates.sort(key=lambda c: float(track.courses[c].get("points") or 0), reverse=True)
            topped_up = []
            for c in candidates:
                if current_points >= min_credits:
                    break
                trial = final_course_numbers + [c]
                if tools.choose_sections(track, trial, excluded_weekdays)["overlaps"]:
                    continue  # would introduce a new collision - skip, nothing runs after this to fix it
                final_course_numbers = trial
                current_points += float(track.courses[c].get("points") or 0)
                topped_up.append(c)
            if topped_up:
                last_verify_result = tools.verify_plan(
                    track, final_course_numbers, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
                )
                names = ", ".join(track.courses[c]["name"] for c in topped_up)
                final_explanation += (
                    f" Note: added {names} - the plan had fallen below the credit floor; these fill "
                    "it back out from the available shortlist."
                )
                tool_log.append({"name": "credit_floor_topup_enforced", "args": {}, "result": {"added": topped_up}})

    # Guaranteed explanation for a requested-but-unavailable course - same
    # "survives even if a later backstop replaces the model's own prose"
    # pattern as the locked-course explanation below. "Not offered this
    # semester" is a real fact nothing can override, so this never gets a
    # force-anyway offer the way a schedule/room trade-off does below.
    for unavailable_course in sorted(requested_adds_unavailable):
        if unavailable_course not in track.courses:
            continue
        unavailable_name = track.courses[unavailable_course]["name"]
        if unavailable_name in final_explanation or unavailable_course in final_explanation:
            continue
        final_explanation += (
            f" On {unavailable_name}: you asked to add it, but it isn't offered this semester at all - "
            "there's no section to register for, so I can't include it no matter what. It'll need to "
            "wait for a semester it's actually offered."
        )

    # Guaranteed explanation for a requested course number that doesn't
    # exist in the catalog at all, otherwise silently dropped with zero
    # trace. No course name to report, so this names the raw number instead.
    for unknown_course in sorted(requested_adds_unknown):
        if unknown_course in final_explanation:
            continue
        final_explanation += (
            f" On {unknown_course}: I couldn't find a course with that number in this track's catalog, "
            "so I couldn't add it - double check the number, or tell me the course name instead."
        )

    # Guaranteed explanation for a requested add that's already passed -
    # falls between requested_add_courses (excludes it, correctly, since
    # it's not new) and requested_retake_course (never fires without
    # explicit "retake" language). Gives a concrete next step instead of
    # silence: confirming "yes, retake it" routes through that path.
    for passed_add in sorted(requested_adds_already_passed):
        if passed_add not in track.courses:
            continue
        passed_add_name = track.courses[passed_add]["name"]
        if passed_add_name in final_explanation or passed_add in final_explanation:
            continue
        final_explanation += (
            f" On {passed_add_name}: you asked to add it, but you've already passed it, so it's not a "
            "new course to take - I left it out. If you meant you want to retake it, just say so and "
            "I'll treat it as a retake next time; if that's not actually passed on your end, let me "
            "know and I'll correct it."
        )

    # Explicit-add negotiation: add_enforced (in-loop, above) already tries
    # to honor a by-name "add X" request, but a course it force-added can
    # still get stripped by a LATER hard backstop (overlap/live-offering)
    # with a generic note that never acknowledges it was an explicit
    # request. Checked here against the truly final list, after every
    # other backstop has had its say. A schedule collision or no room
    # without dropping something else is a trade-off the STUDENT decides
    # on: named plainly, with a standing force-it-in offer that persists
    # across exactly one turn boundary, same lifecycle as a proposed retake.
    new_pending_forced_add = None
    forced_add_course = state.get("forced_add_course")
    # A plain "yes, add it anyway" confirmation reply doesn't re-name the
    # course - requested_add_courses only fires on a BY-NAME request, so a
    # bare confirmation would otherwise never reach this loop. Checked here
    # regardless of whether it was also named again this turn.
    adds_to_check = list(requested_adds)
    if forced_add_course and forced_add_course not in adds_to_check and forced_add_course in track.courses:
        adds_to_check.append(forced_add_course)
    for c in adds_to_check:
        if c in final_course_numbers or c not in track.courses:
            continue
        name = track.courses[c]["name"]
        trial = final_course_numbers + [c]
        trial_overlaps = tools.choose_sections(track, trial, excluded_weekdays)["overlaps"]
        conflict_partner = next(
            (
                o["course_b"] if o["course_a"] == c else o["course_a"]
                for o in trial_overlaps
                if c in (o["course_a"], o["course_b"])
            ),
            None,
        )
        reason_text = (
            f"its class times collide with {track.courses[conflict_partner]['name']} in every available "
            "section combination"
            if conflict_partner and conflict_partner in track.courses
            else "fitting it in means pushing another course out of this plan"
        )
        if forced_add_course == c:
            final_course_numbers = trial
            last_verify_result = tools.verify_plan(
                track, final_course_numbers, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
            )
            final_explanation += f" Added {name} as you confirmed, despite the trade-off: {reason_text}."
            tool_log.append(
                {"name": "forced_add_enforced", "args": {}, "result": {"course_number": c, "reason": reason_text}}
            )
        else:
            final_explanation += (
                f" On {name}: you asked to add it, but I couldn't - {reason_text}. Want me to add it "
                "anyway despite that? Confirm and I will, no matter what."
            )
            if new_pending_forced_add is None:
                new_pending_forced_add = {"course_number": c, "name": name, "reason": reason_text}

    # HARD "semester 1 defaults to exactly the required courses" guarantee:
    # nothing but this track's real semester-1 mandatory courses, no sport/
    # elective padding, unless the student asked for something explicit (a
    # named add, or a pace that signals otherwise). All 3 tracks' semester-1
    # mandatory sets land at 19.5-22 credits on their own, so this never
    # needs filler to reach the credit floor. override_minimums/"light"
    # pace skips it entirely; "fast" or a named add only skips the
    # extras-stripping half.
    if semester_number == 1 and final_course_numbers:
        pace = state["constraints"].get("pace")
        wants_lighter = bool(state["constraints"].get("override_minimums")) or pace == "light"
        semester1_mandatory = {
            c for c in track.mandatory_course_numbers if track.official_semester.get(c) == 1
        }
        if not wants_lighter and semester1_mandatory:
            missing_s1 = sorted(
                c for c in semester1_mandatory
                if c not in final_course_numbers
                and c not in requested_removals
                and track.courses.get(c, {}).get("offered_next_semester")
            )
            added_s1 = []
            for c in missing_s1:
                trial = final_course_numbers + [c]
                if tools.choose_sections(track, trial, excluded_weekdays)["overlaps"]:
                    continue  # a genuine, irreducible collision - disclosed via the normal issue path, not silently forced
                final_course_numbers = trial
                added_s1.append(c)
            if added_s1:
                last_verify_result = tools.verify_plan(
                    track, final_course_numbers, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
                )
                added_names = ", ".join(track.courses[c]["name"] for c in added_s1)
                final_explanation += (
                    f" Note: added {added_names} - real first-semester requirements the plan was missing."
                )
                tool_log.append(
                    {"name": "semester1_completeness_enforced", "args": {}, "result": {"added": added_s1}}
                )

            has_explicit_extra_ask = bool(requested_add_raw) or pace == "fast"
            if not has_explicit_extra_ask:
                retake_protected = (set(failed) | (state.get("confirmed_grade_retakes") or set())) | (
                    {state.get("approved_retake_course")} if state.get("approved_retake_course") else set()
                )
                keep = semester1_mandatory | retake_protected
                extras = [c for c in final_course_numbers if c not in keep]
                if extras:
                    final_course_numbers = [c for c in final_course_numbers if c not in extras]
                    last_verify_result = tools.verify_plan(
                        track, final_course_numbers, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
                    )
                    removed_names = ", ".join(track.courses[c]["name"] for c in extras if c in track.courses)
                    final_explanation += (
                        f" Note: removed {removed_names} - your first semester defaults to exactly the "
                        "required courses, nothing extra, unless you ask for something specific."
                    )
                    tool_log.append(
                        {"name": "semester1_extras_stripped", "args": {}, "result": {"removed": extras}}
                    )

    # Deterministic backstop, not left to the model: it has filled in
    # deliver_plan's proposed_retake without asking the matching question
    # in the explanation prose (or vice versa) - a silent proposal the
    # student never sees is worse than none. If the field is set, the
    # question WILL appear regardless of what the model wrote.
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

    # Deterministic backstop, same reasoning as proposed-retake above: a
    # later backstop (final_safety_swap, mandatory top-up...) can replace
    # the model's own prose with a synthetic one that never saw the
    # instruction to explain a locked-but-requested course. Guarantee the
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
    # course-count/credit figure they embed would go stale if one of those
    # backstops fires afterward. Substituted here, after every backstop has
    # had its say - a no-op replace when no placeholder was ever inserted.
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
                # A failed-course retake and an approved grade-improvement
                # retake (already passed, deliberately retaken) both get
                # the RETAKE badge, not just the failed case.
                "is_retake": (
                    c in failed
                    or c == state.get("approved_retake_course")
                    or c in (state.get("confirmed_grade_retakes") or set())
                ),
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
        # not a from-scratch re-derivation.
        "known_context": {
            "state": {
                "semester_number": semester_number,
                "constraints": state["constraints"],
                "passed_courses": passed,
                "failed_courses": failed,
                "grades": state.get("grades", {}),
                "confirmed_grade_retakes": sorted(state.get("confirmed_grade_retakes") or []),
            },
            "previous_plan": final_course_numbers,
            # Lets the NEXT turn detect "still has the exact same problem"
            # and push back.
            "previous_issues": [i["reason"] for i in verify_result.get("issues", [])],
            # So a later turn can answer "what did you last recommend?"
            # directly - see previous_explanation in _system_prompt.
            "previous_explanation": plan_result.get("explanation"),
            # A retake proposed THIS turn (or None) - next turn's extraction
            # checks it to resolve approved_grade_retake. Replaces, never
            # stacks with, whatever was pending before.
            "proposed_retake": new_proposed_retake,
            # Same single-slot, one-turn-boundary lifecycle as proposed_retake.
            "pending_forced_add": new_pending_forced_add,
        },
    }
