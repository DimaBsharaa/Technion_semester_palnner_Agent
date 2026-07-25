"""
The planning agent's control loop.

Earlier versions gave the model a free-form tool-calling loop and trusted it
to decide when to keep working silently vs. when to stop and ask the
student something. In practice, across repeated testing, the model kept
ending turns early - asking permission for the obvious next step, or asking
the student to supply a course number instead of resolving it - no matter
how forcefully the system prompt said not to. Prompting alone wasn't
reliable enough, so the control flow itself is now code, not the model's
judgment:

1. extract_student_state - one call, given the *entire real course catalog*
   directly in its prompt (not a tool it might skip calling), pulls out
   constraints and passed/failed courses and resolves any course named in
   English/Hebrew/approximately to a real number itself. Only produces a
   question when something genuinely blocks planning. Skipped entirely when
   the caller already has structured form data (see build_state_from_intake) -
   this was the single least reliable step in the whole pipeline, so a form
   the frontend can render directly from known data removes the need for an
   LLM to guess at it at all.
2. draft_plan / repair_plan - proposes a course list; tools.verify_plan (pure
   Python, no LLM) checks it; on failure, the *same kind of call* runs again
   with the specific failure reasons, in a fixed Python loop - the model
   never gets an open turn to ask permission mid-repair, because there is no
   turn there, just a function call with a bounded retry count.
3. explain_plan - writes the student-facing message exactly once, after a
   plan has passed (or repair attempts are exhausted), and is explicitly
   told never to ask a question or offer next steps - but must be honest
   about any unresolved hard-constraint conflict rather than glossing over
   it, when repair attempts genuinely couldn't find a fully valid plan.

No tool-calling API is used here at all - each step is a plain structured
completion (JSON in, JSON or prose out), which is both more predictable and
easier to keep within budget than an open-ended tool loop.

Every function takes the specific Track (see data_bundle.py) a student is
planning for as an explicit parameter, rather than reading module-level
"the current track" state - multiple students in different tracks can be
served by the same running process without their requests interfering with
each other.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

import tools
from data_bundle import Track, get_track
from prereq_parser import collect_prereq_courses


def _load_env_file() -> None:
    """Zero-dependency loader for an `agent/.env` file, so the API key,
    model, and pricing can live in one gitignored file instead of being
    exported by hand in every shell (and so test scripts run via `python3`
    pick them up without a wrapper). Lines are KEY=VALUE; blank lines and
    `#` comments are ignored. .env WINS over any pre-existing environment
    variable - this is deliberate: a leftover OPENAI_BASE_URL from an
    earlier proxy session in the running server's environment must not
    silently override the file that is now the single source of truth."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # An empty value means "unset this" - the OpenAI SDK auto-reads
        # OPENAI_BASE_URL from the environment, and an empty string there is
        # a broken URL, not "use the default." Removing the key is what
        # actually selects OpenAI's default endpoint.
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


_load_env_file()

MODEL = os.environ.get("OPENAI_MODEL", "MB5R2CF-azure/gpt-5.4-mini")
BASE_URL = os.environ.get("OPENAI_BASE_URL") or None
PRICE_INPUT_PER_1M = float(os.environ.get("OPENAI_PRICE_INPUT_PER_1M", "0.25"))
PRICE_OUTPUT_PER_1M = float(os.environ.get("OPENAI_PRICE_OUTPUT_PER_1M", "2.00"))
SESSION_BUDGET_USD_CAP = float(os.environ.get("SESSION_BUDGET_USD_CAP", "0.50"))
MAX_REPAIR_ATTEMPTS = int(os.environ.get("MAX_REPAIR_ATTEMPTS", "5"))

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        # Explicit per-request timeout: the SDK default is 600s, which lets a
        # single hung connection stall an entire turn for ten minutes
        # (observed live during test-plan execution - a turn froze >5min
        # while /health stayed instant). 90s is generous for one completion.
        # max_retries=4: the SDK retries 429 rate-limits with exponential
        # backoff, and a TPM 429 observed live needed under a second of
        # waiting - one retry wasn't always enough and surfaced as a 500.
        _client = OpenAI(base_url=BASE_URL, timeout=90.0, max_retries=4)
    return _client


# --- Per-request LLM-call trace (course-spec /api/execute "steps") ---
# Thread-local so concurrent requests never mix traces. Each entry follows
# the course's required schema: module name (consistent with the
# architecture diagram), the exact prompts, and the raw response.
import threading as _threading

_trace_local = _threading.local()


def start_llm_trace() -> None:
    _trace_local.steps = []


def get_llm_trace() -> list[dict]:
    return getattr(_trace_local, "steps", [])


def record_llm_step(module: str, messages: list[dict], response_payload) -> None:
    steps = getattr(_trace_local, "steps", None)
    if steps is None:
        return
    system = "\n".join(m.get("content") or "" for m in messages if m.get("role") == "system")
    user = "\n".join(
        f"[{m.get('role')}] {m.get('content') or json.dumps(m.get('tool_calls', ''), ensure_ascii=False)}"
        for m in messages
        if m.get("role") != "system"
    )
    steps.append(
        {
            "module": module,
            "prompt": {"System_prompt": system, "User_prompt": user},
            "response": response_payload,
        }
    )


def _call(messages: list[dict], json_mode: bool, module: str = "LLM") -> tuple[str, float]:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = _get_client().chat.completions.create(model=MODEL, messages=messages, **kwargs)
    usage = response.usage
    cost = (usage.prompt_tokens / 1_000_000) * PRICE_INPUT_PER_1M + (
        usage.completion_tokens / 1_000_000
    ) * PRICE_OUTPUT_PER_1M
    content = response.choices[0].message.content
    record_llm_step(module, messages, {"text": content})
    return content, cost


def _parse_json(text: str) -> dict:
    # Some models wrap JSON in a markdown fence even when told not to.
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


# --- Per-track prompt text, cached by track_id (a track's data never changes
# during the process lifetime, so this only needs computing once per track). ---


@lru_cache(maxsize=None)
def _catalog_text(track_id: str) -> str:
    track = get_track(track_id)
    return "\n".join(f"{num} — {c['name']} ({c['points']} pts)" for num, c in sorted(track.courses.items()))


@lru_cache(maxsize=None)
def _catalog_with_meta_text(track_id: str) -> str:
    """Like _catalog_text, but with per-course difficulty, exam date, and
    mandatory-vs-elective status - what the drafting step needs to
    proactively balance a plan (reach for a no-exam sport course to round
    out credits, avoid stacking several heavy courses together, and -
    directly addressing the most common real-advisor mistake named in
    building this - see clustered exam dates coming and route around them
    itself, instead of only reacting to verify_plan's complaint after
    drafting something that collides)."""
    track = get_track(track_id)
    lines = []
    for num, c in sorted(track.courses.items()):
        cf = c["cheesefork"]
        load = f"{cf['avg_difficulty']}/5 difficulty" if cf["avg_difficulty"] is not None else "no review data"
        satisfaction = f", {cf['avg_general']}/5 satisfaction" if cf["avg_general"] is not None else ""
        exam_dates = c["exams"].get("moed_a", [])
        exam_note = f"exam {exam_dates[0]['date']}" if exam_dates else "no exam"
        tag = "MANDATORY" if num in track.mandatory_course_numbers else "elective"
        lines.append(f"{num} — {c['name']} ({c['points']} pts, {load}{satisfaction}, {exam_note}, {tag})")
    return "\n".join(lines)


@lru_cache(maxsize=None)
def _mandatory_text(track_id: str) -> str:
    track = get_track(track_id)
    return "\n".join(
        f"{c} — {track.courses[c]['name']}" for c in sorted(track.mandatory_course_numbers) if c in track.courses
    )


@lru_cache(maxsize=None)
def _foundation_text(track_id: str) -> str:
    """The mandatory list alone isn't the right basis for "what has a
    continuing student already completed" - a mandatory course can have
    prerequisites that Technion files under a completely different category
    (e.g. a math course under a science-elective list), and a student who
    passed the mandatory course necessarily passed those too. Walking the
    real prerequisite trees (not category labels) to find the full closure
    avoids exactly the bug this caught: assuming a retake's own prereqs
    weren't met just because they live outside the mandatory listing."""
    track = get_track(track_id)
    closure = set(track.mandatory_course_numbers)
    frontier = list(track.mandatory_course_numbers)
    while frontier:
        course_number = frontier.pop()
        course = track.courses.get(course_number)
        if course is None:
            continue
        for prereq_number in collect_prereq_courses(course["prerequisites"]):
            if prereq_number not in closure:
                closure.add(prereq_number)
                frontier.append(prereq_number)

    # Some prerequisite courses aren't in this track's own bundle at all
    # (they belong to a different track/faculty's requirement tree, or are
    # older course numbers) - that's not a reason to drop them here, since
    # the model only needs the number to reason about "would a continuing
    # student have passed this," not a display name.
    lines = [
        f"{c} — {track.courses[c]['name']}" if c in track.courses else f"{c} — (prerequisite course, name unavailable)"
        for c in sorted(closure)
    ]
    return "\n".join(lines)


def _extract_student_state(track: Track, messages: list[dict], known_state: dict | None = None) -> tuple[dict, float]:
    known_state_note = ""
    if known_state:
        known_state_note = f"""

Already established from an earlier turn in this same conversation - \
carry every one of these forward EXACTLY as given below unless the \
student's latest message explicitly changes it. Do not re-derive these \
from scratch, do not become less certain of something just because a new \
message was added, and do not ask about anything already settled here:
- semester_number: {known_state.get("semester_number")}
- constraints: {json.dumps(known_state.get("constraints"), ensure_ascii=False)}
- passed_courses: {known_state.get("passed_courses")}
- failed_courses: {known_state.get("failed_courses")}
Only the specific thing the student's newest message actually addresses \
should change - e.g. a message adding one new exam-date conflict should \
update constraints/notes only, not reset passed_courses or semester_number."""

    system = {
        "role": "system",
        "content": f"""\
You extract structured information from a conversation between a Technion \
student and a semester-planning assistant, for the "{track.name}" track, \
planning for {track.target_semester}.
{known_state_note}

Full course catalog for this track (number — name (points)):
{_catalog_text(track.otjid)}

The mandatory/required courses for this track - ask about these by name if \
you need to:
{_mandatory_text(track.otjid)}

The mandatory list's own prerequisite chain also includes these courses \
(from other categories or even other faculties) - don't ask about these \
individually, just include the ones implied by your reasoning below:
{_foundation_text(track.otjid)}

From the conversation, extract a JSON object with exactly these keys:
- "semester_number": the integer semester the student says they're \
starting (e.g. 4). Required to check their progress against a typical \
sequence - ask for this if it's genuinely not stated.
- "constraints": {{"excluded_weekdays": [ints, 0=Sunday..5=Friday] - use \
null when day availability was never mentioned, and [] ONLY when the \
student explicitly says they have no day restrictions or LIFTS a previous \
one ("Sundays are fine now" with no other exclusions -> []), \
"pace": one of "light"/"normal"/"fast"/null, "notes": short string or "", \
"override_minimums": true only if the student explicitly asked for fewer \
than a normal full course load (e.g. "just the one retake", "as few \
credits as possible") - false by default, since a normal semester should \
carry a real course load unless they've said otherwise}}
- "passed_courses": real 8-digit course numbers (from either list above).
- "passed_all_expected": true when the student says they passed \
everything so far / everything expected / everything in all their \
previous semesters (e.g. "I passed everything in my first semester" from \
a student starting semester 2). That statement fully answers the \
course-status question: leave passed_courses empty if you like - the \
expected list is filled in deterministically by code - and "courses" \
must NOT appear in missing_fields. false when they only partially \
described their record or mentioned failures alongside it (failures \
still go in failed_courses either way).
- "failed_courses": real 8-digit course numbers they said they failed or \
still need to take.
- "remove_courses": real 8-digit course numbers the student explicitly \
asked to REMOVE, REPLACE, or SWAP OUT of their existing plan in their \
LATEST message (resolve names to numbers via the catalog above). Empty \
list when they aren't asking to remove anything - this is only for \
explicit change requests against a plan they already received.
- "intent": "question" ONLY when the latest message is purely asking for \
information or opinion ("is X hard?", "what do students say about Y?", \
"which is easier, A or B?") with NO request to build or change a plan; \
otherwise "plan". A change request ("swap X", "make it lighter") is \
"plan", not "question".
- "question_courses": when intent is "question", the 8-digit numbers of \
the course(s) being asked about (resolved via the catalog); else [].
- "ready_to_plan": true once you have enough of the mandatory list's \
pass/fail status to check prerequisites (not necessarily every single \
elective - just the mandatory list) AND the student's semester/day \
constraints are known. false otherwise.
- "clarifying_question": if ready_to_plan is false, ONE short sentence \
introducing what's still needed (e.g. "Just need a couple more things:") - \
the specifics will be shown as button choices, not asked in this text, so \
don't enumerate them here and never ask for a course NUMBER. If \
ready_to_plan is true, this must be null.
- "missing_fields": a list of whichever of these are still genuinely \
unknown after reading the conversation - "semester_number", "weekdays", \
"pace", "courses" (meaning mandatory-course pass/fail status). Empty list \
when ready_to_plan is true. Only include a field here if you truly cannot \
make a reasonable call on it - per the judgment guidance below, that should \
be rare for "courses" specifically.

Use real-world judgment, not just literal statements, to fill in \
passed_courses: a student who tells you their semester number and names \
specific courses they still need to complete is telling you something \
about everything else too, the same way a person who says "I've read every \
book on this shelf except that one" doesn't need to also list every book \
they did read. Reason about what a student at that point in the track would \
realistically have completed by now, and treat asking again as the failure \
mode to avoid - only do it when you genuinely can't make a reasonable call, \
not as a default. A wrong assumption can always be corrected later if the \
student pushes back.

You must resolve every course name yourself using the catalog above - \
translating between English and Hebrew as needed - and never ask the \
student to supply a course number.

Respond with ONLY the JSON object, no other text.""",
    }
    content, cost = _call([system] + messages, json_mode=True, module="Understand")
    state = _parse_json(content)
    return _normalize_state(state), cost


def _normalize_state(state: dict) -> dict:
    """The model's JSON output isn't strictly typed no matter how the schema
    is described in the prompt - this coerces the fields later code does
    arithmetic/comparisons on (semester_number, excluded_weekdays) into the
    types they're actually used as, instead of letting a string where an
    int was expected surface as a confusing TypeError deep in a downstream
    tool call."""

    def to_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    state["semester_number"] = to_int(state.get("semester_number"), default=0)

    constraints = state.get("constraints") or {}
    # None (never mentioned) and [] (explicitly no restrictions / lifted)
    # are DIFFERENT: merge_known_state must be able to tell "unknown, keep
    # the earlier answer" from "the student just cleared this constraint" -
    # collapsing both to [] made relaxing a constraint impossible (found
    # live: "Sundays are fine now" kept excluding Sundays).
    if constraints.get("excluded_weekdays") is None:
        constraints["excluded_weekdays"] = None
    else:
        constraints["excluded_weekdays"] = [
            to_int(d) for d in constraints["excluded_weekdays"] if to_int(d, default=None) is not None
        ]
    constraints["override_minimums"] = bool(constraints.get("override_minimums", False))
    state["constraints"] = constraints

    # Technion course numbers are always 8 digits - the extraction model
    # doesn't always keep leading zeros (e.g. "940219" instead of
    # "00940219"), which would otherwise silently split one real course into
    # two different dict/set keys everywhere downstream (prereq checks,
    # "already passed" checks, etc.).
    state["passed_courses"] = [str(c).zfill(8) for c in (state.get("passed_courses") or [])]
    state["failed_courses"] = [str(c).zfill(8) for c in (state.get("failed_courses") or [])]
    state["remove_courses"] = [str(c).zfill(8) for c in (state.get("remove_courses") or [])]
    state["intent"] = state.get("intent") if state.get("intent") in ("plan", "question") else "plan"
    state["question_courses"] = [str(c).zfill(8) for c in (state.get("question_courses") or [])]

    valid_missing = {"semester_number", "weekdays", "pace", "courses"}
    state["missing_fields"] = [f for f in (state.get("missing_fields") or []) if f in valid_missing]

    # "I passed everything so far" answers the course-status question in
    # full - backfill_passed_courses fills the actual list from the track's
    # expected-by-now set, so re-asking via the checklist (observed live:
    # a semester-2 student who said exactly that still got the passed/failed
    # checklist) is pure re-asking of a settled fact. Enforced here in code
    # rather than trusted to the extraction model's judgment.
    state["passed_all_expected"] = bool(state.get("passed_all_expected", False))
    if state["passed_all_expected"]:
        state["missing_fields"] = [f for f in state["missing_fields"] if f != "courses"]
        if not state["missing_fields"]:
            state["ready_to_plan"] = True
            state["clarifying_question"] = None

    return state


def merge_known_state(extracted: dict, known_state: dict) -> dict:
    """Deterministic state continuity for revision turns (a plan was already
    delivered this conversation). Extraction re-reads the whole transcript
    every turn and - being a probabilistic call - can come back LESS sure
    than the previous turn, up to re-asking the student things they already
    answered (observed live: a student pointing out an exam conflict got
    re-asked which courses they'd passed). The carry-forward instruction in
    the extraction prompt alone is not a guarantee; this merge is.

    Rules, in code, not prompt:
    - A settled fact can be refined by the new message, never un-known:
      extraction's non-empty values win, known_state fills every gap.
    - passed/failed merge as a union with the newest explicit claims
      winning: something the student NOW says they failed is failed even if
      previously counted passed, and vice versa.
    - A revision turn always proceeds to planning - it can never bounce the
      student back to the intake buttons. If the delivered plan's facts were
      complete enough to plan once, they still are."""
    merged = dict(extracted)

    if not merged.get("semester_number") and known_state.get("semester_number"):
        merged["semester_number"] = known_state["semester_number"]

    known_constraints = known_state.get("constraints") or {}
    constraints = dict(merged.get("constraints") or {})
    # None = "this turn didn't mention days" -> keep the earlier answer.
    # [] = "the student explicitly cleared the restriction" -> respect it.
    if constraints.get("excluded_weekdays") is None and known_constraints.get("excluded_weekdays") is not None:
        constraints["excluded_weekdays"] = list(known_constraints["excluded_weekdays"])
    if constraints.get("excluded_weekdays") is None:
        constraints["excluded_weekdays"] = []
    if not constraints.get("pace") and known_constraints.get("pace"):
        constraints["pace"] = known_constraints["pace"]
    if not constraints.get("notes") and known_constraints.get("notes"):
        constraints["notes"] = known_constraints["notes"]
    if known_constraints.get("override_minimums"):
        constraints.setdefault("override_minimums", True)
    merged["constraints"] = constraints

    extracted_passed = set(merged.get("passed_courses") or [])
    extracted_failed = set(merged.get("failed_courses") or [])
    known_passed = set(known_state.get("passed_courses") or [])
    known_failed = set(known_state.get("failed_courses") or [])
    # Union both, newest explicit claims win on conflict: a course the
    # student now says they failed leaves passed (and vice versa - a
    # previously-failed course they now say they passed leaves failed).
    passed = (extracted_passed | known_passed) - extracted_failed
    failed = (extracted_failed | known_failed) - passed
    merged["passed_courses"] = sorted(passed)
    merged["failed_courses"] = sorted(failed)

    merged["missing_fields"] = []
    merged["ready_to_plan"] = True
    merged["clarifying_question"] = None
    return merged


_WEEKDAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def build_state_from_intake(intake: dict) -> dict:
    """Builds the same state shape _extract_student_state produces, but
    from a structured form submission instead of an LLM guessing at free
    text - skips the single least reliable step in this whole pipeline
    entirely when the frontend can just ask directly."""
    state = {
        "semester_number": intake.get("semester_number"),
        "constraints": {
            "excluded_weekdays": intake.get("excluded_weekdays", []),
            "pace": intake.get("pace"),
            "notes": intake.get("notes", ""),
            "override_minimums": intake.get("override_minimums", False),
        },
        "passed_courses": intake.get("passed_courses", []),
        "failed_courses": intake.get("failed_courses", []),
        "ready_to_plan": True,
        "clarifying_question": None,
    }
    return _normalize_state(state)


def summarize_intake_as_message(track: Track, state: dict) -> str:
    """A human-readable stand-in for what the student "said," so the
    conversation transcript still reads naturally for later free-text
    follow-up turns (which do go through _extract_student_state, and
    benefit from seeing this context same as any other prior turn)."""
    excluded = [_WEEKDAY_NAMES[d] for d in state["constraints"]["excluded_weekdays"] if 0 <= d < 6]
    passed_names = [track.courses[c]["name"] for c in state["passed_courses"] if c in track.courses]
    failed_names = [track.courses[c]["name"] for c in state["failed_courses"] if c in track.courses]

    parts = [f"Semester {state['semester_number']}."]
    if excluded:
        parts.append(f"Can't attend classes on: {', '.join(excluded)}.")
    if state["constraints"].get("pace"):
        parts.append(f"Preferred pace: {state['constraints']['pace']}.")
    if passed_names:
        parts.append(f"Passed: {', '.join(passed_names)}.")
    if failed_names:
        parts.append(f"Failed/still need: {', '.join(failed_names)}.")
    if state["constraints"].get("notes"):
        parts.append(state["constraints"]["notes"])
    return " ".join(parts)


def answer_course_question(track: Track, state: dict, messages: list[dict]) -> tuple[str, float]:
    """A question turn ("is X hard?") gets a single grounded answer built
    from the real data in the bundle - crowd difficulty/satisfaction
    numbers and actual review excerpts - instead of spinning up the whole
    planning loop for something that isn't a planning request. Cheap (one
    small call) and honest: the model is told to ground claims in the
    quoted reviews and admit when the sample is thin."""
    course_numbers = [c for c in state.get("question_courses", []) if c in track.courses]
    course_blocks = []
    for c in course_numbers:
        course = track.courses[c]
        cf = course["cheesefork"]
        exams = course["exams"].get("moed_a", [])
        course_blocks.append(
            json.dumps(
                {
                    "course_number": c,
                    "name": course["name"],
                    "points": course["points"],
                    "avg_difficulty_1to5": cf["avg_difficulty"],
                    "avg_satisfaction_1to5": cf["avg_general"],
                    "review_count": cf["review_count"],
                    "review_excerpts": cf.get("sample_reviews", []),
                    "exam_moed_a": exams[0]["date"] if exams else None,
                    "mandatory": c in track.mandatory_course_numbers,
                },
                ensure_ascii=False,
            )
        )
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
    system = {
        "role": "system",
        "content": f"""\
You are an experienced Technion academic advisor answering a student's \
QUESTION - not building a plan. Answer in 2-4 sentences, grounded in the \
data below: cite the crowd numbers, quote a short phrase from a real \
review when it supports your point (in its original language), and be \
honest when the sample is thin (low review_count). If the student compares \
courses, compare directly. Do not offer to build or change a plan - just \
answer.

Course data:
{chr(10).join(course_blocks) if course_blocks else "(no specific course resolved - answer generally from the track's perspective)"}

Student's question: {last_user.get("content", "")}""",
    }
    return _call([system], json_mode=False, module="CourseQA")


def backfill_passed_courses(track: Track, state: dict) -> tuple[list[str], list[str]]:
    """Deterministically fills in passed_courses rather than trusting the
    model to apply "everything except the named failure" correctly every
    time - it's a probabilistic call and has under-included prerequisite
    courses (like a retake's own prereqs living outside the mandatory list)
    in testing even with an explicit instruction to include them. Anything
    this track's own depth heuristic already expects a student at this
    semester to have finished (see assess_progress) gets added here in
    code, transitively through its own prerequisites, unless the student
    explicitly named it as failed/outstanding.

    Shared by both the pipeline (run_agent_turn) and the agent loop
    (agent_react_loop.run_agent_turn_v2) so a fix here - like the
    passed/failed contradiction bug found via live testing - applies to
    both rather than needing to be duplicated and kept in sync."""
    failed = set(state["failed_courses"])
    semester_number = state.get("semester_number") or 0
    expected_by_now = {
        c for c, d in track.mandatory_prereq_depths.items() if d <= semester_number - 2
    } - failed
    # Choose-one-variant requirements due by now: mark ALL variants passed
    # (we can't know which one the student took, and later prerequisites
    # accept any variant of the group), unless the student explicitly named
    # one as failed.
    for group in track.mandatory_choice_groups:
        options = set(group["options"])
        if group["depth"] <= semester_number - 2 and not (options & failed):
            expected_by_now |= options
    auto_passed = set(expected_by_now)
    for c in expected_by_now:
        if c in track.courses:
            auto_passed.update(collect_prereq_courses(track.courses[c]["prerequisites"]))
    # A course walked in as someone ELSE's transitive prerequisite must never
    # override an explicit failure - without this subtraction, a failed
    # course that's also a foundation for other mandatory courses gets
    # silently re-added as "passed," producing a plan that's simultaneously
    # a retake and "already passed."
    passed = sorted((set(state["passed_courses"]) | auto_passed) - failed)
    return passed, sorted(failed)


def resolve_verify_kwargs(state: dict) -> dict:
    """18 credits is the target load for a normal-pace semester; a bit
    above is fine. A student who explicitly asks for "light" gets a lower
    floor (14) and a lower ceiling (18, i.e. never above the normal
    target) rather than the normal range."""
    override_minimums = state["constraints"].get("override_minimums", False)
    pace = state["constraints"].get("pace")
    if override_minimums:
        return {"min_credits": 0, "min_mandatory_courses": 0}
    if pace == "light":
        return {"min_credits": tools.LIGHT_PACE_MIN_CREDITS, "max_credits": tools.LIGHT_PACE_MAX_CREDITS}
    return {"min_credits": tools.DEFAULT_MIN_CREDITS, "max_credits": tools.DEFAULT_MAX_CREDITS}


_REPAIR_STRATEGIES = {
    "swap_elective": "replace a non-mandatory course with a different one to resolve the conflict",
    "defer_course": "drop a non-critical course from this semester entirely, to be taken a later semester",
    "rebalance_credits": "change which electives fill the credit gap, without touching mandatory/retake courses",
    "accept_tradeoff": (
        "ONLY for a genuine constraint conflict you've concluded no swap resolves without creating an equal or "
        "worse problem (e.g. the only section of a required course meets on the one excluded weekday, with no "
        "alternative section) - never for something with a plain fix, like an already-passed course still sitting "
        "in the plan; the student will be told about a real trade-off honestly rather than being handed a "
        "silently-broken constraint"
    ),
}


def _draft_or_repair_plan(
    track: Track,
    state: dict,
    catalog: dict,
    prereq: dict,
    progress: dict,
    previous_plan: list[str] | None,
    previous_verify: dict | None,
    future_impact: dict | None = None,
    bias: str | None = None,
    allow_accept_tradeoff: bool = True,
    delivered_plan: list[str] | None = None,
) -> tuple[list[str], str, str, float]:
    # delivered_plan is the plan the student ALREADY RECEIVED in an earlier
    # turn of this conversation - a completely different thing from
    # previous_plan (this turn's own failed repair attempt). When present,
    # the student's latest message is feedback on an existing plan, and the
    # right behavior is a minimal revision of it, not a fresh draft that
    # forgets everything they were already shown.
    revision_note = ""
    if delivered_plan:
        revision_note = f"""

The student ALREADY RECEIVED this plan in an earlier turn: {delivered_plan}
Their latest message is feedback on it, not a fresh request. Treat this as \
a revision: keep every course from that plan that still fits the updated \
constraints, and change ONLY what the new information actually breaks. A \
student who reports one exam conflict expects one course swapped, not a \
completely different plan."""

    repair_note = ""
    if previous_plan is not None:
        strategy_menu = "\n".join(f'- "{name}": {desc}' for name, desc in _REPAIR_STRATEGIES.items())
        if not allow_accept_tradeoff:
            strategy_menu = "\n".join(
                f'- "{name}": {desc}' for name, desc in _REPAIR_STRATEGIES.items() if name != "accept_tradeoff"
            )
            strategy_menu += (
                '\n(You have not tried enough real fixes yet to give up - "accept_tradeoff" is not available '
                "this attempt.)"
            )
        repair_note = f"""

Your previous candidate plan was: {previous_plan}
It failed verification for these reasons:
{json.dumps(previous_verify["issues"], ensure_ascii=False)}
(workload_score={previous_verify["workload_score"]}, total_credits={previous_verify["total_credits"]})
{"Planning-horizon warning (advisory, not a hard failure): " + json.dumps(future_impact["bottleneck_warnings"], ensure_ascii=False) if future_impact and future_impact.get("bottleneck_warnings") else ""}

Before producing the repaired course list, pick exactly ONE repair \
strategy that best addresses the issues above:
{strategy_menu}

Produce a repaired plan consistent with that strategy - swap or drop the \
specific problematic course(s) rather than starting over completely, \
unless the strategy you picked is "accept_tradeoff", in which case keep \
the plan as it was."""

    bias_note = ""
    if bias:
        bias_note = f"\n\nAdditionally, draft this as a deliberately {bias} alternative to a standard plan - a genuinely different option worth comparing, not just a minor variation."

    system = {
        "role": "system",
        "content": f"""\
You draft a candidate {track.target_semester} course plan for a Technion \
"{track.name}" student, acting like an experienced Technion academic \
advisor, not just a constraint solver.

Hard rule, never negotiable: never include a course from the student's \
passed list below in this plan - it's already completed, retaking it would \
waste a slot. Only a course from the failed/outstanding list may reappear, \
and only as a deliberate retake. If a repair note below flags "already \
passed" as an issue, that is always a straightforward fix (drop the \
course) - never a genuine trade-off to accept.

This is how that advisor actually thinks, in priority order:

1. Retaking a failed course with a high gateway_score (query_prereq_graph \
below) happens now, regardless of its difficulty - this is not up for \
negotiation with the student's stated pace. A blocked degree costs more \
than one hard semester.
2. The student's "pace" preference shapes everything BUILT AROUND that \
retake, not whether it happens: pair a heavy mandatory retake with \
genuinely light electives or "no exam" courses (sport/PE, choir - marked \
"no exam" below); if the mandatory side of the plan is already light, \
there's more room for a substantive elective instead of just the easiest \
option available. Look at each candidate's difficulty and exam presence \
below to make this trade-off directly, not just to react to the credit \
total.
3. Exam-date clustering is the single most common real-advisor mistake to \
avoid: look at the exam date printed for each candidate course below and \
actively avoid putting two exams within a few days of each other - don't \
wait for verify_plan to catch this after the fact, route around it while \
choosing.
4. Unless the student's constraints below say "override_minimums": true: \
include at least {tools.DEFAULT_MIN_MANDATORY_COURSES} mandatory courses if \
that many remain. Target {tools.DEFAULT_MIN_CREDITS} credit points total - \
a bit above ({tools.DEFAULT_MAX_CREDITS} max) is perfectly fine, don't \
trim a good plan just to land exactly on {tools.DEFAULT_MIN_CREDITS}. If \
the student's pace is "light", the target drops to \
{tools.LIGHT_PACE_MIN_CREDITS}-{tools.LIGHT_PACE_MAX_CREDITS} instead - \
still a real course load, just the lighter end of one. Use lighter \
electives or no-exam courses to round out credits rather than leaving the \
semester underloaded.
5. When choosing which elective fills a credit gap, prefer one with a \
decent satisfaction rating (shown below) over a similarly-light one \
students clearly dislike (roughly below 2.5/5) - light workload doesn't \
excuse picking something students regret taking, when an equally light \
alternative with a better rating is available.

Full course catalog (number — name (points, difficulty, exam date, \
mandatory/elective status)) - only use numbers from here:
{_catalog_with_meta_text(track.otjid)}

Student's semester: {state.get("semester_number")}
Student's passed courses: {state["passed_courses"]}
Student's failed/outstanding required courses: {state["failed_courses"]}
Student's constraints: {json.dumps(state["constraints"], ensure_ascii=False)}

Progress vs. a typical student at this point in the track: \
{json.dumps(progress, ensure_ascii=False)}

Degree requirement audit:
{json.dumps(catalog, ensure_ascii=False)}

Prerequisite graph analysis (what failed courses block, gateway scores):
{json.dumps(prereq, ensure_ascii=False)}\
{revision_note}\
{repair_note}\
{bias_note}

Respond with ONLY a JSON object: {{"course_numbers": ["...", ...], \
"repair_strategy": "{'one of the strategy names above' if previous_plan is not None else 'initial_draft'}", \
"repair_reasoning": "one short sentence, or empty string on the first draft"}}.""",
    }
    content, cost = _call([system], json_mode=True, module="PlannerDraft")
    parsed = _parse_json(content)
    course_numbers = [c for c in parsed.get("course_numbers", []) if c in track.courses]
    repair_strategy = parsed.get("repair_strategy") or ("initial_draft" if previous_plan is None else "swap_elective")
    repair_reasoning = parsed.get("repair_reasoning") or ""
    return course_numbers, repair_strategy, repair_reasoning, cost


def _choose_best_plan(track: Track, state: dict, comparison: list[dict]) -> tuple[str, str, float]:
    """A genuine judgment-under-alternatives call: given two already-verified
    candidate plans and their objective metrics (computed by
    tools.compare_plans, not by this model), pick which one actually serves
    this specific student better and say why - a fixed formula could rank
    them, but "why" often depends on things a formula can't weigh, like
    whether the student's own phrasing suggested they're stretched thin this
    semester. Kept as its own structured decision, separate from the prose
    explanation, so it's a distinct visible choice rather than something
    baked silently into the final write-up."""
    system = {
        "role": "system",
        "content": f"""\
Two candidate plans for the same student both passed or were the best \
achievable option. Pick the one that actually serves this student better \
given everything you know about them - don't just default to whichever has \
fewer issues if the other represents a genuinely better trade-off for their \
situation.

Student's constraints: {json.dumps(state["constraints"], ensure_ascii=False)}
Student's failed/outstanding required courses: {state["failed_courses"]}

Candidates:
{json.dumps(comparison, ensure_ascii=False)}

Respond with ONLY a JSON object: {{"winner": "<label of the winning \
candidate, exactly as given>", "reasoning": "one short sentence"}}.""",
    }
    content, cost = _call([system], json_mode=True, module="PlanChooser")
    parsed = _parse_json(content)
    labels = {c["label"] for c in comparison}
    winner = parsed.get("winner") if parsed.get("winner") in labels else comparison[0]["label"]
    reasoning = parsed.get("reasoning") or ""
    return winner, reasoning, cost


def _explain_plan(
    track: Track,
    state: dict,
    plan: list[str],
    verify_result: dict,
    exam_dates: dict,
    cheesefork: dict,
    progress: dict,
    future_impact: dict | None = None,
    alternative_note: str | None = None,
) -> tuple[str, float]:
    plan_details = [
        {
            "course_number": c,
            "name": track.courses[c]["name"],
            "points": track.courses[c]["points"],
            "mandatory": c in track.mandatory_course_numbers,
        }
        for c in plan
    ]
    system = {
        "role": "system",
        "content": f"""\
Write the final message to the student presenting their {track.target_semester} \
semester plan. The student already sees a visual breakdown with every \
course's name, credits, difficulty, satisfaction rating, and a review \
excerpt, plus an exam calendar - your job is NOT to repeat that in prose. \
Write only what the visual can't show: the reasoning behind the choices.

Plan: {json.dumps(plan_details, ensure_ascii=False)}
Verification result: {json.dumps(verify_result, ensure_ascii=False)}
CheeseFork summary per course (avg_general is a 1-5 satisfaction rating, \
not an official grade): {json.dumps(cheesefork, ensure_ascii=False)}
Student's stated constraints: {json.dumps(state["constraints"], ensure_ascii=False)}
Student's failed/outstanding courses going in: {state["failed_courses"]}
Progress vs. a typical student at this point (approximate, since Technion \
doesn't publish an official sequence): {json.dumps(progress, ensure_ascii=False)}
{f"Courses left for a later semester and what they block (only mention if a real bottleneck - gateway_score 2+): {json.dumps(future_impact['bottleneck_warnings'], ensure_ascii=False)}" if future_impact and future_impact.get("bottleneck_warnings") else ""}
{f"An alternative plan was considered and not chosen: {alternative_note}" if alternative_note else ""}

Write 3-5 short sentences, not a report. Cover, only briefly: whether it \
verified and the credit/workload numbers in one clause, why this specific \
combination (which retake mattered and why, what got deferred and why - \
name it once, don't re-justify it course by course), and the progress \
comparison ONLY if "behind" or "slightly behind" and relevant to urgency \
(skip it entirely if on track - that's noise). If one elective has a \
notably poor satisfaction rating (roughly below 2.5/5) and a better \
same-credit alternative existed, say so in one clause - otherwise don't \
comment on elective quality at all, the cards already show it. If a real \
bottleneck warning or alternative-plan note is given above, mention it in \
one clause only if it's genuinely relevant to this student's decision - \
skip it entirely otherwise, don't force it in.

If verification did NOT pass (check the "pass" field above): be completely \
honest about it rather than glossing over it. Name the specific unresolved \
issue(s) in plain language (e.g. "the only section of X meets on a day you \
excluded, and no alternative section exists this semester") so the student \
has the whole picture, not just a vague "didn't verify." Say plainly that \
this is the best option found given every constraint, and that it's their \
call how to proceed (drop the constraint, accept the conflict, or take the \
course a different semester) - this is a genuine trade-off to disclose, not \
a permission-seeking question about an obvious next step.

Do not ask the student any other question. Do not offer to build an \
alternative as a pending option beyond disclosing the trade-off above - if \
a real trade-off exists, state it as already-considered context in this \
same message. Plain prose, no JSON, no headers, no bullet-pointed \
interrogation of the student.""",
    }
    return _call([system], json_mode=False, module="Explainer")


def run_agent_turn(
    track_id: str,
    messages: list[dict],
    cost_so_far: float,
    state_override: dict | None = None,
    known_context: dict | None = None,
) -> dict:
    """Runs one full turn: either one clarifying question, or a completely
    silent plan/verify/repair loop ending in one final explained plan.

    state_override skips the LLM extraction step entirely - used when the
    caller already has structured, validated intake data (a form
    submission) rather than free text to interpret.

    known_context carries forward what an EARLIER turn in this same
    conversation already resolved ({"state": {...}, "previous_plan": [...]})
    - without it, a free-text follow-up ("I have an exam conflict on
    23.2") has no way to know it's a change request against an existing
    plan rather than a brand new drafting task, and extraction has to
    re-derive passed/failed from scratch every turn instead of treating
    it as already-settled fact."""
    track = get_track(track_id)
    tool_log = []
    cost = cost_so_far
    known_state = (known_context or {}).get("state")
    previous_plan = (known_context or {}).get("previous_plan")

    if state_override is not None:
        state = state_override
        tool_log.append({"name": "structured_intake", "args": {}, "result": state})
    else:
        state, c = _extract_student_state(track, messages, known_state=known_state)
        cost += c
        tool_log.append({"name": "extract_student_state", "args": {}, "result": state})
        if previous_plan and known_state:
            # Same continuity guarantee as the react loop: once a plan has
            # been delivered, a revision turn can refine facts but never
            # un-know them, and never bounces the student back to intake.
            state = _normalize_state(merge_known_state(state, known_state))
            tool_log.append({"name": "merge_known_state", "args": {}, "result": state})

    if not state.get("ready_to_plan"):
        reply = state.get("clarifying_question") or "Just need a couple more things:"
        messages = messages + [{"role": "assistant", "content": reply}]
        return {
            "messages": messages,
            "cost_usd": cost,
            "tool_log": tool_log,
            "stopped_reason": "done",
            "plan_result": None,
            # Whatever was already figured out from free text, so the
            # frontend only needs to show button widgets for genuine gaps
            # (missing_fields) rather than re-asking everything - talking
            # first should mean less clicking after, not more.
            "needs_input": {
                "missing_fields": state.get("missing_fields", []),
                "semester_number": state.get("semester_number"),
                "constraints": state.get("constraints"),
                "passed_courses": state.get("passed_courses"),
                "failed_courses": state.get("failed_courses"),
            },
            # Carry forward whatever was already known even on a
            # not-ready-yet turn, so the next turn doesn't lose ground
            # either.
            "known_context": {
                "state": {
                    "semester_number": state.get("semester_number"),
                    "constraints": state.get("constraints"),
                    "passed_courses": state.get("passed_courses"),
                    "failed_courses": state.get("failed_courses"),
                },
                "previous_plan": previous_plan,
            },
        }

    semester_number = state.get("semester_number") or 0
    passed, failed = backfill_passed_courses(track, state)
    state["passed_courses"] = passed
    state["failed_courses"] = failed

    excluded_weekdays = state["constraints"].get("excluded_weekdays") or []
    verify_kwargs = resolve_verify_kwargs(state)
    pace = state["constraints"].get("pace")

    # First diagnostic step, same as a human advisor: how does this student's
    # actual progress compare to a typical student at this point in the
    # track? This shapes urgency (a "slightly behind" semester-4 student
    # planning around one recent gap reads differently than someone with a
    # long-accumulated backlog), even though it doesn't gate planning the
    # way missing pass/fail data does.
    progress = tools.assess_progress(track, semester_number, passed)
    tool_log.append(
        {"name": "assess_progress", "args": {"semester_number": semester_number}, "result": progress}
    )

    catalog = tools.fetch_catalog(track, passed)
    tool_log.append({"name": "fetch_catalog", "args": {"passed_courses": passed}, "result": catalog})

    prereq = tools.query_prereq_graph(track, passed, failed)
    tool_log.append(
        {"name": "query_prereq_graph", "args": {"passed_courses": passed, "failed_courses": failed}, "result": prereq}
    )

    plan, verify_result, future_impact = [], None, None
    stopped_reason = "repair_attempts_exhausted"
    for attempt in range(MAX_REPAIR_ATTEMPTS):
        if (cost - cost_so_far) >= SESSION_BUDGET_USD_CAP:
            stopped_reason = "budget_cap"
            break

        # accept_tradeoff is only offered once the model has made at least
        # two real repair attempts - it can voluntarily end its own loop
        # early, but only after genuinely trying, not as a first resort.
        # Python still owns both the floor and the ceiling here; the model
        # only ever gets to shorten the loop, never lengthen it past
        # MAX_REPAIR_ATTEMPTS.
        allow_accept_tradeoff = attempt >= 2
        plan, repair_strategy, repair_reasoning, c = _draft_or_repair_plan(
            track,
            state,
            catalog,
            prereq,
            progress,
            plan if attempt else None,
            verify_result,
            future_impact=future_impact,
            allow_accept_tradeoff=allow_accept_tradeoff,
            delivered_plan=previous_plan,
        )
        cost += c
        tool_log.append(
            {
                "name": f"draft_plan (attempt {attempt + 1})",
                "args": {"repair_strategy": repair_strategy, "repair_reasoning": repair_reasoning},
                "result": {"course_numbers": plan},
            }
        )

        if not plan:
            continue

        verify_result = tools.verify_plan(track, plan, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs)
        tool_log.append({"name": "verify_plan", "args": {"plan_course_numbers": plan}, "result": verify_result})

        future_impact = tools.simulate_future_impact(track, plan, passed, failed)
        tool_log.append(
            {"name": "simulate_future_impact", "args": {"plan_course_numbers": plan}, "result": future_impact}
        )

        if verify_result["pass"]:
            stopped_reason = "passed"
            break

        if repair_strategy == "accept_tradeoff" and allow_accept_tradeoff:
            stopped_reason = "model_accepted_tradeoff"
            break

    verify_result = verify_result or {"pass": False, "total_credits": 0, "workload_score": 0, "issues": []}

    # A genuine judgment-under-alternatives step, not just more repair: once
    # the primary plan has settled, draft one deliberately different-biased
    # alternative (lighter if the primary leaned normal/fast, more ambitious
    # if the primary was already light), score both on the same objective
    # metrics, and let a dedicated model call pick a winner with reasoning -
    # rather than Python silently keeping whichever came first.
    alternative_note = None
    # Skipped on revision turns (previous_plan set): when the student is
    # giving feedback on a plan they already have, continuity beats
    # exploring a differently-biased alternative - swapping their whole
    # plan out from under them is exactly the failure being fixed here.
    if plan and (cost - cost_so_far) < SESSION_BUDGET_USD_CAP and not previous_plan:
        bias = "lighter-workload" if pace != "light" else "faster-progress, even if heavier"
        alt_plan, _, _, c = _draft_or_repair_plan(
            track, state, catalog, prereq, progress, None, None, bias=bias
        )
        cost += c
        if alt_plan and alt_plan != plan:
            alt_verify = tools.verify_plan(track, alt_plan, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs)
            tool_log.append(
                {"name": "draft_alternative_plan", "args": {"bias": bias}, "result": {"course_numbers": alt_plan}}
            )
            tool_log.append(
                {"name": "verify_plan (alternative)", "args": {"plan_course_numbers": alt_plan}, "result": alt_verify}
            )

            comparison = tools.compare_plans(
                track,
                [
                    {"label": "primary", "course_numbers": plan, "verify": verify_result},
                    {"label": "alternative", "course_numbers": alt_plan, "verify": alt_verify},
                ],
            )
            tool_log.append({"name": "compare_plans", "args": {}, "result": comparison})

            if (cost - cost_so_far) < SESSION_BUDGET_USD_CAP:
                winner, reasoning, c = _choose_best_plan(track, state, comparison)
                cost += c
                tool_log.append(
                    {"name": "choose_best_plan", "args": {}, "result": {"winner": winner, "reasoning": reasoning}}
                )

                if winner == "alternative":
                    alternative_note = f"a more standard-progress option was considered and not chosen: {reasoning}"
                    plan, verify_result = alt_plan, alt_verify
                    future_impact = tools.simulate_future_impact(track, plan, passed, failed)
                else:
                    alternative_note = f"a {bias} alternative was considered and not chosen: {reasoning}"

    # Deterministic backstop, independent of whatever the drafting model
    # claims about its own plan: a course flagged both a retake and
    # "already passed" is a data-integrity bug, not a trade-off, so it's
    # corrected here in code rather than left for the model to notice on a
    # 6th attempt it doesn't get. This runs regardless of which attempt
    # produced the final plan (including the alternative, if it won).
    violations = tools.check_invariants(track, plan, passed, failed)
    if violations:
        tool_log.append({"name": "check_invariants", "args": {}, "result": {"violations": violations}})
        seen: set[str] = set()
        plan = [c for c in plan if c not in passed and c in track.courses and not (c in seen or seen.add(c))]
        verify_result = tools.verify_plan(track, plan, passed, excluded_weekdays=excluded_weekdays, **verify_kwargs)
        future_impact = tools.simulate_future_impact(track, plan, passed, failed)
        tool_log.append({"name": "verify_plan (post-backstop)", "args": {}, "result": verify_result})

    exam_dates = tools.fetch_exam_dates(track, plan) if plan else {}
    cheesefork = tools.summarize_cheesefork(track, plan) if plan else {}

    final_text, c = _explain_plan(
        track, state, plan, verify_result, exam_dates, cheesefork, progress, future_impact, alternative_note
    )
    cost += c

    messages = messages + [{"role": "assistant", "content": final_text}]

    plan_result = {
        "track_id": track.otjid,
        "track_name": track.name,
        "target_semester": track.target_semester,
        "constraints": state["constraints"],
        "verify": verify_result,
        "progress": progress,
        "explanation": final_text,
        "courses": [
            {
                "course_number": c,
                "name": track.courses[c]["name"],
                "points": track.courses[c]["points"],
                "is_retake": c in failed,
                "is_mandatory": c in track.mandatory_course_numbers,
                "schedule": track.courses[c]["schedule"],
                "exams": exam_dates.get(c, {}),
                "cheesefork": cheesefork.get(c, {}),
            }
            for c in plan
        ],
    }

    return {
        "messages": messages,
        "cost_usd": cost,
        "tool_log": tool_log,
        "stopped_reason": stopped_reason,
        "plan_result": plan_result,
        "needs_input": None,
        # The resolved facts and delivered plan, for the frontend to echo
        # back on the next request - so a follow-up like "I have a wedding
        # on an exam date" is treated as a revision of THIS plan against
        # THIS student state, instead of re-deriving everything from raw
        # text and drafting a stranger's plan from scratch.
        "known_context": {
            "state": {
                "semester_number": semester_number,
                "constraints": state["constraints"],
                "passed_courses": passed,
                "failed_courses": failed,
            },
            "previous_plan": plan,
        },
    }
