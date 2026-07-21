"""
Deterministic tool implementations the agent can call. Each function takes
a Track (see data_bundle.py - one degree track's static data) plus plain
JSON-able arguments, and returns a plain JSON-able dict - no LLM calls, no
network calls, just lookups/math over the bundled track data plus whatever
the student has stated about themselves this session.

Keeping this logic here (not left to the LLM to reason about) is a
deliberate cost control: exam-gap arithmetic, graph traversal, and category
auditing are exactly the kind of thing a model gets subtly wrong under
token pressure, and every token spent redoing it here in the tool result
would be a token not spent on judgment calls the model is actually good at.

Every function takes the track explicitly rather than reading a module-level
"current track" global - the latter would corrupt concurrent requests for
different students planning different tracks at the same time.
"""

from datetime import date

from data_bundle import Track, _bottom_categories, _leaf_courses
from prereq_parser import collect_prereq_courses

DEFAULT_MIN_CREDITS = 18  # 18 is the target load for a normal-pace semester; a bit above is fine
DEFAULT_MAX_CREDITS = 22
LIGHT_PACE_MIN_CREDITS = 14  # a student who explicitly asks for "light" gets a lower floor...
LIGHT_PACE_MAX_CREDITS = 18  # ...and a lower ceiling too, never above the normal target
DEFAULT_MIN_EXAM_GAP_DAYS = 3
DEFAULT_WORKLOAD_CAP = 75  # 0-100 scale; see _workload_score
DEFAULT_MIN_MANDATORY_COURSES = 3  # a semester carrying only 1-2 mandatory courses wastes a term

# CheeseFork's difficultyRank is 1-5; bucket boundaries for summarize_cheesefork.
_LIGHT_MAX = 2.5
_MEDIUM_MAX = 3.75


def search_courses(track: Track, query: str) -> list[dict]:
    """Looks up course numbers by (partial) name or by number, so the agent
    never has to guess or invent a course number - a student naming a course
    in English or Hebrew, or half-remembering its name, should always be
    resolved here before it's used in any other tool call.

    Matches on either the whole query as a substring, or - since a model's
    Hebrew translation of an English course name may be close but not
    exact - on any individual word of the query appearing in the course
    name, so an approximate translation still finds the right course."""
    query = query.strip().lower()
    if not query:
        return []

    query_words = [w for w in query.split() if len(w) >= 3]

    def is_match(name: str, course_number: str) -> bool:
        name_lower = name.lower()
        if query in name_lower or query in course_number:
            return True
        return any(word in name_lower for word in query_words)

    matches = [
        {"course_number": c, "name": course["name"], "points": course["points"]}
        for c, course in track.courses.items()
        if is_match(course["name"], c)
    ]
    return matches[:20]


def _course_or_error(track: Track, course_number: str) -> dict:
    course = track.courses.get(course_number)
    if course is None:
        raise ValueError(f"Unknown course number: {course_number}")
    return course


def _prereq_satisfied(tree: dict | None, passed: set[str]) -> bool:
    if tree is None:
        return True
    if "course" in tree:
        return tree["course"] in passed
    if tree["op"] == "and":
        return all(_prereq_satisfied(arg, passed) for arg in tree["args"])
    return any(_prereq_satisfied(arg, passed) for arg in tree["args"])


def assess_progress(track: Track, semester_number: int, passed_courses: list[str]) -> dict:
    """The first diagnostic step, before anything else: how does this
    student's actual completion compare to a normal student at this point
    in the track? Since courses at prerequisite-depth D are typically not
    reachable before roughly semester D+2 (one semester per prerequisite
    layer, plus the first semester itself), a student in semester N should
    typically have finished depth <= N-2. Approximate on purpose - see
    Track._compute_mandatory_prereq_depths.
    """
    passed = set(passed_courses)
    depths = track.mandatory_prereq_depths
    expected_completed = {c for c, d in depths.items() if d <= semester_number - 2}
    missing_vs_expected = sorted(
        {c: track.courses[c]["name"] for c in (expected_completed - passed) if c in track.courses}.items()
    )
    ahead_of_schedule = sorted(
        c for c in passed if c in depths and depths[c] > semester_number - 2
    )

    if len(missing_vs_expected) >= 3:
        pace = "behind"
    elif missing_vs_expected:
        pace = "slightly behind"
    else:
        pace = "on track"

    return {
        "pace": pace,
        "missing_vs_expected": [{"course_number": c, "name": n} for c, n in missing_vs_expected],
        "ahead_of_schedule_count": len(ahead_of_schedule),
        "note": (
            "This compares against an inferred typical sequence (Technion "
            "doesn't publish an official one), so treat it as a useful "
            "signal, not a precise verdict."
        ),
    }


def get_intake_options(track: Track) -> dict:
    """Static, deterministic data for a structured intake form, instead of
    asking the student to type free text for things that are really just a
    fixed choice. The mandatory course list in particular is the same set
    every time - there's no reason to make an LLM re-derive names and
    numbers from a student's typed description of them, the single biggest
    source of unreliable behavior seen in this build. depth is included per
    course so a frontend can pre-check "probably passed" vs "probably not
    reached yet" once the student picks a semester number, the same
    heuristic assess_progress and the drafting step already use."""
    mandatory = [
        {"course_number": c, "name": track.courses[c]["name"], "depth": track.mandatory_prereq_depths.get(c, 0)}
        for c in sorted(track.mandatory_course_numbers)
        if c in track.courses
    ]
    mandatory.sort(key=lambda c: (c["depth"], c["course_number"]))
    return {
        "track_id": track.otjid,
        "track_name": track.name,
        "mandatory_courses": mandatory,
        "weekdays": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "pace_options": ["light", "normal", "fast"],
    }


def fetch_catalog(track: Track, passed_courses: list[str]) -> dict:
    """Degree requirement audit: walks the track's requirement tree and
    reports, per bottom-level category, how much is done vs. left."""
    passed = set(passed_courses)
    categories = []
    total_remaining_points = 0.0
    total_remaining_requirements = 0

    for category in _bottom_categories(track.requirement_tree):
        leaves = _leaf_courses(category)
        done = [c for c in leaves if c in passed]
        remaining = [c for c in leaves if c not in passed]

        remaining_points = sum(
            float(track.courses[c]["points"])
            for c in remaining
            if c in track.courses and track.courses[c]["points"]
        )
        total_remaining_points += remaining_points
        total_remaining_requirements += len(remaining)

        if not leaves:
            status = "n/a"
        elif not remaining:
            status = "complete"
        elif not done:
            status = "not started"
        else:
            status = "in progress"

        categories.append(
            {
                "category": category["name"],
                "status": status,
                "total_courses": len(leaves),
                "completed": len(done),
                "remaining_courses": [
                    {"course_number": c, "name": track.courses[c]["name"]} for c in remaining if c in track.courses
                ],
                "remaining_points": round(remaining_points, 1),
            }
        )

    return {
        "target_semester": track.target_semester,
        "categories": categories,
        "total_remaining_requirements": total_remaining_requirements,
        "total_remaining_points": round(total_remaining_points, 1),
    }


def query_prereq_graph(track: Track, passed_courses: list[str], failed_courses: list[str]) -> dict:
    """For each failed/incomplete required course, reports what it directly
    and transitively blocks (via the precomputed reverse-prerequisite graph),
    plus a gateway score = size of the transitive block set. Also reports
    which candidate courses (those with a failed/unattempted prereq) are
    currently unlocked given what the student has actually passed."""
    passed = set(passed_courses)

    blockages = []
    for course_number in failed_courses:
        if course_number not in track.courses:
            continue

        visited: set[str] = set()
        frontier = [course_number]
        while frontier:
            current = frontier.pop()
            for dependent in track.reverse_prereqs.get(current, []):
                if dependent not in visited:
                    visited.add(dependent)
                    frontier.append(dependent)

        direct = track.reverse_prereqs.get(course_number, [])
        blockages.append(
            {
                "course": course_number,
                "name": track.courses[course_number]["name"],
                "blocks_directly": direct,
                "blocks_transitively": sorted(visited),
                "gateway_score": len(visited),
            }
        )

    blockages.sort(key=lambda b: -b["gateway_score"])

    unlocked, blocked = [], []
    all_blocked_numbers = {c for b in blockages for c in b["blocks_transitively"]}
    for course_number in all_blocked_numbers:
        course = track.courses.get(course_number)
        if course is None:
            continue
        if _prereq_satisfied(course["prerequisites"], passed):
            unlocked.append(course_number)
        else:
            blocked.append(course_number)

    return {"blockages": blockages, "unlocked": sorted(unlocked), "blocked": sorted(blocked)}


def fetch_exam_dates(track: Track, course_numbers: list[str]) -> dict:
    """Exam dates for a set of candidate courses, for exam-spacing checks."""
    return {c: _course_or_error(track, c)["exams"] for c in course_numbers}


def _difficulty_bucket(avg_difficulty: float | None) -> str:
    if avg_difficulty is None:
        return "unknown"
    if avg_difficulty <= _LIGHT_MAX:
        return "light"
    if avg_difficulty <= _MEDIUM_MAX:
        return "medium"
    return "heavy"


def summarize_cheesefork(track: Track, course_numbers: list[str]) -> dict:
    """CheeseFork crowd-review difficulty summary per course."""
    result = {}
    for c in course_numbers:
        cf = _course_or_error(track, c)["cheesefork"]
        result[c] = {
            "avg_difficulty": cf["avg_difficulty"],
            "avg_general": cf["avg_general"],
            "review_count": cf["review_count"],
            "load": _difficulty_bucket(cf["avg_difficulty"]),
            "sample_reviews": cf.get("sample_reviews", []),
        }
    return result


def _workload_score(track: Track, course_numbers: list[str]) -> float:
    """0-100 heuristic: normalized credit load plus normalized difficulty.
    Not a validated formula - a transparent, tunable stand-in so the critic
    has something concrete to check against."""
    total_points = sum(
        float(track.courses[c]["points"]) for c in course_numbers if track.courses[c]["points"]
    )
    difficulties = [
        track.courses[c]["cheesefork"]["avg_difficulty"]
        for c in course_numbers
        if track.courses[c]["cheesefork"]["avg_difficulty"] is not None
    ]
    avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 3.0

    points_component = min(total_points / 20.0, 1.0) * 60  # 20 pts -> full 60
    difficulty_component = min(avg_difficulty / 5.0, 1.0) * 40  # 5.0 -> full 40
    return round(points_component + difficulty_component, 1)


def simulate_future_impact(
    track: Track, plan_course_numbers: list[str], passed_courses: list[str], failed_courses: list[str]
) -> dict:
    """Look-ahead check: among the required courses NOT in this plan (and not
    already passed), which ones have a high gateway_score - i.e. block many
    other still-outstanding courses transitively? Leaving a low-gateway
    course for next semester costs little; leaving a high-gateway one delays
    everything behind it. This is the planning-horizon judgment a human
    advisor makes instinctively but a single-semester optimizer would miss -
    it's advisory, not a hard constraint, so the drafting step can weigh it
    against everything else rather than being forced to act on it."""
    passed = set(passed_courses)
    plan = set(plan_course_numbers)
    remaining_mandatory = [c for c in sorted(track.mandatory_course_numbers) if c not in passed and c not in plan]

    deferred = []
    for c in remaining_mandatory:
        visited: set[str] = set()
        frontier = [c]
        while frontier:
            current = frontier.pop()
            for dependent in track.reverse_prereqs.get(current, []):
                if dependent not in visited:
                    visited.add(dependent)
                    frontier.append(dependent)
        deferred.append(
            {
                "course_number": c,
                "name": track.courses[c]["name"] if c in track.courses else c,
                "gateway_score": len(visited),
                "is_current_failure": c in failed_courses,
            }
        )

    deferred.sort(key=lambda d: -d["gateway_score"])
    bottleneck_warnings = [d for d in deferred if d["gateway_score"] >= 2]

    return {
        "deferred_mandatory_count": len(remaining_mandatory),
        "deferred_courses": deferred[:10],
        "bottleneck_warnings": bottleneck_warnings[:5],
    }


def compare_plans(track: Track, candidates: list[dict]) -> list[dict]:
    """Deterministic side-by-side metrics for two or more already-verified
    candidate plans - computes the numbers, but does NOT pick a winner.
    That judgment call (which trade-off is actually better for this
    specific student) belongs to a model call that can weigh things a
    fixed formula can't, like whether the student sounded schedule-stressed
    in conversation - this tool just makes sure that judgment is made on
    accurate, consistent numbers rather than the model re-deriving them."""
    rows = []
    for candidate in candidates:
        course_numbers = candidate["course_numbers"]
        verify = candidate["verify"]
        mandatory_in_plan = [c for c in course_numbers if c in track.mandatory_course_numbers]
        rows.append(
            {
                "label": candidate["label"],
                "course_numbers": course_numbers,
                "pass": verify["pass"],
                "total_credits": verify["total_credits"],
                "workload_score": verify["workload_score"],
                "issue_count": len(verify["issues"]),
                "mandatory_count": len(mandatory_in_plan),
            }
        )
    return rows


def roadmap_to_graduation(
    track: Track,
    passed_courses: list[str],
    failed_courses: list[str],
    plan_course_numbers: list[str] | None = None,
    max_mandatory_per_semester: int = 4,
) -> dict:
    """Projects the remaining MANDATORY requirements across future semesters,
    so this semester's choices can be explained by what they unlock later -
    the judgment a real advisor makes ("take X now or semester 7 collapses")
    that a single-semester optimizer can't see. Deterministic layering: a
    course lands in the earliest future semester after all its mandatory
    prerequisites are scheduled, with a realistic per-semester cap.

    plan_course_numbers, when given, counts as "completed after this
    semester" - so the roadmap shows the future assuming the current plan
    goes well."""
    done = set(passed_courses) | set(plan_course_numbers or [])
    remaining = [c for c in sorted(track.mandatory_course_numbers) if c not in done and c in track.courses]

    # Which mandatory courses does each remaining course require?
    mandatory_set = track.mandatory_course_numbers
    prereqs_of = {
        c: [p for p in collect_prereq_courses(track.courses[c]["prerequisites"]) if p in mandatory_set]
        for c in remaining
    }

    semesters: list[list[str]] = []
    scheduled: set[str] = set(done)
    pending = list(remaining)
    # Bounded loop: every pass schedules at least one course or force-places
    # the stragglers, so this always terminates.
    while pending and len(semesters) < 10:
        ready = [c for c in pending if all(p in scheduled for p in prereqs_of[c])][:max_mandatory_per_semester]
        if not ready:
            ready = pending[:max_mandatory_per_semester]
        semesters.append(ready)
        scheduled.update(ready)
        pending = [c for c in pending if c not in scheduled]

    return {
        "semesters_left_estimate": len(semesters),
        "remaining_mandatory_count": len(remaining),
        "future_semesters": [
            {
                "semester_offset": i + 1,
                "courses": [
                    {"course_number": c, "name": track.courses[c]["name"]} for c in batch
                ],
            }
            for i, batch in enumerate(semesters)
        ],
    }


def risk_report(
    track: Track, plan_course_numbers: list[str], passed_courses: list[str], failed_courses: list[str]
) -> dict:
    """Stress-tests a candidate plan: what could realistically go wrong, and
    how bad would it be? Deterministic - graph math and date arithmetic over
    data already in the bundle - so it costs nothing per turn. The agent is
    told to surface the TOP risk with a mitigation, turning "here's a plan"
    into "here's a plan and here's its weak spot," which is what a candid
    human advisor does."""
    passed = set(passed_courses)
    risks = []

    # Fragile courses: failing this plan course again would delay everything
    # transitively behind it.
    for c in plan_course_numbers:
        if c not in track.courses:
            continue
        visited: set[str] = set()
        frontier = [c]
        while frontier:
            current = frontier.pop()
            for dependent in track.reverse_prereqs.get(current, []):
                if dependent not in visited and dependent not in passed:
                    visited.add(dependent)
                    frontier.append(dependent)
        if len(visited) >= 2:
            risks.append(
                {
                    "type": "fragile_course",
                    "course_number": c,
                    "name": track.courses[c]["name"],
                    "detail": f"failing it would delay {len(visited)} downstream course(s)",
                    "severity": min(len(visited), 10),
                }
            )

    # Exam crunch windows: moed A dates within 4 days of each other.
    exam_dates = []
    for c in plan_course_numbers:
        if c not in track.courses:
            continue
        for exam in track.courses[c]["exams"].get("moed_a", []):
            exam_dates.append((exam["date"], c))
    exam_dates.sort()
    for (d1, c1), (d2, c2) in zip(exam_dates, exam_dates[1:]):
        gap = (date.fromisoformat(d2) - date.fromisoformat(d1)).days
        if gap <= 4:
            risks.append(
                {
                    "type": "exam_crunch",
                    "course_number": c2,
                    "name": track.courses[c2]["name"],
                    "detail": f"exam only {gap} day(s) after {track.courses[c1]['name']}",
                    "severity": 5 - gap,
                }
            )

    # Workload spike: multiple heavy courses stacked together.
    heavy = [
        c for c in plan_course_numbers
        if c in track.courses
        and track.courses[c]["cheesefork"]["avg_difficulty"] is not None
        and track.courses[c]["cheesefork"]["avg_difficulty"] > _MEDIUM_MAX
    ]
    if len(heavy) >= 2:
        risks.append(
            {
                "type": "workload_spike",
                "course_number": None,
                "name": None,
                "detail": f"{len(heavy)} heavy courses in the same semester: "
                + ", ".join(track.courses[c]["name"] for c in heavy),
                "severity": len(heavy) + 1,
            }
        )

    risks.sort(key=lambda r: -r["severity"])
    return {"risks": risks[:5], "top_risk": risks[0] if risks else None}


def _slots_by_group(course: dict) -> dict:
    """Groups a course's schedule slots by registration group (section).
    Slots without a group number all share one pseudo-group."""
    groups: dict = {}
    for slot in course.get("schedule", []):
        groups.setdefault(slot.get("group") or 0, []).append(slot)
    return groups


def pick_section(course: dict, excluded_weekdays: list[int] | None = None) -> list[dict]:
    """The slots of the best registration group for this course: the first
    group with no slot on an excluded weekday, else the group with the
    fewest conflicts. Used both by the weekday check (a course only truly
    conflicts if EVERY group conflicts - the data has multiple sections per
    course, so 'some section meets Sunday' is not 'you must attend Sunday')
    and by the frontend timetable (which should show one coherent section,
    not every group's slots overlaid)."""
    excluded = set(excluded_weekdays or [])
    groups = _slots_by_group(course)
    if not groups:
        return []
    best = None
    best_conflicts = None
    for group_id in sorted(groups):
        slots = groups[group_id]
        conflicts = sum(1 for s in slots if s.get("weekday") in excluded)
        if conflicts == 0:
            return slots
        if best_conflicts is None or conflicts < best_conflicts:
            best, best_conflicts = slots, conflicts
    return best or []


def check_invariants(
    track: Track, plan_course_numbers: list[str], passed_courses: list[str], failed_courses: list[str]
) -> list[str]:
    """A deterministic backstop, separate from verify_plan's plan-quality
    checks (schedule, credits, workload): these are data-integrity
    invariants that must hold no matter what produced the plan or the
    student's state, and are checked independently of whether the caller
    correctly acted on verify_plan's issues. A course flagged both a
    "RETAKE" and "already passed" in the same plan - the exact bug this was
    written to catch, found via live testing - is a state-integrity bug,
    not a scheduling trade-off, so it's checked here rather than left to
    the drafting model to notice.

    Returns a list of plain-language violation strings; empty means clean.
    This function raises nothing and calls no other tool - callers decide
    what to do with a non-empty result (e.g. force a repair, or refuse to
    deliver the plan at all)."""
    violations = []

    passed = set(passed_courses)
    failed = set(failed_courses)

    overlap = passed & failed
    if overlap:
        violations.append(f"course(s) {sorted(overlap)} are marked both passed and failed - contradictory state")

    already_passed_in_plan = [c for c in plan_course_numbers if c in passed]
    if already_passed_in_plan:
        violations.append(f"plan re-includes already-passed course(s) {sorted(already_passed_in_plan)}")

    unknown = [c for c in plan_course_numbers if c not in track.courses]
    if unknown:
        violations.append(f"plan includes unknown course number(s) {sorted(unknown)}")

    seen = set()
    duplicates = {c for c in plan_course_numbers if c in seen or seen.add(c)}
    if duplicates:
        violations.append(f"plan lists course(s) {sorted(duplicates)} more than once")

    return violations


def verify_plan(
    track: Track,
    plan_course_numbers: list[str],
    passed_courses: list[str],
    excluded_weekdays: list[int] | None = None,
    min_exam_gap_days: int = DEFAULT_MIN_EXAM_GAP_DAYS,
    max_workload_score: float = DEFAULT_WORKLOAD_CAP,
    min_credits: float = DEFAULT_MIN_CREDITS,
    max_credits: float = DEFAULT_MAX_CREDITS,
    min_mandatory_courses: int = DEFAULT_MIN_MANDATORY_COURSES,
) -> dict:
    """The critic: checks a candidate plan against prerequisites, semester
    offering, credit range, day-of-week constraints, exam spacing, workload
    cap, and real degree progress (enough mandatory courses to make the
    semester worthwhile). Returns pass/fail plus itemized issues so the
    agent can repair a rejected plan rather than just being told "no".

    The mandatory-course and credit floors are enforced here, not just
    suggested in the drafting prompt - a rule that must actually hold
    belongs in deterministic code, the same lesson that applies to every
    other check in this function."""
    excluded_weekdays = set(excluded_weekdays or [])
    passed = set(passed_courses)
    issues = []

    unknown = [c for c in plan_course_numbers if c not in track.courses]
    if unknown:
        return {"pass": False, "issues": [{"course": c, "reason": "unknown course number"} for c in unknown]}

    for c in plan_course_numbers:
        course = track.courses[c]
        if c in passed:
            issues.append({"course": c, "reason": "already passed - can't be retaken or counted again"})
        if not course["offered_next_semester"]:
            issues.append({"course": c, "reason": f"not offered in {track.target_semester}"})
        if not _prereq_satisfied(course["prerequisites"], passed):
            issues.append({"course": c, "reason": "prerequisites not met"})
        # Section-aware weekday check: the schedule data has multiple
        # registration groups per course, so a course only genuinely
        # conflicts if its BEST group still has a slot on an excluded day -
        # "some section meets Sunday" is not "you must attend Sunday".
        best_slots = pick_section(course, list(excluded_weekdays))
        for slot in best_slots:
            if slot["weekday"] in excluded_weekdays:
                issues.append(
                    {
                        "course": c,
                        "reason": f"{slot['type']} lands on an excluded weekday ({slot['weekday']}) in every section",
                    }
                )
                break

    total_credits = sum(float(track.courses[c]["points"]) for c in plan_course_numbers if track.courses[c]["points"])
    if not (min_credits <= total_credits <= max_credits):
        issues.append({"course": None, "reason": f"total credits {total_credits} outside [{min_credits}, {max_credits}]"})

    remaining_mandatory = track.mandatory_course_numbers - passed
    effective_min_mandatory = min(min_mandatory_courses, len(remaining_mandatory))
    mandatory_in_plan = [c for c in plan_course_numbers if c in track.mandatory_course_numbers and c not in passed]
    if len(mandatory_in_plan) < effective_min_mandatory:
        issues.append(
            {
                "course": None,
                "reason": (
                    f"only {len(mandatory_in_plan)} mandatory course(s) in the plan, "
                    f"expected at least {effective_min_mandatory} - a semester should make "
                    "real degree progress unless the student asked for fewer"
                ),
            }
        )

    exam_dates = []
    for c in plan_course_numbers:
        for exam in track.courses[c]["exams"].get("moed_a", []):
            exam_dates.append((exam["date"], c))
    exam_dates.sort()
    for (d1, c1), (d2, c2) in zip(exam_dates, exam_dates[1:]):
        gap = (date.fromisoformat(d2) - date.fromisoformat(d1)).days
        if gap < min_exam_gap_days:
            issues.append(
                {"course": c2, "reason": f"exam only {gap} day(s) after {c1}'s exam (minimum {min_exam_gap_days})"}
            )

    workload = _workload_score(track, plan_course_numbers)
    if workload > max_workload_score:
        issues.append({"course": None, "reason": f"workload score {workload} exceeds cap {max_workload_score}"})

    return {
        "pass": len(issues) == 0,
        "total_credits": total_credits,
        "workload_score": workload,
        "issues": issues,
    }
