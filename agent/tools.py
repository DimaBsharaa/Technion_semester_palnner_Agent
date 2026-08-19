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

DEFAULT_MIN_CREDITS = 16  # 18 is the target load for a normal-pace semester; a bit above is fine
DEFAULT_MAX_CREDITS = 24
LIGHT_PACE_MIN_CREDITS = 14  # a student who explicitly asks for "light" gets a lower floor...
LIGHT_PACE_MAX_CREDITS = 18  # ...and a lower ceiling too, never above the normal target
DEFAULT_MIN_EXAM_GAP_DAYS = 3
DEFAULT_WORKLOAD_CAP = 80  # 0-100 scale; see _workload_score
DEFAULT_MIN_MANDATORY_COURSES = 3  # a semester carrying only 1-2 mandatory courses wastes a term

# How many semesters of buffer beyond a course's prereq depth before it's
# assumed already completed (assess_progress / backfill_passed_courses).
# Was 2 - live testing found real semester-3+ mandatory courses sitting at
# depth 1 (shallow prereq chain, but NOT actually an early-semester course
# in Technion's real curriculum), silently excluded from every plan as a
# result. See assess_progress's docstring for the full evidence.
EXPECTED_BY_NOW_BUFFER = 3

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
    in the track?

    "depth" is a PREREQUISITE-GRAPH property (how many prerequisite hops
    deep a course sits), not a real semester-sequence number - it's a
    proxy, and a confirmed-wrong one at the old N-2 cutoff: live testing
    found real semester-3-and-later mandatory courses (Software
    Engineering, DB Management, Deterministic Models, the Economics
    requirement) sitting at depth 1, meaning they'd be silently assumed
    ALREADY DONE by semester 3 under the old formula - a shallow
    prerequisite chain does not mean Technion's real curriculum schedules
    a course that early; pacing/credit-balancing spreads mandatory courses
    across many semesters independent of how few prerequisites they need.
    EXPECTED_BY_NOW_BUFFER widens that gap - still an approximation (there
    is no real per-semester program-sequence data in this codebase, see
    docs/enhancement-checklist.md item 4's DDS-diagram note), but a
    meaningfully less aggressive one, calibrated against that live evidence.
    """
    passed = set(passed_courses)
    depths = track.mandatory_prereq_depths
    expected_completed = {c for c, d in depths.items() if d <= semester_number - EXPECTED_BY_NOW_BUFFER}
    missing_vs_expected = sorted(
        {c: track.courses[c]["name"] for c in (expected_completed - passed) if c in track.courses}.items()
    )
    # Choose-one-variant requirements (calculus, algebra, probability...):
    # due when the group's earliest variant sits at an expected-done depth
    # and NO variant was passed.
    for group in track.mandatory_choice_groups:
        if group["depth"] <= semester_number - EXPECTED_BY_NOW_BUFFER and not (set(group["options"]) & passed):
            missing_vs_expected.append((group["options"][0], f"{group['label']} (one of these is required)"))
    ahead_of_schedule = sorted(
        c for c in passed if c in depths and depths[c] > semester_number - EXPECTED_BY_NOW_BUFFER
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
    # Choose-one-variant requirements appear as ONE row each (the student
    # answers for the group; the first option stands in as its id).
    for group in track.mandatory_choice_groups:
        mandatory.append(
            {"course_number": group["options"][0], "name": f"{group['label']} (אחת מהאפשרויות)", "depth": group["depth"]}
        )
    mandatory.sort(key=lambda c: (c["depth"], c["course_number"]))
    return {
        "track_id": track.otjid,
        "track_name": track.name,
        "mandatory_courses": mandatory,
        "weekdays": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "pace_options": ["light", "normal", "fast"],
        # The frontend's own "expected done by now" checklist filter must
        # use this SAME number, not a hardcoded copy of its own - found
        # live: EXPECTED_BY_NOW_BUFFER was recalibrated from 2 to 3 here,
        # but a second, independent "- 2" hardcoded in site/index.html was
        # never updated, so the checklist silently drifted out of sync with
        # what assess_progress/backfill_passed_courses actually use.
        "expected_by_now_buffer": EXPECTED_BY_NOW_BUFFER,
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


def analyze_grade_improvement_candidates(track: Track, passed_courses: list[str], grades: dict) -> list[dict]:
    """Raw comparison data for the model to REASON about - not a verdict.
    For each passed course with a known grade, reports it against TWO
    baselines: the course's own historical average (technion-histograms,
    see pipeline/histogram_client.py) and the student's OWN average across
    every grade known about them. A 70 means something very different for
    a student who otherwise averages 90 than for one who averages 72 - a
    single fixed threshold against the course average alone (the previous
    version of this function) couldn't see that difference; this hands
    both numbers over and leaves the judgment call - whether either gap is
    actually meaningful, combined with whatever review sentiment the model
    separately checks via summarize_cheesefork - to the model itself.

    The only filtering done here is noise reduction, not a suggestion
    decision: a course where the grade is at or above BOTH baselines is
    dropped, since there is no plausible improvement story for it. Nothing
    stronger than that - a course below only ONE baseline still surfaces,
    with both numbers shown, not hidden.

    Advisory data only. Technion's real grade-improvement policy (שיפור
    ציון) has its own eligibility rules - which courses qualify, how many
    attempts allowed, which grade counts, time limits - that are NOT
    implemented or verified here; the caller (the system prompt) is
    responsible for framing anything built on this as a dismissible
    suggestion the student must explicitly accept, never a rule the agent
    applied on its own.

    Silently skips (not an error) any course with no known grade or no
    historical grade_stats - most electives won't have both, and that's
    the expected common case, not a failure. Returns [] entirely when
    `grades` is empty - this function is never a reason to ask a student
    for a grade they haven't volunteered."""
    if not grades:
        return []
    known_grades = list(grades.values())
    your_own_average = round(sum(known_grades) / len(known_grades), 1)

    out = []
    for c in passed_courses:
        grade = grades.get(c)
        if grade is None or c not in track.courses:
            continue
        stats = track.courses[c].get("grade_stats")
        if not stats:
            continue
        course_average = stats["avg_grade"]
        if grade >= course_average and grade >= your_own_average:
            continue  # no plausible improvement story either way - noise, not a decision
        out.append(
            {
                "course_number": c,
                "name": track.courses[c]["name"],
                "your_grade": grade,
                "course_average": course_average,
                "your_own_average": your_own_average,
                "gap_vs_course_average": round(course_average - grade, 1),
                "gap_vs_your_average": round(your_own_average - grade, 1),
            }
        )
    return out


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

    # Moed B crunches (advisory): moed B dates are fixed by the Technion,
    # so tight gaps are often unavoidable - but a student who lands in two
    # moed B exams a day apart deserves to know before it happens.
    b_dates = []
    for c in plan_course_numbers:
        if c not in track.courses:
            continue
        for exam in track.courses[c]["exams"].get("moed_b", []):
            b_dates.append((exam["date"], c))
    b_dates.sort()
    for (d1, c1), (d2, c2) in zip(b_dates, b_dates[1:]):
        gap = (date.fromisoformat(d2) - date.fromisoformat(d1)).days
        if gap <= 1:
            risks.append(
                {
                    "type": "moed_b_crunch",
                    "course_number": c2,
                    "name": track.courses[c2]["name"],
                    "detail": f"moed B exam only {gap} day(s) after {track.courses[c1]['name']}'s moed B",
                    "severity": 3 - gap,
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


def _to_minutes(hhmm: str) -> int:
    parts = str(hhmm or "0:0").split(":")
    try:
        return int(parts[0]) * 60 + int(parts[1] if len(parts) > 1 else 0)
    except ValueError:
        return 0


def _slots_overlap(a: dict, b: dict) -> bool:
    if a.get("weekday") != b.get("weekday"):
        return False
    return _to_minutes(a.get("start")) < _to_minutes(b.get("end")) and _to_minutes(b.get("start")) < _to_minutes(a.get("end"))


def choose_sections(
    track: Track, course_numbers: list[str], excluded_weekdays: list[int] | None = None
) -> dict:
    """Coordinated section assignment for a WHOLE plan: picks one
    registration group per course so that, as far as the data allows, no
    two courses collide in time and no slot lands on an excluded day.

    This exists because sections were previously chosen per-course in
    isolation, which produced weekly timetables with lectures stacked on
    top of each other - a defect the verifier never flagged because time
    collisions weren't checked anywhere. Greedy with least-flexible-first
    ordering: courses with the fewest sections choose first, so a
    single-section course never gets boxed out by a flexible one.

    Returns {"assignments": {course: [slots]}, "overlaps": [...],
    "excluded_day_hits": [...]} - overlaps/hits list only IRREDUCIBLE
    problems (no combination of sections avoids them)."""
    excluded = set(excluded_weekdays or [])
    valid = [c for c in course_numbers if c in track.courses]
    # Least flexible first: fewest section options, then most weekly slots.
    ordered = sorted(
        valid,
        key=lambda c: (len(_slots_by_group(track.courses[c])) or 1, -len(track.courses[c].get("schedule", []))),
    )

    assignments: dict[str, list[dict]] = {}
    overlaps: list[dict] = []
    excluded_day_hits: list[dict] = []

    for c in ordered:
        groups = _slots_by_group(track.courses[c])
        if not groups:
            assignments[c] = []
            continue
        best_slots, best_cost = None, None
        for group_id in sorted(groups):
            slots = groups[group_id]
            excl_hits = sum(1 for s in slots if s.get("weekday") in excluded)
            clashes = sum(
                1
                for s in slots
                for other_slots in assignments.values()
                for o in other_slots
                if _slots_overlap(s, o)
            )
            # Excluded-day violations dominate; then collisions.
            cost = (excl_hits, clashes)
            if best_cost is None or cost < best_cost:
                best_slots, best_cost = slots, cost
        assignments[c] = best_slots
        if best_cost[0]:
            hit_days = sorted({s["weekday"] for s in best_slots if s.get("weekday") in excluded})
            excluded_day_hits.append({"course": c, "weekdays": hit_days})
        if best_cost[1]:
            for s in best_slots:
                for other_c, other_slots in assignments.items():
                    if other_c == c:
                        continue
                    for o in other_slots:
                        if _slots_overlap(s, o):
                            overlaps.append(
                                {
                                    "course_a": c,
                                    "course_b": other_c,
                                    "weekday": s.get("weekday"),
                                    "time": f"{s.get('start')}-{s.get('end')}",
                                }
                            )
    # Restore caller order in assignments output
    return {
        "assignments": {c: assignments.get(c, []) for c in valid},
        "overlaps": overlaps,
        "excluded_day_hits": excluded_day_hits,
    }


def weekly_schedule_analyzer(
    track: Track, plan_course_numbers: list[str], excluded_weekdays: list[int] | None = None
) -> dict:
    """The agent's 'look at the actual week' tool: coordinates sections for
    the candidate plan and reports what the week really looks like -
    unavoidable time collisions, excluded-day violations, how many days the
    student must be on campus, which days stay free, and the heaviest day.
    Deterministic; the model calls it to sanity-check a candidate's weekly
    life before delivering, and the UI renders the same assignment."""
    result = choose_sections(track, plan_course_numbers, excluded_weekdays)
    day_minutes: dict[int, int] = {}
    for slots in result["assignments"].values():
        for s in slots:
            wd = s.get("weekday")
            if wd is None:
                continue
            day_minutes[wd] = day_minutes.get(wd, 0) + max(_to_minutes(s.get("end")) - _to_minutes(s.get("start")), 0)
    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    def name_of(c):
        return track.courses[c]["name"] if c in track.courses else c

    return {
        "overlap_free": not result["overlaps"],
        "unavoidable_overlaps": [
            {
                "courses": f"{name_of(o['course_a'])} + {name_of(o['course_b'])}",
                "weekday": day_names[o["weekday"]] if o.get("weekday") is not None and 0 <= o["weekday"] < 6 else o.get("weekday"),
                "time": o["time"],
            }
            for o in result["overlaps"]
        ],
        "excluded_day_violations": [
            {"course": name_of(h["course"]), "weekdays": [day_names[d] for d in h["weekdays"] if 0 <= d < 6]}
            for h in result["excluded_day_hits"]
        ],
        "days_on_campus": len(day_minutes),
        "free_days": [day_names[d] for d in range(6) if d not in day_minutes],
        "busiest_day": (
            {"day": day_names[max(day_minutes, key=day_minutes.get)], "hours": round(max(day_minutes.values()) / 60, 1)}
            if day_minutes
            else None
        ),
    }


def exam_study_planner(track: Track, plan_course_numbers: list[str]) -> dict:
    """The agent's 'can this exam period actually be studied for' tool: the
    full chronological exam timeline - moed A AND moed B - with the study
    days available before each exam, crunch flags, and which single course
    most relieves the worst crunch if swapped. Moed B matters: students who
    miss or fail a moed A sit the moed B, and two moed B exams zero days
    apart is a real (previously invisible) planning defect."""
    timeline = []
    for c in plan_course_numbers:
        if c not in track.courses:
            continue
        for moed, label in (("moed_a", "A"), ("moed_b", "B")):
            for exam in track.courses[c]["exams"].get(moed, []):
                timeline.append({"date": exam["date"], "course_number": c, "name": track.courses[c]["name"], "moed": label})
    timeline.sort(key=lambda e: e["date"])

    crunches = []
    for moed in ("A", "B"):
        seq = [e for e in timeline if e["moed"] == moed]
        for prev, cur in zip(seq, seq[1:]):
            gap = (date.fromisoformat(cur["date"]) - date.fromisoformat(prev["date"])).days
            cur["study_days_before"] = gap
            if gap < 3:
                crunches.append(
                    {
                        "moed": moed,
                        "course_number": cur["course_number"],
                        "name": cur["name"],
                        "after": prev["name"],
                        "study_days": gap,
                    }
                )

    # Which course's removal most improves the worst moed-A crunch?
    relief = None
    a_seq = [e for e in timeline if e["moed"] == "A"]
    if len(a_seq) >= 2:
        def min_gap(entries):
            gaps = [
                (date.fromisoformat(b["date"]) - date.fromisoformat(a["date"])).days
                for a, b in zip(entries, entries[1:])
            ]
            return min(gaps) if gaps else 99
        base = min_gap(a_seq)
        best_course, best_gain = None, 0
        for c in {e["course_number"] for e in a_seq}:
            without = [e for e in a_seq if e["course_number"] != c]
            gain = min_gap(without) - base
            if gain > best_gain:
                best_course, best_gain = c, gain
        if best_course:
            relief = {
                "swap_candidate": best_course,
                "name": track.courses[best_course]["name"],
                "min_study_days_now": base,
                "min_study_days_without_it": base + best_gain,
            }

    return {
        "exam_timeline": timeline,
        "crunches": crunches,
        "crunch_free": not crunches,
        "best_single_swap_to_relieve": relief,
    }


def prereq_unmet_in(track: Track, plan_course_numbers: list[str], passed_courses: list[str], failed_courses: list[str]) -> list[str]:
    """Courses in a plan the student literally cannot register for -
    prerequisites unmet (retakes exempt: a failed course's own prereqs were
    met when they first took it). Used by the delivery backstop: a locked
    course is as undeliverable as an already-passed one, and the model was
    observed knowingly including one as an 'anchor' despite the menu."""
    passed = set(passed_courses)
    failed = set(failed_courses)
    return [
        c
        for c in plan_course_numbers
        if c in track.courses
        and c not in failed
        and not _prereq_satisfied(track.courses[c]["prerequisites"], passed)
    ]


def check_invariants(
    track: Track,
    plan_course_numbers: list[str],
    passed_courses: list[str],
    failed_courses: list[str],
    approved_retake_course: str | None = None,
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

    approved_retake_course: the ONE narrow, explicit exception - a passed
    course the student just explicitly approved retaking for grade
    improvement this turn (see agent_react_loop.py's proposed_retake /
    approved_grade_retake flow). Every OTHER passed course still triggers
    this check exactly as before; this is a single-course, single-turn
    carve-out, never a general loophole.

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

    already_passed_in_plan = [
        c for c in plan_course_numbers if c in passed and c != approved_retake_course
    ]
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
    approved_retake_course: str | None = None,
) -> dict:
    """The critic: checks a candidate plan against prerequisites, semester
    offering, credit range, day-of-week constraints, exam spacing, workload
    cap, and real degree progress (enough mandatory courses to make the
    semester worthwhile). Returns pass/fail plus itemized issues so the
    agent can repair a rejected plan rather than just being told "no".

    The mandatory-course and credit floors are enforced here, not just
    suggested in the drafting prompt - a rule that must actually hold
    belongs in deterministic code, the same lesson that applies to every
    other check in this function.

    approved_retake_course: same single-course, single-turn exception as
    check_invariants - a passed course the student just explicitly approved
    retaking for grade improvement. Every other passed course still fails
    this check exactly as before."""
    excluded_weekdays = set(excluded_weekdays or [])
    passed = set(passed_courses)
    issues = []

    unknown = [c for c in plan_course_numbers if c not in track.courses]
    if unknown:
        return {"pass": False, "issues": [{"course": c, "reason": "unknown course number"} for c in unknown]}

    for c in plan_course_numbers:
        course = track.courses[c]
        if c in passed and c != approved_retake_course:
            issues.append({"course": c, "reason": "already passed - can't be retaken or counted again"})
        if not course["offered_next_semester"]:
            issues.append({"course": c, "reason": f"not offered in {track.target_semester}"})
        if not _prereq_satisfied(course["prerequisites"], passed):
            issues.append({"course": c, "reason": "prerequisites not met"})

    # Coordinated section check for the WHOLE plan at once: sections are
    # chosen to dodge excluded days AND each other, so only genuinely
    # irreducible problems become issues. This is what catches "two
    # lectures stacked on Tuesday 11:30" - a real delivered-plan defect
    # that was previously invisible because time collisions were never
    # checked anywhere.
    section_plan = choose_sections(track, plan_course_numbers, list(excluded_weekdays))
    for hit in section_plan["excluded_day_hits"]:
        issues.append(
            {
                "course": hit["course"],
                "reason": f"meets on an excluded weekday ({', '.join(str(d) for d in hit['weekdays'])}) in every section",
            }
        )
    seen_pairs: set[frozenset] = set()
    for o in section_plan["overlaps"]:
        pair = frozenset((o["course_a"], o["course_b"]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        other = o["course_b"]
        issues.append(
            {
                "course": o["course_a"],
                "reason": (
                    f"class time overlaps with {track.courses[other]['name'] if other in track.courses else other} "
                    f"({o['time']}) in every section combination"
                ),
            }
        )

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

    # Choose-one-variant requirements (calculus / algebra / probability...):
    # a group the student could take RIGHT NOW (a variant is offered and its
    # prerequisites are met) but hasn't passed and isn't taking is a real
    # gap - these are the gateway courses that block half the degree, and
    # skipping them is exactly the failure seen live (a semester-1 plan
    # without infi or algebra). Skipped when the student explicitly asked
    # for a minimal load (min_mandatory_courses == 0).
    if min_mandatory_courses > 0:
        plan_set = set(plan_course_numbers)
        for group in track.mandatory_choice_groups:
            options = set(group["options"])
            if options & passed or options & plan_set:
                continue
            takeable = [
                o
                for o in group["options"]
                if track.courses[o]["offered_next_semester"]
                and _prereq_satisfied(track.courses[o]["prerequisites"], passed)
            ]
            if takeable:
                issues.append(
                    {
                        "course": takeable[0],
                        "reason": (
                            f"required choice group not covered - must take one of: {group['label']} "
                            "(gateway requirement, don't postpone)"
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
