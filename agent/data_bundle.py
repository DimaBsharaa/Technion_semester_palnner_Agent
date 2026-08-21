"""Loads every pre-fetched track data bundle once at process start.

No live Technion/CheeseFork calls happen here or anywhere in the agent
backend - all of that ran ahead of time in pipeline/fetch_track_bundle.py.
Keeping the agent purely a reader of static data keeps every tool call fast
and free, which matters given the tight LLM budget this backend runs under.

Each track's data (courses, requirement tree, and the derived structures
that used to be single-track module globals - the mandatory course set and
inferred prerequisite depths) lives on its own Track object. Callers pass
the specific Track for a student's selected track explicitly through the
whole call chain, rather than mutating shared global state to point at
"the active track" - the latter would silently corrupt concurrent requests
for different students in different tracks.
"""

import json
from pathlib import Path


_DATA_DIR = Path(__file__).parent.parent / "data"
_MANDATORY_CATEGORY_NAMES = {"מקצועות חובה", "Mandatory", "Mandatory option"}

# The mirror image of the missing-courses problem: Technion's requirement-
# tree API tags some courses "Mandatory" that the official DDS diagram
# never lists at all (a duplicate course number for a requirement already
# covered elsewhere). Per-track, per-course, since this is confirmed
# against each track's own diagram, not a general rule.
_CONFIRMED_NON_REQUIREMENTS: dict[str, set[str]] = {
    # הנדסת מערכות מידע - 00940347: diagram wants 00940345 (ת') only.
    # 01040022 is the same course as 01040044 (the one the diagram
    # actually places in semester 2) under a second course number -
    # confirmed by the near-identical name and identical credit points.
    "SC00001416": {"00940347", "01040022"},
}


def _leaf_courses(node: dict) -> list[str]:
    if "course" in node:
        return [node["course"]]
    result = []
    for child in node.get("children", []):
        result.extend(_leaf_courses(child))
    return result


def _bottom_categories(node: dict) -> list[dict]:
    """Finds the shallowest nodes whose direct children are all course
    leaves - the real reportable categories. See tools.fetch_catalog for
    why this walk (rather than a fixed depth) is needed."""
    children = node.get("children")
    if children is None:
        return []
    if all("course" in child for child in children):
        return [node]

    result = []
    direct_leaves = [child for child in children if "course" in child]
    if direct_leaves:
        result.append({"name": node["name"], "children": direct_leaves})
    for child in children:
        if "course" not in child:
            result.extend(_bottom_categories(child))
    return result


class Track:
    """One degree track's full static dataset, plus derived structures
    computed once when the track loads rather than per-request."""

    def __init__(self, bundle: dict):
        self.otjid: str = bundle["track"]["otjid"]
        self.name: str = bundle["track"]["name"]
        self.faculty: str = bundle["track"]["faculty"]
        self.target_semester: str = bundle["target_semester"]
        self.courses: dict = bundle["courses"]
        self.requirement_tree: dict = bundle["requirement_tree"]
        self.reverse_prereqs: dict = bundle["reverse_prereqs"]
        # A course pulled in only as a prerequisite-tree dependency (not a
        # tree leaf fetch_track_bundle.py fetched CheeseFork/histogram data
        # for) can be missing these keys entirely, crashing any course-
        # listing tool with a bare KeyError. Backfilled once here, in the
        # same "no reviews yet" shape cheesefork_client.py returns for a
        # real course with zero reviews, so downstream readers need no
        # defensive code.
        for course in self.courses.values():
            course.setdefault(
                "cheesefork",
                {"review_count": 0, "avg_difficulty": None, "avg_general": None, "sample_reviews": []},
            )
            course.setdefault("grade_stats", None)

        # Ground truth, where we have it: Technion's own official DDS
        # ("track diagram") PDF places each course in a specific semester -
        # a real answer the prerequisite-depth heuristic below can only
        # approximate. Not exposed by Technion's requirement-tree API, so
        # populated by hand from digitized diagrams, necessarily partial.
        # See assess_progress/backfill_passed_courses: official data wins
        # whenever it exists, the heuristic only fills the gaps.
        self.official_semester: dict[str, int] = {
            c: course["official_semester"] for c, course in self.courses.items() if course.get("official_semester")
        }

        self.mandatory_course_numbers: set[str] = {
            c
            for category in _bottom_categories(self.requirement_tree)
            if category["name"] in _MANDATORY_CATEGORY_NAMES
            for c in _leaf_courses(category)
        }
        # Technion's requirement-tree API is missing real mandatory courses
        # the official DDS diagram shows - not stale data, the tree
        # endpoint just doesn't cover everything the diagram does. Any
        # course the diagram places in a semester is mandatory by
        # definition, tree or no tree.
        self.mandatory_course_numbers |= set(self.official_semester.keys())
        self.mandatory_course_numbers -= _CONFIRMED_NON_REQUIREMENTS.get(self.otjid, set())
        # Required courses that come as "pick one variant" groups (e.g.
        # calculus 1מ' OR 1מ2) live under the Mandatory category but NOT in
        # flat "Mandatory option" leaves - missing them would let the agent
        # treat a gateway requirement like calculus as an ordinary elective.
        self.mandatory_choice_groups: list[dict] = self._extract_mandatory_choice_groups()
        all_depths = self._compute_prereq_depths(
            self.mandatory_course_numbers | {o for g in self.mandatory_choice_groups for o in g["options"]}
        )
        self.mandatory_prereq_depths: dict[str, int] = {
            c: d for c, d in all_depths.items() if c in self.mandatory_course_numbers
        }
        for group in self.mandatory_choice_groups:
            group["depth"] = min(all_depths.get(o, 0) for o in group["options"])

    def _extract_mandatory_choice_groups(self) -> list[dict]:
        """Walks the requirement tree; every non-mandatory-named node that
        sits under a mandatory category and whose children are all course
        leaves is one choose-one-of group. Options are filtered to courses
        this track's bundle actually has (and never a confirmed
        non-requirement - see _CONFIRMED_NON_REQUIREMENTS).

        A group is skipped entirely if ANY of its options is already a
        confirmed flat-mandatory course (from official_semester, i.e. the
        real digitized diagram) - not just when ALL of them are. Technion's
        requirement tree is shared across tracks/faculties, so it pairs
        each real, already-required course variant with a DIFFERENT
        track's variant of the same subject (not part of this curriculum
        at all) under one shared "OR" node - every choice group across all
        3 tracks turned out to be one real option plus one phantom, never a
        genuine open choice once official_semester coverage exists. Left
        as "pick one," the agent could deliver the phantom option or
        surface a fake dilemma as an open issue. Still matters for whatever
        official_semester doesn't yet cover - that's where a real
        unresolved choice would show up."""
        excluded = _CONFIRMED_NON_REQUIREMENTS.get(self.otjid, set())
        groups: list[dict] = []
        seen: set[frozenset] = set()

        def walk(node: dict, in_mandatory: bool):
            name = node.get("name") or ""
            here_mandatory = in_mandatory or name in _MANDATORY_CATEGORY_NAMES
            children = node.get("children") or []
            if (
                in_mandatory
                and name not in _MANDATORY_CATEGORY_NAMES
                and children
                and all("course" in c for c in children)
            ):
                options = [c for c in _leaf_courses(node) if c in self.courses and c not in excluded]
                key = frozenset(options)
                already_resolved = key & self.mandatory_course_numbers
                if options and key not in seen and not already_resolved:
                    seen.add(key)
                    groups.append(
                        {
                            "label": " / ".join(self.courses[o]["name"] for o in options),
                            "options": options,
                        }
                    )
                return
            for child in children:
                if "course" not in child:
                    walk(child, here_mandatory)

        walk(self.requirement_tree, False)
        return groups

    def _compute_prereq_depths(self, seed_courses: set[str]) -> dict[str, int]:
        """Technion's own data has no "recommended semester" field for any
        course - this infers a rough ordering instead from the prerequisite
        chain across the WHOLE track catalog, honoring the AND/OR
        structure: an OR group needs only its easiest-to-reach alternative
        (min), an AND group needs all parts (max). depth 0 = takeable in
        semester 1; depth N = at least N semesters of prerequisites first.

        Only counting prerequisites that are themselves on the mandatory
        list would put courses whose prereqs live in other requirement
        categories at depth 0 ("semester 1"), wrongly asking a later-
        semester student about them. Alternatives outside the track
        catalog are ignored inside OR groups (can't be sequenced) rather
        than counted as depth 0, which would have the same effect. A
        heuristic approximation, not ground truth."""
        depths: dict[str, int] = {}

        def depth(course_number: str, stack: frozenset = frozenset()) -> int:
            if course_number in depths:
                return depths[course_number]
            if course_number in stack:
                return 0  # defensive: a cycle would be a data error, not a real case
            course = self.courses.get(course_number)
            if course is None:
                return 0

            def layers_before(tree) -> int | None:
                """Semester-layers needed before this course per the prereq
                tree; None = unknowable (references only courses outside
                this track's catalog)."""
                if tree is None:
                    return 0
                if "course" in tree:
                    p = tree["course"]
                    if p not in self.courses:
                        return None
                    return depth(p, stack | {course_number}) + 1
                known = [v for v in (layers_before(a) for a in tree.get("args", [])) if v is not None]
                if not known:
                    return None
                return min(known) if tree.get("op") == "or" else max(known)

            result = layers_before(course["prerequisites"])
            depths[course_number] = 0 if result is None else result
            return depths[course_number]

        for c in seed_courses:
            depth(c)
        return depths


def _load_all_tracks() -> dict[str, Track]:
    tracks = {}
    for path in sorted(_DATA_DIR.glob("track_*.json")):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        track = Track(bundle)
        tracks[track.otjid] = track
    return tracks


TRACKS: dict[str, Track] = _load_all_tracks()


def get_track(track_id: str) -> Track:
    track = TRACKS.get(track_id)
    if track is None:
        raise ValueError(f"Unknown track_id {track_id!r}. Available: {sorted(TRACKS)}")
    return track


def list_tracks() -> list[dict]:
    """Only tracks where a flat mandatory-course list was actually detected
    - some tracks are built around specialization "chains" instead of one
    universal required list, which this data model doesn't represent yet.
    Excluded by the structural signal, not a hardcoded track ID, so a
    future track with the same mismatch is silently skipped rather than
    silently broken."""
    return [
        {"track_id": t.otjid, "name": t.name, "faculty": t.faculty}
        for t in TRACKS.values()
        if t.mandatory_course_numbers
    ]
