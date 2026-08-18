"""
Fetches everything the planning agent's tools need for one degree track:
its requirement tree, full details for every course referenced anywhere in
that tree (regardless of faculty), CheeseFork difficulty data,
technion-histograms grade stats, and the reverse-prerequisite graph - and
writes it all to one consolidated JSON file.

Usage:
    python3 fetch_track_bundle.py --faculty "הפקולטה למדעי הנתונים וההחלטות" \\
        --track "הנדסת נתונים ומידע"
"""

import argparse
import json
from pathlib import Path

import technion_api as api
from prereq_parser import collect_prereq_courses, parse_prereq_tree

DEFAULT_FACULTY = "הפקולטה למדעי הנתונים וההחלטות"
DEFAULT_TRACK = "הנדסת נתונים ומידע"

# How many semesters back to search for a course's general info if it's not
# offered in the target (next) semester - covers courses offered only every
# other year, or ones this track requires but doesn't itself teach often.
FALLBACK_SEMESTER_LOOKBACK = 8


def semester_key(year: int, semester: int) -> str:
    return f"{year}_{semester}"


def fetch_course_data(course_number: str, semesters: list[dict], target: dict) -> dict:
    """Fetch a course's general info from the target (next) semester if
    offered there, else fall back to the most recent semester it was
    offered in. Schedule/exam data always reflects whichever semester the
    general info came from."""
    candidates = [target] + list(
        reversed(
            [s for s in semesters if (s["year"], s["semester"]) < (target["year"], target["semester"])]
        )
    )[:FALLBACK_SEMESTER_LOOKBACK]

    last_error = None
    for semester in candidates:
        try:
            general = api.get_course_general_info(semester["year"], semester["semester"], course_number)
        except RuntimeError as e:
            last_error = e
            continue

        schedule = api.get_course_schedule(semester["year"], semester["semester"], course_number)
        prereq_tree = parse_prereq_tree(general.pop("prereq_tokens"))

        return {
            **general,
            "prerequisites": prereq_tree,
            "info_semester": semester_key(semester["year"], semester["semester"]),
            "offered_next_semester": semester is target,
            "schedule": schedule,
        }

    raise RuntimeError(
        f"Could not find {course_number} in any of the last "
        f"{FALLBACK_SEMESTER_LOOKBACK} semesters: {last_error}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--faculty", default=DEFAULT_FACULTY)
    parser.add_argument("--track", default=DEFAULT_TRACK)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent.parent / "data"))
    parser.add_argument("--cache-dir", default=str(Path(__file__).parent.parent / ".cache"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-cheesefork", action="store_true", help="Skip fetching CheeseFork reviews (faster iteration)")
    parser.add_argument("--no-histograms", action="store_true", help="Skip fetching technion-histograms grade stats (faster iteration)")
    args = parser.parse_args()

    if not args.no_cache:
        api.set_cache_dir(Path(args.cache_dir))

    print("Fetching semester list...")
    semesters = api.get_semesters()
    target = semesters[-1]
    print(f"  target (next) semester: {semester_key(target['year'], target['semester'])}")

    print(f"Resolving track '{args.track}' under '{args.faculty}'...")
    track = api.find_track(target["year"], target["semester"], args.faculty, args.track)
    print(f"  {track['otjid']} | {track['name']}")

    print("Fetching requirement tree...")
    tree = api.get_track_requirement_tree(target["year"], target["semester"], track["otjid"])
    course_numbers = api.flatten_tree_course_numbers(tree)
    print(f"  {len(course_numbers)} unique courses referenced in the tree")

    print("Fetching course details...")
    courses = {}
    failures = []
    for i, course_number in enumerate(course_numbers, 1):
        if i % 25 == 0 or i == len(course_numbers):
            print(f"  {i}/{len(course_numbers)}")
        try:
            courses[course_number] = fetch_course_data(course_number, semesters, target)
        except RuntimeError as e:
            print(f"  FAILED {course_number}: {e}")
            failures.append(course_number)

    if not args.no_cheesefork:
        import cheesefork_client as cf

        print("Fetching CheeseFork reviews...")
        for i, course_number in enumerate(courses, 1):
            if i % 25 == 0 or i == len(courses):
                print(f"  {i}/{len(courses)}")
            courses[course_number]["cheesefork"] = cf.get_course_feedback(course_number)

    if not args.no_histograms:
        import histogram_client as hc

        print("Fetching technion-histograms grade stats...")
        for i, course_number in enumerate(courses, 1):
            if i % 25 == 0 or i == len(courses):
                print(f"  {i}/{len(courses)}")
            courses[course_number]["grade_stats"] = hc.get_grade_stats(course_number)

    print("Building reverse-prerequisite graph...")
    reverse_prereqs: dict[str, list[str]] = {c: [] for c in courses}
    for course_number, course in courses.items():
        for prereq_course in collect_prereq_courses(course["prerequisites"]):
            reverse_prereqs.setdefault(prereq_course, [])
            if course_number not in reverse_prereqs[prereq_course]:
                reverse_prereqs[prereq_course].append(course_number)

    bundle = {
        "track": {"otjid": track["otjid"], "name": track["name"], "faculty": args.faculty},
        "target_semester": semester_key(target["year"], target["semester"]),
        "requirement_tree": tree,
        "courses": courses,
        "reverse_prereqs": reverse_prereqs,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"track_{track['otjid']}.json"
    output_file.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nWrote {len(courses)} courses to {output_file}")
    if failures:
        print(f"{len(failures)} courses failed entirely: {failures}")


if __name__ == "__main__":
    main()
