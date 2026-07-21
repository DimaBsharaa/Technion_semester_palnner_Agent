"""
Fetches the full course catalog + schedule history + prerequisite graph for
one Technion faculty, across every semester SAP has data for, and writes it
to clean JSON files the planner app can read directly.

Usage:
    python3 fetch_faculty_data.py
    python3 fetch_faculty_data.py --faculty "הפקולטה למדעי המחשב"
    python3 fetch_faculty_data.py --last-semesters 6   # only the 6 most recent

Output (in --output-dir, default ../data):
    semesters.json  - every semester fetched, with start/end dates
    catalog.json    - one entry per course: name, points, syllabus, a
                      structured prerequisite tree, and which semesters it
                      was offered in
    offerings.json  - per-semester schedule data (groups, days, times, rooms,
                      staff) for every course offered that semester
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import technion_api
from prereq_parser import parse_prereq_tree

DEFAULT_FACULTY = "הפקולטה למדעי הנתונים וההחלטות"


def semester_key(year: int, semester: int) -> str:
    return f"{year}_{semester}"


def fetch_all(faculty: str, last_semesters: int | None, workers: int):
    print("Fetching semester list...")
    semesters = technion_api.get_semesters()
    if last_semesters:
        semesters = semesters[-last_semesters:]
    print(f"  {len(semesters)} semesters: "
          f"{semester_key(semesters[0]['year'], semesters[0]['semester'])} .. "
          f"{semester_key(semesters[-1]['year'], semesters[-1]['semester'])}")

    print(f"Finding '{faculty}' courses per semester...")
    course_numbers_by_semester: dict[str, list[str]] = {}
    for s in semesters:
        key = semester_key(s["year"], s["semester"])
        numbers = technion_api.get_faculty_course_numbers(s["year"], s["semester"], faculty)
        course_numbers_by_semester[key] = numbers
        print(f"  {key}: {len(numbers)} courses")

    tasks = [
        (s, course_number)
        for s in semesters
        for course_number in course_numbers_by_semester[semester_key(s["year"], s["semester"])]
    ]
    print(f"Fetching full details for {len(tasks)} (course, semester) pairs "
          f"with {workers} workers...")

    offerings: dict[str, list[dict]] = defaultdict(list)
    # Keep every semester a course was seen in, and its most recent general info.
    course_semesters: dict[str, list[str]] = defaultdict(list)
    latest_general: dict[str, tuple[str, dict]] = {}
    failures: list[tuple[str, str, str]] = []

    def fetch_one(semester: dict, course_number: str):
        key = semester_key(semester["year"], semester["semester"])
        full = technion_api.get_course_full(semester["year"], semester["semester"], course_number)
        return key, course_number, full

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_one, semester, course_number): (semester, course_number)
            for semester, course_number in tasks
        }
        done = 0
        for future in as_completed(futures):
            semester, course_number = futures[future]
            key = semester_key(semester["year"], semester["semester"])
            done += 1
            if done % 25 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}")

            try:
                _, _, full = future.result()
            except Exception as e:
                print(f"  FAILED {key}/{course_number}: {e}")
                failures.append((key, course_number, str(e)))
                continue

            course_semesters[course_number].append(key)
            # Semesters are processed in chronological order in `tasks`, but
            # futures complete out of order - compare to keep the latest.
            if key >= latest_general.get(course_number, ("", None))[0]:
                latest_general[course_number] = (key, full["general"])

            offerings[key].append(
                {"course_number": course_number, "schedule": full["schedule"]}
            )

    catalog = {}
    prereq_parse_errors = []
    for course_number, (_, general) in latest_general.items():
        tokens = general.pop("prereq_tokens")
        try:
            prereq_tree = parse_prereq_tree(tokens)
        except ValueError as e:
            print(f"  prereq parse failed for {course_number}: {e}")
            prereq_parse_errors.append((course_number, str(e)))
            prereq_tree = None
            general["prereq_tokens_raw"] = tokens  # keep for manual review

        catalog[course_number] = {
            **general,
            "prerequisites": prereq_tree,
            "offered_in": sorted(course_semesters[course_number]),
        }

    if prereq_parse_errors:
        print(f"{len(prereq_parse_errors)} course(s) kept raw prereq tokens "
              f"instead of a parsed tree (see prereq_tokens_raw in catalog.json)")

    return semesters, catalog, offerings, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--faculty", default=DEFAULT_FACULTY)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent.parent / "data"))
    parser.add_argument("--cache-dir", default=str(Path(__file__).parent.parent / ".cache"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--last-semesters", type=int, default=None,
                         help="Only fetch the N most recent semesters (default: all available)")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if not args.no_cache:
        technion_api.set_cache_dir(Path(args.cache_dir))

    semesters, catalog, offerings, failures = fetch_all(
        args.faculty, args.last_semesters, args.workers
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "semesters.json").write_text(
        json.dumps(semesters, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "offerings.json").write_text(
        json.dumps(offerings, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"\nWrote {len(catalog)} courses across {len(semesters)} semesters to {output_dir}")
    if failures:
        print(f"{len(failures)} (course, semester) pairs failed - see above")


if __name__ == "__main__":
    main()
