"""
One-off patch: adds "grade_stats" (technion-histograms averages, see
histogram_client.py) to every course already in the existing data/track_*.json
bundles, WITHOUT re-running the full fetch_track_bundle.py pipeline.

Deliberately separate from a full pipeline re-run: re-fetching everything
would also re-touch SAP/CheeseFork data with no reviewable diff process yet
(a separate unsolved problem) - this script only ever adds the one new key,
nothing else in the file changes.

Any *future* full pipeline run already includes this automatically
(fetch_track_bundle.py's --no-histograms flag), so this script only needs
to run once per already-existing bundle.

Usage:
    python3 backfill_grade_stats.py
"""

import json
from pathlib import Path

import histogram_client as hc

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    bundle_files = sorted(DATA_DIR.glob("track_*.json"))
    if not bundle_files:
        print(f"No track_*.json files found in {DATA_DIR}")
        return

    for bundle_file in bundle_files:
        bundle = json.loads(bundle_file.read_text(encoding="utf-8"))
        courses = bundle.get("courses", {})
        print(f"{bundle_file.name}: {len(courses)} courses")

        found = 0
        for i, (course_number, course) in enumerate(courses.items(), 1):
            if i % 25 == 0 or i == len(courses):
                print(f"  {i}/{len(courses)}")
            stats = hc.get_grade_stats(course_number)
            course["grade_stats"] = stats
            if stats is not None:
                found += 1

        bundle_file.write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"  wrote grade_stats for {found}/{len(courses)} courses -> {bundle_file}")


if __name__ == "__main__":
    main()
