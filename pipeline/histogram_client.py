"""
Client for technion-histograms - per-course grade distribution summaries
(github.com/michael-maltsev/technion-histograms), used to compare a
student's own grade in a passed course against how the course actually
graded historically (see agent/tools.py's suggest_grade_improvements).

The project publishes a static site on its `gh-pages` branch, one directory
per 8-digit course number, each containing an index.min.json shaped like:
{"<semester_code>": {"<exam_session>": {"students": "44", "passFail":
"43/1", "passPercent": "98", "min": "53", "max": "100", "average":
"83.932", "median": "85"}, ...}, ...} - all values are strings, top-level
keys are semester codes (higher = more recent), and exam-session key names
aren't guaranteed uniform across every course (only confirmed against one
real course during research: Final_A/Final_B, sometimes a combined
"Finals"). No auth, plain GET, served off GitHub's raw-content CDN.

This is public aggregate academic data (course-wide statistics), not a
student's own record - unrelated to the transcript privacy rule elsewhere
in this codebase.
"""

import time

import requests

RAW_BASE = "https://raw.githubusercontent.com/michael-maltsev/technion-histograms/gh-pages"
REQUEST_TIMEOUT = 30
MAX_RETRY_DELAY = 60

_session = requests.Session()

# Tried in this order within whichever semester is picked - a combined
# "Finals" session (when a course reports one) is preferred over a single
# moed, since it reflects the whole cohort rather than just one sitting.
_SESSION_KEY_PRIORITY = ("Finals", "Final_A", "Final_B", "Final")


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_grade_stats(course_number: str) -> dict | None:
    """Most-recent-semester grade summary for one course number.

    Returns {"avg_grade": float, "median_grade": float, "student_count":
    int, "semester": str} or None when there's nothing usable (course not
    in the dataset at all, or no session with a parseable average) - this
    is the common/expected case for many electives, not an error.
    """
    url = f"{RAW_BASE}/{course_number}/index.min.json"

    delay = 2
    while True:
        try:
            response = _session.get(url, timeout=REQUEST_TIMEOUT)
            break
        except requests.RequestException as e:
            print(f"  retrying histogram fetch for {course_number} ({e})")
            time.sleep(delay)
            delay = min(delay * 2, MAX_RETRY_DELAY)

    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise RuntimeError(
            f"Bad status {response.status_code} fetching histogram for {course_number}"
        )

    try:
        by_semester = response.json()
    except ValueError:
        return None
    if not isinstance(by_semester, dict) or not by_semester:
        return None

    for semester in sorted(by_semester, reverse=True):
        sessions = by_semester[semester]
        if not isinstance(sessions, dict):
            continue
        for key in _SESSION_KEY_PRIORITY:
            session = sessions.get(key)
            if not session:
                continue
            avg = _to_float(session.get("average"))
            median = _to_float(session.get("median"))
            if avg is None:
                continue
            return {
                "avg_grade": round(avg, 1),
                "median_grade": round(median, 1) if median is not None else None,
                "student_count": int(_to_float(session.get("students")) or 0),
                "semester": semester,
            }
    return None
