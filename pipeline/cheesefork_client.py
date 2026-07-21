"""
Client for CheeseFork's crowd-sourced course reviews.

CheeseFork (an open-source Technion scheduling tool) stores student course
feedback in a public Firestore database. Its own frontend reads this
client-side with no authentication - confirmed the same works through
Firestore's plain REST API, so no SDK is needed here.

Firestore project: cheesefork-de9af
Collection: courseFeedback, one document per course number, holding a
`posts` array of {generalRank, difficultyRank, text, author, semester,
timestamp}. generalRank/difficultyRank are 1-5 student ratings ("overall"
and "course workload" respectively - "עומס הקורס" in the feedback form).
"""

import time
import urllib.parse

import requests

FIRESTORE_BASE = (
    "https://firestore.googleapis.com/v1/projects/cheesefork-de9af/"
    "databases/(default)/documents/courseFeedback"
)
REQUEST_TIMEOUT = 30
MAX_RETRY_DELAY = 60

_session = requests.Session()


def _extract(value: dict):
    """Unwrap one Firestore typed value, e.g. {"integerValue": "4"} -> 4."""
    if "integerValue" in value:
        return int(value["integerValue"])
    if "doubleValue" in value:
        return value["doubleValue"]
    if "stringValue" in value:
        return value["stringValue"]
    if "mapValue" in value:
        return {k: _extract(v) for k, v in value["mapValue"].get("fields", {}).items()}
    return None


MAX_SAMPLE_REVIEWS = 3
MAX_SAMPLE_REVIEW_CHARS = 280

_EMPTY_FEEDBACK = {
    "review_count": 0,
    "avg_difficulty": None,
    "avg_general": None,
    "sample_reviews": [],
}


def _truncate(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_SAMPLE_REVIEW_CHARS:
        return text
    return text[:MAX_SAMPLE_REVIEW_CHARS].rsplit(" ", 1)[0] + "…"


def get_course_feedback(course_number: str) -> dict:
    """Aggregate CheeseFork reviews for one course number.

    Returns {"review_count": int, "avg_difficulty": float | None,
    "avg_general": float | None, "sample_reviews": [str, ...]}.
    avg_general is CheeseFork's 1-5 student *satisfaction* rating, not an
    official grade - Technion doesn't publish per-student grades, and
    CheeseFork doesn't collect them; this is "how much students liked the
    course," which is what's actually available here. A course with no
    reviews yet (common for newer/rare electives) returns review_count 0,
    both averages None, and an empty sample list - this is expected, not an
    error.
    """
    url = f"{FIRESTORE_BASE}/{urllib.parse.quote(course_number)}"

    delay = 2
    while True:
        try:
            response = _session.get(url, timeout=REQUEST_TIMEOUT)
            break
        except requests.RequestException as e:
            print(f"  retrying cheesefork fetch for {course_number} ({e})")
            time.sleep(delay)
            delay = min(delay * 2, MAX_RETRY_DELAY)

    if response.status_code == 404:
        return dict(_EMPTY_FEEDBACK)
    if response.status_code != 200:
        raise RuntimeError(
            f"Bad status {response.status_code} fetching feedback for {course_number}"
        )

    doc = response.json()
    posts_field = doc.get("fields", {}).get("posts")
    if posts_field is None:
        return dict(_EMPTY_FEEDBACK)

    posts = [_extract(v) for v in posts_field["arrayValue"].get("values", [])]

    # Mirrors CheeseFork's own client-side aggregation: only average posts
    # that have both ranks set (a post can be text-only feedback).
    ranked = [p for p in posts if p.get("generalRank") and p.get("difficultyRank")]

    # Most recent reviews with actual text, as a short "what students say"
    # sample - not an aggregate, just real quotes for the final explanation
    # to draw on.
    with_text = sorted(
        (p for p in posts if p.get("text", "").strip()),
        key=lambda p: p.get("timestamp", 0),
        reverse=True,
    )
    sample_reviews = [_truncate(p["text"]) for p in with_text[:MAX_SAMPLE_REVIEWS]]

    if not ranked:
        return {
            "review_count": len(posts),
            "avg_difficulty": None,
            "avg_general": None,
            "sample_reviews": sample_reviews,
        }

    avg_difficulty = sum(p["difficultyRank"] for p in ranked) / len(ranked)
    avg_general = sum(p["generalRank"] for p in ranked) / len(ranked)

    return {
        "review_count": len(posts),
        "avg_difficulty": round(avg_difficulty, 2),
        "avg_general": round(avg_general, 2),
        "sample_reviews": sample_reviews,
    }
