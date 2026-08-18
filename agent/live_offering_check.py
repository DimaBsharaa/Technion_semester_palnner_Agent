"""
One last, real-time check against Technion's own live course-catalog
system, run only on the FINAL candidate courses right before delivering a
plan - so a course that got closed/cancelled/moved since data/track_*.json
was last refreshed doesn't get confidently recommended anyway. This is
what makes course availability something the agent VERIFIES at the moment
of delivery, not just something baked into a stale snapshot.

Deliberately reimplements a minimal slice of pipeline/technion_api.py's
SAP OData query here, in agent/, rather than importing that module:
1. agent/ is Vercel's deployed unit (see vercel.json's includeFiles) -
   pipeline/ isn't bundled for deployment at all.
2. agent/'s live runtime deliberately avoids the `requests` library (see
   student_store.py's docstring) so the root requirements.txt Vercel
   actually installs from doesn't need a new dependency for this - same
   urllib-only convention, same reasoning.

Also deliberately does NOT reuse technion_api.py's infinite-retry
send_request - fine for a supervised offline pipeline run, dangerous
inside a live student-facing request under Vercel's 300s hard cap. This
tries once per course, with a short hard timeout, and gives up cleanly.

Same public, unauthenticated endpoint the official course-search site
(portalex.technion.ac.il/ovv/) uses - no login required for read access.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

BATCH_URL = "https://portalex.technion.ac.il/sap/opu/odata/sap/Z_CM_EV_CDIR_DATA_SRV/$batch?sap-client=700"
BATCH_BOUNDARY = "batch_live_check"
REQUEST_TIMEOUT_SECONDS = 8  # short and hard, no retry - see module docstring


def _batch_body(query: str) -> bytes:
    return (
        f"--{BATCH_BOUNDARY}\r\n"
        "Content-Type: application/http\r\n"
        "Content-Transfer-Encoding: binary\r\n"
        "\r\n"
        f"GET {query} HTTP/1.1\r\n"
        "Accept: application/json\r\n"
        "Accept-Language: he\r\n"
        "DataServiceVersion: 2.0\r\n"
        "MaxDataServiceVersion: 2.0\r\n"
        "\r\n"
        "\r\n"
        f"--{BATCH_BOUNDARY}--\r\n"
    ).encode("utf-8")


def _is_offered(year: int, semester: int, course_number: str) -> bool | None:
    """One request, no retry, hard-capped at REQUEST_TIMEOUT_SECONDS.

    Returns True/False if the check completed, or None if anything went
    wrong (timeout, network error, unparseable response). None is NOT a
    failure to propagate - it means "couldn't verify this one," and the
    caller must treat that as "leave it alone," never as "drop it."
    """
    params = {
        "sap-client": "700",
        "$filter": f"Peryr eq '{year}' and Perid eq '{semester}' and Otjid eq 'SM{course_number}'",
        "$select": "Otjid",
    }
    query = f"SmObjectSet?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        BATCH_URL,
        data=_batch_body(query),
        method="POST",
        headers={
            "MaxDataServiceVersion": "2.0",
            "DataServiceVersion": "2.0",
            "Accept": "multipart/mixed",
            "Accept-Language": "he",
            "Content-Type": f"multipart/mixed;boundary={BATCH_BOUNDARY}",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
            "Origin": "https://portalex.technion.ac.il",
            "Referer": "https://portalex.technion.ac.il/ovv/",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            if response.status != 202:
                return None
            text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None

    chunks = text.replace("\r\n", "\n").strip().split("\n\n")
    if len(chunks) != 3:
        return None
    try:
        result = json.loads(chunks[2].split("\n", 1)[0])
    except (ValueError, IndexError):
        return None

    results = result.get("d", {}).get("results")
    if results is None:
        return None
    return len(results) >= 1


def check_still_offered(year: int, semester: int, course_numbers: list[str]) -> dict[str, bool]:
    """Live-checks each course number, sequentially. Returns a dict
    containing ONLY the courses the check actually completed for - a
    course number absent from the result means "couldn't verify," which
    the caller must treat exactly like a course it never doubted.

    Only ever called with the final, already-small candidate list (never
    the whole catalog), so worst case (every check times out) is bounded
    by len(course_numbers) * REQUEST_TIMEOUT_SECONDS."""
    out = {}
    for c in course_numbers:
        offered = _is_offered(year, semester, c)
        if offered is not None:
            out[c] = offered
    return out
