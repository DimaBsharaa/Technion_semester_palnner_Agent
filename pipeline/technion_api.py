"""
Low-level client for Technion's public course-catalog data.

Technion's registration system (SAP) exposes course/schedule/prerequisite data
through an OData service that is used by the official course-search site
(portalex.technion.ac.il/ovv/). The service does not require login for read
access to course catalog data - it's the same data any student sees when
browsing the course search page, just fetched directly instead of through the
UI.

Endpoint: https://portalex.technion.ac.il/sap/opu/odata/sap/Z_CM_EV_CDIR_DATA_SRV
Protocol: SAP OData v2, queried via its $batch endpoint (single GET wrapped in
a multipart/mixed batch body - this is how the SAP UI itself talks to the
service, and mirroring it avoids extra round trips).
"""

import hashlib
import json
import re
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Optional

import requests

BASE_URL = "https://portalex.technion.ac.il/sap/opu/odata/sap/Z_CM_EV_CDIR_DATA_SRV"
BATCH_URL = f"{BASE_URL}/$batch?sap-client=700"
BATCH_BOUNDARY = "batch_planner_fetch"
REQUEST_TIMEOUT = 60
MAX_RETRY_DELAY = 300

_session = requests.Session()
_cache_dir: Optional[Path] = None


def set_cache_dir(path: Optional[Path]) -> None:
    """Cache every raw query response on disk, keyed by query hash. Lets a
    re-run pick up new semesters/courses without re-fetching ones already on
    disk, and makes the pipeline resilient to interrupted runs."""
    global _cache_dir
    _cache_dir = path
    if _cache_dir:
        _cache_dir.mkdir(parents=True, exist_ok=True)


def _cache_path(query: str) -> Path:
    digest = hashlib.sha256(query.encode()).hexdigest()[:16]
    return _cache_dir / f"{digest}.json"


def _send_request(query: str, allow_empty: bool = False) -> dict:
    """Send one OData query wrapped as a single-request SAP batch call."""
    if _cache_dir:
        cached = _cache_path(query)
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
    headers = {
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
    }

    body = (
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
    )

    response = _session.post(
        BATCH_URL, headers=headers, data=body, timeout=REQUEST_TIMEOUT
    )
    if response.status_code != 202:
        raise RuntimeError(
            f"Bad status code {response.status_code} for query: {query}"
        )

    chunks = response.text.replace("\r\n", "\n").strip().split("\n\n")
    if len(chunks) != 3:
        raise RuntimeError(f"Unexpected batch response shape for: {query}")

    json_str = chunks[2].split("\n", 1)[0]
    result = json.loads(json_str)

    if not allow_empty and result == {"d": {"results": []}}:
        raise RuntimeError(f"Empty response for query: {query}")

    if _cache_dir:
        _cache_path(query).write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )

    return result


def send_request(query: str, allow_empty: bool = False) -> dict:
    """`_send_request` with exponential backoff - the endpoint occasionally
    times out or drops the connection under load."""
    delay = 5
    while True:
        try:
            return _send_request(query, allow_empty)
        except Exception as e:
            print(f"  retrying ({e})")
            time.sleep(delay)
            delay = min(delay * 2, MAX_RETRY_DELAY)


def _sap_date(date_str: str) -> str:
    match = re.fullmatch(r"/Date\((\d+)\)/", date_str)
    if not match:
        raise RuntimeError(f"Invalid SAP date: {date_str}")
    ts = int(match.group(1))
    return datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%Y-%m-%d")


def get_semesters() -> list[dict]:
    """All semesters SAP currently has data for, oldest first.

    Each entry: {"year": int, "semester": int, "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    `semester` is SAP's code: 200 = Winter, 201 = Spring, 202 = Summer.
    """
    params = {
        "sap-client": "700",
        "$select": "PiqYear,PiqSession,Begda,Endda",
    }
    raw = send_request(f"SemesterSet?{urllib.parse.urlencode(params)}")
    results = raw["d"]["results"]
    if not results:
        raise RuntimeError("No semesters returned by SAP")

    semesters = []
    for r in results:
        session_code = int(r["PiqSession"])
        if session_code not in (200, 201, 202):
            continue
        semesters.append(
            {
                "year": int(r["PiqYear"]),
                "semester": session_code,
                "start": _sap_date(r["Begda"]),
                "end": _sap_date(r["Endda"]),
            }
        )

    semesters.sort(key=lambda s: (s["year"], s["semester"]))
    return semesters


def get_faculty_course_numbers(year: int, semester: int, faculty_name: str) -> list[str]:
    """Course numbers (8-digit, no 'SM' prefix) offered by one faculty in one
    semester. Uses a single bulk request selecting only Otjid+OrgText, then
    filters client-side - much cheaper than fetching full details for every
    course just to check its faculty."""
    params = {
        "sap-client": "700",
        "$skip": "0",
        "$top": "10000",
        "$filter": f"Peryr eq '{year}' and Perid eq '{semester}'",
        "$select": "Otjid,OrgText",
    }
    raw = send_request(f"SmObjectSet?{urllib.parse.urlencode(params)}", allow_empty=True)
    results = raw["d"]["results"]

    return sorted(
        r["Otjid"].removeprefix("SM")
        for r in results
        if r["OrgText"] == faculty_name
    )


@cache
def _get_building_name(year: int, semester: int, room_id: str) -> str:
    params = {"sap-client": "700", "$select": "Building"}
    raw = send_request(
        f"GObjectSet(Otjid='{urllib.parse.quote(room_id)}',Peryr='{year}',"
        f"Perid='{semester}')?{urllib.parse.urlencode(params)}"
    )
    building = raw["d"]["Building"]
    return re.sub(r"\s+", " ", (building or "").strip())


@dataclass(frozen=True)
class ScheduleTime:
    weekday: int
    begin: str
    end: str


_WEEKDAYS = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי"]


def _parse_schedule_text(text: str) -> Optional[list[ScheduleTime]]:
    """Parse SAP's ScheduleSummary/ScheduleText field, e.g.
    'יום ראשון 14:30 - 16:30, יום שני 10:30 - 12:30'. Returns None for
    irregular schedules SAP can't express as a simple weekly recurrence."""
    if not text:
        return []
    if text.strip() in ("לא סדיר", "מועדי מפגשים"):
        return None

    times = []
    for part in text.split(","):
        part = part.strip()
        match = re.fullmatch(
            r"יום (ראשון|שני|שלישי|רביעי|חמישי|שישי) (\d\d:\d\d)\s*-\s*(\d\d:\d\d)",
            part,
        )
        if not match:
            # Unrecognized fragment (date ranges, exceptions, etc.) - skip
            # rather than fail the whole course.
            continue
        times.append(
            ScheduleTime(
                weekday=_WEEKDAYS.index(match.group(1)),
                begin=match.group(2),
                end=match.group(3),
            )
        )
    return times


def get_course_schedule(year: int, semester: int, course_number: str) -> list[dict]:
    """Weekly schedule slots (lecture/tutorial/lab groups) for one course."""
    params = {
        "sap-client": "700",
        "$select": ",".join(
            [
                "ZzSeSeqnr",
                "EObjectSet/CategoryText",
                "EObjectSet/RoomText",
                "EObjectSet/RoomId",
                "EObjectSet/ScheduleSummary",
                "EObjectSet/Persons/Title",
                "EObjectSet/Persons/FirstName",
                "EObjectSet/Persons/LastName",
            ]
        ),
        "$expand": "EObjectSet,EObjectSet/Persons",
    }
    raw = send_request(
        f"SmObjectSet(Otjid='SM{course_number}',Peryr='{year}',Perid='{semester}',"
        f"ZzCgOtjid='',ZzPoVersion='',ZzScOtjid='')/SeObjectSet?"
        f"{urllib.parse.urlencode(params)}",
        allow_empty=True,
    )
    groups = raw["d"]["results"]

    schedule = []
    for group in groups:
        group_id = int(group["ZzSeSeqnr"])
        for item in group["EObjectSet"]["results"]:
            category = item["CategoryText"]
            if category not in ("הרצאה", "תרגול", "מעבדה", "פרויקט", "סמינר"):
                continue

            building, room = "", 0
            room_text = item["RoomText"]
            if room_text and room_text != "ראה פרטים":
                room_match = re.fullmatch(r"(\d\d\d)-(\d\d\d\d)", room_text)
                if room_match:
                    building = _get_building_name(year, semester, item["RoomId"])
                    room = int(room_match.group(2))

            staff = ""
            for person in item["Persons"]["results"]:
                title = person["Title"].strip()
                staff += (f"{title} " if title and title != "-" else "") + (
                    f"{person['FirstName']} {person['LastName']}\n"
                )
            staff = staff.rstrip("\n")

            times = _parse_schedule_text(item["ScheduleSummary"])
            for t in times or []:
                if t.weekday == 6:  # Saturday - shouldn't normally happen
                    continue
                schedule.append(
                    {
                        "group": group_id,
                        "type": category,
                        "weekday": t.weekday,
                        "start": t.begin,
                        "end": t.end,
                        "building": building,
                        "room": room,
                        "staff": staff,
                    }
                )

    return schedule


def _sap_time(time_str: str) -> str:
    match = re.fullmatch(r"PT(\d\d)H(\d\d)M\d\dS", time_str)
    if not match:
        raise RuntimeError(f"Invalid SAP time: {time_str}")
    return f"{match.group(1)}:{match.group(2)}"


# SAP's exam category codes -> our field names. FI/FB are the two regular
# exam slots ("מועד א"/"מועד ב"), MI/M2 are midterms ("בוחן מועד א/ב").
_EXAM_CATEGORIES = {
    "FI": "moed_a",
    "FB": "moed_b",
    "MI": "quiz_a",
    "M2": "quiz_b",
}


def _parse_exam_dates(exams: list[dict]) -> dict[str, list[dict]]:
    """Group raw Exams results by category into {category: [{date, time}]}."""
    result: dict[str, list[dict]] = {name: [] for name in _EXAM_CATEGORIES.values()}
    for exam in exams:
        category = _EXAM_CATEGORIES.get(exam["CategoryCode"])
        if category is None or not exam["ExamDate"]:
            continue
        entry = {"date": _sap_date(exam["ExamDate"])}
        if exam["ExamBegTime"] and exam["ExamBegTime"] != "PT00H00M00S":
            entry["time"] = _sap_time(exam["ExamBegTime"])
        if entry not in result[category]:
            result[category].append(entry)
    return {k: v for k, v in result.items() if v}


def get_course_general_info(year: int, semester: int, course_number: str) -> dict:
    """Course-level metadata: name, credit points, syllabus, faculty,
    responsible staff, exam dates, raw prerequisite tokens, and related-course
    links."""
    params = {
        "sap-client": "700",
        "$filter": (
            f"Peryr eq '{year}' and Perid eq '{semester}' and Otjid eq 'SM{course_number}'"
        ),
        "$select": ",".join(
            [
                "Otjid",
                "Points",
                "Name",
                "StudyContentDescription",
                "OrgText",
                "ZzAcademicLevelText",
                "ZzSemesterNote",
                "Responsible",
                "SmRelations",
                "SmPrereq",
                "Exams",
            ]
        ),
        "$expand": "Responsible,SmRelations,SmPrereq,Exams",
    }
    raw = send_request(f"SmObjectSet?{urllib.parse.urlencode(params)}")
    results = raw["d"]["results"]
    if len(results) != 1:
        raise RuntimeError(f"Expected 1 result for {course_number}, got {len(results)}")
    course = results[0]

    points = re.sub(r"(\.[1-9]+)0+$", r"\1", course["Points"])
    points = re.sub(r"\.0+$", "", points)

    responsible = "\n".join(
        f"{p['Title']} {p['FirstName']} {p['LastName']}".strip()
        for p in course["Responsible"]["results"]
    )

    no_additional_credit_with = [
        rel["Otjid"].removeprefix("SM")
        for rel in course["SmRelations"]["results"]
        if rel["ZzRelationshipKey"] in ("AZEC", "AZID")
    ]

    # Raw prereq tokens, kept as-is; prereq_parser.py turns these into a tree.
    prereq_tokens = [
        {
            "course": p["ModuleId"].lstrip("0") and p["ModuleId"],
            "operator": p["Operator"],
            "bracket": p["Bracket"],
        }
        for p in course["SmPrereq"]["results"]
    ]

    exams = _parse_exam_dates(course["Exams"]["results"])

    return {
        "course_number": course_number,
        "name": re.sub(r"\s+", " ", course["Name"].strip()),
        "faculty": course["OrgText"],
        "academic_level": course["ZzAcademicLevelText"],
        "points": points,
        "syllabus": course["StudyContentDescription"],
        "responsible": responsible,
        "notes": course["ZzSemesterNote"],
        "no_additional_credit_with": no_additional_credit_with,
        "prereq_tokens": prereq_tokens,
        "exams": exams,
    }


def get_course_full(year: int, semester: int, course_number: str) -> dict:
    general = get_course_general_info(year, semester, course_number)
    schedule = get_course_schedule(year, semester, course_number)
    return {"general": general, "schedule": schedule}


def get_faculties(year: int, semester: int) -> list[dict]:
    """All faculties/study-fields with course offerings in one semester.
    Each entry: {"org_id": "00002190", "name": "הפקולטה למדעי הנתונים וההחלטות"}."""
    params = {
        "sap-client": "700",
        "$skip": "0",
        "$top": "10000",
        "$filter": f"PiqYear eq '{year}' and PiqSession eq '{semester}'",
        "$select": "ZzOrgId,ZzOrgName",
    }
    raw = send_request(f"StudyFieldTilesSet?{urllib.parse.urlencode(params)}")
    return [
        {"org_id": r["ZzOrgId"], "name": r["ZzOrgName"]} for r in raw["d"]["results"]
    ]


def get_degree_tracks(year: int, semester: int, org_id: str) -> list[dict]:
    """Undergraduate degree tracks (e.g. "הנדסת נתונים ומידע") under one
    faculty org_id. Each entry: {"otjid": "SC00001453", "name": "..."}."""
    params = {
        "sap-client": "700",
        "$skip": "0",
        "$top": "10000",
        "$filter": (
            f"Peryr eq '{year}' and Perid eq '{semester}' and OrgId eq '{org_id}' "
            "and ZzAcademicLevel eq '1'"
        ),
        "$select": "Name,Otjid",
    }
    raw = send_request(f"ScObjectSet?{urllib.parse.urlencode(params)}")
    return [{"otjid": r["Otjid"], "name": r["Name"]} for r in raw["d"]["results"]]


def find_track(year: int, semester: int, faculty_name: str, track_name: str) -> dict:
    """Resolve human-readable faculty+track names to a track's Otjid, by
    substring match (Technion's own naming is inconsistent about exact
    wording/whitespace across catalog years)."""
    faculties = [f for f in get_faculties(year, semester) if faculty_name in f["name"]]
    if not faculties:
        raise RuntimeError(f"No faculty matching {faculty_name!r}")

    for faculty in faculties:
        all_tracks = get_degree_tracks(year, semester, faculty["org_id"])

        exact = [t for t in all_tracks if t["name"] == track_name]
        if exact:
            return exact[0]

        tracks = [t for t in all_tracks if track_name in t["name"]]
        if tracks:
            if len(tracks) > 1:
                raise RuntimeError(f"Ambiguous track match: {tracks}")
            return tracks[0]

    raise RuntimeError(f"No track matching {track_name!r} under {faculty_name!r}")


def _tree_children(year: int, semester: int, parent_otjid: str) -> list[dict]:
    params = {
        "sap-client": "700",
        "$filter": (
            f"ParentOtjid eq '{parent_otjid}' and Peryr eq '{year}' and Perid eq '{semester}'"
        ),
        "$orderby": "Otjid asc",
        "$select": "Stext,Otjid",
    }
    raw = send_request(f"TreeOnDemandSet?{urllib.parse.urlencode(params)}", allow_empty=True)
    return raw["d"]["results"]


def get_track_requirement_tree(year: int, semester: int, track_otjid: str) -> dict:
    """The degree track's requirement structure: a tree of category groups
    bottoming out at course-number leaves. Each node:
    {"otjid": ..., "name": ..., "children": [...] } for a category, or
    {"otjid": "SM...", "name": ..., "course": "12345678"} for a course leaf.
    """

    def build(otjid: str, name: str) -> dict:
        if otjid.startswith("SM"):
            return {"otjid": otjid, "name": name, "course": otjid.removeprefix("SM")}

        children = _tree_children(year, semester, otjid)
        return {
            "otjid": otjid,
            "name": name,
            "children": [build(c["Otjid"], c["Stext"]) for c in children],
        }

    return build(track_otjid, "root")


def flatten_tree_course_numbers(tree: dict) -> list[str]:
    """All course numbers referenced anywhere in a requirement tree, deduped."""
    numbers: set[str] = set()

    def walk(node: dict):
        if "course" in node:
            numbers.add(node["course"])
        else:
            for child in node.get("children", []):
                walk(child)

    walk(tree)
    return sorted(numbers)
