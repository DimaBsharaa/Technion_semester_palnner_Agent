"""
Parses a Technion undergraduate transcript PDF into the same structured
shape build_state_from_intake already consumes: passed_courses,
failed_courses, and a grades dict - replacing the least reliable step in
the whole pipeline (LLM free-text guessing) with exact ground truth
whenever a student has the PDF.

Text-layer extraction only (pdfplumber, pure Python - no OCR, no
poppler/native-binary dependency, unlike the pdftoppm tool that isn't
available in this environment) - so this works identically on Vercel's
serverless runtime and locally. Technion's transcript is a genuine text
PDF, not a scan.

PRIVACY - read this before touching this file: a transcript's header
carries the student's full name and Technion ID number. This module MUST
NEVER return, log, or persist those - only the course rows (course number,
pass/fail, grade). The PDF bytes and any extracted raw text are processed
in memory only and never written to disk, logged, or cached. This is a
hard rule, restated here because it's easy to violate by accident later
(e.g. logging the full extracted text while debugging a parsing miss).
"""

import io
import re

import pdfplumber

MIN_PASSING_GRADE = 55

# One course row, e.g.:
#   "00940224 Data Structures and Algorithms 4 81 2023-2024 Winter"
#   "00940345 Discrete Mathematics (for I.E) 4 Pass 2022-2023 Spring"
#   "01130013 Introductory Physics 1 Exemption without points 2022-2023 Winter"
# Course number: 8 digits. Semester: "YYYY-YYYY <Season>". Grade: a 1-3
# digit number, "Pass", or "Exemption without points". Credits (a decimal)
# are present for graded/Pass rows but absent for Exemption rows - treated
# as optional since it isn't needed for passed/failed/grades.
_COURSE_ROW_RE = re.compile(
    r"^(?P<course_number>\d{8})\s+"
    r"(?P<name>.*?)\s+"
    r"(?:(?P<credits>[\d.]+)\s+)?"
    r"(?P<grade>Pass|Exemption without points|\d{1,3})\s+"
    r"(?P<semester>\d{4}-\d{4}\s+\S+)\s*$"
)

_NON_GRADE_TOKENS = {"Pass", "Exemption without points"}


def parse_transcript_text(text: str) -> dict:
    """Core parsing logic, given already-extracted text - kept separate
    from the PDF-reading step so it's directly unit-testable without a
    real/fake PDF binary fixture. Tests must use a hand-fabricated fake
    transcript, never a real one (see module docstring)."""
    passed: set[str] = set()
    failed: set[str] = set()
    grades: dict[str, int] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _COURSE_ROW_RE.match(line)
        if not match:
            continue
        course_number = match.group("course_number")
        grade_token = match.group("grade")
        if grade_token in _NON_GRADE_TOKENS:
            passed.add(course_number)
            continue
        grade = int(grade_token)
        grades[course_number] = grade
        if grade >= MIN_PASSING_GRADE:
            passed.add(course_number)
        else:
            failed.add(course_number)

    return {
        "passed_courses": sorted(passed),
        "failed_courses": sorted(failed),
        "grades": grades,
    }


def parse_transcript_pdf(pdf_bytes: bytes) -> dict:
    """Opens pdf_bytes entirely in memory (never written to disk) and
    extracts course rows. Raises ValueError if nothing recognizable as a
    transcript row was found, so the caller can report a clear "doesn't
    look like a transcript" instead of silently returning empty lists."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    result = parse_transcript_text(text)
    if not result["passed_courses"] and not result["failed_courses"]:
        raise ValueError("No recognizable course rows found in this PDF.")
    return result
