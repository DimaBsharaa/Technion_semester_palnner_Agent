"""
HTTP entry point. Run locally with:
    uvicorn main:app --reload --port 8787

Note the --reload flag above: without it, editing this code while uvicorn
is already running has no effect until the process is restarted, which has
already caused real confusion once (a stale process kept answering with the
previous architecture's behavior for over an hour after a rewrite). Check
GET /health's "version" field against AGENT_VERSION below if behavior ever
seems inconsistent with the code you're reading.

GET /tracks
  Which degree tracks can actually be planned right now - only tracks with a
  usable mandatory-course structure (see data_bundle.list_tracks).

GET /intake-options?track_id=...
  Static data for rendering a structured intake form for one track: its
  mandatory course list (with prerequisite depth, for smart "probably passed
  by now" defaults), weekday labels, and pace options.

POST /transcript
  multipart/form-data, field "file": a Technion transcript PDF. Returns
  {"found": true, "passed_courses": [...], "failed_courses": [...],
  "grades": {course_number: int}} on success, or {"found": false, "error":
  "..."} on anything unreadable - never a 500, never a traceback. See
  transcript_parser.py. The frontend folds a successful result straight
  into a normal `intake` submission below, in place of the manual course
  checklist.

POST /chat
  body: {"track_id": "...", "messages": [...], "cost_usd": 0.0, "intake": null}
    - track_id: which track (from GET /tracks) this conversation is for.
    - messages: plain {"role": "user"|"assistant", "content": ...} history
      (the caller keeps it - see student_key below for the one optional
      exception); on the very first call this should just be
      [{"role": "user", "content": "<what the student said>"}]. No system
      message needed - each internal step builds its own.
    - cost_usd: running total from the previous response, echoed back so the
      budget guard is cumulative across a whole conversation.
    - intake: optional structured form submission (semester_number,
      excluded_weekdays, pace, passed_courses, failed_courses, notes,
      override_minimums). When present, this replaces free-text extraction
      entirely for this turn - the single least reliable step in the whole
      pipeline, skipped whenever the frontend can just ask directly instead
      of an LLM guessing at typed text.
    - student_key: optional SHA-256 hex hash of a student's email, computed
      client-side (the server never sees the raw email). When present, this
      turn's resolved state is also saved to Supabase (student_store.py) so
      GET /session can restore it later on a different browser/device. This
      is a lookup key, not authentication - see README.md's "Session
      persistence" note for the honest security boundary. Omit entirely for
      the original fully-stateless behavior (still the default).
      Also: when a caller sends student_key but no known_context (a brand
      new conversation), the server checks for a previously saved profile
      under that key and seeds known_context from it automatically
      (_recall_known_context) - so a returning student doesn't need to
      re-state facts they already gave in an earlier conversation, on any
      device. The response then carries "recalled_profile": true.
  response: {"messages": [...], "cost_usd": ..., "tool_log": [...], "stopped_reason": ...}

GET /session?student_key=...
  Look up a previously saved conversation/plan by the same hash described
  above. {"found": false} if nothing's saved (or Supabase isn't configured,
  or the lookup fails for any reason) - never an error; a student who never
  saved a session just sees "not found," same as never having used the app.
"""

import json
import os
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

import student_store
import tools
import transcript_parser
from agent_loop import (
    build_state_from_intake,
    get_llm_trace,
    run_agent_turn,
    start_llm_trace,
    summarize_intake_as_message,
)
from agent_react_loop import run_agent_turn_v2
from data_bundle import get_track, list_tracks

_HERE = Path(__file__).parent
SITE_INDEX = _HERE.parent / "site" / "index.html"
ARCH_PNG = _HERE / "static" / "architecture.png"

# Bump this string on every architecturally-significant change, so a stale
# running process is immediately obvious from GET /health.
AGENT_VERSION = "schedule-intelligence-v1-2026-07-21"

# "pipeline" = agent_loop.run_agent_turn (hardcoded draft/verify/repair/
# compare/explain stages) - kept available as a fallback. "react" =
# agent_react_loop.run_agent_turn_v2 (model picks its own tool sequence,
# ending in a terminal deliver_plan call) - the default after Stage 2
# validation showed no reliability regressions. Both share the identical
# /chat response contract, so the frontend needed zero changes either way.
AGENT_MODE_DEFAULT = os.environ.get("AGENT_MODE", "react")
AGENT_IMPLEMENTATIONS = {"pipeline": run_agent_turn, "react": run_agent_turn_v2}

app = FastAPI()

# GitHub Pages origin for the frontend - set via env var so this isn't
# hardcoded to one specific username/repo path.
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    track_id: str
    messages: list[dict] = []
    cost_usd: float = 0.0
    intake: dict | None = None
    # Optional per-request override of AGENT_MODE, for side-by-side testing
    # without restarting the server. Omit to use the server's default.
    agent_mode: str | None = None
    # Echo of the previous response's known_context field (resolved state +
    # delivered plan). The caller keeps it - there's no server-side session
    # store. When present, a follow-up is treated as a revision rather than
    # a from-scratch request.
    known_context: dict | None = None
    # Optional: SHA-256 hex digest of the student's email, computed
    # CLIENT-SIDE (server never sees the raw email - see site/index.html's
    # sha256hex + student_store.py). When present, this turn's resolved
    # state is also saved to Supabase so GET /session can restore it on a
    # different device. Purely additive: omit it and behavior is unchanged.
    student_key: str | None = None


def _persist_session(track_id: str, student_key: str | None, result: dict) -> None:
    """Best-effort save of this turn's resolved state to Supabase (see
    student_store.py) so a later GET /session on another browser/device can
    restore it - the server-side counterpart to site/index.html's
    `planner_session` localStorage blob, not a replacement for it. No-op
    whenever student_key is absent or Supabase isn't configured; never
    raises - a Supabase hiccup must never affect the response the student
    actually gets."""
    if not student_key:
        return
    student_store.save_session(
        student_key,
        {
            "track_id": track_id,
            "track_name": get_track(track_id).name,
            "schema_version": AGENT_VERSION,
            "messages": result.get("messages"),
            "cost_usd": result.get("cost_usd"),
            "known_context": result.get("known_context"),
            "plan_result": result.get("plan_result"),
        },
    )


def _recall_known_context(track_id: str, student_key: str | None) -> dict | None:
    """Best-effort pull of a returning student's saved facts into the exact
    known_context shape run_agent_turn already understands - so a BRAND-NEW
    conversation (new device, cleared browser, or just "hi" a week later)
    gets the same deterministic carry-forward guarantee merge_known_state
    already gives an ordinary same-session revision turn, instead of
    starting cold and re-asking things this student already answered.

    Only ever called by the /chat handlers when the caller sent no
    known_context of its own - i.e. strictly the first turn of a fresh
    conversation; an ongoing conversation is never overridden by this.

    state (passed/failed/constraints) carries over regardless of track -
    Technion course numbers are global. previous_plan/previous_issues/
    previous_explanation only carry over when the saved row's track_id
    matches the one requested now - a plan built for a different degree
    track isn't a valid revision baseline for this one.

    Never raises - same posture as student_store's own functions; a
    Supabase hiccup or a malformed saved row must never break the turn,
    just silently look like "nothing saved yet"."""
    if not student_key:
        return None
    try:
        saved = student_store.load_session(student_key)
        if not saved:
            return None
        saved_context = saved.get("known_context") or {}
        state = saved_context.get("state")
        if not state:
            return None
        recalled = {
            "state": state,
            "previous_plan": None,
            "previous_issues": [],
            "previous_explanation": None,
        }
        if saved.get("track_id") == track_id:
            recalled["previous_plan"] = saved_context.get("previous_plan")
            recalled["previous_issues"] = saved_context.get("previous_issues") or []
            recalled["previous_explanation"] = (saved.get("plan_result") or {}).get("explanation")
        return recalled
    except Exception:
        return None


@app.get("/tracks")
def tracks():
    return list_tracks()


@app.get("/intake-options")
def intake_options(track_id: str):
    return tools.get_intake_options(get_track(track_id))


@app.post("/transcript")
async def transcript(file: UploadFile):
    """Parses an uploaded Technion transcript PDF (see transcript_parser.py)
    into passed_courses/failed_courses/grades - the frontend folds this
    straight into a normal `intake` submission on /chat, same as the manual
    course checklist. Optional, additive: a student without the PDF just
    uses the checklist instead, unaffected by this endpoint's existence.

    Deliberately a generic error message on failure, never a traceback -
    unlike /chat's TEMPORARY debug behavior, this endpoint handles
    PII-adjacent content (a transcript's header carries a name and Technion
    ID, even though this parser never returns them) and must not risk
    echoing back a fragment of parsed text in an error."""
    try:
        result = transcript_parser.parse_transcript_pdf(await file.read())
        return {"found": True, **result}
    except Exception:
        return {
            "found": False,
            "error": "Could not read course rows from this PDF - you can enter courses manually instead.",
        }


@app.post("/chat")
def chat(request: ChatRequest):
    # Each internal step builds its own system message, so strip any stray
    # one the caller might send rather than expecting it here.
    messages = [m for m in request.messages if m.get("role") != "system"]

    state_override = None
    if request.intake is not None:
        track = get_track(request.track_id)
        state_override = build_state_from_intake(request.intake)
        # Keep the transcript readable for later free-text follow-up turns,
        # which do go through LLM extraction and benefit from seeing this
        # as normal prior context.
        messages = messages + [{"role": "user", "content": summarize_intake_as_message(track, state_override)}]

    mode = request.agent_mode or AGENT_MODE_DEFAULT
    implementation = AGENT_IMPLEMENTATIONS.get(mode)
    if implementation is None:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown agent_mode {mode!r}. Valid: {sorted(AGENT_IMPLEMENTATIONS)}"},
        )

    try:
        start_llm_trace()
        known_context = request.known_context or _recall_known_context(request.track_id, request.student_key)
        result = implementation(
            request.track_id,
            messages,
            request.cost_usd,
            state_override=state_override,
            known_context=known_context,
        )
        result["llm_steps"] = get_llm_trace()
        if known_context is not None and request.known_context is None:
            result["recalled_profile"] = True
            result.setdefault("tool_log", []).insert(
                0,
                {
                    "name": "recall_student_profile",
                    "args": {},
                    "result": {"found": True, "same_track": bool(known_context.get("previous_plan"))},
                },
            )
        _persist_session(request.track_id, request.student_key, result)
        return result
    except Exception as e:
        # The traceback is useful for local debugging but must never reach
        # a caller of a deployed, public endpoint. Vercel sets VERCEL=1 in
        # every deployed environment, so this needs no manual flag.
        content = {"error": str(e)}
        if not os.environ.get("VERCEL"):
            import traceback

            content["traceback"] = traceback.format_exc()
        return JSONResponse(status_code=500, content=content)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Same contract as /chat but streams NDJSON: {"type":"step","name":...}
    for every real tool call as it happens (so the frontend's working card
    shows the agent actually thinking, not canned text), then a final
    {"type":"result","data":<the usual /chat response>}. Always uses the
    react loop - it's the only implementation that emits events."""
    messages = [m for m in request.messages if m.get("role") != "system"]

    state_override = None
    if request.intake is not None:
        track = get_track(request.track_id)
        state_override = build_state_from_intake(request.intake)
        messages = messages + [{"role": "user", "content": summarize_intake_as_message(track, state_override)}]

    events: queue.Queue = queue.Queue()

    def on_event(entry):
        events.put({"type": "step", "name": entry.get("name", "")})

    def worker():
        try:
            start_llm_trace()  # thread-local: must start inside the worker
            known_context = request.known_context or _recall_known_context(request.track_id, request.student_key)
            result = run_agent_turn_v2(
                request.track_id,
                messages,
                request.cost_usd,
                state_override=state_override,
                known_context=known_context,
                on_event=on_event,
            )
            result["llm_steps"] = get_llm_trace()
            if known_context is not None and request.known_context is None:
                result["recalled_profile"] = True
                result.setdefault("tool_log", []).insert(
                    0,
                    {
                        "name": "recall_student_profile",
                        "args": {},
                        "result": {"found": True, "same_track": bool(known_context.get("previous_plan"))},
                    },
                )
            _persist_session(request.track_id, request.student_key, result)
            events.put({"type": "result", "data": result})
        except Exception as e:
            events.put({"type": "error", "error": str(e)})
        events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        while True:
            item = events.get()
            if item is None:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/session")
def get_session(student_key: str):
    """Restore a previously saved conversation/plan on a fresh browser or a
    different device, keyed by the SHA-256 hash of the student's email (see
    site/index.html's "Restore my last session" flow and student_store.py).
    Never errors on a missing/invalid key or a Supabase hiccup - a student
    who never saved a session, or whose lookup fails for any reason, just
    sees "not found," same as if they'd never used the app before."""
    saved = student_store.load_session(student_key)
    if saved is None:
        return {"found": False}
    return {"found": True, "session": saved}


@app.get("/health")
def health():
    return {"status": "ok", "version": AGENT_VERSION, "agent_mode_default": AGENT_MODE_DEFAULT}


# ===================== Course-spec required endpoints =====================

@app.get("/")
def root_gui():
    """The GUI, served at the root URL as the course spec requires."""
    return FileResponse(SITE_INDEX, media_type="text/html")


@app.get("/api/team_info")
def team_info():
    return {
        "group_batch_order_number": os.environ.get("GROUP_BATCH_ORDER", "2_5"),
        "team_name": os.environ.get("TEAM_NAME", "APEX"),
        "students": [
            {"name": "Manhal Ghoummaid", "email": "manhal@campus.technion.ac.il"},
            {"name": "Dima Bshara", "email": "dima.bshara@campus.technion.ac.il"},
            {"name": "Diyar Husayyan", "email": "diyarh@campus.technion.ac.il"},
        ],
    }


@app.get("/api/agent_info")
def agent_info():
    """Static agent metadata; the worked example (with its real captured
    steps) lives in static/agent_info_example.json, generated from an
    actual /api/execute run."""
    example_path = _HERE / "static" / "agent_info_example.json"
    examples = []
    if example_path.exists():
        examples = json.loads(example_path.read_text(encoding="utf-8"))
    return {
        "description": (
            "APEX plans a Technion student's next semester from natural-language input in Hebrew or "
            "English. It reads the student's semester, track, passed and failed courses, blocked days, "
            "pace preference, requested courses, and grades, then builds a checked plan with course "
            "reasoning, weekly timetable, moed A and moed B exam dates, workload notes, risk analysis, "
            "and a roadmap to graduation.\n\n"
            "During planning, APEX works through real course data and planning tools: catalog and "
            "requirement lookup, prerequisite graph checks, section-level schedule analysis, exam "
            "spacing, CheeseFork review summaries, historical grade averages, future-impact analysis, "
            "risk reporting, and final invariant checks. It supports Data & Information Engineering, "
            "Information Systems Engineering, and Industrial Engineering and Management.\n\n"
            "The system handles the cases students actually ask about: failed-course retakes, "
            "grade-improvement retake suggestions, unavailable weekdays, light or fast pacing, course "
            "questions, transcript PDF grades, and follow-up changes such as adding, removing, or "
            "swapping a course. When constraints conflict, it explains the trade-off clearly instead "
            "of hiding it."
        ),
        "purpose": (
            "Give every Technion student the quality of semester planning an experienced human academic "
            "advisor provides - retake priority, exam-spacing awareness, workload balance, honest "
            "trade-offs - in seconds instead of a scarce office-hours appointment."
        ),
        "prompt_template": {
            "template": (
                "Semester: <number you are starting, e.g. 5>\n"
                "Passed: <'everything expected so far' or list what you passed>\n"
                "Failed / still need: <courses you failed or must retake, by name>\n"
                "Days you cannot attend: <e.g. Sunday, or 'none'>\n"
                "Pace: <light | normal | fast>\n"
                "Anything else: <optional - e.g. 'make sure course X is included', "
                "'I want the safest plan', a question about a course>"
            )
        },
        "prompt_examples": examples,
    }


@app.get("/api/model_architecture")
def model_architecture():
    return FileResponse(ARCH_PNG, media_type="image/png")


class ExecuteRequest(BaseModel):
    # Optional (default "") so a missing/empty prompt returns the spec's
    # {status:error,...} envelope from the handler rather than FastAPI's
    # default 422 validation error, which wouldn't match the required shape.
    prompt: str = ""


def _plan_to_text(result: dict) -> str:
    """Flatten a turn result into the single response string the course
    spec requires."""
    pr = result.get("plan_result")
    if result.get("needs_input"):
        missing = ", ".join(result["needs_input"].get("missing_fields", []))
        last = (result.get("messages") or [{}])[-1].get("content") or ""
        return f"{last} (Missing to proceed: {missing}. Please include these details in your prompt.)"
    if not pr:
        last = (result.get("messages") or [{}])[-1].get("content") or ""
        return last
    lines = [f"Semester plan for {pr['track_name']} - {pr['target_semester']}:"]
    for c in pr["courses"]:
        tag = " [RETAKE]" if c.get("is_retake") else ""
        why = f" - {c['why']}" if c.get("why") else ""
        lines.append(f"- {c['course_number']} {c['name']} ({c['points']} pts){tag}{why}")
    v = pr["verify"]
    lines.append(f"Total: {v['total_credits']} credits | verified: {'PASS' if v['pass'] else 'NEEDS REVIEW'}")
    if v.get("issues"):
        lines.append("Open issues: " + "; ".join(i["reason"] for i in v["issues"]))
    lines.append("")
    lines.append(pr.get("explanation") or "")
    return "\n".join(lines)


@app.post("/api/execute")
def execute(request: ExecuteRequest):
    """Course-spec main entry point: single prompt in, response + full
    LLM-call steps trace out. Stateless single shot; the rich session UI
    uses /chat/stream, but both run the same agent."""
    start_llm_trace()
    try:
        prompt = (request.prompt or "").strip()
        if not prompt:
            return {"status": "error", "error": "Empty prompt.", "response": None, "steps": []}
        # Optional track switch: mention a track by name to use it instead of the default.
        track_id = "SC00001453"
        low = prompt.lower()
        if "information systems" in low or "מערכות מידע" in prompt:
            track_id = "SC00001416"
        elif "industrial engineering" in low or "תעשיה" in prompt or "תעשייה" in prompt:
            track_id = "SC00001383"
        result = run_agent_turn_v2(track_id, [{"role": "user", "content": prompt}], 0.0)
        return {
            "status": "ok",
            "error": None,
            "response": _plan_to_text(result),
            "steps": get_llm_trace(),
        }
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "response": None, "steps": get_llm_trace()}
