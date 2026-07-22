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

POST /chat
  body: {"track_id": "...", "messages": [...], "cost_usd": 0.0, "intake": null}
    - track_id: which track (from GET /tracks) this conversation is for.
    - messages: plain {"role": "user"|"assistant", "content": ...} history
      (the caller keeps it, since there's no server-side session store); on
      the very first call this should just be
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
  response: {"messages": [...], "cost_usd": ..., "tool_log": [...], "stopped_reason": ...}
"""

import json
import os
import queue
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

import tools
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
# running process is immediately obvious from GET /health instead of being
# diagnosed after the fact from file timestamps.
AGENT_VERSION = "schedule-intelligence-v1-2026-07-21"

# "pipeline" = agent_loop.run_agent_turn (hardcoded draft/verify/repair/
# compare/explain stages) - kept available as a fallback, not removed.
# "react" = agent_react_loop.run_agent_turn_v2 (the model picks its own tool
# sequence, ending in a terminal deliver_plan call) - now the default after
# Stage 2 validation (regression suite + live adversarial testing) showed no
# reliability regressions. Both share the identical /chat response
# contract, so the frontend needed zero changes either way.
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
    # Echo of the previous response's known_context field (resolved student
    # state + the plan already delivered). The caller keeps it, same as
    # messages - there's no server-side session store. When present, a
    # follow-up message is treated as a revision of the existing plan
    # rather than a from-scratch request.
    known_context: dict | None = None


@app.get("/tracks")
def tracks():
    return list_tracks()


@app.get("/intake-options")
def intake_options(track_id: str):
    return tools.get_intake_options(get_track(track_id))


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
        result = implementation(
            request.track_id,
            messages,
            request.cost_usd,
            state_override=state_override,
            known_context=request.known_context,
        )
        result["llm_steps"] = get_llm_trace()
        return result
    except Exception as e:
        # TEMPORARY: surface the real error for local debugging. Remove
        # before this is ever reachable from outside localhost - a
        # traceback shouldn't be handed to arbitrary callers.
        import traceback

        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()},
        )


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
            result = run_agent_turn_v2(
                request.track_id,
                messages,
                request.cost_usd,
                state_override=state_override,
                known_context=request.known_context,
                on_event=on_event,
            )
            result["llm_steps"] = get_llm_trace()
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
        "group_batch_order_number": os.environ.get("GROUP_BATCH_ORDER", "TODO_batch_order"),
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
            "An academic-advisor agent that plans a Technion student's next semester. "
            "The student describes their situation in free text (Hebrew or English) - semester number, "
            "courses passed or failed, days they cannot attend, preferred pace - and the agent builds a "
            "complete, verified semester plan: courses with per-course reasoning, a clash-free weekly "
            "timetable, an exam calendar (moed A+B), risk analysis, and a roadmap to graduation.\n\n"
            "What it CAN do: plan a semester for the supported Technion tracks (Data & Information "
            "Engineering by default; Information Systems Engineering on request); prioritize retakes of "
            "blocking courses; respect excluded weekdays with section-level precision; balance credits, "
            "workload and exam spacing; answer course questions grounded in real student reviews; and "
            "revise an existing plan minimally on follow-up ('swap course X', 'I have a wedding on an "
            "exam date').\n\n"
            "What it CANNOT do (constraints): it does not register the student to courses or access any "
            "personal Technion account (it works from the public catalog only); it never delivers a plan "
            "containing a schedule collision, an already-passed course, or a course whose prerequisites "
            "are unmet; it caps sport-course filler at one; and when constraints make a full plan "
            "impossible it says so honestly instead of hiding the problem."
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
    prompt: str


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
        # Optional track switch: mention Information Systems to use that track.
        track_id = "SC00001453"
        low = prompt.lower()
        if "information systems" in low or "מערכות מידע" in prompt:
            track_id = "SC00001416"
        result = run_agent_turn_v2(track_id, [{"role": "user", "content": prompt}], 0.0)
        return {
            "status": "ok",
            "error": None,
            "response": _plan_to_text(result),
            "steps": get_llm_trace(),
        }
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "response": None, "steps": get_llm_trace()}
