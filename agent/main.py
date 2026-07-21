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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import tools
from agent_loop import build_state_from_intake, run_agent_turn, summarize_intake_as_message
from agent_react_loop import run_agent_turn_v2
from data_bundle import get_track, list_tracks

# Bump this string on every architecturally-significant change, so a stale
# running process is immediately obvious from GET /health instead of being
# diagnosed after the fact from file timestamps.
AGENT_VERSION = "creative-upgrades-v1-2026-07-21"

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
        return implementation(
            request.track_id,
            messages,
            request.cost_usd,
            state_override=state_override,
            known_context=request.known_context,
        )
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
            result = run_agent_turn_v2(
                request.track_id,
                messages,
                request.cost_usd,
                state_override=state_override,
                known_context=request.known_context,
                on_event=on_event,
            )
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
