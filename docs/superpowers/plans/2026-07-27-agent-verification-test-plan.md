# Agent Verification Test Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three small, runnable test scripts (plus one doc pointer) that give concrete, repeatable evidence for the three things the project owner asked to be sure of: (1) the shipped agent is a genuine model-driven tool-calling loop, not the legacy hardcoded pipeline also present in this codebase, (2) the system runs correctly end-to-end, (3) the agent produces good plans, including on stateful follow-up turns.

**Architecture:** This project already ships two planner implementations behind one interface — `agent_loop.run_agent_turn` ("pipeline": fixed JSON-mode stages, always `Understand → PlannerDraft → [PlanChooser] → Explainer`) and `agent_react_loop.run_agent_turn_v2` ("react": a bounded loop where the model chooses tool calls via real OpenAI `tools=[...]` function-calling, terminating in `deliver_plan`). `main.py` selects between them via `AGENT_MODE_DEFAULT` (env `AGENT_MODE`, defaults to `"react"`) and exposes a per-request `agent_mode` override on `POST /chat` for side-by-side comparison. This plan writes tests that call both implementations directly (in-process, same pattern as the existing `agent/tests/test_scenarios.py`) and inspect their `llm_steps`/`tool_log` traces to prove the architectural and behavioral properties an "agent" (vs. a pipeline) must have. No production code changes.

**Tech Stack:** Python 3.11+, stdlib `unittest`-free plain scripts (matching the existing `agent/tests/test_scenarios.py` style — a `main()` that prints PASS/FAIL and returns an exit code), `urllib` for the one HTTP-level check, no new dependencies.

## Global Constraints

- **No production code changes.** This plan only adds test files and a short doc pointer. If a test uncovers a real bug, stop and report it — do not silently patch `agent/` source as part of this plan.
- **Do not implement on `main`/`master` without explicit user consent.** Create a branch or worktree first (see `superpowers:using-git-worktrees`) unless the user has already said work should happen directly on the current branch.
- **Live-call budget:** every live scenario in this plan reuses `TESTING.md`'s own per-turn cost model (~$0.03-0.08/turn with the project's configured model). Total across all three new/reused live suites is roughly $0.40-0.60 (Task 2: ~3 live calls ≈ $0.10-0.25; Task 3: ~2 live calls ≈ $0.10-0.15; existing `test_scenarios.py react`, reused not rewritten: ~$0.20). Stop and report if any single script's total reported `cost_usd` exceeds 3x this estimate.
- **Live tests require `agent/.env`** with a real `OPENAI_API_KEY` (see `README.md` setup). `test_smoke.py`'s HTTP checks additionally require both dev servers running (`uvicorn main:app --port 8787 --app-dir agent` and `python3 -m http.server 4173 --directory site`).
- **Reuse, don't duplicate:** `agent/tests/test_scenarios.py` already defines `SCENARIOS`, `TRACK_ID`, and a working pattern for calling both implementations with `build_state_from_intake`. Every new file below imports from it rather than redefining scenarios. `agent/tests/run_mocked.sh` already covers deterministic tool logic (54 checks, $0) — do not reimplement it, just invoke it.

---

### Task 1: Smoke test - "does it run correctly"

**Files:**
- Create: `agent/tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing from other tasks (runs first, standalone).
- Produces: nothing consumed by later tasks — this is a leaf check.

- [ ] **Step 1: Write the smoke test script**

```python
"""
Level 0 smoke test - proves the system is alive before any paid checks run.
Requires both dev servers running (see README.md setup):
    uvicorn main:app --port 8787 --app-dir agent
    python3 -m http.server 4173 --directory site
Cost: $0 (no LLM calls). Run: python3 agent/tests/test_smoke.py
"""
import json
import sys
import urllib.error
import urllib.request

BACKEND = "http://127.0.0.1:8787"
FRONTEND = "http://127.0.0.1:4173"


def _get(url: str):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read()
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach {url} - is the server running? ({e})") from e


def test_health() -> list[str]:
    failures = []
    status, data = _get(f"{BACKEND}/health")
    if status != 200:
        failures.append(f"/health returned {status}, expected 200")
        return failures
    if data.get("status") != "ok":
        failures.append(f"/health status field: {data.get('status')!r}, expected 'ok'")
    if data.get("agent_mode_default") != "react":
        failures.append(
            f"/health agent_mode_default: {data.get('agent_mode_default')!r}, expected 'react' "
            "(the legacy pipeline should not be the shipped default)"
        )
    if "version" not in data:
        failures.append("/health missing 'version' field")
    return failures


def test_tracks() -> list[str]:
    failures = []
    status, data = _get(f"{BACKEND}/tracks")
    if status != 200:
        failures.append(f"/tracks returned {status}, expected 200")
        return failures
    if not isinstance(data, list) or len(data) < 2:
        failures.append(f"/tracks returned {data!r}, expected a list of at least 2 tracks")
    return failures


def test_frontend_serves() -> list[str]:
    failures = []
    status, body = _get(FRONTEND)
    if status != 200:
        failures.append(f"frontend at {FRONTEND} returned {status}, expected 200")
        return failures
    if isinstance(body, bytes) and b"<title>" not in body:
        failures.append("frontend HTML missing a <title> tag - page may not have rendered")
    return failures


def main() -> int:
    checks = [test_health, test_tracks, test_frontend_serves]
    total_failures = 0
    for check in checks:
        name = check.__name__
        try:
            failures = check()
        except RuntimeError as e:
            failures = [str(e)]
        if failures:
            total_failures += len(failures)
            print(f"FAIL {name}:")
            for f in failures:
                print(f"  - {f}")
        else:
            print(f"PASS {name}")
    print()
    if total_failures:
        print(f"{total_failures} smoke check failure(s). Fix before running any paid levels.")
        return 1
    print("All smoke checks passed ($0 spent).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it with both dev servers up**

Run: `python3 agent/tests/test_smoke.py`
Expected: `PASS test_health`, `PASS test_tracks`, `PASS test_frontend_serves`, exit code 0. If a server isn't running, the relevant check reports "Could not reach ... is the server running?" rather than a raw traceback.

- [ ] **Step 3: Run the existing mocked suite (reused, not rewritten)**

Run: `bash agent/tests/run_mocked.sh`
Expected: ends with "ALL MOCKED SUITES PASSED" (54 checks, $0).

- [ ] **Step 4: Commit**

```bash
git add agent/tests/test_smoke.py
git commit -m "test: add Level 0 smoke script for backend/frontend liveness"
```

---

### Task 2: Architecture proof - "is it really an agent, not a pipeline"

**Files:**
- Create: `agent/tests/test_agent_architecture.py`

**Interfaces:**
- Consumes: `SCENARIOS`, `TRACK_ID` from `agent/tests/test_scenarios.py` (existing); `build_state_from_intake`, `get_llm_trace`, `run_agent_turn`, `start_llm_trace` from `agent_loop`; `run_agent_turn_v2` from `agent_react_loop`; `AGENT_MODE_DEFAULT`, `AGENT_IMPLEMENTATIONS` from `main`.
- Produces: nothing consumed by later tasks.

**Why these specific checks, grounded in the actual code (not guessed):**
- `agent_react_loop._call_with_tools` calls the OpenAI API with `tools=REACT_TOOL_SCHEMAS, tool_choice="auto"` and logs each response via `record_llm_step("PlanningLoop", messages, {"text": ..., "tool_calls": [...]})` — a `PlanningLoop` step with a non-empty `tool_calls` list is direct proof of genuine per-step model tool selection.
- `agent_loop._call` (used by the pipeline's `Understand`/`PlannerDraft`/`PlanChooser`/`Explainer` stages) calls the OpenAI API with `response_format={"type":"json_object"}` when `json_mode=True` and logs `{"text": content}` only — it never has a `tool_calls` key, because it never passes `tools=[...]`. This is the literal difference between "agent" and "pipeline" in this codebase.
- The pipeline's stage sequence is fixed Python control flow (`run_agent_turn`'s body always proceeds `draft → verify → repair → compare → explain` in the same order); the react loop's `for step in range(MAX_STEPS)` lets the model decide how many iterations and which tools, so step count and tool sequence should differ across scenarios of different difficulty.

- [ ] **Step 1: Write the architecture test script**

```python
"""
Architectural proof: the shipped agent (react mode) makes genuine per-step
OpenAI tool-calling decisions, and its step count / tool sequence vary with
input complexity - it is not a hardcoded draft/verify/repair/compare/explain
pipeline with a fixed shape, even though a pipeline implementation
(agent_loop.run_agent_turn) also exists in this codebase for comparison.

Runs in-process (no server needed, but agent/.env must have OPENAI_API_KEY).
Cost: ~3 live model calls, roughly $0.10-0.25 total (see agent/TESTING.md
Level 1 estimate: ~$0.05/turn).
Run: cd agent && python3 tests/test_agent_architecture.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_loop import (  # noqa: E402
    build_state_from_intake,
    get_llm_trace,
    run_agent_turn,
    start_llm_trace,
)
from agent_react_loop import run_agent_turn_v2  # noqa: E402
from test_scenarios import SCENARIOS, TRACK_ID  # noqa: E402

TRIVIAL = next(s for s in SCENARIOS if s["name"] == "clean_normal_pace")
COMPLEX = next(s for s in SCENARIOS if s["name"] == "heavily_constrained_weekdays")


def _run(implementation, scenario) -> dict:
    start_llm_trace()
    state = build_state_from_intake(scenario["intake"])
    result = implementation(TRACK_ID, messages=[], cost_so_far=0.0, state_override=state)
    result["llm_steps"] = get_llm_trace()
    return result


def test_react_makes_real_tool_calls() -> list[str]:
    failures = []
    result = _run(run_agent_turn_v2, TRIVIAL)
    planning_steps = [s for s in result["llm_steps"] if s["module"] == "PlanningLoop"]
    if not planning_steps:
        failures.append("no 'PlanningLoop' steps found in react mode's llm_steps trace")
        return failures
    tool_calling_steps = [s for s in planning_steps if s["response"].get("tool_calls")]
    if not tool_calling_steps:
        failures.append(
            "no PlanningLoop step had a non-empty 'tool_calls' list - the model never made "
            "a real OpenAI tool-calling decision"
        )
    return failures


def test_pipeline_never_uses_tool_calling() -> list[str]:
    failures = []
    result = _run(run_agent_turn, TRIVIAL)
    steps_with_tool_calls = [s for s in result["llm_steps"] if s["response"].get("tool_calls")]
    if steps_with_tool_calls:
        failures.append(
            f"pipeline mode produced {len(steps_with_tool_calls)} step(s) with tool_calls - "
            "expected zero, since agent_loop.run_agent_turn never passes tools=[...]"
        )
    return failures


def test_react_step_count_and_tool_sequence_vary_with_complexity() -> list[str]:
    failures = []
    trivial_result = _run(run_agent_turn_v2, TRIVIAL)
    complex_result = _run(run_agent_turn_v2, COMPLEX)

    trivial_steps = sum(1 for s in trivial_result["llm_steps"] if s["module"] == "PlanningLoop")
    complex_steps = sum(1 for s in complex_result["llm_steps"] if s["module"] == "PlanningLoop")

    trivial_tool_seq = tuple(t["name"] for t in trivial_result["tool_log"])
    complex_tool_seq = tuple(t["name"] for t in complex_result["tool_log"])

    if trivial_steps == complex_steps and trivial_tool_seq == complex_tool_seq:
        failures.append(
            f"identical step count ({trivial_steps}) AND identical tool sequence for a trivial "
            f"vs. heavily-constrained scenario - suspicious for a model-driven loop; "
            f"trace: {trivial_tool_seq}"
        )
    return failures


def test_health_default_mode_is_react() -> list[str]:
    import main  # noqa: E402 - only inspects module-level constants, no server needed

    failures = []
    if main.AGENT_MODE_DEFAULT != "react":
        failures.append(
            f"AGENT_MODE_DEFAULT is {main.AGENT_MODE_DEFAULT!r}, expected 'react' - the shipped "
            "default must be the agent, not agent_loop's pipeline"
        )
    if main.AGENT_IMPLEMENTATIONS.get("react") is not run_agent_turn_v2:
        failures.append("main.AGENT_IMPLEMENTATIONS['react'] is not agent_react_loop.run_agent_turn_v2")
    return failures


def run_all_checks() -> int:
    checks = [
        test_health_default_mode_is_react,  # $0, run first
        test_react_makes_real_tool_calls,
        test_pipeline_never_uses_tool_calling,
        test_react_step_count_and_tool_sequence_vary_with_complexity,
    ]
    total_failures = 0
    for check in checks:
        name = check.__name__
        print(f"--- {name} ---")
        try:
            failures = check()
        except Exception as e:
            failures = [f"raised {type(e).__name__}: {e}"]
        if failures:
            total_failures += len(failures)
            for f in failures:
                print(f"  FAIL: {f}")
        else:
            print("  PASS")
    print()
    if total_failures:
        print(f"{total_failures} failure(s) - this system may not be a real agent.")
        return 1
    print("All architecture checks passed: react mode is a genuine tool-calling agent.")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_checks())
```

- [ ] **Step 2: Run it**

Run: `cd agent && python3 tests/test_agent_architecture.py`
Expected: all four checks PASS, exit code 0. `test_health_default_mode_is_react` costs $0 and should be checked first even on failure of later (paid) checks.

- [ ] **Step 3: Commit**

```bash
git add agent/tests/test_agent_architecture.py
git commit -m "test: prove react mode is genuine tool-calling, not the legacy pipeline"
```

---

### Task 3: Plan-quality proof, including stateful revision - "does it give good plans"

**Files:**
- Create: `agent/tests/test_agent_quality_live.py`

**Interfaces:**
- Consumes: `SCENARIOS`, `TRACK_ID` from `agent/tests/test_scenarios.py`; `build_state_from_intake` from `agent_loop`; `run_agent_turn_v2` from `agent_react_loop` (its exact contract: returns `known_context = {"state": {...}, "previous_plan": [...], "previous_issues": [...]}`, which a follow-up call passes back via the `known_context=` kwarg to get revision behavior instead of a from-scratch replan); `check_invariants` from `tools`; `get_track` from `data_bundle`.
- Produces: nothing consumed by later tasks.

The six one-shot scenarios in `test_scenarios.py` already check the acceptance contract (no passed/duplicate/unknown course, retake always present) for single-turn plans. What they do not cover is **TESTING.md Level 3.2 — exam-conflict revision**: a follow-up message must not re-ask settled facts and must change only what's actually broken, not silently rebuild the whole plan. That is the one live behavior this task adds.

- [ ] **Step 1: Write the quality/revision test script**

```python
"""
Live plan-quality proof: 'the agent runs and gives good plans' per
agent/TESTING.md's acceptance contract, focused on what the six-scenario
regression suite (test_scenarios.py) does NOT cover - multi-turn revision
continuity (TESTING.md Level 3.2, exam-conflict revision).

Run test_scenarios.py react separately for the broader one-shot quality net
(~$0.20, six scenarios); this file adds one stateful follow-up scenario
(~$0.10-0.15, 2 live calls).

Run: cd agent && python3 tests/test_agent_quality_live.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools  # noqa: E402
from agent_loop import build_state_from_intake  # noqa: E402
from agent_react_loop import run_agent_turn_v2  # noqa: E402
from data_bundle import get_track  # noqa: E402
from test_scenarios import SCENARIOS, TRACK_ID  # noqa: E402

SEED_SCENARIO = next(s for s in SCENARIOS if s["name"] == "retake_that_is_also_a_foundation")


def _first_exam_date(plan_result: dict):
    for course in plan_result["courses"]:
        moed_a = course.get("exams", {}).get("moed_a") or []
        if moed_a:
            return course["course_number"], moed_a[0]["date"]
    return None


def test_exam_conflict_revision_is_minimal_and_settled() -> list[str]:
    failures = []
    track = get_track(TRACK_ID)
    state = build_state_from_intake(SEED_SCENARIO["intake"])

    first = run_agent_turn_v2(TRACK_ID, messages=[], cost_so_far=0.0, state_override=state)
    if first["plan_result"] is None:
        return ["initial turn produced no plan_result - cannot test revision"]

    target = _first_exam_date(first["plan_result"])
    if target is None:
        return ["initial plan had no course with a moed_a exam date to conflict with"]
    conflict_course, conflict_date = target
    conflict_name = track.courses[conflict_course]["name"]
    original_courses = {c["course_number"] for c in first["plan_result"]["courses"]}

    followup_text = (
        f"I have a wedding on {conflict_date} and can't sit the {conflict_name} exam that day - "
        "please fix it, keep the rest of the plan the same."
    )
    second = run_agent_turn_v2(
        TRACK_ID,
        messages=[{"role": "user", "content": followup_text}],
        cost_so_far=first["cost_usd"],
        known_context=first["known_context"],
    )

    if second["needs_input"] is not None:
        failures.append(
            f"revision turn re-asked for input ({second['needs_input']['missing_fields']}) instead "
            "of treating the message as feedback on the existing plan"
        )
    if second["plan_result"] is None:
        return failures + ["revision turn produced no plan_result"]

    new_courses = {c["course_number"] for c in second["plan_result"]["courses"]}
    if conflict_course in new_courses:
        failures.append(
            f"conflicting course {conflict_course} ({conflict_name}) is still in the revised plan "
            f"unchanged despite the reported exam-date conflict on {conflict_date}"
        )

    kept = original_courses & new_courses
    min_expected_overlap = max(1, (len(original_courses) - 1) // 2)
    if len(kept) < min_expected_overlap:
        failures.append(
            f"only {len(kept)}/{len(original_courses)} original courses kept after a single "
            f"course-swap request - looks like a from-scratch replan, not a minimal revision "
            f"(kept: {sorted(kept)}, original: {sorted(original_courses)})"
        )

    violations = tools.check_invariants(
        track,
        list(new_courses),
        SEED_SCENARIO["intake"]["passed_courses"],
        SEED_SCENARIO["intake"]["failed_courses"],
    )
    failures.extend(violations)

    return failures


def main() -> int:
    checks = [test_exam_conflict_revision_is_minimal_and_settled]
    total_failures = 0
    for check in checks:
        name = check.__name__
        print(f"--- {name} ---")
        try:
            failures = check()
        except Exception as e:
            failures = [f"raised {type(e).__name__}: {e}"]
        if failures:
            total_failures += len(failures)
            for f in failures:
                print(f"  FAIL: {f}")
        else:
            print("  PASS")
    print()
    if total_failures:
        print(f"{total_failures} failure(s).")
        return 1
    print("Revision-continuity check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it**

Run: `cd agent && python3 tests/test_agent_quality_live.py`
Expected: `PASS test_exam_conflict_revision_is_minimal_and_settled`, exit code 0.

- [ ] **Step 3: Run the existing broader regression suite (reused, not rewritten)**

Run: `cd agent && python3 tests/test_scenarios.py react`
Expected: `All 6 scenarios passed (react)`.

- [ ] **Step 4: Commit**

```bash
git add agent/tests/test_agent_quality_live.py
git commit -m "test: add exam-conflict revision-continuity check for plan quality"
```

---

### Task 4: Point TESTING.md at the new scripts

**Files:**
- Modify: `TESTING.md` (repo root)

**Interfaces:**
- Consumes: file paths created in Tasks 1-3.
- Produces: nothing (documentation only).

- [ ] **Step 1: Add a short pointer section right after the Level 0 table**

Insert this block into `TESTING.md` immediately after the `**Gate: all 5 pass or stop here.**` line that closes Level 0:

```markdown

## Automated proof scripts (in addition to the levels above)

| Script | Proves | Cost |
|---|---|---|
| `agent/tests/test_smoke.py` | Backend/frontend are alive (same as Level 0, scripted) | $0 |
| `agent/tests/test_agent_architecture.py` | react mode makes genuine OpenAI tool-calling decisions and its step count/tool sequence vary with input complexity; the legacy pipeline mode never does | ~$0.10-0.25 |
| `agent/tests/test_agent_quality_live.py` | A follow-up exam-conflict message gets a minimal, settled revision, not a from-scratch replan (Level 3.2) | ~$0.10-0.15 |
```

- [ ] **Step 2: Verify the doc renders sensibly**

Run: `grep -A 8 "Automated proof scripts" TESTING.md`
Expected: the table above appears once, directly after the Level 0 gate line.

- [ ] **Step 3: Commit**

```bash
git add TESTING.md
git commit -m "docs: point TESTING.md at the automated agent-vs-pipeline proof scripts"
```

---

## Self-Review

**Spec coverage:** "really an agent not a pipeline" → Task 2 (four checks: real tool-calling present in react mode, absent in pipeline mode, step count/sequence varies with complexity, default mode is react). "runs correctly" → Task 1 (health, tracks, frontend, existing mocked suite). "gives good plans" → Task 3 (new revision-continuity check) plus the existing `test_scenarios.py react` reused as-is (not rewritten, per Global Constraints). Task 4 wires it all into the doc a human would actually read next. No gaps identified.

**Placeholder scan:** every step contains complete, real code grounded in the actual source (`agent_react_loop.py`, `agent_loop.py`, `main.py`, `tool_schemas.py`, `test_scenarios.py` were all read directly, not guessed) or an exact shell command. No TBD/TODO/"add appropriate handling" anywhere.

**Type consistency:** `run_agent_turn` / `run_agent_turn_v2` are always called with the exact signature already proven live in `test_scenarios.py` (`implementation(TRACK_ID, messages=[], cost_so_far=0.0, state_override=state)`); `known_context` is passed through exactly as `run_agent_turn_v2` returns it; `llm_steps` entries are read as `{"module", "prompt", "response"}` and `tool_log` entries as `{"name", "args", "result"}`, matching `agent_loop.record_llm_step` and `agent_react_loop.py`'s `tool_log.append(...)` calls respectively.
