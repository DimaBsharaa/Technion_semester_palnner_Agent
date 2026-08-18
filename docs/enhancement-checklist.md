# Enhancement Checklist

Combined list from the architecture review + team brainstorm on 2026-08-05.
Nothing here is implemented yet — this is the plan to work from, in
proposed priority order. Check items off as they land; update the "Open
question" notes as decisions get made instead of deleting the history.

## 0. Testing setup (unblocks everything else)

- [ ] **Switch to a personal API key for free live testing.** `agent/.env.example`
  already documents this — it's a config swap, not new infra:
  ```
  OPENAI_API_KEY=sk-...your own key...
  OPENAI_BASE_URL=            # leave EMPTY to use real OpenAI, not the course LLMod proxy
  OPENAI_MODEL=<a real model name available on your account>
  ```
  Steps: create an account at platform.openai.com (or whichever provider), add a
  payment method, **set a hard monthly spend limit in Billing settings** (this
  is the real "credit card limit" control), create an API key, paste it into
  `agent/.env`.
  - [ ] Also update `OPENAI_PRICE_INPUT_PER_1M` / `OPENAI_PRICE_OUTPUT_PER_1M`
    in `.env` to match the real model's actual published pricing — the
    in-app cost badge reads these env vars directly (`agent_loop.py`), so
    leaving the LLMod placeholder prices in place will show a wrong number
    once the model changes.
  - Open question: which model? The course used `gpt-5.4-mini` through the
    LLMod proxy — confirm the equivalent name/tier available on a personal
    account before assuming it's identical.

## 1. Transcript ingestion (highest value, no new infra)

- [ ] Parse an official Technion grade sheet PDF (the format the attached
  example is in — Hebrew, fixed columns: course number / name / points /
  grade / semester) directly into the same structured shape
  `agent_loop.build_state_from_intake` already consumes: `passed_courses`,
  `failed_courses`, and per-course grades.
- [ ] This **replaces the least reliable step in the whole system**
  (`_extract_student_state`'s free-text guessing) with exact ground truth
  whenever a student has the PDF — same philosophy that already motivated
  `build_state_from_intake` for the structured form.
- [ ] A course row with grade below 55 ("נכשל"/fail) or literally marked
  fail → `failed_courses`. Passed ("עובר"/exempt "פטור" too) → `passed_courses`.
  Keep the raw grade per course — needed for item 2 below.
- [ ] **Do not persist or log the raw PDF or its extracted text anywhere in
  the repo.** It contains name + national ID + full grade history. Process
  in memory for the request, store only what the planning logic needs
  (course numbers + pass/fail, optionally grade if item 2 ships), never the
  ID number. No test fixture may ever contain a real transcript — synthesize
  a fake one for tests.
- Open question: PDF text extraction — Technion transcripts are a fixed
  layout, so a plain text-layer parser (pdfplumber/pypdf) should work without
  OCR; confirm once one real (redacted) sample is tested against.

## 2. Grade-improvement retake suggestions

- [ ] New advisory tool, e.g. `suggest_grade_improvements(track, passed_with_grades)`:
  flags a **passed** course as a *candidate* (not a requirement) for retake
  when its grade is meaningfully below some benchmark.
- [ ] The missing data piece: a per-course grade distribution/average to
  compare against — this is exactly what
  [technion-histograms](https://github.com/michael-maltsev/technion-histograms)
  provides and CheeseFork doesn't. Add it as a new pipeline source
  (`pipeline/histogram_client.py`, same shape as `cheesefork_client.py`) and
  bundle `avg_grade`/`median_grade` per course alongside `cheesefork` in
  `data/track_*.json`.
- Open question — the actual retake rule: Technion's real "שיפור ציון"
  (grade improvement) policy has specific eligibility constraints (which
  courses qualify, how many attempts allowed, which grade counts, time
  limits). **Don't hardcode a guessed threshold** — this needs the real
  regulation text before it ships as anything stronger than "advisory,
  clearly labeled as not verified against official policy." Suggest scoping
  v1 as: surface it only as a labeled suggestion in the explanation/risk
  report, never auto-include the retake in the delivered plan, and never
  claim it's guaranteed-eligible.

## 3. Persistent chat / student memory across sessions

- [ ] Currently fully stateless server-side — the client re-sends the whole
  `messages` array and `known_context` every request (`main.py`'s
  `ChatRequest`). A returning student starts cold.
- [ ] Add a lightweight server-side store keyed by some student identifier
  (not the national ID — see privacy note) holding: conversation history,
  resolved state, last delivered plan. SQLite is enough at this scale; no
  need for a hosted DB.
- [ ] Decide the identity model before building storage: anonymous
  browser-local ID (simplest, matches current "Continue / Start fresh"
  localStorage pattern in `site/index.html`) vs. a real login. Given this
  has no auth today, browser-local ID + optional "save my plan" is the
  lower-risk v1; a real account system is a much bigger scope add.
- [ ] **Privacy**: if transcript upload (item 1) and persistent storage
  (item 3) combine, the stored profile could contain a student's real
  course history tied to an identifier — treat that store like it holds PII
  even without the ID number attached, and don't let it leak into any
  client-visible debug/trace output.

## 4. Data freshness / sync pipeline

- [ ] Problem as described: track bundles were hand-run/copied once rather
  than kept in a repeatable, documented refresh process — so "what happens
  next year" currently has no answer.
- [ ] `pipeline/fetch_track_bundle.py` already does the right thing
  end-to-end (fetch → cache → write `data/track_*.json`) — the gap isn't
  the fetch logic, it's that there's no **runbook or schedule** for re-running
  it. Add:
  - [ ] A short `pipeline/README.md`: exact command per track, when to run
    it (start of every semester, before the course-selection period opens),
    and what to check in the diff before committing refreshed data.
  - [ ] Consider a `--diff-only` or summary mode that reports what changed
    vs. the last committed bundle (new courses, removed courses, schedule
    changes) so a re-run is reviewable, not a silent full overwrite.
- [ ] Live seat/offering status (open/closed, seats left) is a **separate,
  harder problem** from the yearly catalog refresh — that's near-real-time
  data, not something to bake into the static bundle. Needs its own
  investigation of what `technion_api.py`'s OData service actually exposes
  for enrollment counts (only the first ~60 lines were read so far — check
  the rest before promising this is available) before committing to it as a
  feature.
- Sources named for this: [CheeseFork](https://github.com/michael-maltsev/cheese-fork)
  (already integrated) and [technion-histograms](https://github.com/michael-maltsev/technion-histograms)
  (not yet integrated — see item 2).

## 5. New agent actions beyond "show a plan"

- `.ics` calendar export already exists (📅 button, per `TESTING.md` 2.5).
  New ideas, roughly in order of how self-contained they are:
  - [ ] **Email-the-plan** (send the same explanation + course list to the
    student's email) — no external auth needed, just an outbound mail
    provider.
  - [ ] **Direct Google Calendar sync** (vs. downloading an .ics) — needs
    OAuth, meaningfully more infra than the current export.
  - [ ] **Add-drop deadline reminder** — needs a scheduling/notification
    layer that doesn't exist yet (this is a bigger add, pairs naturally
    with item 3's persistent storage since it needs somewhere to remember
    "remind this student on date X").
  Recommend starting with email export if this is wanted at all — smallest
  new moving part, reuses the plan data structure that already exists.

## 6. More tracks (not just Data Eng / Info Systems Eng)

- [ ] `pipeline/fetch_track_bundle.py` already takes `--faculty`/`--track`
  as arguments — expanding coverage for tracks with the same flat
  mandatory-list shape is just running it again per track and committing
  the output, no code change needed.
- [ ] Tracks built around **specialization chains** instead of a flat
  mandatory list (confirmed in `data_bundle.list_tracks`'s comment:
  Industrial Engineering & Management is one) are structurally excluded
  right now — `Track.__init__` assumes one universal mandatory-course set.
  Supporting those needs an actual data-model change in `data_bundle.py`,
  not just a new fetch run. Scope this as a separate, bigger task from
  "just add more of the same-shaped tracks."

## 7. API endpoints — what exists, what a spec might ask for

Current surface (`agent/main.py`): `GET /tracks`, `GET /intake-options`,
`POST /chat`, `POST /chat/stream` (NDJSON live steps), `GET /health`, plus
course-spec-required `GET /`, `GET /api/team_info`, `GET /api/agent_info`,
`GET /api/model_architecture`, `POST /api/execute`.

- Open question: what specifically prompted "API endpoints - idk what that
  means" — a rubric line item, or genuine interest in exposing this for
  external consumption? If it's the latter, worth deciding things like: does
  a new endpoint need auth/rate-limiting before being public, should
  `/chat` support pagination for long conversations, etc. Nothing to build
  until the actual need is clearer.

## 8. System prompt size

- [ ] **Recommendation: leave it, don't prioritize shrinking it.** The
  react-loop system prompt (`agent_react_loop._system_prompt`) is long
  (~150 lines), but nearly every clause in it traces back to a specific,
  named, previously-observed live failure in its own comments (sport-course
  padding, dropped retakes, exam clustering, etc.). Input tokens are also
  the cheap side of the pricing (`PRICE_INPUT_PER_1M=$0.25` vs.
  `PRICE_OUTPUT_PER_1M=$2.00` in `.env.example`) — prompt length isn't
  the dominant cost driver here, output length and step count are.
  If cost ever needs trimming, measure first (the mocked/live test suite
  already exists to check for regressions) rather than cutting rules
  blind — removing a clause without testing risks reintroducing the exact
  bug it was written to prevent.

## Suggested build order

1. Item 0 (personal API key) — unblocks free live testing of everything else.
2. Item 1 (transcript ingestion) — biggest single quality win, no new infra.
3. Item 4 (data sync runbook) — prevents "next year is broken," mostly docs + a script flag.
4. Item 6 (more same-shaped tracks) — cheap, same pipeline, just more runs.
5. Item 3 (persistent memory) — real infra decision, do after the above are stable.
6. Item 2 (grade-improvement suggestions) — needs the histogram data source first.
7. Item 5 (new actions) — additive, low urgency.
8. Item 7 (API endpoints) — only once the actual need is known.
9. Item 8 (system prompt) — no action planned.
