# Enhancement Checklist

Combined list from the architecture review + team brainstorm on 2026-08-05.
Items 0-3 have since shipped (2026-08-18/19) - see the "shipped" notes under
each. Check items off as they land; update the "Open question" notes as
decisions get made instead of deleting the history.

## 0. Testing setup (unblocks everything else)

**Shipped 2026-08-17.** Running on a personal OpenAI key (`gpt-5.4-mini`,
`OPENAI_BASE_URL` empty), verified live against the real API.

- [x] **Switch to a personal API key for free live testing.** `agent/.env.example`
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

**Shipped 2026-08-18.** `agent/transcript_parser.py` (`pdfplumber`, text-layer
only, no OCR) + `POST /transcript` + a frontend upload option inside the
existing gap-fill course checklist. Feeds `intake.passed_courses`/
`failed_courses`/`grades` through the exact existing intake pipeline.
Grade capture also works from free text now (not just the PDF) - see item 2.
Tested against a hand-fabricated fixture only, per the privacy rule below -
**not yet validated against a real Technion transcript's exact layout**;
that's still a real open item if you want full confidence in the parser.

- [x] Parse an official Technion grade sheet PDF (the format the attached
  example is in — Hebrew, fixed columns: course number / name / points /
  grade / semester) directly into the same structured shape
  `agent_loop.build_state_from_intake` already consumes: `passed_courses`,
  `failed_courses`, and per-course grades.
- [x] This **replaces the least reliable step in the whole system**
  (`_extract_student_state`'s free-text guessing) with exact ground truth
  whenever a student has the PDF — same philosophy that already motivated
  `build_state_from_intake` for the structured form.
- [x] A course row with grade below 55 (fail) or literally marked
  `Pass`/`Exemption without points` → `passed_courses`/`failed_courses`
  accordingly. Raw grade kept per course — used by item 2.
- [x] **Do not persist or log the raw PDF or its extracted text anywhere in
  the repo.** It contains name + national ID + full grade history. Process
  in memory for the request, store only what the planning logic needs
  (course numbers + pass/fail, optionally grade if item 2 ships), never the
  ID number. No test fixture may ever contain a real transcript — synthesize
  a fake one for tests. (Confirmed followed: no real transcript was ever
  written to any file, fixture, or commit during implementation.)
- Open question, still open: PDF text extraction was only verified against a
  hand-built fake PDF (via `reportlab`) - Technion's real layout (multi-line
  wrapped course names, watermark/signature text) may extract differently.
  First real upload through the running app is the real test.

## 2. Grade-improvement retake suggestions

**Shipped 2026-08-19.** `tools.suggest_grade_improvements` + `pipeline/histogram_client.py`
(real technion-histograms data, verified live) + `pipeline/backfill_grade_stats.py`
(already run against all 3 real `data/track_*.json` bundles - 115/123,
128/139, 126/137 courses got real grade_stats). Preinjected (like
`assess_progress`) only when the student has ≥1 known grade, so it costs
nothing on the common turn where no grade was ever given. The "never
hardcode the real retake-eligibility rule" caution below was followed
exactly - it's phrased in the prompt as an advisory, "worth checking with
the registrar" suggestion, never auto-added to the delivered plan.

- [x] New advisory tool, `tools.suggest_grade_improvements(track, passed_courses, grades)`:
  flags a **passed** course as a *candidate* (not a requirement) for retake
  when its grade is meaningfully below some benchmark (10 points below the
  historical average, `GRADE_IMPROVEMENT_GAP_THRESHOLD` in `tools.py`).
- [x] The missing data piece: a per-course grade distribution/average to
  compare against — this is exactly what
  [technion-histograms](https://github.com/michael-maltsev/technion-histograms)
  provides and CheeseFork doesn't. Added as a new pipeline source
  (`pipeline/histogram_client.py`, same shape as `cheesefork_client.py`) and
  bundled as `grade_stats` per course alongside `cheesefork` in
  `data/track_*.json`.
- Open question — the actual retake rule: **still genuinely open, intentionally
  not implemented.** Technion's real "שיפור ציון" (grade improvement) policy
  has specific eligibility constraints (which courses qualify, how many
  attempts allowed, which grade counts, time limits) that this system does
  not know or verify. The shipped version deliberately never claims
  guaranteed eligibility and never auto-includes the retake in the delivered
  plan - only ever a labeled, dismissible suggestion in the explanation.

## 3. Persistent chat / student memory across sessions

**Shipped 2026-08-18, verified against a real (not mocked) Supabase project
2026-08-19.** `agent/student_store.py` (Supabase REST, `sessions` table,
keyed by SHA-256 of the student's email, hashed client-side - raw email
never reaches the server) + `main._recall_known_context` (automatic recall
on a brand-new conversation, reusing the existing `known_context`/
`merge_known_state` continuity machinery rather than a new system) +
`previous_explanation` threading so the agent can answer "what did you
recommend last time?" directly. Identity model decided: email-as-lookup-key,
no password, no login screen - matches the course spec's no-auth-guard
requirement (see README's "Session persistence" note for the honest
security boundary this implies).

- [x] Currently fully stateless server-side — the client re-sends the whole
  `messages` array and `known_context` every request (`main.py`'s
  `ChatRequest`). A returning student starts cold. *(No longer true on a
  brand-new conversation when a student_key is present - see above.)*
- [x] Add a lightweight server-side store keyed by some student identifier
  (not the national ID — see privacy note) holding: conversation history,
  resolved state, last delivered plan. *(Supabase, not SQLite - already
  required by the course spec as the primary database, so used directly
  rather than adding a second storage system.)*
- [x] Decide the identity model before building storage — decided: hashed
  email, no login (see above).
- [x] **Privacy**: transcript-derived data (item 1) and persistent storage
  (item 3) do now combine (grades flow into the same saved state) - treated
  as PII throughout: never logged, never in client-visible debug output
  beyond the same fields the student themselves already provided.

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
- [x] **Partially shipped 2026-08-19**: `agent/live_offering_check.py` does a
  real-time check (existence only - "is this course still offered this
  semester") against Technion's own live system on the final 4-8 candidate
  courses, right before delivery - so a course closed/cancelled since the
  last bundle refresh doesn't get confidently recommended anyway.
  Deliberately its own minimal `urllib`-based module rather than importing
  `pipeline/technion_api.py` (which has an unbounded retry loop, fine for an
  offline script, unsafe inside a live request under Vercel's 300s cap; also
  `pipeline/` isn't bundled for Vercel deployment at all). Fail-open: a
  course is only ever dropped when positively confirmed gone, never on a
  timeout/network issue.
  - [ ] Still open: actual **seat count** (open/closed with N seats left) is
    a different, harder question than "is it offered at all" - not
    attempted. Would need investigating what `technion_api.py`'s OData
    service exposes for enrollment counts specifically.
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

Actual build order ended up different from this original plan - items 1, 3,
and 2 shipped together (2026-08-18/19) because building memory first made
transcript/grade data worth persisting, rather than each being useful only
in isolation. Status as of 2026-08-19:

1. ~~Item 0 (personal API key)~~ — **done**.
2. ~~Item 1 (transcript ingestion)~~ — **done**; real-layout validation still open.
3. ~~Item 3 (persistent memory)~~ — **done**, verified against real Supabase.
4. ~~Item 2 (grade-improvement suggestions)~~ — **done**.
5. Item 4 (data sync runbook) — **partially done**: the live offering-check
   covers "is this course still offered," a written runbook for periodic
   full refreshes is still not written.
6. Item 6 (more same-shaped tracks) — not started; still cheap, same pipeline.
7. Item 5 (new actions) — not started, additive, low urgency.
8. Item 7 (API endpoints) — the required course-spec endpoints all exist
   and are deployed; the "what else might be wanted" open question is
   otherwise unresolved.
9. Item 8 (system prompt) — no action planned, recommendation unchanged.
