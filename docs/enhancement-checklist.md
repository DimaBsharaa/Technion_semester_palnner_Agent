# Enhancement Checklist

Combined list from the architecture review + team brainstorm on 2026-08-05.
Items 0-3 shipped 2026-08-18/19; item 2 was reworked and item 4/6 got a
major real-data pass on 2026-08-19/20 - see the "shipped" notes under each.
Check items off as they land; update the "Open question" notes as
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

## 2. Grade-improvement retakes

**Shipped 2026-08-19, reworked from a fixed threshold into a genuine
propose→approve loop after live critique that the original version wasn't
actually agentic.** `tools.analyze_grade_improvement_candidates` (renamed
from `suggest_grade_improvements` - returns raw comparison data, course
average AND the student's own average, no hardcoded "worth suggesting"
threshold, that judgment moved entirely to the model) + `pipeline/histogram_client.py`
(real technion-histograms data, verified live) + `pipeline/backfill_grade_stats.py`
(already run against all 3 real `data/track_*.json` bundles).

- [x] The agent may PROPOSE a retake (`deliver_plan`'s `proposed_retake`
  field) and ask directly whether to include it next turn - the one
  deliberate exception to "never end with a question." Encouraged to also
  call `summarize_cheesefork` on a promising candidate first (real
  multi-step reasoning, not a single lookup) before proposing.
- [x] A student can also bring up a retake themselves, unprompted, with no
  prior proposal (`requested_retake_course` at extraction time) - handled
  identically to an approved proposal from there.
- [x] Either path sets `state["approved_retake_course"]`, the single
  narrow, single-turn exception to "never re-include a passed course"
  (`tools.verify_plan`/`check_invariants`). **Hard-enforced, not just a
  prompt suggestion** - found live, a one-shot nudge alone wasn't reliable
  enough: a one-shot nudge first, then code-level enforcement, checked
  across EVERY delivery path (normal `deliver_plan` call, forced wrap-up,
  the final safety swap) so an approved retake can't silently vanish
  depending on which path the loop happened to resolve through - see
  `approved_retake_guaranteed`/`approved_retake_enforced` in
  `agent_react_loop.py`. Shows the same RETAKE badge as a failed-course
  retake (`is_retake`).
- Open question — the actual retake rule: **still genuinely open, intentionally
  not implemented.** Technion's real "שיפור ציון" (grade improvement) policy
  has specific eligibility constraints (which courses qualify, how many
  attempts allowed, which grade counts, time limits) that this system does
  not know or verify. The agent never claims guaranteed eligibility - it
  always frames a proposal as worth confirming with the registrar.

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

**Major data-quality pass, 2026-08-19/20**, prompted by the user uploading
the official DDS ("track diagram") PDFs for all 3 tracks and cross-checking
the app's output against them directly:

- [x] **Official per-course semester placement, digitized by hand from each
  track's diagram** (every semester's credit total cross-checked against the
  diagram's own printed total - all matched exactly) - see
  `Track.official_semester` in `agent/data_bundle.py`. Overrides the
  prerequisite-depth heuristic wherever known; the heuristic still fills gaps
  for courses not yet transcribed. Directly fixed multiple live bugs the
  heuristic alone couldn't: courses silently mis-bucketed into the wrong
  "semester," a semester-4 course auto-marked already-passed for a
  semester-3 student purely because it was someone else's transitive
  prerequisite.
- [x] **Confirmed Technion's requirement-tree API is neither complete nor
  precise** - both directions found live: it's missing real mandatory
  courses the diagrams show (12+ courses fetched by hand across the 3
  tracks), AND it separately over-includes some courses as "Mandatory" that
  the diagrams never list at all (duplicate course-number variants of the
  same real subject - e.g. two different course numbers both meaning
  "Discrete Math," only one of which the diagram actually requires). See
  `_CONFIRMED_NON_REQUIREMENTS` in `data_bundle.py` for the per-track,
  per-course exclusions, each verified against that track's own diagram
  before being added.
- [x] **Fixed a real bug in the fetch pipeline itself**:
  `pipeline/technion_api.py`'s `send_request` retried a failed query FOREVER
  (5-minute backoff, indefinitely) on "empty response" - which is what the
  API returns for "this course isn't offered in this semester," a real
  permanent answer, not a transient network error. This is very likely why
  earlier data-fetch attempts silently never completed. Added
  `EmptyResponseError`, never retried, so the caller's own semester-fallback
  loop gets it immediately instead of hanging.
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

- [x] **הנדסת תעשיה וניהול (Industrial Engineering & Management) shipped
  2026-08-20** - previously excluded from `/tracks` entirely
  (`list_tracks()` only lists tracks with a non-empty flat
  `mandatory_course_numbers`, and this track's SAP tree has none). Digitizing
  its official diagram gave it real `official_semester` data for every
  course through semester 8, which populates `mandatory_course_numbers`
  directly (union of the tree-derived set and `official_semester`'s keys) -
  no separate data-model change needed after all, the diagram data itself
  turned out to be enough of a flat mandatory-course source. Smoke-tested: a
  real plan builds successfully.
  - [ ] **Still not modeled**: the diagram's semester 5+ "specialization
    chain" requirement (3 courses from a track-specific chain by degree end,
    per the diagram's own red-boxed note) isn't represented at all - one
    course (`00960324`) is deliberately left untagged rather than guessed,
    since its exact semester couldn't be pinned down without contradicting
    the diagram's own printed semester-6 total. Semesters 1-4 for this
    track are fully accurate; later-semester chain planning is not.
- [ ] `pipeline/fetch_track_bundle.py` already takes `--faculty`/`--track`
  as arguments — expanding coverage to a genuinely new track (beyond the 3
  now supported) is running it again and committing the output, then
  digitizing that track's own diagram the same way (see item 4) rather than
  trusting the tree walk alone - now the established, proven pattern.

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
in isolation. Item 4's real-data pass and item 6's third track then shipped
together (2026-08-19/20), prompted directly by the user uploading the
official diagrams for all 3 tracks. Status as of 2026-08-20:

1. ~~Item 0 (personal API key)~~ — **done**.
2. ~~Item 1 (transcript ingestion)~~ — **done**; real-layout validation still open.
3. ~~Item 3 (persistent memory)~~ — **done**, verified against real Supabase.
4. ~~Item 2 (grade-improvement retakes)~~ — **done**, reworked into a real
   propose→approve loop with hard cross-path enforcement.
5. Item 4 (data freshness) — **mostly done**: official per-course semester
   data digitized for all 3 tracks, real tree over/under-inclusion bugs
   found and fixed, a real fetch-pipeline retry bug fixed. A written runbook
   for periodic full refreshes still doesn't exist.
6. ~~Item 6 (more tracks)~~ — **done** for a 3rd track (הנדסת תעשיה וניהול);
   its semester 5+ specialization-chain requirement still isn't modeled.
7. Item 5 (new actions) — not started, additive, low urgency.
8. Item 7 (API endpoints) — the required course-spec endpoints all exist
   and are deployed; the "what else might be wanted" open question is
   otherwise unresolved.
9. Item 8 (system prompt) — no action planned, recommendation unchanged.
