# Semester Planning Agent

An AI agent that plans a Technion semester like an experienced academic advisor.
A student describes their situation in free text ("starting semester 5, failed
Data Structures, can't come on Sundays") and the agent builds, verifies, and
revises a personalized plan: courses, weekly timetable, exam calendar, risks —
with honest trade-offs when something can't be satisfied.

**Start here: read [`REPORT.pdf`](REPORT.pdf)** — the story,
the architecture diagram, the data, the tools, testing. **[`BUILD_RECORD.pdf`](BUILD_RECORD.pdf)**
logs what was built and fixed session by session. **[`TESTING.md`](TESTING.md)**
has a leveled test plan with exact prompts to try.

## Setup (5 minutes)

Requirements: Python 3.11+, `pip install fastapi uvicorn openai pdfplumber python-multipart`
(the last two are for transcript-PDF upload — see `agent/requirements.txt`).

1. **API key:** copy the template and paste in the group's shared LLMod.ai
   key (see `agent/.env.example` for the exact base URL/model to pair with
   it). Deliberately not committed to the repo:

   ```bash
   cp agent/.env.example agent/.env
   # edit agent/.env and fill OPENAI_API_KEY
   ```

   `agent/.env` is gitignored — real keys never get committed.

2. **Run** (two terminals):

   ```bash
   uvicorn main:app --port 8787 --app-dir agent        # backend
   python3 -m http.server 4173 --directory site         # frontend
   ```

3. Open **http://127.0.0.1:4173**, pick a track, and describe your semester.
   Sanity check: `curl http://127.0.0.1:8787/health`.

## Tests

```bash
bash agent/tests/run_mocked.sh          # 115+ checks, no API cost - run before any demo
python3 agent/tests/test_scenarios.py react   # live suite (~$0.20) - only after model/prompt changes
```

## Repo map

| Path | What it is |
|---|---|
| `agent/agent_react_loop.py` | **The agent** — the bounded tool-choosing loop and its guards |
| `agent/tools.py` | 15 model-callable tools (pure Python, no AI inside) - incl. the weekly-schedule and exam-study analyzers, plus `analyze_grade_improvement_candidates` (preinjected, not model-callable - see `docs/enhancement-checklist.md` item 2) |
| `agent/agent_loop.py` | Understanding/extraction + shared helpers + legacy pipeline (fallback) |
| `agent/student_store.py` | Supabase-backed session persistence, keyed by hashed student email (item 3) |
| `agent/transcript_parser.py` | Parses an uploaded Technion transcript PDF into courses/grades (item 1) |
| `agent/live_offering_check.py` | Real-time "is this course still offered" check on the final candidates, right before delivery |
| `agent/main.py` | Web server — see its own docstring for the full endpoint list: `/chat`, `/chat/stream`, `/transcript`, `/session`, `/tracks`, `/health`, plus the course-spec-required `/api/*` endpoints |
| `agent/tests/` | Automated checks + live scenario suite |
| `site/index.html` | The whole website — one file, no build step |
| `pipeline/` | Offline data fetching (Technion SAP API + CheeseFork + technion-histograms) |
| `pipeline/backfill_grade_stats.py` | One-off patch script that added `grade_stats` to the already-committed `data/track_*.json` bundles |
| `data/track_*.json` | Packaged per-track course bundles the server loads |

## Data sources

Everything the agent reasons over is real, fetched ahead of time (never live
during a request, except the one narrow offering check below):

| Source | What it provides | Where |
|---|---|---|
| [Technion's own SAP/OData course API](https://students.technion.ac.il) | Course catalog, requirement trees, prerequisites, weekly schedule/sections, exam dates | `pipeline/technion_api.py` → `data/track_*.json` |
| Official DDS ("track diagram") PDFs, hand-digitized per track | Ground-truth per-course semester placement, overriding the requirement tree wherever the tree is wrong or incomplete | `Track.official_semester` in `agent/data_bundle.py` — see the "Official semester placement" note below |
| [CheeseFork](https://github.com/michael-maltsev/cheese-fork) | Crowd-sourced difficulty/satisfaction ratings + real review excerpts | `pipeline/cheesefork_client.py`, surfaced via `summarize_cheesefork` |
| [technion-histograms](https://github.com/michael-maltsev/technion-histograms) | Real historical grade distributions (course average) | `pipeline/histogram_client.py` → `grade_stats`, used by grade-improvement retakes |
| Technion's live system, one real-time check per delivery | Confirms the final candidate courses are still actually offered (the cached bundle is a snapshot) | `agent/live_offering_check.py` |

A student's own transcript/grades are the one source that's never fetched —
only ever what the student uploads (PDF) or states directly in chat; see
"Transcript upload" below.

## Grade-improvement retakes — how it actually behaves

Three distinct paths, all requiring the student to be the one who decides —
the agent never adds a retake on its own judgment alone:

1. **Student asks outright** ("I want to retake Data Structures," a course
   already passed) — included immediately, no negotiation needed.
2. **Agent proposes, student approves** — if a grade is known (only ever
   because the student volunteered it — never solicited) and it's notably
   below the course's real historical average, the agent *may* propose a
   retake via `deliver_plan`'s `proposed_retake` field and asks directly
   whether to include it — the one deliberate exception to "never end with a
   question." Only included if the student says yes on a *later* turn.
3. **A failed course** isn't a "retake decision" at all — it's just mandatory
   again, automatically.

**Example (short):**
```
Student: "...I got a 65 in Linear Algebra."
Agent:   [builds the plan] "...one more thing — your grade in Linear
          Algebra (65) is well below both the course average and your own
          average. Want me to add a retake next time?"
Student: "yes"
Agent:   [next plan includes Linear Algebra, tagged RETAKE]
```

**Persists correctly across the whole conversation, not just one turn** —
found live and fixed: the exemption that lets an already-passed retake
back into the plan used to reset every turn, so an accepted retake could
get silently stripped out by a completely unrelated later revision
(`check_invariants` treating it as "already passed"). `state["confirmed_grade_retakes"]`
now persists it for the rest of the conversation, the same way
passed/failed courses already do — see `agent_react_loop.py`.

Never claims guaranteed eligibility under Technion's actual grade-improvement
policy (שיפור ציון) — see `docs/enhancement-checklist.md` item 2.

## Explicit "add this course" requests

A by-name request to add a specific course is a hard directive, not a
suggestion the model may weigh against difficulty or reviews — but it can
genuinely fail two different ways, and the agent is required to say which:

- **Not offered this semester at all** — explained plainly by name; there's
  no section to register for, so this can never be forced in, no matter what.
- **Genuinely collides with another course, or needs room** — the agent
  names the specific conflict and asks whether to force it in anyway despite
  the trade-off. Confirming (even a bare "yes, add it anyway" with no course
  name) adds it next turn, with the resulting conflict disclosed as an open
  issue rather than hidden.

## Notes

- The per-turn spend cap lives in `agent/.env` (`SESSION_BUDGET_USD_CAP`).
- Never commit `agent/.env`. If a key ever leaks into a commit, rotate it immediately.
- **Session persistence (optional):** set `SUPABASE_URL` and
  `SUPABASE_SERVICE_ROLE_KEY` in `agent/.env` (see `.env.example`) to let a
  student restore a saved plan across browsers/devices by re-entering the
  email they used before (`GET /session`, `agent/student_store.py`). Left
  empty, everything works exactly as before this existed - purely additive.
  Also set both in the Vercel project's Environment Variables for
  production; they're not read from `agent/.env` there.
  **Identity boundary, stated plainly**: the app never asks for a password
  and never verifies an email belongs to whoever typed it - per the course
  spec, the GUI must have no login/auth guard at all. The email is hashed
  (SHA-256, client-side, in `site/index.html`) before it's ever sent, so a
  raw address never sits in Supabase or in Vercel's request logs - but
  that's leak-hygiene, not access control: anyone who already knows or
  guesses a specific student's email can compute the same hash themselves
  and look up that saved plan directly, with or without this UI. That's an
  inherent consequence of the spec's no-auth requirement, not a bug in the
  implementation - documented here so it's a known, accepted trade-off
  rather than something discovered later.
- **Transcript upload (optional):** a student can upload their official
  Technion transcript PDF instead of clicking through the course checklist -
  see `POST /transcript`, `agent/transcript_parser.py`. Never stores the raw
  PDF or its extracted text anywhere; only course numbers, pass/fail, and
  grades survive, and those only ever live in memory for the request unless
  `student_key` is also set (in which case they're saved the same way any
  other resolved state is - see above).
- **Grade-improvement retakes and explicit add requests:** see the two
  dedicated sections above - both are hard-enforced in code across every
  delivery path, not just prompted, and a student's own retake question
  ("do you recommend retaking X?") is answered from their real known grade
  and the course's real historical average, never a guess.
- **Official semester placement (`track.official_semester`):** Technion's own
  requirement-tree API is neither complete (missing real mandatory courses
  the official DDS diagrams show) nor precise (also tags some courses
  "Mandatory" that the diagrams never list, and the prerequisite-depth
  heuristic used as a fallback is only ever an approximation of real
  curriculum pacing). Real per-course semester placement, hand-digitized
  from each track's official DDS diagram and cross-checked against its
  printed credit totals, now overrides the heuristic wherever it's known -
  see `agent/data_bundle.py`'s `Track.official_semester` and
  `_CONFIRMED_NON_REQUIREMENTS`. Currently covers all 3 tracks, including
  הנדסת תעשיה וניהול (previously excluded from `/tracks` entirely because it
  has no flat mandatory-course list in Technion's tree - digitizing its
  diagram gave it one, so it's selectable now; semesters 1-4 are fully
  accurate, its later-semester "specialization chain" requirement isn't
  modeled yet - see `docs/enhancement-checklist.md` item 6).
- **Live course-offering check:** right before delivering a plan, the agent
  makes one real-time check against Technion's own system to confirm the
  final candidate courses are still actually offered - `data/track_*.json`
  is a snapshot, only refreshed when someone reruns the pipeline; this
  catches a course that changed in between. Fails open (never blocks
  delivery on a network hiccup) - see `agent/live_offering_check.py`.
- **Requirement-tree "choice group" phantoms:** Technion's requirement tree
  is shared across tracks/faculties, so it sometimes pairs a track's real,
  already-known-required course variant (e.g. the Calculus a specific track
  actually requires) with a DIFFERENT track's variant of the same subject
  under one shared "pick one" node. A choice group is only ever surfaced to
  the student when NEITHER option is already a confirmed real requirement -
  see `_extract_mandatory_choice_groups` in `agent/data_bundle.py`.
- **Session save/restore requires a Technion campus email**
  (`@campus.technion.ac.il`) - client-side only, since the backend never
  sees the raw address (only its SHA-256 hash - see the identity boundary
  note above), so this is a courtesy check for honest students, not access
  control.
- **Back to previous plan:** once a plan has been revised at least once, a
  button restores the prior version in full - same rendering path as any
  other plan, so Print/Add to calendar/Share image all work on it exactly
  as normal - and rewinds the conversation's memory so the next message
  revises from the restored plan, not the abandoned one. Repeatable
  (steps one plan further back each click), client-side only (`site/index.html`'s
  `planHistory`), survives a page reload via the existing localStorage
  session blob.
