# APEX — Semester Planning Agent

**Live demo: [technion-semester-palnner-agent.vercel.app](https://technion-semester-palnner-agent.vercel.app)**

An AI agent that plans a university semester the way a good academic advisor
would — not a form that fills in a template, but a model that reasons its way
there. It calls real tools to check prerequisites, build a clash-free weekly
timetable, price out exam-period risk, and weigh trade-offs, then explains
*why* every course is in the plan. Point it at a messy, human description of
a situation and it comes back with something a person would actually approve.

```
"Starting semester 5. Failed Data Structures. Can't come in on Sundays."
```

...and it returns a verified plan: the failed course retaken first, a
clash-free timetable around the Sunday restriction, exam dates spaced out
where possible, and a plain-English explanation of every trade-off it had to
make — never silently hiding a problem it couldn't solve.

## Why this exists

Academic advising doesn't scale — a human advisor gives every student
individual attention, but there are far more students than advising hours.
Most "smart" schedule builders are really just constraint solvers wearing a
chat interface: they optimize a spreadsheet, they don't *advise*. APEX is
built the other way around — an LLM in the driver's seat, deciding which of
15 real tools to call and in what order, with deterministic code holding the
hard guarantees (no schedule collisions, no re-including a passed course, no
unmet prerequisite) so the model's judgment gets to focus on the parts that
actually need it: priority, trade-offs, and honest communication.

**What it can do:**
- Build a full semester plan from free text (Hebrew or English) — courses,
  a clash-free weekly timetable, an exam calendar (moed A + B), a workload
  score, and a roadmap to graduation
- Prioritize retaking a failed course, and separately, *propose* — never
  silently add — a grade-improvement retake when a student's own grade is
  well below the course average, asking directly before ever including it
- Respect hard constraints (days off, lighter pace) down to the section level
- Revise an existing plan minimally on follow-up ("swap the Tuesday course,"
  "I have a wedding on that exam date") instead of starting over
- Answer grounded questions about a specific course from real student
  reviews, without derailing into a full replan
- Say plainly when something can't be satisfied, instead of hiding it

**What it deliberately never does:** register a student for anything, touch
a real Technion account, or deliver a plan with a schedule collision, an
unmet prerequisite, or a course already passed.

## Try it

Open the [live demo](https://technion-semester-palnner-agent.vercel.app),
pick a track, and describe your semester in your own words — or paste the
example above. No login, no setup.

## Run it locally

**Requirements:** Python 3.11+

```bash
pip install fastapi uvicorn openai pdfplumber python-multipart
```

**1. API key** — copy the template and fill in a key (LLMod proxy or a real
OpenAI key both work — see `agent/.env.example` for both options):

```bash
cp agent/.env.example agent/.env
# edit agent/.env and fill OPENAI_API_KEY
```

`agent/.env` is gitignored — a real key never gets committed.

**2. Run** (two terminals):

```bash
uvicorn main:app --port 8787 --app-dir agent        # backend
python3 -m http.server 4173 --directory site        # frontend
```

**3.** Open **http://127.0.0.1:4173**, pick a track, and describe your
semester. Sanity check the backend directly: `curl http://127.0.0.1:8787/health`.

## Tests

```bash
bash agent/tests/run_mocked.sh                # deterministic, zero API cost - run before any demo
python3 agent/tests/test_scenarios.py react    # live scenario suite (~$0.20) - after model/prompt changes
```

The mocked suite drives the real agent loop against scripted model
responses, so every guarantee below (schedule integrity, retake handling,
revision continuity) is regression-tested without spending anything.
[`TESTING.md`](TESTING.md) has a leveled test plan with exact prompts to try
against the live app.

## How it works

The model runs inside a bounded ReAct-style loop, choosing among 15 tools —
catalog lookup, prerequisite graph, schedule builder, exam-date fetcher,
workload/risk analyzers, review lookup — and ends the turn by calling
`deliver_plan`. Nothing it delivers reaches the student unchecked: Python
re-verifies the final plan against every hard invariant regardless of what
the model claims, and repairs or explains any violation before it ever goes
out. See [`REPORT.pdf`](REPORT.pdf) for the full architecture write-up and
diagram, and [`BUILD_RECORD.pdf`](BUILD_RECORD.pdf) for a session-by-session
log of what got built and fixed along the way.

## Data sources

Everything the agent reasons over is real, fetched ahead of time (never live
mid-request, except the one narrow check below):

| Source | What it provides | Where |
|---|---|---|
| [Technion's own course API](https://students.technion.ac.il) | Course catalog, requirement trees, prerequisites, weekly schedule/sections, exam dates | `pipeline/technion_api.py` → `data/track_*.json` |
| Official per-track requirement diagrams, hand-digitized | Ground-truth per-course semester placement, overriding the requirement tree wherever it's wrong or incomplete | `Track.official_semester` in `agent/data_bundle.py` |
| [CheeseFork](https://github.com/michael-maltsev/cheese-fork) | Crowd-sourced difficulty/satisfaction ratings and real review excerpts | `pipeline/cheesefork_client.py` |
| [technion-histograms](https://github.com/michael-maltsev/technion-histograms) | Real historical grade distributions | `pipeline/histogram_client.py` → used by grade-improvement retakes |
| Technion's live system, one check per delivery | Confirms the final candidate courses are still actually offered (the cached bundle is a snapshot) | `agent/live_offering_check.py` |

A student's own transcript/grades are the one thing never fetched — only
ever what the student uploads (PDF) or states directly in chat.

## Project structure

| Path | What it is |
|---|---|
| `agent/agent_react_loop.py` | **The agent** — the bounded tool-choosing loop and its guards |
| `agent/tools.py` | 15 model-callable tools (pure Python, no AI inside) |
| `agent/agent_loop.py` | Free-text understanding/extraction, shared helpers |
| `agent/student_store.py` | Optional session persistence, keyed by a hashed student email |
| `agent/transcript_parser.py` | Parses an uploaded transcript PDF into courses/grades |
| `agent/live_offering_check.py` | Real-time "is this course still offered" check before delivery |
| `agent/main.py` | Web server — REST endpoints for the frontend and for programmatic use |
| `agent/tests/` | Automated checks + live scenario suite |
| `site/index.html` | The whole frontend — one file, no build step |
| `pipeline/` | Offline data fetching (course API + CheeseFork + grade histograms) |
| `data/track_*.json` | Packaged per-track course bundles the server loads |

## A few engineering details worth knowing about

**Grade-improvement retakes are a proposal, never a silent add.** If a
student volunteers a grade (never solicited) and it's notably below both the
course average and their own average, the agent *may* propose a retake and
ask directly whether to include it — the one deliberate exception to "never
end with a question." Only added if the student says yes on a later turn,
and that acceptance persists for the rest of the conversation, not just one
turn.

**An explicit "add this course" is a hard directive**, not a suggestion the
model can weigh against difficulty — but it can genuinely fail two ways, and
the agent has to say which: not offered at all (can never be forced), or a
real schedule collision (negotiated — the agent names the conflict and asks
before forcing it in).

**A plan is re-verified after delivery, not just before.** The model's own
explanation is written before certain deterministic corrections run; if
Python changes what's actually being delivered, the explanation is
regenerated from the corrected plan's real state rather than left describing
something that no longer matches.

**Official per-course semester placement is hand-corrected**, because
Technion's own requirement-tree API is neither complete nor precise on its
own — some real mandatory courses are missing from it, others are tagged
mandatory that shouldn't be. Real per-course placement, digitized from each
track's official diagram, overrides the heuristic wherever it's known.

**Session save/restore, transcript upload, and the live offering check are
all optional and fail open** — leave the relevant environment variables
unset and the app behaves exactly as if they didn't exist; a hiccup in any
of them never blocks a plan from being delivered.
