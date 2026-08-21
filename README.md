# APEX - Semester Planning Agent

**Live demo: [technion-semester-palnner-agent.vercel.app](https://technion-semester-palnner-agent.vercel.app)**

APEX is an AI agent for planning a Technion semester. It takes a student's
free-text situation, calls real tools, checks the result, and returns a course
plan with a timetable, exam dates, workload notes, and short reasons for each
course.

Example:

```text
"Starting semester 5. Failed Data Structures. Can't come in on Sundays."
```

The agent should return a plan that puts the failed course first, builds a
clash-free timetable around the Sunday restriction, spaces exams where possible,
and explains any trade-off it could not fully solve.

## Why this exists

Academic advising is hard to scale. Students need individual planning, but
there are not enough advising hours for every small case. Many schedule tools
mainly solve constraints. APEX adds an LLM decision loop on top of real planning
tools, so the model can decide what to check next while deterministic code keeps
the hard rules in place: no schedule collisions, no re-including a passed
course, and no unmet prerequisites.

**What it can do:**
- Build a full semester plan from free text in Hebrew or English, including
  courses, a weekly timetable, an exam calendar with moed A and moed B, a
  workload score, and a roadmap to graduation
- Prioritize retaking a failed course, and separately propose, not silently add,
  a grade-improvement retake when a student's grade is well below both the
  course average and their own average
- Respect hard constraints, such as unavailable days and lighter pace, down to
  the section level
- Revise an existing plan minimally on follow-up, for example "swap the Tuesday
  course" or "I have a wedding on that exam date"
- Answer grounded questions about a specific course from real student reviews,
  without turning every question into a full replan
- Say plainly when something cannot be satisfied

**What it deliberately never does:** register a student for anything, touch a
real Technion account, or deliver a plan with a schedule collision, an unmet
prerequisite, or a course already passed.

## Try it

Open the [live demo](https://technion-semester-palnner-agent.vercel.app), pick a
track, and describe your semester in your own words. No login, no setup.

## Run it locally

**Requirements:** Python 3.11+

```bash
pip install fastapi uvicorn openai pdfplumber python-multipart
```

**1. API key:** copy the template and fill in a key. LLMod proxy or a real
OpenAI key both work. See `agent/.env.example` for both options.

```bash
cp agent/.env.example agent/.env
# edit agent/.env and fill OPENAI_API_KEY
```

`agent/.env` is gitignored, so a real key does not get committed.

**2. Run** in two terminals:

```bash
uvicorn main:app --port 8787 --app-dir agent        # backend
python3 -m http.server 4173 --directory site        # frontend
```

**3.** Open **http://127.0.0.1:4173**, pick a track, and describe your semester.
Sanity check the backend directly: `curl http://127.0.0.1:8787/health`.

## Tests

```bash
bash agent/tests/run_mocked.sh                # deterministic, zero API cost - run before any demo
python3 agent/tests/test_scenarios.py react    # live scenario suite (~$0.20) - after model/prompt changes
```

The mocked suite drives the real agent loop against scripted model responses.
It regression-tests schedule integrity, retake handling, and revision
continuity without spending API budget. [`TESTING.md`](TESTING.md) has a leveled
test plan with exact prompts to try against the live app.

## How it works

The model runs inside a bounded ReAct-style loop. It chooses among 15 tools:
catalog lookup, prerequisite graph, schedule builder, exam-date fetcher,
workload and risk analyzers, review lookup, and the final delivery tool. The
turn ends when the model calls `deliver_plan`.

The delivered plan is checked again by Python. The code re-verifies the final
plan against hard invariants regardless of what the model claims, then repairs
or explains any remaining issue before responding. See [`REPORT.pdf`](REPORT.pdf)
for the full architecture write-up and diagram, and [`BUILD_RECORD.pdf`](BUILD_RECORD.pdf)
for a session-by-session log of what got built and fixed.

## Data sources

Everything the agent reasons over is real and fetched ahead of time, except for
one narrow live offering check before delivery.

| Source | What it provides | Where |
|---|---|---|
| [Technion's own course API](https://students.technion.ac.il) | Course catalog, requirement trees, prerequisites, weekly schedule/sections, exam dates | `pipeline/technion_api.py` -> `data/track_*.json` |
| Official per-track requirement diagrams, hand-digitized | Ground-truth per-course semester placement, overriding the requirement tree wherever it is wrong or incomplete | `Track.official_semester` in `agent/data_bundle.py` |
| [CheeseFork](https://github.com/michael-maltsev/cheese-fork) | Crowd-sourced difficulty/satisfaction ratings and real review excerpts | `pipeline/cheesefork_client.py` |
| [technion-histograms](https://github.com/michael-maltsev/technion-histograms) | Real historical grade distributions | `pipeline/histogram_client.py` -> used by grade-improvement retakes |
| Technion's live system, one check per delivery | Confirms the final candidate courses are still actually offered, since the cached bundle is a snapshot | `agent/live_offering_check.py` |

A student's own transcript and grades are never fetched from Technion. They only
come from what the student uploads as a PDF or states directly in chat.

## Project structure

| Path | What it is |
|---|---|
| `agent/agent_react_loop.py` | **The agent**, the bounded tool-choosing loop and its guards |
| `agent/tools.py` | 15 model-callable tools, pure Python and no AI inside |
| `agent/agent_loop.py` | Free-text understanding/extraction, shared helpers |
| `agent/student_store.py` | Optional session persistence, keyed by a hashed student email |
| `agent/transcript_parser.py` | Parses an uploaded transcript PDF into courses/grades |
| `agent/live_offering_check.py` | Real-time "is this course still offered" check before delivery |
| `agent/main.py` | Web server, REST endpoints for the frontend and for programmatic use |
| `agent/tests/` | Automated checks and live scenario suite |
| `site/index.html` | The whole frontend, one file and no build step |
| `pipeline/` | Offline data fetching, course API + CheeseFork + grade histograms |
| `data/track_*.json` | Packaged per-track course bundles the server loads |

## A few engineering details worth knowing about

**Grade-improvement retakes are a proposal, never a silent add.** If a student
volunteers a grade and it is notably below both the course average and their own
average, the agent may propose a retake and ask directly whether to include it.
This is the one deliberate exception to "never end with a question." The retake
is only added if the student says yes on a later turn, and that acceptance
persists for the rest of the conversation.

**An explicit "add this course" is a hard directive.** The model should not
drop it just because it is hard or unpopular. It can genuinely fail in two ways:
the course is not offered at all, or every available section creates a real
schedule collision. In those cases, the agent has to say what happened.

**A plan is re-verified after delivery, not just before.** The model writes its
explanation before some deterministic corrections run. If Python changes the
actual delivered plan, the explanation is regenerated from the corrected plan's
real state.

**Official per-course semester placement is hand-corrected.** Technion's
requirement-tree API is useful, but it is not complete or precise enough on its
own. Some real mandatory courses are missing from it, and some courses are
tagged mandatory when they should not be. The official diagrams override the
heuristic wherever they are known.

**Session save/restore, transcript upload, and the live offering check are
optional and fail open.** If the relevant environment variables are missing, the
app behaves as if those features do not exist. A storage or upload hiccup should
not block the planner from delivering a response.
