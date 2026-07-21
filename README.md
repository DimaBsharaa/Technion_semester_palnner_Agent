# Semester Planning Agent

An AI agent that plans a Technion semester like an experienced academic advisor.
A student describes their situation in free text ("starting semester 5, failed
Data Structures, can't come on Sundays") and the agent builds, verifies, and
revises a personalized plan: courses, weekly timetable, exam calendar, risks —
with honest trade-offs when something can't be satisfied.

**Start here: read [`REPORT.pdf`](REPORT.pdf)** — the full team report: the story,
the architecture diagram, the data, the tools, testing. **[`TESTING.md`](TESTING.md)**
has a leveled test plan with exact prompts to try.

## Setup (5 minutes)

Requirements: Python 3.11+, `pip install fastapi uvicorn openai`.

1. **API key:** copy the template and paste the shared course (LLMod) key —
   ask Manhal for it, it is deliberately not in the repo:

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
bash agent/tests/run_mocked.sh          # 54 checks, no API cost - run before any demo
python3 agent/tests/test_scenarios.py react   # live suite (~$0.20) - only after model/prompt changes
```

## Repo map

| Path | What it is |
|---|---|
| `agent/agent_react_loop.py` | **The agent** — the bounded tool-choosing loop and its guards |
| `agent/tools.py` | All 15 tools (pure Python, no AI inside) - incl. the weekly-schedule and exam-study analyzers |
| `agent/agent_loop.py` | Understanding/extraction + shared helpers + legacy pipeline (fallback) |
| `agent/main.py` | Web server: `/chat`, `/chat/stream` (live steps), `/tracks`, `/health` |
| `agent/tests/` | Automated checks + live scenario suite |
| `site/index.html` | The whole website — one file, no build step |
| `pipeline/` | Offline data fetching (Technion SAP API + CheeseFork) |
| `data/track_*.json` | Packaged per-track course bundles the server loads |

## Notes

- The per-turn spend cap lives in `agent/.env` (`SESSION_BUDGET_USD_CAP`).
- Never commit `agent/.env`. If a key ever leaks into a commit, rotate it immediately.
