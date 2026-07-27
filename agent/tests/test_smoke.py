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
