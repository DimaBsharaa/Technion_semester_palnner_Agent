"""
Optional server-side session persistence via Supabase (PostgREST REST API),
so a student's conversation/plan survives a cleared browser or a different
device - the counterpart to the client-side `planner_session` localStorage
blob site/index.html already keeps, not a replacement for it.

Uses stdlib `urllib.request`, not the `requests` package or the full
supabase-py SDK: Vercel's Python builder installs from the ROOT
requirements.txt (fastapi/openai/pydantic - no `requests`), while
agent/requirements.txt is a separate, already-drifted file only used for
local dev. Adding a new dependency here would work locally and silently
ModuleNotFoundError in production. agent/tests/test_smoke.py already uses
urllib.request for the identical reason - follow that precedent.

Both SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are read lazily (inside each
call, not as module-level constants) so this module's import order relative
to agent_loop.py's _load_env_file() doesn't matter. If either is unset,
every function here is a silent no-op - persistence is opt-in, and local
dev without Supabase configured must keep working exactly as before this
module existed.

The service role key is used exclusively server-side (this module never
runs in a browser); the browser never talks to Supabase directly, so no
row-level-security policy is needed for it to work - RLS is still enabled
on the table with zero policies, as defense in depth in case an anon key
ever gets exposed client-side by a future change.

A Supabase outage or malformed input must NEVER break a chat turn or leak a
traceback to the student - every failure here is caught and printed to
stderr (visible in Vercel's function logs), same swallow-and-log posture as
agent_react_loop.py's _EmittingList.append ("a broken stream consumer must
never break the turn itself").

HARD RULE: never call Supabase via a detached/fire-and-forget thread here.
Vercel's Python runtime has no guarantee a background task started after a
response begins actually completes - both call sites in main.py call these
functions synchronously, BEFORE their response is finalized, which is the
only safe pattern on this runtime.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

_TIMEOUT_SECONDS = 3
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")


def _config() -> tuple[str, str] | None:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    return url.rstrip("/"), key


def _valid_key(student_key: str) -> bool:
    return bool(student_key) and bool(_KEY_RE.match(student_key))


def save_session(student_key: str, payload: dict) -> None:
    if not _valid_key(student_key):
        return
    config = _config()
    if config is None:
        return
    url, key = config
    body = dict(payload)
    body["student_key"] = student_key
    body["updated_at"] = datetime.now(timezone.utc).isoformat()

    request = urllib.request.Request(
        f"{url}/rest/v1/sessions?on_conflict=student_key",
        data=json.dumps([body], ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        },
    )
    try:
        urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS)
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"student_store.save_session failed: {e}", file=sys.stderr)


def load_session(student_key: str) -> dict | None:
    if not _valid_key(student_key):
        return None
    config = _config()
    if config is None:
        return None
    url, key = config

    request = urllib.request.Request(
        f"{url}/rest/v1/sessions?student_key=eq.{student_key}&select=*",
        method="GET",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            rows = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"student_store.load_session failed: {e}", file=sys.stderr)
        return None
    return rows[0] if rows else None
