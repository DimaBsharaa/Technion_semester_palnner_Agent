"""Vercel entrypoint: exposes the FastAPI app in agent/main.py as an ASGI
function. The agent package uses flat imports (import tools, import agent_loop),
so agent/ must be on sys.path before importing main."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from main import app  # noqa: E402,F401  (Vercel picks up the ASGI `app`)
