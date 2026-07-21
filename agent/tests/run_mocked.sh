#!/usr/bin/env bash
# Runs every zero-cost mocked suite. Exit code 0 = safe to demo.
set -e
cd "$(dirname "$0")/.."
echo "=== cross-turn ==="
python3 tests/test_cross_turn_mocked.py
echo
echo "=== agent guarantees ==="
python3 tests/test_agent_guarantees_mocked.py
echo
echo "=== revision continuity ==="
python3 tests/test_revision_continuity_mocked.py
echo
echo "ALL MOCKED SUITES PASSED — safe to demo."
