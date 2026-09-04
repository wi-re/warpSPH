"""Runs the scripts/gradcheck_*.py canary scripts as pytest cases.

Phase 4.1's gradcheck rollout (see docs/historic_plans/CLEANUP_PLAN.md) produced these as
standalone scripts, one per kernel-bearing module, rather than in-process
pytest functions -- for two independent reasons, both fatal to running them
in-process here:

  * warpSPHCore_PRECISION is baked into every compiled kernel at first
    warpSPHCore import and cannot change mid-process, so importing more than
    one gradcheck script into the same interpreter would have the later ones
    silently reuse the first script's precision.
  * This repo's own tests/conftest.py already calls
    ``warpSPHBootstrap.bootstrap(precision='float32')`` at collection time,
    before any test module is imported -- so the main pytest process is
    locked to float32 before a gradcheck script would even get a chance to
    request float64. gradcheck's numerical Jacobian needs float64 headroom;
    at float32 it does not just get less precise, it produces spurious
    failures.

Subprocess isolation sidesteps both: each script gets its own fresh
interpreter, exactly as if a user ran it by hand, so this file just shells
out and checks the exit code. Follows warpSPHCore's own
tests/operations/test_gradcheck_scripts.py, same rationale and shape.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

# Discovered rather than listed: a hardcoded list silently stops covering a
# module the moment someone adds a gradcheck script and forgets this file, and
# the failure mode is invisible (the suite still passes, just proving less).
GRADCHECK_SCRIPTS = sorted(p.name for p in SCRIPTS_DIR.glob("gradcheck_*.py"))

# A refactor that moves or renames the scripts should fail loudly here rather
# than quietly parametrizing over nothing.
assert GRADCHECK_SCRIPTS, f"no gradcheck_*.py scripts found in {SCRIPTS_DIR}"


@pytest.mark.parametrize("script_name", GRADCHECK_SCRIPTS)
def test_gradcheck_script(script_name):
    script_path = SCRIPTS_DIR / script_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"{script_name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
