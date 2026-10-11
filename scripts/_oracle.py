"""Guard for the oracle probes (`scripts/probe_*Oracle*.py`, `probe_omniMatchOriginal.py`, `probe_omniOmniSPHLattice.py`).

They compare warpSPH against the boundaries repo's `DFSPH2D` / `DeltaSPH2D`, which were retired from `warpSPHBoundaries` on 2026-10-11 (ANALYTIC_BOUNDARIES_PLAN E7). The solver layer lives on at the git tag
`oracle-solvers-2026-10-11` of `~/dev/curvatureBoundaries`; importing this module exits with the checkout instructions when `warpSPHBoundaries.sim` is not importable.
"""
import importlib.util
import sys

TAG = 'oracle-solvers-2026-10-11'


def _hasSim() -> bool:
    try:
        return importlib.util.find_spec('warpSPHBoundaries.sim.dfsph2d') is not None
    except ModuleNotFoundError:
        return False


if not _hasSim():
    sys.exit(f"This probe needs the retired oracle solvers (warpSPHBoundaries.sim). Check out tag {TAG} of ~/dev/curvatureBoundaries "
             f"(e.g. `git -C ~/dev/curvatureBoundaries worktree add /tmp/oracle {TAG}` and put /tmp/oracle/src on PYTHONPATH); see docs/analytic_boundaries/retirement_2026-10-11/README.md.")
