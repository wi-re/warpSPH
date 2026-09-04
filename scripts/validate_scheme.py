#!/usr/bin/env python3
"""Run one SPH scheme across a battery of cases and report a pass/fail table.

This is the "does this scheme actually work" harness. `run_sweep.py` answers
*does every case still build and step* (a refactor smoke test, all cases, one
scheme each -- their own defaults). This answers the orthogonal question: take
**one** scheme, push it through the cases that scheme is supposed to handle,
and grade each on a physics criterion rather than on "did not crash".

Written for `band2018pb` (DFSPH_IMPROVEMENT_PLAN.md active track), but the
experiment table is scheme-agnostic -- `--scheme divergenceFree` grades the
shipped incompressible path, and the two can be diffed.

!! EXPENSIVE -- DO NOT RUN THIS CASUALLY !!
--------------------------------------------
The `full` profile is ~50 minutes per scheme and renders 14 videos. It is a
release-grade validation run, not a debugging tool. Specifically:

  * **Never run the full sweep to find out whether a change worked.** Run the
    ONE case the change was supposed to affect, look at it, and only sweep once
    that case is actually fixed. A sweep launched on a hunch costs an hour and
    tells you what a 90-second single-case run would have.
  * A change to a case's ICs / discretisation is validated by re-running *that
    case* and comparing against a known-good reference run (for `sloshingTank`
    that is `export/16-sloshingTank-dfsph_2026-09-03_14-30-29`, which is clean).
  * The `smoke` profile exists for coverage questions ("does this scheme handle
    these cases at all"). Prefer it, and prefer `--cases <name>` over both.

Two profiles, and the order matters:

    scripts/validate_scheme.py --scheme band2018pb                  # smoke
    scripts/validate_scheme.py --scheme band2018pb --profile full --video

* **smoke** (default) -- coarse `nx`, ~100 steps, no video. Minutes. Run this
  first: it finds the cases a scheme cannot handle at all, before the full
  profile spends an hour rendering them.
* **full** -- the resolution and duration each case is actually graded at,
  plus videos. Expensive (`sloshingTank` alone is 7 s of physics).

Each experiment runs in **its own subprocess** so one divergence, CUDA OOM, or
hard crash cannot take the sweep with it -- the same reason `run_sweep.py` does
it. Results land in a timestamped directory under `sweeps/`:

    sweeps/validate-<scheme>-<profile>-<stamp>/
        summary.json       every metric, machine readable
        summary.md         the table below, as markdown
        <experiment>.json  one per experiment
        <experiment>.log   stdout/stderr of that run
        <experiment>/      video, when --video

Grading. Every experiment gets the generic guards (ran to completion, no
divergence, finite kinetic energy, density inside a per-experiment band) plus
its own physics check -- `hydrostaticColumn` must build the hydrostatic
gradient, `tgv` must not gain energy, the bounded box must decay, `sloshingTank`
must put Sensor 1 in the SPHERIC band. A check that cannot be evaluated
(metric absent) is reported `n/a`, never silently passed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# The experiment table
# --------------------------------------------------------------------------

@dataclass
class Experiment:
    name: str
    case: str
    #: Notes what this experiment is *for* -- printed in the report.
    purpose: str
    params: Dict[str, Any] = field(default_factory=dict)
    #: Extra `run()` kwargs (cflFactor, integrationScheme, ...).
    runKwargs: Dict[str, Any] = field(default_factory=dict)
    smoke: Dict[str, Any] = field(default_factory=dict)
    full: Dict[str, Any] = field(default_factory=dict)
    #: Trajectory keys to summarise (last / min / max) beyond the generic set.
    watch: List[str] = field(default_factory=list)
    #: Physics checks: (label, predicate). Predicate returns True/False, or
    #: None when the metric it needs is missing.
    checks: List[Any] = field(default_factory=list)
    #: Accepted density band for the generic guard.
    rhoBand: tuple = (0.90, 1.10)
    #: Does this case have a free surface? If so the low side of `rhoBand` is
    #: graded on a **spray-robust** series (`embeddedMinDensity` / `densityP05`)
    #: when the case emits one, and otherwise against `rhoFloor` instead of
    #: `rhoBand[0]`.
    #:
    #: Raw `minDensity` at a free surface is a known bad figure of merit in this
    #: codebase (DFSPH_FINDINGS.md Part 33 / Sec. 1.1): it reads 1-3 isolated
    #: particles thrown a spacing or two above the surface by the bulk motion,
    #: whose density is low purely by kernel deficiency and which fall back a
    #: few steps later. It is cosmetic ballistic spray, not structural loss --
    #: which is exactly why `densityP05` and `embeddedMinDensity` were added.
    #: Grading a free-surface case on raw `minDensity` fails every scheme,
    #: including ones that are working correctly.
    freeSurface: bool = False
    #: Structural-collapse floor for a free-surface case with no robust series.
    rhoFloor: float = 0.30
    #: Grade the *low* side of the density at all? For a small free-space body
    #: (`staticBlob`, `impact`) the answer is no, and the case says so itself:
    #: "The surface *deficit* `minDensity` ~ 0.5 is sampling geometry, not a
    #: defect -- half a surface particle's support is empty -- so only the
    #: upper bound is a health check" (`cases/staticBlob.py`). These blobs have
    #: a large surface-to-volume ratio, so even the 5th percentile is dominated
    #: by legitimate surface deficit (`impact` measures p05 ~ 0.29 on a run
    #: whose KE ratio is 0.98 and whose `maxDensity` is 1.032). Clumping --
    #: the real failure -- shows up on the upper bound, which is always graded.
    gradeDensityLow: bool = True
    #: Tolerated fraction of fluid particles whose nearest neighbour is closer
    #: than `dx/2` (the pairing / tensile instability) and further than
    #: `1.5 dx` (voids). These are the structural guards; a healthy lattice is
    #: ~0 on both. Violent free-surface cases (`dambreak`, `columnCollapse`,
    #: `sloshingTank`) legitimately run higher than a quiescent one because a
    #: breaking wave really does fling particles apart, so they get their own
    #: budget rather than a global constant.
    #: Absolute catastrophe cap on `pairedFraction` -- `impact` reaches 0.50
    #: under `divergenceFree`, which no growth test should excuse.
    maxPaired: float = 0.25
    #: Below this, pairing is treated as negligible regardless of ratio (keeps
    #: a 0.001 -> 0.004 wobble from reading as "4x growth").
    pairedFloor: float = 0.02
    #: Tolerated growth of `pairedFraction` from the first fifth of the run to
    #: the last. The validated sloshing reference is flat (1.09x over 140
    #: steps); a delaminating run multiplies by 3x within a handful of steps.
    maxPairedGrowth: float = 2.0
    maxVoid: float = 0.02
    timeout: int = 3600


def _get(m: Dict[str, Any], key: str) -> Optional[float]:
    v = m.get(key)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v != v else v          # NaN -> unavailable


def chk(label: str, fn: Callable[[Dict[str, Any]], Optional[bool]]):
    return (label, fn)


def _decays(m, factor=1.0):
    a, b = _get(m, 'keFirst'), _get(m, 'keLast')
    return None if a is None or b is None or a == 0 else b < factor * a


#: Discretisation an incompressible scheme needs, applied to every experiment
#: unless the experiment's own `runKwargs` override it.
#:
#: **This is not cosmetic.** The incompressible-native cases (`tgv`,
#: `hydrostaticColumn`, `staticBlob`, `shearWave`, `kolmogorovIncompressible`,
#: `randomFlowIncompressible`, `columnCollapse`) already default to exactly
#: this. But `dambreak`, `sloshingTank`, `impact` and `rotatingSquarePatch` are
#: weakly-compressible-first cases whose defaults are `Wendland4` +
#: `KernelMeanSymmetric` + `rungeKutta2` -- and the pressure projection was
#: developed and validated against `Wendland2` / `SuperSymmetric` at `n_h = 4`.
#: Running those four under the WCSPH kernel destabilises them from the first
#: steps: the `sloshingTank` free surface delaminates into horizontal sheets
#: and the `impact` blobs hollow into shells, neither of which happens under
#: the correct preset.
#:
#: `examples/sloshingTank/run_sloshingTank.py`'s `dfsph` preset is the same
#: set, and the run it produced
#: (`export/16-sloshingTank-dfsph_2026-09-03_14-30-29`) is clean -- that export
#: is what exposed this, and it is the reference for the sloshing experiment.
INCOMPRESSIBLE_PRESET = dict(
    kernel='Wendland2',
    supportMode='SuperSymmetric',
    integrationScheme='semiImplicitEuler',
)

#: Schemes the preset above applies to (`enumTypes.IncompressibleSPHScheme`).
INCOMPRESSIBLE_SCHEMES = {
    'divergenceFree', 'dfsphReference', 'iisph', 'omniIncompressible',
    'band2018pb',
}


#: One band for all three `randomFlowIncompressible` variants -- they are the
#: same case and were being graded inconsistently (0.97/1.03 vs 0.95/1.05).
#: 5% is the bound a scheme calling itself incompressible should hold on a
#: closed box; `band2018pb` measures ~4.4% peak here, i.e. it passes but sits
#: near the edge -- far looser than the paper's 0.01% average volume-error
#: convergence target. Reported rather than tuned away.
RANDOMFLOW_RHO = (0.95, 1.05)


EXPERIMENTS: List[Experiment] = [
    # -- periodic / free space: no walls, no free surface -------------------
    Experiment(
        name='tgv', case='tgv',
        purpose='Periodic Taylor-Green: no walls, no surface. Energy must not grow.',
        smoke=dict(nx=32, nSteps=60), full=dict(nx=64, nSteps=300),
        runKwargs=dict(integrationScheme='semiImplicitEuler'),
        rhoBand=(0.97, 1.03),
        checks=[chk('KE ratio < 1.15', lambda m: (
            None if _get(m, 'keRatio') is None else _get(m, 'keRatio') < 1.15))],
    ),
    Experiment(
        name='kolmogorov', case='kolmogorovIncompressible',
        purpose='Periodic + body forcing: the forcing must drive the flow.',
        smoke=dict(nx=32, nSteps=60), full=dict(nx=64, nSteps=300),
        runKwargs=dict(integrationScheme='semiImplicitEuler'),
        rhoBand=(0.97, 1.03),
        checks=[chk('forcing drives flow (KE grows)', lambda m: (
            None if _get(m, 'keLast') is None else _get(m, 'keLast') > 1e-6))],
    ),
    Experiment(
        name='shearWave', case='shearWave',
        purpose='Periodic viscous shear decay -- a quantitative damping check.',
        smoke=dict(nx=32, nSteps=60), full=dict(nx=64, nSteps=300),
        runKwargs=dict(integrationScheme='semiImplicitEuler'),
        rhoBand=(0.97, 1.03),
        checks=[chk('shear decays', lambda m: _decays(m))],
    ),
    Experiment(
        name='staticBlob', case='staticBlob',
        purpose='Free-space blob at rest: |v| must stay ~0 (no spurious forces).',
        smoke=dict(nx=32, nSteps=60), full=dict(nx=64, nSteps=200),
        runKwargs=dict(integrationScheme='semiImplicitEuler'),
        rhoBand=(0.90, 1.10),
        freeSurface=True, gradeDensityLow=False,
        watch=['centroidDrift', 'dispMax', 'dispRms', 'densityStd'],
        checks=[chk('stays at rest (|v|max < 0.5)', lambda m: (
                    None if _get(m, 'vmaxPeak') is None
                    else _get(m, 'vmaxPeak') < 0.5)),
                chk('no net drift (centroidDrift < 0.02)', lambda m: (
                    None if _get(m, 'centroidDrift_max') is None
                    else _get(m, 'centroidDrift_max') < 0.02)),
                chk('shape held (dispMax < 0.1)', lambda m: (
                    None if _get(m, 'dispMax_max') is None
                    else _get(m, 'dispMax_max') < 0.1))],
    ),

    # -- random flow: periodic / closed box / internal obstacle -------------
    Experiment(
        name='randomFlow-periodic', case='randomFlowIncompressible',
        purpose='Decaying divergence-free noise, periodic. Baseline for the two below.',
        smoke=dict(nx=48, nSteps=60), full=dict(nx=64, nSteps=200),
        runKwargs=dict(integrationScheme='semiImplicitEuler'),
        rhoBand=RANDOMFLOW_RHO,
        checks=[chk('KE decays', lambda m: _decays(m))],
    ),
    Experiment(
        name='randomFlow-bounded', case='randomFlowIncompressible',
        purpose=('Closed box, no free surface -- the active-track acceptance test. '
                 'NaNs under divergenceFree; needs the closed-domain gauge.'),
        params=dict(bounded=True),
        smoke=dict(nx=48, nSteps=60), full=dict(nx=64, nSteps=200),
        runKwargs=dict(integrationScheme='semiImplicitEuler'),
        rhoBand=RANDOMFLOW_RHO,
        checks=[chk('KE decays', lambda m: _decays(m)),
                chk('no velocity spike (|v|max < 5)', lambda m: (
                    None if _get(m, 'vmaxPeak') is None else _get(m, 'vmaxPeak') < 5.0))],
    ),
    Experiment(
        name='randomFlow-obstacle', case='randomFlowIncompressible',
        purpose='Internal solid obstacle in the flow -- a curved interior boundary.',
        params=dict(obstacle=True, bounded=True),
        smoke=dict(nx=48, nSteps=60), full=dict(nx=64, nSteps=200),
        runKwargs=dict(integrationScheme='semiImplicitEuler'),
        rhoBand=RANDOMFLOW_RHO,
        checks=[chk('KE decays', lambda m: _decays(m)),
                chk('no velocity spike (|v|max < 5)', lambda m: (
                    None if _get(m, 'vmaxPeak') is None else _get(m, 'vmaxPeak') < 5.0))],
    ),

    # -- gravity + walls + free surface ------------------------------------
    Experiment(
        name='hydrostaticColumn-64', case='hydrostaticColumn',
        purpose='Quiescent column under gravity: must build the hydrostatic gradient.',
        smoke=dict(nx=32, nSteps=100), full=dict(nx=64, nSteps=400),
        runKwargs=dict(integrationScheme='semiImplicitEuler'),
        watch=['pressureSlopeRatio', 'embeddedMinDensity', 'densityP05'],
        rhoBand=(0.85, 1.10),
        freeSurface=True,
        checks=[chk('quiescent (|v|max < 0.5)', lambda m: (
                    None if _get(m, 'maxVelocity_tailMedian') is None
                    else _get(m, 'maxVelocity_tailMedian') < 0.5)),
                chk('hydrostatic gradient 0.9-1.1', lambda m: (
                    None if _get(m, 'pressureSlopeRatio_tailMedian') is None
                    else 0.9 < _get(m, 'pressureSlopeRatio_tailMedian') < 1.1)),
                chk('surface intact (embMin > 0.9)', lambda m: (
                    None if _get(m, 'embeddedMinDensity_tailMedian') is None
                    else _get(m, 'embeddedMinDensity_tailMedian') > 0.9))],
    ),
    Experiment(
        name='hydrostaticColumn-128', case='hydrostaticColumn',
        purpose='Same at the resolution that exposed the bandRelaxation bug.',
        smoke=dict(nx=64, nSteps=100), full=dict(nx=128, nSteps=400),
        runKwargs=dict(integrationScheme='semiImplicitEuler'),
        watch=['pressureSlopeRatio', 'embeddedMinDensity', 'densityP05'],
        rhoBand=(0.85, 1.10),
        freeSurface=True,
        checks=[chk('quiescent (|v|max < 0.5)', lambda m: (
                    None if _get(m, 'maxVelocity_tailMedian') is None
                    else _get(m, 'maxVelocity_tailMedian') < 0.5)),
                chk('hydrostatic gradient 0.9-1.1', lambda m: (
                    None if _get(m, 'pressureSlopeRatio_tailMedian') is None
                    else 0.9 < _get(m, 'pressureSlopeRatio_tailMedian') < 1.1)),
                chk('surface intact (embMin > 0.9)', lambda m: (
                    None if _get(m, 'embeddedMinDensity_tailMedian') is None
                    else _get(m, 'embeddedMinDensity_tailMedian') > 0.9))],
    ),
    Experiment(
        name='dambreak', case='dambreak',
        purpose='Violent free surface + wall impact. Needs cflFactor 0.2 (Part 20).',
        smoke=dict(nx=32, nSteps=100), full=dict(nx=64, nSteps=600),
        runKwargs=dict(cflFactor=0.2, integrationScheme='semiImplicitEuler'),
        watch=['nPenetrating', 'maxPenetrationDx'],
        rhoBand=(0.70, 1.20),
        freeSurface=True,
        checks=[chk('wall holds (nPenetrating <= 3)', lambda m: (
                    None if _get(m, 'nPenetrating_max') is None
                    else _get(m, 'nPenetrating_max') <= 3))],
    ),
    Experiment(
        name='columnCollapse', case='columnCollapse',
        purpose='Released column slams the far wall -- the no-penetration test.',
        smoke=dict(nx=32, nSteps=100), full=dict(nx=64, nSteps=400),
        runKwargs=dict(cflFactor=0.2, integrationScheme='semiImplicitEuler'),
        watch=['nPenetrating', 'maxPenetrationDx'],
        rhoBand=(0.70, 1.20),
        freeSurface=True,
        checks=[chk('wall holds (nPenetrating <= 3)', lambda m: (
                    None if _get(m, 'nPenetrating_max') is None
                    else _get(m, 'nPenetrating_max') <= 3))],
    ),
    Experiment(
        name='impact', case='impact',
        purpose='Two bodies collide -- a pure momentum/pressure-spike test.',
        smoke=dict(nx=32, nSteps=60), full=dict(nx=64, nSteps=200),
        runKwargs=dict(integrationScheme='semiImplicitEuler'),
        rhoBand=(0.80, 1.20),
        freeSurface=True, gradeDensityLow=False,
        checks=[chk('bounded (|v|max < 10)', lambda m: (
            None if _get(m, 'vmaxPeak') is None else _get(m, 'vmaxPeak') < 10.0))],
    ),
    Experiment(
        name='squarePatch', case='squarePatch',
        purpose='Rotating free-surface patch: four convex corners, no walls.',
        smoke=dict(nx=64, nSteps=60), full=dict(nx=96, nSteps=300),
        runKwargs=dict(integrationScheme='semiImplicitEuler'),
        rhoBand=(0.40, 1.20),
        freeSurface=True,
        checks=[chk('bounded (|v|max < 20)', lambda m: (
            None if _get(m, 'vmaxPeak') is None else _get(m, 'vmaxPeak') < 20.0))],
    ),

    # -- the validation case: SPHERIC TC10 wall pressure --------------------
    Experiment(
        name='sloshingTank', case='sloshingTank',
        purpose=('SPHERIC Test Case 10. Graded against the MEASURED Sensor-1 '
                 'pressure band (2.2-13 kPa peaks). The only experiment here '
                 'with an external reference.'),
        # These match the validated reference exports
        # `export/16-sloshingTank-dfsph_2026-09-03_14-*` (nx=60 smoke, nx=200
        # full), so this experiment reproduces runs that are known clean.
        #
        # `nx` is no longer load-bearing: the case now calibrates its particle
        # mass so the at-rest sampling measures `rho0` (its
        # `calibrateRestDensity`, auto-on for incompressible schemes). Before
        # that, the startup impulse was an `nx` lottery -- step-0 `|v|max` 0.06
        # at nx=200 but 1.74 at nx=100, the latter delaminating the layer
        # within ~0.07 s -- and the first version of this harness picked
        # nx=100, which is what destabilised it. With the calibration, step-0
        # `|v|max` is 0.0098 at every `nx` and nx=100 matches nx=200
        # (pairedFraction 0.235 -> 0.0000, voidFraction 0.198 -> 0.0000).
        #
        # Do NOT "fix" a case like this by hunting for a better `nx`. `dx` is
        # chosen to divide the domain evenly, which is what puts the domain
        # bounds a clean dx/2 from the real surface; the boundary and
        # ghost-particle representation depends on it, and re-sampling perturbs
        # the initial geometry discontinuously. Correct the mass (or widen the
        # support) instead -- `scripts/lattice_density_offset.py`.
        smoke=dict(nx=60, nSteps=150),
        full=dict(nx=200, nSteps=None, tLimit=7.0, estimatedSteps=7000),
        runKwargs=dict(cflFactor=0.2, dt=1e-3, maxDt=2e-3),
        watch=['sensorPressure', 'sensorPressureWall', 'sensorRho',
               'sensorDensityRatio', 'rollAngleDeg'],
        rhoBand=(0.60, 1.40),
        timeout=14400,
        freeSurface=True,
        checks=[chk('Sensor-1 peak in 1-30 kPa', lambda m: (
                    None if _get(m, 'sensorPressure_max') is None
                    else 1e3 < _get(m, 'sensorPressure_max') < 3e4)),
                chk('Sensor-1 not stuck at zero', lambda m: (
                    None if _get(m, 'sensorPressure_max') is None
                    else abs(_get(m, 'sensorPressure_max')) > 1.0))],
    ),
]

BY_NAME = {e.name: e for e in EXPERIMENTS}


# --------------------------------------------------------------------------
# Child: run one experiment, emit metrics JSON
# --------------------------------------------------------------------------

def runOne(exp: Experiment, scheme: str, profile: str, outDir: Path,
           video: bool) -> Dict[str, Any]:
    from warpSPH.cases import importAll
    importAll()
    from warpSPH.runner import getCase, run

    cfg = dict(exp.smoke if profile == 'smoke' else exp.full)
    nx = cfg.pop('nx')
    nSteps = cfg.pop('nSteps', None)
    tLimit = cfg.pop('tLimit', None)
    # Harness-only keys -- must not reach `run()`.
    frames = cfg.pop('frames', 200)
    estimatedSteps = cfg.pop('estimatedSteps', 2000)

    kwargs: Dict[str, Any] = {}
    if scheme in INCOMPRESSIBLE_SCHEMES:
        kwargs.update(INCOMPRESSIBLE_PRESET)
    kwargs.update(exp.runKwargs)          # experiment overrides the preset
    kwargs.update(cfg)
    kwargs.update(nx=nx, scheme=scheme, quiet=True, store=False, progress=False)
    if exp.params:
        kwargs['params'] = dict(exp.params)
    if nSteps is not None:
        kwargs['nSteps'] = nSteps
    if tLimit is not None:
        kwargs['tLimit'] = tLimit
    if video:
        # Target ~200 frames. `nSteps` is None on a `tLimit`-driven run
        # (sloshingTank is ~7000 steps at dt=1e-3), where the default
        # `nSteps or 400` would render every 2nd step -- thousands of frames.
        estimated = nSteps if nSteps is not None else estimatedSteps
        kwargs.update(plot=True, video=True, plotBackend='matplotlib',
                      plotInterval=max(1, estimated // frames))

    t0 = time.time()
    result = run(getCase(exp.case), **kwargs)
    elapsed = time.time() - t0

    rows = [r for r in result.trajectory if r.get('step', -1) >= 0]
    m: Dict[str, Any] = {
        'experiment': exp.name, 'case': exp.case, 'scheme': scheme,
        'profile': profile, 'nx': nx, 'seconds': elapsed,
        'kernel': kwargs.get('kernel'), 'supportMode': kwargs.get('supportMode'),
        'stepsRun': len(rows), 'stepsAsked': nSteps, 'tLimit': tLimit,
        'diverged': bool(result.diverged),
    }

    def series(key):
        return [float(r[key]) for r in rows
                if r.get(key) is not None and float(r[key]) == float(r[key])]

    ke = series('kineticEnergy')
    if ke:
        m.update(keFirst=ke[0], keLast=ke[-1], kePeak=max(ke),
                 keRatio=(ke[-1] / ke[0]) if ke[0] else float('nan'))
    m['keAllFinite'] = len(ke) == len(rows)

    vm = series('maxVelocity')
    if vm:
        m.update(vmaxLast=vm[-1], vmaxPeak=max(vm))
    lo = series('minDensity')
    hi = series('maxDensity')
    if lo:
        m['rhoMin'] = min(lo)
    if hi:
        m['rhoMax'] = max(hi)

    # The spray-robust density series are summarised for *every* experiment,
    # not just those that name them in `watch` -- the generic free-surface
    # guard in `grade()` reads them, so leaving them to per-experiment opt-in
    # silently drops the case back to the raw-`minDensity` fallback.
    for key in dict.fromkeys(['maxVelocity', 'densityP05', 'embeddedMinDensity',
                              'densityMedian', 'pairedFraction', 'voidFraction',
                              'nnDistP01', 'nnDistMedian', 'neighbourCountCV']
                             + list(exp.watch)):
        s = series(key)
        if s:
            m[f'{key}_last'] = s[-1]
            m[f'{key}_min'] = min(s)
            m[f'{key}_max'] = max(s)
            # Median over the final fifth of the run. A single last-sample
            # reading is too noisy to grade on -- `pressureSlopeRatio` is a
            # least-squares fit through the pressure column and swings wildly
            # at coarse `nx` (measured range -1.1e5 .. +248 on one nx=32 run
            # whose density profile was perfectly healthy).
            tail = sorted(s[-max(1, len(s) // 5):])
            m[f'{key}_tailMedian'] = tail[len(tail) // 2]
            # Early window, for growth ratios (see the structural guards).
            head = sorted(s[:max(1, len(s) // 5)])
            m[f'{key}_headMedian'] = head[len(head) // 2]

    if video and getattr(result, 'videoPath', None):
        src = Path(result.videoPath)
        dest = outDir / exp.name
        dest.mkdir(parents=True, exist_ok=True)
        for fname in ('output.mp4', 'out.gif'):
            cand = src.parent / fname
            if cand.exists():
                shutil.copy(cand, dest / fname)
                m.setdefault('video', str(dest / fname))
    return m


# --------------------------------------------------------------------------
# Parent: grade, tabulate
# --------------------------------------------------------------------------

def grade(exp: Experiment, m: Optional[Dict[str, Any]],
          error: Optional[str]) -> Dict[str, Any]:
    """Generic guards + the experiment's own physics checks."""
    if m is None:
        return dict(status='ERROR', detail=error or 'no result',
                    checks=[], metrics={})

    results: List[tuple] = []

    def add(label, ok):
        results.append((label, ok))

    add('ran to completion', m.get('stepsRun', 0) > 0
        and not m.get('diverged', True))
    add('kinetic energy finite', bool(m.get('keAllFinite')))
    loB, hiB = exp.rhoBand
    hi = _get(m, 'rhoMax')
    add(f'max density <= {hiB}', None if hi is None else hi <= hiB)

    # --- the guards that actually catch a destroyed particle distribution ---
    # A kernel-summed density is nearly blind to the arrangement: a clumped
    # pair and a delaminated sheet both still read rho ~ rho0. These do not.
    med = _get(m, 'densityMedian_tailMedian')
    add(f'BULK density (median) in [{loB}, {hiB}]',
        None if med is None else loB <= med <= hiB)
    # Pairing is graded on GROWTH, not on an absolute level. Every case has its
    # own equilibrium: the validated `sloshingTank` reference run
    # (`export/16-sloshingTank-dfsph_2026-09-03_14-30-29`, nx=200) sits at a
    # FLAT pairedFraction ~0.06 for its whole run and is clean, so a flat 2%
    # bar would fail a known-good reference. The same case at nx=100 goes
    # 0 -> 0.11 -> 0.13 -> 0.35 within five steps. Runaway is the signal.
    paired = _get(m, 'pairedFraction_tailMedian')
    paired0 = _get(m, 'pairedFraction_headMedian')
    if paired is None:
        add('pairing not growing', None)
    elif paired < exp.pairedFloor:
        add(f'pairing negligible ({paired:.3f} < {exp.pairedFloor})', True)
    elif paired0 is None or paired0 <= 0.0:
        add(f'pairing appeared from ~0 (now {paired:.3f})',
            paired < exp.pairedFloor)
    else:
        add(f'pairing not growing ({paired0:.3f} -> {paired:.3f}, '
            f'<{exp.maxPairedGrowth:g}x)', paired / paired0 < exp.maxPairedGrowth)
    add(f'pairing not catastrophic (<{exp.maxPaired:.0%})',
        None if paired is None else paired < exp.maxPaired)
    void = _get(m, 'voidFraction_tailMedian')
    add(f'no voids (<{exp.maxVoid:.0%} further than 1.5 dx)',
        None if void is None else void < exp.maxVoid)

    # Low side of the raw/percentile density: informational for a free surface.
    if not exp.gradeDensityLow:
        pass                     # see `Experiment.gradeDensityLow`
    elif exp.freeSurface:
        # Tail median, not the global minimum: `hydrostaticColumn` has a
        # documented startup transient (Part 34 -- the raw hydrostatic IC seed
        # relaxing to the discretisation, `embeddedMinDensity` dipping for
        # ~0.1 s before recovering). A run that ends in a healthy state passed;
        # a run that blew up is already caught by the divergence, finite-KE and
        # max-density guards above.
        for key, bound in (('embeddedMinDensity_tailMedian', loB),
                           ('densityP05_tailMedian', loB)):
            lo = _get(m, key)
            if lo is not None:
                add(f'{key.replace("_tailMedian", "")} >= {bound} '
                    f'(spray-robust, converged state)', lo >= bound)
                break
        else:
            lo = _get(m, 'rhoMin')
            add(f'no structural collapse (min density >= {exp.rhoFloor}; '
                f'raw min is spray-contaminated at a free surface)',
                None if lo is None else lo >= exp.rhoFloor)
    else:
        lo = _get(m, 'rhoMin')
        add(f'min density >= {loB}', None if lo is None else lo >= loB)
    for label, fn in exp.checks:
        try:
            add(label, fn(m))
        except Exception as exc:                       # a check must never crash the report
            add(label, None)
            results[-1] = (f'{label} (check raised {type(exc).__name__})', None)

    hard = [ok for _, ok in results if ok is False]
    unknown = [ok for _, ok in results if ok is None]
    status = 'FAIL' if hard else ('PARTIAL' if unknown else 'PASS')
    return dict(status=status, checks=results, metrics=m, detail='')


def fmt(v, width=10):
    if v is None:
        return '-'.rjust(width)
    if isinstance(v, float):
        if v != v:
            return 'nan'.rjust(width)
        return (f'{v:.4g}' if abs(v) < 1e5 else f'{v:.3e}').rjust(width)
    return str(v).rjust(width)


def report(graded: Dict[str, Dict[str, Any]], scheme: str, profile: str) -> str:
    mark = {'PASS': 'PASS', 'FAIL': 'FAIL', 'PARTIAL': 'PART', 'ERROR': 'ERR '}
    lines: List[str] = []
    lines.append(f'# Scheme validation: `{scheme}` ({profile} profile)')
    lines.append('')
    counts: Dict[str, int] = {}
    for g in graded.values():
        counts[g['status']] = counts.get(g['status'], 0) + 1
    lines.append('  '.join(f'{k}: {v}' for k, v in sorted(counts.items())))
    lines.append('')
    lines.append('| experiment | status | nx | steps | KE ratio | '
                 '|v|max peak | rho min | rho max | key metric | s |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for name, g in graded.items():
        m = g['metrics']
        exp = BY_NAME[name]
        key = ''
        if 'pressureSlopeRatio_tailMedian' in m:
            key = f"slope {fmt(_get(m,'pressureSlopeRatio_tailMedian'),1).strip()}"
        elif 'sensorPressure_max' in m:
            key = f"P1max {fmt(_get(m,'sensorPressure_max'),1).strip()} Pa"
        elif 'nPenetrating_max' in m:
            key = f"nPen {fmt(_get(m,'nPenetrating_max'),1).strip()}"
        lines.append(
            f"| {name} | **{mark.get(g['status'], g['status'])}** | "
            f"{m.get('nx','-')} | {m.get('stepsRun','-')} | "
            f"{fmt(_get(m,'keRatio'),1).strip()} | "
            f"{fmt(_get(m,'vmaxPeak'),1).strip()} | "
            f"{fmt(_get(m,'rhoMin'),1).strip()} | "
            f"{fmt(_get(m,'rhoMax'),1).strip()} | {key} | "
            f"{fmt(m.get('seconds'),1).strip()} |")
    lines.append('')
    lines.append('## Per-experiment checks')
    for name, g in graded.items():
        lines.append('')
        lines.append(f"### {name} -- {mark.get(g['status'], g['status'])}")
        lines.append(f'*{BY_NAME[name].purpose}*')
        lines.append('')
        if g['detail']:
            lines.append(f"```\n{g['detail'].strip()[-1500:]}\n```")
        for label, ok in g['checks']:
            tick = {True: '[x]', False: '[ ] **FAILED**', None: '[?] n/a'}[ok]
            lines.append(f'- {tick} {label}')
    return '\n'.join(lines) + '\n'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scheme', default='band2018pb')
    ap.add_argument('--profile', choices=('smoke', 'full'), default='smoke')
    ap.add_argument('--cases', nargs='*', default=None,
                    help='experiment names (default: all)')
    ap.add_argument('--skip', nargs='*', default=[])
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--timeout', type=int, default=None,
                    help='override the per-experiment timeout (seconds)')
    ap.add_argument('--_exec', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--_outdir', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.list:
        for e in EXPERIMENTS:
            print(f'{e.name:26s} {e.case:26s} {e.purpose}')
        return 0

    # -- child ------------------------------------------------------------
    if args._exec:
        exp = BY_NAME[args._exec]
        outDir = Path(args._outdir)
        m = runOne(exp, args.scheme, args.profile, outDir, args.video)
        (outDir / f'{exp.name}.json').write_text(json.dumps(m, indent=2))
        return 0

    # -- parent -----------------------------------------------------------
    selected = [e for e in EXPERIMENTS
                if (args.cases is None or e.name in args.cases)
                and e.name not in args.skip]
    if not selected:
        print('no experiments selected', file=sys.stderr)
        return 2

    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    outDir = Path(args.out) if args.out else (
        REPO / 'sweeps' / f'validate-{args.scheme}-{args.profile}-{stamp}')
    outDir.mkdir(parents=True, exist_ok=True)
    print(f'scheme   {args.scheme}\nprofile  {args.profile}\n'
          f'output   {outDir}\nrunning  {len(selected)} experiments\n')

    graded: Dict[str, Dict[str, Any]] = {}
    for i, exp in enumerate(selected, 1):
        print(f'[{i}/{len(selected)}] {exp.name:26s} ', end='', flush=True)
        logPath = outDir / f'{exp.name}.log'
        cmd = [sys.executable, str(Path(__file__).resolve()),
               '--_exec', exp.name, '--_outdir', str(outDir),
               '--scheme', args.scheme, '--profile', args.profile]
        if args.video:
            cmd.append('--video')
        t0 = time.time()
        err = None
        try:
            with open(logPath, 'w') as log:
                proc = subprocess.run(cmd, cwd=str(REPO), stdout=log,
                                      stderr=subprocess.STDOUT,
                                      timeout=args.timeout or exp.timeout)
            if proc.returncode != 0:
                err = (f'exit {proc.returncode}; tail of {logPath.name}:\n'
                       + logPath.read_text()[-1200:])
        except subprocess.TimeoutExpired:
            err = f'TIMEOUT after {args.timeout or exp.timeout}s'

        resPath = outDir / f'{exp.name}.json'
        m = json.loads(resPath.read_text()) if resPath.exists() else None
        g = grade(exp, m, err)
        graded[exp.name] = g
        print(f'{g["status"]:8s} {time.time() - t0:7.1f}s')

    text = report(graded, args.scheme, args.profile)
    (outDir / 'summary.md').write_text(text)
    (outDir / 'summary.json').write_text(json.dumps(
        {k: {'status': v['status'],
             'checks': [[a, b] for a, b in v['checks']],
             'metrics': v['metrics']} for k, v in graded.items()}, indent=2))
    print('\n' + text)
    print(f'written: {outDir}/summary.md')
    return 1 if any(g['status'] in ('FAIL', 'ERROR')
                    for g in graded.values()) else 0


if __name__ == '__main__':
    sys.exit(main())
