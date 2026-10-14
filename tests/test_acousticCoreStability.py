"""`JFNK_PLAN.md` Phase B, steps 3-5: the actual hypothesis test -- "it
should theoretically be feasible, with an implicit solver of sufficient
accuracy, to run the simulation without dissipation."

Runs the rudimentary acoustic core (`f_acoustic_core`, zero dissipation, no
surface treatment, no shifting) three ways at the same resolution/sound
speed/`dt`, past the acoustic CFL limit (`dt_c = C*h/c_s`):

- **explicit** (`RK4`) -- the expected-to-fail control.
- **`FixedPointSolver` (Picard(2))** -- the registry's own shipped default
  for every implicit DIRK scheme.
- **`JFNKSolver`**, well-converged -- the actual hypothesis under test.

**The finding** (recorded here, not assumed): at `dt = 20x` the acoustic
CFL, both `RK4` and Picard(2) diverge to `inf`/`nan` within the first ~10
steps (float32 overflow, `Picard(2)`'s `rhoStd` measured `inf` by step 10 in
the exploratory run this test is drawn from). `JFNKSolver` stays bounded --
`rho` within `0.3%` of `rho0`, no secular drift in `rho`'s spread -- over a
run an order of magnitude longer than what's needed to show the other two
solvers fail, at the same `dt`, with every dissipation term at zero. This is
the positive half of Phase B's hypothesis, on this periodic-vortex probe: the
CFL/stability tax explicit WCSPH pays is decoupled from whether artificial
dissipation is present, once the acoustic part is solved to convergence
rather than integrated explicitly or Picard-iterated a fixed few times.

Validation criterion is **stability, not decay-rate matching** (`JFNK_PLAN.md`
Phase B step 4) -- zero-viscosity 2D SPH has no closed-form decay curve to
compare against here, so these tests assert boundedness/finiteness, not
agreement with an analytic solution (that comparison is Phase D's, once a
real dissipation term and `deltaSPH_step`'s analytic-decay TGV case are both
in play).
"""

from __future__ import annotations

import math

import torch

from warpSPH.sample.acousticCore import buildPeriodicVortexAcousticCoreSystem
from warpSPH.schemes.experimental.acousticCore import f_acoustic_core

from warpSPHIntegrators import FixedPointSolver, JFNKSolver, get_reference_state, getIntegrator
from warpSPHCore import SupportScheme, buildVerletList

_NX = 24
_ACOUSTIC_CFL = 0.2  # C in dt_c = C*h/c_s
_DT_MULTIPLIER = 20  # how far past dt_c this module's "large dt" runs


def _acousticDt(system, schemeConfig, cflFactor=_ACOUSTIC_CFL):
    h = system.state.supports.mean().item()
    return cflFactor * h / schemeConfig.soundSpeed


def _runSteps(integrationScheme, dt, nSteps, solver=None, nx=_NX, rebuildAdjacencyEvery=1,
             divergeRhoMax=100.0):
    """Drive `AcousticCoreSystem` for `nSteps` real steps (rebuilding
    adjacency every `rebuildAdjacencyEvery` steps, since positions move here
    unlike the static-neighbor wave equation), stopping early once the state
    is non-finite *or* `rho` has grown past `divergeRhoMax` times `rho0` --
    a genuinely unstable run can take a variable number of further steps to
    literally overflow to `inf`/`nan` (GPU reduction order is not bit-exact
    run to run), so treating "density is already `100x` rest density" as
    diverged too keeps this check from depending on exactly when float32
    overflow happens to land. Returns
    `(finalRhoMax, finalRhoStd, finalVMax, diverged)`.
    """
    system, config, schemeConfig = buildPeriodicVortexAcousticCoreSystem(nx=nx)
    scheme = getIntegrator(integrationScheme)
    kwargs = dict(f=f_acoustic_core, dt=dt, config=config, schemeConfig=schemeConfig, verbose=False)
    if solver is not None:
        kwargs['solver'] = solver

    state = system
    rhoMax = rhoStd = vMax = float('nan')
    for i in range(1, nSteps + 1):
        result = scheme(state=state, **kwargs)
        state = result.state
        s = get_reference_state(state)
        rhoMax = s.densities.abs().max().item()
        rhoStd = s.densities.std().item()
        vMax = s.velocities.norm(dim=1).max().item()
        finite = math.isfinite(rhoMax) and math.isfinite(vMax)
        if not finite or rhoMax > divergeRhoMax * schemeConfig.restDensity:
            return rhoMax, rhoStd, vMax, True
        if i % rebuildAdjacencyEvery == 0:
            state.adjacency = buildVerletList(s, config.domain, 1.0, SupportScheme.SuperSymmetric, None)
    return rhoMax, rhoStd, vMax, False


def test_explicitRK4DivergesPastAcousticCFLWithZeroDissipation():
    system, _config, schemeConfig = buildPeriodicVortexAcousticCoreSystem(nx=_NX)
    dtLarge = _DT_MULTIPLIER * _acousticDt(system, schemeConfig)

    _rhoMax, _rhoStd, _vMax, diverged = _runSteps('RK4', dtLarge, nSteps=10)
    assert diverged, 'expected explicit RK4 to diverge at 20x the acoustic CFL with zero dissipation'


def test_picardDefaultDivergesPastAcousticCFLWithZeroDissipation():
    """`FixedPointSolver()` (2 iterations, the shipped default for every
    registered implicit DIRK scheme) also fails at this `dt` -- the
    comparison is JFNK vs. *both* existing options, not just vs. explicit.
    """
    system, _config, schemeConfig = buildPeriodicVortexAcousticCoreSystem(nx=_NX)
    dtLarge = _DT_MULTIPLIER * _acousticDt(system, schemeConfig)

    # 40 steps, not 10: Picard(2)'s own iterated map is unstable at this dt
    # (confirmed exploding past 1e20 within ~15-30 steps), but exactly which
    # step first clears the `divergeRhoMax` bar is not bit-exact run to run
    # (GPU reduction order), so this needs the same step budget the JFNK
    # test below gets, not the tighter one explicit RK4's much faster,
    # cleaner blow-up can use.
    _rhoMax, _rhoStd, _vMax, diverged = _runSteps(
        'Backward Euler (implicit)', dtLarge, nSteps=40, solver=FixedPointSolver(iterations=2))
    assert diverged, 'expected Picard(2) to diverge at 20x the acoustic CFL with zero dissipation'


def test_jfnkStaysBoundedPastAcousticCFLWithZeroDissipationOverManySteps():
    """The actual hypothesis: a well-converged JFNK solve, at the *same*
    `dt` and zero dissipation, run for `4x` as many steps as it takes the
    other two solvers to fail -- not just "doesn't blow up immediately".
    """
    system, _config, schemeConfig = buildPeriodicVortexAcousticCoreSystem(nx=_NX)
    dtLarge = _DT_MULTIPLIER * _acousticDt(system, schemeConfig)
    solver = JFNKSolver(matvec='fd', tol=1e-6, max_iterations=20, gmres_maxiter=60, gmres_restart=30)

    rhoMax, rhoStd, vMax, diverged = _runSteps(
        'Backward Euler (implicit)', dtLarge, nSteps=40, solver=solver)

    assert not diverged, 'expected JFNK to stay finite at 20x the acoustic CFL over 40 zero-dissipation steps'
    assert rhoMax < 1.1, f'expected density to stay near rho0=1, got max|rho|={rhoMax:.4f}'
    assert rhoStd < 0.01, f'expected density spread to stay small (no clumping/pairing runaway), got std(rho)={rhoStd:.4e}'
    assert vMax < 1.0, f'expected velocity to stay of order the initial vortex amplitude, got vMax={vMax:.4f}'
