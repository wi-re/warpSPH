"""`JFNK_PLAN.md` Phase E1.5: the Laplacian-based density/velocity
dissipation terms added to the rudimentary acoustic core
(`AcousticCoreConfig.densityDiffusionCoefficient`/
`velocityDiffusionCoefficient`, `f_acoustic_core`'s two optional
`warpOperation(..., WarpOperation.Laplacian, ...)` terms).

Two things this module checks, both flagged as "not yet done" when Phase
E1.5 first landed:

1. `test_dissipationDoesNotDegradeFDvsJVPAgreementRelativeToBaseline`
   (originally attempted as a direct `assert_close` on the whole flattened
   stage-residual matvec, the same shape as Phase A's own
   `test_exactJVPMatvecAgreesWithFDAndUsesNoMoreGMRESIterations`
   (`test_implicitWaveEquation.py`) -- **a real finding surfaced fixing
   that attempt, not a dissipation bug**): the acoustic core's FD matvec
   (`fd_matvec`'s single global Knoll-Keyes step `h`, shared across
   positions/velocities/densities lumped into one flat vector) disagrees
   with the exact JVP by up to `~0.0047` absolute (`nx=24`, `dt=1e-3`,
   concentrated in the velocity slice) **even with both dissipation
   coefficients at `0`** -- confirmed by isolating the comparison down to
   Phase B's own already-validated quiescent, unforced, zero-dissipation
   core, which reproduces the same-order disagreement. This is a
   pre-existing characteristic of applying one shared FD step size to a
   nonlinear, multi-field, multi-scale (position ~O(1), velocity ~O(0.05),
   density ~O(1), `soundSpeed=10`) stage system, not something the
   Laplacian dissipation terms introduce: a direct, same-particle-
   configuration comparison (dissipation on vs. off, otherwise identical
   inputs and the same random probe vector) found `nBad`/max-abs-diff
   essentially unchanged (`615`/`0.0047453` without dissipation vs.
   `609`/`0.00475144` with), confirmed at the per-field level too (position/
   velocity/density slices individually near-identical between the two).
   The exact-JVP path's own correctness is what `warpSPHCore`'s
   operator-level gradcheck tests already establish (`torch.autograd.
   functional.jacobian` as ground truth, not a coarse fixed-`h` FD probe);
   what this test actually checks is the meaningful claim -- that adding
   dissipation doesn't make FD/JVP agreement *worse* than this pre-existing
   baseline. `test_jfnkThroughGenericDIRKConvergesWithBothDissipationTermsLive`
   below separately confirms both matvec modes still reach the same kind of
   converged, physically sane fixed point despite FD's per-entry noise --
   Newton/GMRES evidently tolerates it well enough in practice.

2. `test_sufficientVelocityDiffusionAvoidsTheKolmogorovDivergence`/
   `test_smallVelocityDiffusionStillDivergesLikeE1`: the actual hypothesis
   Phase E1's own finding (`test_acousticCoreForcing.py`) left open --
   E1 measured JFNK *delaying* the Kolmogorov shear instability's
   divergence at zero dissipation, but never asked whether nonzero
   dissipation could do better than delay. **It can, once past a real
   threshold.** An exploratory sweep (`velocityDiffusionCoefficient` from
   `0` to `0.3`, `nx=24`, `dt=20x` the acoustic CFL, `JFNKSolver(matvec='fd')`,
   80-step budget) found: `nu=0`/`1e-4`/`1e-3`/`3e-3`/`1e-2` all diverge
   within step 13-16 (the same order as E1's own zero-dissipation finding);
   `nu=0.015`/`0.02`/`0.025` push the divergence step later but still fail
   within the budget (steps 20/25/27 respectively -- a real but partial
   effect, still "delay" not "avoid"); `nu=0.03` and above never diverge in
   80 steps at all -- a genuinely different, sharp transition, not a
   continuation of the delay trend. Confirmed not a budget artifact
   (`nu=0.03` still hadn't diverged after 200 steps in the exploratory run)
   and not GPU-reduction-order flakiness (3 repeated 80-step runs at
   `nu=0.03` all stayed bounded). This is consistent with the real physics:
   Kolmogorov flow has a genuine viscous stability threshold (a critical
   Reynolds number below which the linear instability is fully damped, not
   just slowed), and `nu=0.03` is evidently past it for this probe's
   wavenumber/forcing/resolution. `densityDiffusionCoefficient` alone (no
   velocity diffusion, swept at `0.05`/`0.1`/`0.2`) stayed in the same
   step-14-to-20 "still diverges" band with no monotonic trend -- expected,
   since the Kolmogorov instability is a shear/vorticity mechanism the
   density-diffusion term (which smooths the acoustic/density field, not
   velocity) doesn't directly damp; not asserted here as a dedicated test
   since it is the same "no effect" finding as the zero-dissipation
   baseline already establishes.
"""

from __future__ import annotations

import math

import torch

from warpSPH.sample.acousticCore import buildPeriodicVortexAcousticCoreSystem
from warpSPH.schemes.experimental.acousticCore import f_acoustic_core

from warpSPHIntegrators.solvers import JFNKSolver
from warpSPHIntegrators.state import get_reference_state
from warpSPHIntegrators import get_scheme
from warpSPHIntegrators.state import flatten_integrated
from warpSPHIntegrators.solvers import fd_matvec, jvp_matvec
from warpSPHIntegrators.state import updateStateEuler
from warpSPHIntegrators.state.lifecycle import updateStep
from warpSPHCore import SupportScheme, buildVerletList

_NX = 24
_FORCING_AMPLITUDE = 1.0
_FORCING_WAVENUMBER = 4.0


def _buildDissipativeForcedSystem(nx=_NX, densityDiffusionCoefficient=0.02, velocityDiffusionCoefficient=0.01):
    return buildPeriodicVortexAcousticCoreSystem(
        nx=nx, uMag=0.05, forcingAmplitude=_FORCING_AMPLITUDE, forcingWavenumber=_FORCING_WAVENUMBER,
        densityDiffusionCoefficient=densityDiffusionCoefficient,
        velocityDiffusionCoefficient=velocityDiffusionCoefficient)


def _backwardEulerStageStepFn(system, dt, config, schemeConfig):
    """Same stage map `dirk.py`'s `DIRK` builds internally for backward
    Euler's one implicit stage (`a_ii = 1`): `Y -> y0 + dt * f(Y)`, same
    helper shape as `test_implicitWaveEquation.py`'s own
    `_backwardEulerStageStepFn`.
    """
    y0 = system.initializeNewState()
    y0.t = float(system.t) + dt

    def step_fn(Y):
        Y.t = y0.t
        k, r = updateStep(system, Y, dt, f_acoustic_core, config, schemeConfig)
        return updateStateEuler(y0, k, dt, copyState=True)

    return y0, step_fn


def _matvecDisagreement(densityDiffusionCoefficient, velocityDiffusionCoefficient, dt=1e-3, seed=0):
    """Max absolute disagreement between the FD and exact-JVP matvecs on one
    backward-Euler stage, for a fixed particle configuration
    (`uMag=0.05`+forcing, matching `_buildDissipativeForcedSystem`) and a
    fixed random probe vector -- used to compare dissipation-on vs.
    dissipation-off apples-to-apples (see the module docstring's finding).
    """
    system, config, schemeConfig = buildPeriodicVortexAcousticCoreSystem(
        nx=_NX, uMag=0.05, forcingAmplitude=_FORCING_AMPLITUDE, forcingWavenumber=_FORCING_WAVENUMBER,
        densityDiffusionCoefficient=densityDiffusionCoefficient,
        velocityDiffusionCoefficient=velocityDiffusionCoefficient)
    y0, step_fn = _backwardEulerStageStepFn(system, dt, config, schemeConfig)
    y_flat = flatten_integrated(y0)
    G_y = y_flat - flatten_integrated(step_fn(y0))
    torch.manual_seed(seed)
    v = torch.randn_like(y_flat)
    jv_fd = fd_matvec(step_fn, y0, y_flat, G_y)(v)
    jv_jvp = jvp_matvec(step_fn, y0)(v)
    return (jv_fd - jv_jvp).abs().max().item()


def test_dissipationDoesNotDegradeFDvsJVPAgreementRelativeToBaseline():
    """See the module docstring's finding: even the zero-dissipation
    baseline has real FD-vs-exact-JVP disagreement on this multi-field,
    multi-scale nonlinear stage system (a pre-existing `fd_matvec`
    single-step-size characteristic, not a Laplacian/dissipation bug). What
    matters for this phase is that the new dissipation terms don't make
    that disagreement *worse* -- checked here as a same-configuration,
    same-probe-vector, dissipation-on-vs-off comparison, allowing generous
    headroom (3x) since this is a qualitative "doesn't regress" check, not
    a numerical-accuracy claim about either matvec mode.
    """
    baseline = _matvecDisagreement(0.0, 0.0)
    withDissipation = _matvecDisagreement(0.02, 0.01)
    assert withDissipation < 3.0 * baseline + 1e-4, (
        f'expected dissipation to not meaningfully worsen FD-vs-JVP matvec agreement, '
        f'got baseline={baseline:.6g}, withDissipation={withDissipation:.6g}'
    )


def test_jfnkThroughGenericDIRKConvergesWithBothDissipationTermsLive():
    """Companion sanity check, same pattern as
    `test_acousticCore.py::test_acousticCoreWiredThroughGenericDIRKWithJFNK`:
    one full backward-Euler step through the generic driver, both matvec
    modes, should converge to a finite, positive-density state with
    dissipation and forcing all active simultaneously.
    """
    for matvec in ('fd', 'jvp'):
        system, config, schemeConfig = _buildDissipativeForcedSystem()
        scheme = get_scheme('Backward Euler (implicit)')
        solver = JFNKSolver(matvec=matvec, tol=1e-8, max_iterations=20)
        result = scheme(system, 1e-3, f_acoustic_core, config, schemeConfig, solver=solver)
        resultState = get_reference_state(result.state)
        assert torch.isfinite(resultState.positions).all()
        assert torch.isfinite(resultState.velocities).all()
        assert torch.isfinite(resultState.densities).all()
        assert (resultState.densities > 0).all()


def _acousticDt(system, schemeConfig, cflFactor=0.2):
    h = system.state.supports.mean().item()
    return cflFactor * h / schemeConfig.soundSpeed


def _runUntilDiverged(integrationScheme, dt, nSteps, solver=None, nx=_NX, divergeRhoMax=100.0,
                       densityDiffusionCoefficient=0.0, velocityDiffusionCoefficient=0.0):
    """Same shape as `test_acousticCoreForcing.py`'s own
    `_runUntilDiverged`, generalized to accept dissipation coefficients:
    returns the 1-indexed step at which the run first went non-finite or
    `rho` exceeded `divergeRhoMax * rho0`, or `nSteps + 1` if it never did.
    """
    system, config, schemeConfig = buildPeriodicVortexAcousticCoreSystem(
        nx=nx, uMag=0.0, forcingAmplitude=_FORCING_AMPLITUDE, forcingWavenumber=_FORCING_WAVENUMBER,
        densityDiffusionCoefficient=densityDiffusionCoefficient,
        velocityDiffusionCoefficient=velocityDiffusionCoefficient)
    scheme = get_scheme(integrationScheme)
    kwargs = dict(f=f_acoustic_core, dt=dt, config=config, schemeConfig=schemeConfig, verbose=False)
    if solver is not None:
        kwargs['solver'] = solver

    state = system
    for i in range(1, nSteps + 1):
        result = scheme(state=state, **kwargs)
        state = result.state
        s = get_reference_state(state)
        rhoMax = s.densities.abs().max().item()
        vMax = s.velocities.norm(dim=1).max().item()
        finite = math.isfinite(rhoMax) and math.isfinite(vMax)
        if not finite or rhoMax > divergeRhoMax * schemeConfig.restDensity:
            return i
        state.adjacency = buildVerletList(s, config.domain, 1.0, SupportScheme.SuperSymmetric, None)
    return nSteps + 1


def test_smallVelocityDiffusionStillDivergesLikeE1():
    """Below the stability threshold this module's docstring records
    (`nu=0.01`), the Kolmogorov shear instability still wins -- same
    qualitative outcome as `test_acousticCoreForcing.py`'s own
    zero-dissipation finding, not a fluke of that specific coefficient.
    """
    system, _config, schemeConfig = buildPeriodicVortexAcousticCoreSystem(
        nx=_NX, uMag=0.0, forcingAmplitude=_FORCING_AMPLITUDE, forcingWavenumber=_FORCING_WAVENUMBER)
    dtLarge = 20 * _acousticDt(system, schemeConfig)

    step = _runUntilDiverged(
        'Backward Euler (implicit)', dtLarge, 80,
        solver=JFNKSolver(matvec='fd', tol=1e-6, max_iterations=20, gmres_maxiter=60, gmres_restart=30),
        velocityDiffusionCoefficient=0.01)
    assert step <= 25, f'expected nu=0.01 to still diverge within the same order as E1s zero-dissipation case (~13-16), got step {step}'


def test_sufficientVelocityDiffusionAvoidsTheKolmogorovDivergence():
    """The actual E1.5 hypothesis, and a genuinely different answer from
    E1's own "JFNK delays but never avoids divergence" finding: past a real
    viscous stability threshold, JFNK-with-dissipation does not merely
    delay the Kolmogorov shear instability, it avoids triggering it at all
    within this budget -- a `4x`-longer run (`80` steps, vs. E1's `40`) than
    the zero-dissipation comparison this mirrors.
    """
    system, _config, schemeConfig = buildPeriodicVortexAcousticCoreSystem(
        nx=_NX, uMag=0.0, forcingAmplitude=_FORCING_AMPLITUDE, forcingWavenumber=_FORCING_WAVENUMBER)
    dtLarge = 20 * _acousticDt(system, schemeConfig)
    nSteps = 80

    step = _runUntilDiverged(
        'Backward Euler (implicit)', dtLarge, nSteps,
        solver=JFNKSolver(matvec='fd', tol=1e-6, max_iterations=20, gmres_maxiter=60, gmres_restart=30),
        velocityDiffusionCoefficient=0.03)
    assert step == nSteps + 1, f'expected nu=0.03 to stay bounded for the full {nSteps}-step budget, diverged at step {step}'
