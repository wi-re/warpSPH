"""`JFNK_PLAN.md` Phase E1: the Kolmogorov-flow-style forcing term added to
the rudimentary acoustic core (`AcousticCoreConfig.forcingAmplitude`/
`forcingWavenumber`, `f_acoustic_core`'s optional `dv_x/dt += forcingAmplitude
* sin(forcingWavenumber * pi * y)` term).

**The finding, recorded here** (this module's own docstring is the "record
it either way" step, mirroring `test_acousticCoreStability.py`'s Phase B
writeup): unlike Phase B's periodic vortex -- a smooth rotational flow with
no instability mechanism, where a well-converged JFNK solve stayed bounded
indefinitely at 20x the acoustic CFL with zero dissipation -- the Kolmogorov
base flow `v_x = xi*sin(k*pi*y)` is a textbook example of a *linearly
unstable* shear flow (the Meshalkin-Sinai/Kolmogorov-flow instability).
With zero dissipation, nothing damps the exponentially growing perturbation
mode, so **every** solver tested here eventually diverges once that mode
reaches nonlinear/numerical blow-up -- confirmed empirically, and confirmed
*not* to be an artifact of the fixed acoustic-CFL-scaled `dt` becoming
advectively invalid as the flow accelerates (a velocity-tracking adaptive
`dt` was tried and diverged at essentially the same step count). `JFNKSolver`
does not prevent this, but it measurably delays it relative to the other two
solvers at the same `dt` -- `RK4` and `FixedPointSolver` (Picard(2), the
registry default) both diverge by step 5-6, `JFNKSolver` by step ~13,
consistent across repeated runs (unlike Phase B's Picard(2) comparison,
this timing is *not* GPU-reduction-order-flaky, confirmed by 3 repeats
landing on the exact same step each time). This is a genuinely different
answer from Phase B's, and it is the expected one: real `cases/periodic/kolmogorov.py`
both seeds symmetry-breaking noise deliberately (to trigger transition on
purpose) and relies on δ-SPH's artificial/physical viscosity to keep the
resulting turbulent cascade numerically bounded -- dissipation here is
physically load-bearing against a real hydrodynamic instability, a different
concern from the acoustic-stiffness problem JFNK actually solves. The two
findings are complementary, not contradictory: JFNK's payoff is real and
substantial (removing the acoustic `dt` tax), but it is not a general
substitute for dissipation on every flow -- only on ones (like Phase B's
vortex) with no other instability mechanism at zero dissipation.
"""

from __future__ import annotations

import math

import torch

from warpSPH.sample.acousticCore import buildPeriodicVortexAcousticCoreSystem
from warpSPH.schemes.experimental.acousticCore import f_acoustic_core

from warpSPHIntegrators import FixedPointSolver, JFNKSolver, get_reference_state, getIntegrator
from warpSPHCore import SupportScheme, buildVerletList

_NX = 24
_FORCING_AMPLITUDE = 1.0
_FORCING_WAVENUMBER = 4.0


def _buildForcedSystem(nx=_NX):
    return buildPeriodicVortexAcousticCoreSystem(
        nx=nx, uMag=0.0, forcingAmplitude=_FORCING_AMPLITUDE, forcingWavenumber=_FORCING_WAVENUMBER)


def _acousticDt(system, schemeConfig, cflFactor=0.2):
    h = system.state.supports.mean().item()
    return cflFactor * h / schemeConfig.soundSpeed


def _runUntilDiverged(integrationScheme, dt, nSteps, solver=None, nx=_NX, divergeRhoMax=100.0):
    """Returns the 1-indexed step at which the run first went non-finite or
    `rho` exceeded `divergeRhoMax * rho0`, or `nSteps + 1` if it never did.
    """
    system, config, schemeConfig = _buildForcedSystem(nx=nx)
    scheme = getIntegrator(integrationScheme)
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


def test_forcingTermMatchesKolmogorovSinusoidInXOnly():
    """Sanity check on the term itself, independent of any solver: `dv_x/dt`
    should correlate perfectly with `sin(forcingWavenumber * pi * y)`, and
    the unforced `y`-component should be exactly zero (this core omits the
    real case's symmetry-breaking `y`-noise, per the module docstring).
    """
    system, config, schemeConfig = _buildForcedSystem()
    update, _adjacency = f_acoustic_core(system, dt=1e-4, config=config, schemeConfig=schemeConfig)

    y = system.state.positions[:, 1]
    expected = torch.sin(_FORCING_WAVENUMBER * math.pi * y)
    corr = torch.corrcoef(torch.stack([update.dvdt[:, 0], expected]))[0, 1].item()
    assert corr > 0.9999, f'expected dvdt_x to correlate with sin(k*pi*y), got corr={corr:.6f}'
    assert update.dvdt[:, 1].abs().max().item() == 0.0


def test_forcedFlowGrowsSmoothlyAtSmallDt():
    """Correctness check on the early (linear, not-yet-unstable) growth
    phase: at the acoustic-CFL-scaled `dt`, a well-converged JFNK solve
    should integrate the forced, momentum-injecting flow smoothly -- `vMax`
    growing to the right order of magnitude (`~forcingAmplitude * t`, before
    nonlinear/pressure feedback bends the curve), staying finite throughout.
    """
    system, config, schemeConfig = _buildForcedSystem()
    dt = _acousticDt(system, schemeConfig)
    nSteps = 20
    solver = JFNKSolver(matvec='fd', tol=1e-6, max_iterations=15, gmres_maxiter=60)

    divergedAt = _runUntilDiverged('Backward Euler (implicit)', dt, nSteps, solver=solver)
    assert divergedAt == nSteps + 1, f'expected the small-dt forced run to stay finite, diverged at step {divergedAt}'

    # Re-run to inspect the final vMax (kept separate from the diverge-scan
    # loop above for clarity).
    finalSystem, _config2, _schemeConfig2 = _buildForcedSystem()
    scheme = getIntegrator('Backward Euler (implicit)')
    state = finalSystem
    for _ in range(nSteps):
        result = scheme(state=state, f=f_acoustic_core, dt=dt, config=config, schemeConfig=schemeConfig,
                        solver=JFNKSolver(matvec='fd', tol=1e-6, max_iterations=15, gmres_maxiter=60))
        state = result.state
    vMax = get_reference_state(state).velocities.norm(dim=1).max().item()
    expectedOrderOfMagnitude = _FORCING_AMPLITUDE * dt * nSteps
    assert 0.3 * expectedOrderOfMagnitude < vMax < 3.0 * expectedOrderOfMagnitude, (
        f'expected vMax~{expectedOrderOfMagnitude:.4f} (forcingAmplitude*dt*nSteps order), got {vMax:.4f}'
    )


def test_jfnkDelaysButDoesNotPreventTheShearInstabilityPastAcousticCFL():
    """The actual Phase E1 finding: at 20x the acoustic CFL with zero
    dissipation, `RK4` and Picard(2) both diverge almost immediately (by
    step ~5-6); `JFNKSolver` survives measurably longer (by step ~13,
    confirmed non-flaky across repeats) but still eventually diverges --
    the Kolmogorov shear instability, not acoustic stiffness, is what's
    growing. This test asserts the *comparison* (JFNK outlasts both), not
    that JFNK avoids divergence outright, because it doesn't -- see the
    module docstring.
    """
    system, _config, schemeConfig = _buildForcedSystem()
    dtLarge = 20 * _acousticDt(system, schemeConfig)
    nSteps = 40

    stepRK4 = _runUntilDiverged('RK4', dtLarge, nSteps)
    stepPicard = _runUntilDiverged('Backward Euler (implicit)', dtLarge, nSteps,
                                   solver=FixedPointSolver(iterations=2))
    stepJFNK = _runUntilDiverged(
        'Backward Euler (implicit)', dtLarge, nSteps,
        solver=JFNKSolver(matvec='fd', tol=1e-6, max_iterations=20, gmres_maxiter=60, gmres_restart=30))

    assert stepRK4 <= 10, f'expected RK4 to diverge quickly (observed step 5), got step {stepRK4}'
    assert stepPicard <= 10, f'expected Picard(2) to diverge quickly (observed step 6), got step {stepPicard}'
    assert stepJFNK > stepRK4, f'expected JFNK to outlast RK4 (diverged step {stepRK4}), got JFNK step {stepJFNK}'
    assert stepJFNK > stepPicard, f'expected JFNK to outlast Picard(2) (diverged step {stepPicard}), got JFNK step {stepJFNK}'
    assert stepJFNK >= 10, f'expected JFNK to survive at least 10 steps (observed step 13), got step {stepJFNK}'
