"""Phase 3 of `warpSPHCore/warpier_forward_mode_plan.md`: the smallest
possible demonstration that a Tier-1 forward-mode JVP bridge
(`warpOperationJVP`, landed in Phase 2) is enough to stand up an implicit SPH
solve with *no per-problem Hessian derivation at all*.

Backward Euler applied to the wave system
`du/dt = v`, `dv/dt = c**2 * Laplacian(u) - damping * v`
eliminates `v^{n+1}` algebraically (the PDE is linear, so Newton's method
degenerates to exactly one linear solve) down to a single stage equation for
`u^{n+1}`:

    u^{n+1} - alpha * Laplacian(u^{n+1}) = rhs,
    alpha = dt**2 * c**2 / (1 + dt * damping)  (per-particle, but see below)
    rhs   = u^n + dt * v^n / (1 + dt * damping)

`Laplacian` is exactly linear in its `queryValues`/`referenceValues` argument
for fixed positions/adjacency (Tier 1's whole premise), so this stage
equation's matvec -- `p -> p - alpha * Laplacian(p)` -- is built entirely
from `warpOperationJVP(Laplacian, tangentQueryValues=p)` calls: no
hand-derived Jacobian, no finite differences, no bespoke kernel. `v^{n+1}` is
then read off algebraically once `u^{n+1}` is known.

`alpha` varies per query particle whenever `c`/`damping` do, which makes the
stage operator `I - diag(alpha) A` generically *non-symmetric* even when the
underlying Laplacian matrix `A` is symmetric (`diag(alpha) A` transposes to
`A diag(alpha)`, not itself, unless `alpha` is constant). The test cases
below deliberately use constant `c` and zero `damping` -- the plan's
"verify symmetry empirically rather than assume it" case -- so `alpha`
collapses to a true scalar and CG applies; a spatially varying `c`/`damping`
case would need `bicgstab.py`'s existing generic BiCGStab instead (same
`matvec` closure, no new derivation there either).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import pytest
import torch

from warpSPH.configurations import WaveEquationConfig, buildConfig
from warpSPH.sample.bySamplingScheme import sampleParticles
from warpSPH.schemes.waveEquation import f_wave_equation
from warpSPH.systems.waveSystem import WaveSystemStatev3, WaveSystemv3
from warpSPH.utils import buildDomainDescription
from warpSPHCore import OperationProperties, SupportScheme, WarpOperation, buildVerletList, warpOperationJVP

_CFL_FACTOR = 0.1


# --- A from-scratch matrix-free CG, mirroring bicgstab.py's matvec-closure --
# shape (no CG solver existed in this repo to reuse; BiCGStab does, and is
# the documented fallback for the non-symmetric case this module's docstring
# describes). ------------------------------------------------------------

def _conjugateGradientSolve(matvec: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor,
                            x0: Optional[torch.Tensor] = None, tol: float = 1e-8,
                            maxiter: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    x = x0.clone() if x0 is not None else torch.zeros_like(b)
    r = b - matvec(x)
    p = r.clone()
    rsOld = torch.dot(r, r)
    atol = tol * max(float(torch.linalg.norm(b)), 1.0)
    if maxiter is None:
        maxiter = b.shape[0] * 10

    for iteration in range(maxiter):
        if torch.sqrt(rsOld) < atol:
            return x, iteration
        Ap = matvec(p)
        alpha = rsOld / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsNew = torch.dot(r, r)
        if torch.sqrt(rsNew) < atol:
            return x, iteration + 1
        p = r + (rsNew / rsOld) * p
        rsOld = rsNew
    return x, maxiter


def _laplacianProperties(schemeConfig: WaveEquationConfig) -> OperationProperties:
    return OperationProperties(
        operation=WarpOperation.Laplacian, kernel=schemeConfig.kernel,
        supportMode=schemeConfig.supportMode, laplacianMode=schemeConfig.laplacianMode,
        gradientMode=schemeConfig.gradientMode,
    )


def _laplacianMatvec(system: WaveSystemv3, config, schemeConfig: WaveEquationConfig):
    """`p -> Laplacian(p)`, entirely via `warpOperationJVP` -- the whole
    "matvec" this stage solve needs, and the only warpSPHCore call in this
    module.
    """
    props = _laplacianProperties(schemeConfig)

    def laplacian(p: torch.Tensor) -> torch.Tensor:
        return warpOperationJVP(system.state, props, system.domain, tangentQueryValues=p,
                                adjacency=system.adjacency)

    return laplacian


def implicitBackwardEulerStep(system: WaveSystemv3, dt: float, config, schemeConfig: WaveEquationConfig,
                               tol: float = 1e-8) -> WaveSystemv3:
    """One fully implicit (backward Euler) step -- see module docstring for
    the stage-equation derivation. Solved with CG; `bicgstab.py`'s
    `bicgstabSolve` is the drop-in replacement for a case where `alpha`
    (and hence the stage operator) isn't constant.
    """
    state = system.state
    laplacian = _laplacianMatvec(system, config, schemeConfig)

    denom = 1.0 + dt * state.damping
    alpha = dt * dt * state.c ** 2 / denom
    rhs = state.u + dt * state.v / denom

    def matvec(p: torch.Tensor) -> torch.Tensor:
        return p - alpha * laplacian(p)

    uNext, _iters = _conjugateGradientSolve(matvec, rhs, x0=state.u, tol=tol)
    vNext = (state.v + dt * state.c ** 2 * laplacian(uNext)) / denom

    newState = WaveSystemStatev3(
        positions=state.positions, supports=state.supports, masses=state.masses,
        densities=state.densities, kinds=state.kinds, materials=state.materials,
        UIDs=state.UIDs, UIDcounter=state.UIDcounter,
        u=uNext, v=vNext, c=state.c, damping=state.damping,
    )
    return WaveSystemv3(state=newState, adjacency=system.adjacency, domain=system.domain,
                        t=system.t + dt)


def _buildStandingWaveSystem(nx: int, dim: int, c: float = 1.0, L: float = 2.0):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    domain = buildDomainDescription(l=L, dim=dim, periodic=True, device=device, dtype=dtype)
    config, integrator = buildConfig(dim=dim, nx=nx, domain=domain, device=device,
                                     dtype=dtype, dx=L / nx, cflFactor=_CFL_FACTOR)

    k = math.pi
    particles = sampleParticles(nx, config)
    x = particles.positions[:, 0]
    u0 = torch.sin(k * x)
    v0 = torch.zeros_like(u0)

    n = x.shape[0]
    state = WaveSystemStatev3(
        positions=particles.positions, supports=particles.supports,
        masses=particles.masses, densities=particles.densities,
        kinds=torch.zeros(n, device=device, dtype=torch.int32),
        materials=torch.zeros(n, device=device, dtype=torch.int32),
        UIDs=torch.arange(n, device=device, dtype=torch.int32),
        UIDcounter=n,
        u=u0, v=v0, c=torch.full_like(u0, c), damping=torch.zeros_like(u0),
    )
    adjacency = buildVerletList(state, config.domain, 1.0, SupportScheme.SuperSymmetric, None)
    system = WaveSystemv3(state=state, adjacency=adjacency, domain=config.domain,
                          t=torch.tensor(0.0, device=device, dtype=dtype))
    return system, config, k


# --- Step 2: the stage operator is symmetric on this (constant-c, zero- ----
# damping) case, so CG is a valid choice rather than an assumption. --------

def test_stageOperatorIsSymmetricForConstantSpeedNoDamping():
    system, config, _k = _buildStandingWaveSystem(nx=32, dim=1)
    schemeConfig = WaveEquationConfig()
    dt = 0.01

    laplacian = _laplacianMatvec(system, config, schemeConfig)
    denom = 1.0 + dt * system.state.damping
    alpha = dt * dt * system.state.c ** 2 / denom

    def matvec(p):
        return p - alpha * laplacian(p)

    n = system.state.u.shape[0]
    torch.manual_seed(0)
    a = torch.randn(n, dtype=system.state.u.dtype, device=system.state.u.device)
    b = torch.randn(n, dtype=system.state.u.dtype, device=system.state.u.device)

    lhs = torch.dot(matvec(a), b)
    rhs = torch.dot(a, matvec(b))
    torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-5)


# --- Step 3(a): agrees with the explicit rollout at small dt --------------

@pytest.mark.parametrize('dim,nx', [(1, 64), (2, 24)])
def test_implicitStepAgreesWithExplicitAtSmallDt(dim, nx):
    # Two independent builds, not two names for the same system: the
    # explicit loop below mutates its system's state in place, and
    # `_buildStandingWaveSystem`'s sampling is deterministic, so a shared
    # system would have the explicit rollout silently become the implicit
    # rollout's initial condition too.
    explicitSystem, config, _k = _buildStandingWaveSystem(nx=nx, dim=dim)
    implicitSystem, _config2, _k2 = _buildStandingWaveSystem(nx=nx, dim=dim)
    schemeConfig = WaveEquationConfig()
    dt = 2e-4  # small relative to the CFL-limited step, so both first-order
               # schemes' local truncation errors stay small.
    nSteps = 5

    for _ in range(nSteps):
        update, adjacency = f_wave_equation(explicitSystem, dt, config, schemeConfig)
        explicitSystem.adjacency = adjacency
        explicitSystem.state.u = explicitSystem.state.u + dt * update.dudt
        explicitSystem.state.v = explicitSystem.state.v + dt * update.dvdt

    for _ in range(nSteps):
        implicitSystem = implicitBackwardEulerStep(implicitSystem, dt, config, schemeConfig)

    torch.testing.assert_close(implicitSystem.state.u, explicitSystem.state.u, rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(implicitSystem.state.v, explicitSystem.state.v, rtol=2e-3, atol=2e-4)


# --- Step 3(b): convergence to the closed-form standing wave, mirroring ---
# test_waveEquation.py's explicit-scheme check. -----------------------------

def _standingWaveErrorImplicit(nx: int, dim: int, c: float = 1.0, L: float = 2.0) -> float:
    system, config, k = _buildStandingWaveSystem(nx, dim, c, L)
    schemeConfig = WaveEquationConfig()

    hMax = system.state.supports.max().item()
    dt = _CFL_FACTOR * hMax / c

    tEnd = (2 * math.pi / (c * k)) * 0.25
    nSteps = max(1, round(tEnd / dt))
    dtActual = tEnd / nSteps

    for _ in range(nSteps):
        system = implicitBackwardEulerStep(system, dtActual, config, schemeConfig)

    x = system.state.positions[:, 0]
    analytic = torch.sin(k * x) * math.cos(c * k * tEnd)
    return torch.sqrt(torch.mean((system.state.u - analytic) ** 2)).item()


def test_implicitStandingWaveErrorShrinksWithResolution():
    resolutions = (32, 64, 128)
    errors = [_standingWaveErrorImplicit(nx, dim=1) for nx in resolutions]
    assert all(after < before for before, after in zip(errors, errors[1:])), (
        f'errors {errors} did not shrink monotonically with resolution')
    assert errors[-1] < 0.5 * errors[0], (
        f'error only went from {errors[0]:.3e} to {errors[-1]:.3e} over resolutions {resolutions}')


# --- JFNK_PLAN.md Phase A6: the generic warpSPHIntegrators JFNK solver, -----
# validated against this module's own hand-rolled CG reference. -------------

from warpSPHIntegrators import (
    FixedPointSolver, JFNKSolver, get_reference_state, getIntegrator,
)
from warpSPHIntegrators.dirk import DIRK, getDIRKTableau
from warpSPHIntegrators.fields import flatten_integrated
from warpSPHIntegrators.jfnk import fd_matvec, gmres, jvp_matvec
from warpSPHIntegrators.util import updateStateEuler, updateStep


def _backwardEulerStageStepFn(system: WaveSystemv3, dt: float, config, schemeConfig: WaveEquationConfig):
    """The same stage map `dirk.py`'s `DIRK` builds internally for backward
    Euler's one implicit stage (`a_ii = 1`): `Y -> y0 + dt * f(Y)`. Used here to
    drive `fd_matvec`/`jvp_matvec`/`gmres` directly, for the iteration-count
    comparison below -- `JFNKSolver` itself only reports outer Newton
    iterations, not GMRES's own per-solve count.
    """
    y0 = system.initializeNewState()
    y0.t = float(system.t) + dt

    def step_fn(Y):
        Y.t = y0.t
        k, r = updateStep(system, Y, dt, f_wave_equation, config, schemeConfig)
        return updateStateEuler(y0, k, dt, copyState=True)

    return y0, step_fn


# --- Step 4: `getIntegrator('Backward Euler (implicit)')` + JFNKSolver -----
# reproduces the hand-eliminated CG reference, for both matvec modes. -------

@pytest.mark.parametrize('matvec', ['fd', 'jvp'])
def test_jfnkThroughGenericDIRKAgreesWithHandRolledCG(matvec):
    """`implicitBackwardEulerStep` solves the hand-eliminated `N`-dimensional
    equation for `u` alone; `getIntegrator(...)` + `JFNKSolver` solves the full
    `2N`-dimensional coupled `(u, v)` stage system a generic driver has to use,
    since it has no way to know the problem-specific algebraic elimination is
    available. Both solve for the same fixed point, so they should agree to
    solver tolerance -- that agreement is itself most of the point of this
    check (JFNK_PLAN.md A6).
    """
    dt = 0.01
    cgSystem, config, _k = _buildStandingWaveSystem(nx=32, dim=1)
    schemeConfig = WaveEquationConfig()
    cgResult = implicitBackwardEulerStep(cgSystem, dt, config, schemeConfig, tol=1e-10)

    jfnkSystem, _config2, _k2 = _buildStandingWaveSystem(nx=32, dim=1)
    scheme = getIntegrator('Backward Euler (implicit)')
    solver = JFNKSolver(matvec=matvec, tol=1e-10, max_iterations=15)
    result = scheme(jfnkSystem, dt, f_wave_equation, config, schemeConfig, solver=solver)
    jfnkState = get_reference_state(result.state)

    torch.testing.assert_close(jfnkState.u, cgResult.state.u, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(jfnkState.v, cgResult.state.v, rtol=1e-4, atol=1e-5)


# --- Step 5: JFNK must succeed somewhere Picard measurably fails, mirroring
# test_dirk.py::test_picard_diverges_on_a_stiff_problem_regardless_of_tableau_stability,
# but driven by the wave equation's own stiffness (dt*omega ~ dt*c/h growing
# with dt here) rather than a tunable k. ------------------------------------

def test_picardDivergesWhereJFNKStaysBoundedOnAStiffStep():
    dt = 2.0  # far past the CFL-scaled dt (~0.006 at nx=32) this case normally uses
    schemeConfig = WaveEquationConfig()
    scheme = getIntegrator('Backward Euler (implicit)')

    picardSystem, config, _k = _buildStandingWaveSystem(nx=32, dim=1)
    picardResult = scheme(picardSystem, dt, f_wave_equation, config, schemeConfig,
                          solver=FixedPointSolver(iterations=20))
    picardMax = get_reference_state(picardResult.state).u.abs().max().item()
    assert picardMax > 1e10, (
        f'expected Picard(20) to have blown up at this stiffness, got max|u|={picardMax:.3e}'
    )

    for matvec in ('fd', 'jvp'):
        jfnkSystem, config2, _k2 = _buildStandingWaveSystem(nx=32, dim=1)
        solver = JFNKSolver(matvec=matvec, tol=1e-8, max_iterations=15)
        result = scheme(jfnkSystem, dt, f_wave_equation, config2, schemeConfig, solver=solver)
        jfnkMax = get_reference_state(result.state).u.abs().max().item()
        # L-stable backward Euler damps hard at this stiffness; the initial
        # standing-wave amplitude is 1, so a converged solve should be small,
        # not just "not astronomical".
        assert jfnkMax < 1.0, f'JFNK({matvec}) should be damped well below the initial amplitude, got {jfnkMax:.3e}'


# --- Step 6: exact-JVP vs FD matvec -- agreement AND the concrete payoff ---
# (fewer Krylov iterations, no eps to tune), not just agreement. ------------

def test_exactJVPMatvecAgreesWithFDAndUsesNoMoreGMRESIterations():
    dt = 0.01
    system, config, _k = _buildStandingWaveSystem(nx=32, dim=1)
    schemeConfig = WaveEquationConfig()
    y0, step_fn = _backwardEulerStageStepFn(system, dt, config, schemeConfig)

    y_flat = flatten_integrated(y0)
    G_y = y_flat - flatten_integrated(step_fn(y0))

    mv_fd = fd_matvec(step_fn, y0, y_flat, G_y)
    mv_jvp = jvp_matvec(step_fn, y0)

    torch.manual_seed(0)
    v = torch.randn_like(y_flat)
    jv_fd, jv_jvp = mv_fd(v), mv_jvp(v)
    # FD's own truncation tolerance -- not exact agreement, per JFNK_PLAN.md A6.
    torch.testing.assert_close(jv_fd, jv_jvp, rtol=1e-3, atol=1e-4)

    _delta_fd, iters_fd = gmres(mv_fd, -G_y, tol=1e-8, maxiter=y_flat.numel())
    _delta_jvp, iters_jvp = gmres(mv_jvp, -G_y, tol=1e-8, maxiter=y_flat.numel())
    assert iters_jvp <= iters_fd, (
        f'expected the exact matvec to need no more Krylov iterations than FD, '
        f'got jvp={iters_jvp}, fd={iters_fd}'
    )


# --- Step 7: warpSPHIntegrators IMPLICIT_ROADMAP.md Phase 2 -- the ---------
# preconditioner hook, exercised on this real SPH operator: the block ------
# lower-triangular factor of the backward-Euler stage Jacobian, applied ----
# through warpOperationJVP (operator-based, no dense matrix). -------------
#
# For backward Euler (a_ii = 1) the stage residual is G(Y) = Y - y0 - dt*f(Y)
# and its Jacobian -- the GMRES operator -- is
#     G' = [[I,           -dt*I        ],
#           [-dt*c^2*Lap,  I + dt*damping*I]]
# whose lower block-triangular factor
#     L = [[I,           0             ],
#          [-dt*c^2*Lap,  I + dt*damping*I]]
# inverts as  L^{-1} b = [b_u, (b_v + dt*c^2*Lap(b_u)) / (1 + dt*damping)],
# one Laplacian apply -- a preconditioner in the "sparse/operator-based, no
# dense Jacobian" sense the roadmap asks for. The PDE is linear, so G' is
# constant and Newton is a single correction: the GMRES iteration count is
# the whole solve cost, and it is what the preconditioner buys down.

def _blockLowerTriangularWavePreconditioner(system: WaveSystemv3, dt: float, config,
                                            schemeConfig: WaveEquationConfig):
    """Build ``(preconditioner, context)`` for the backward-Euler wave stage.

    Returns the 3-arg hook callable plus the extra ``preconditioner_context``
    keys it reads. ``c`` and ``damping`` are read from the current Newton
    iterate ``state`` (the 3-arg contract: the preconditioner approximates
    ``J_G(Y)^{-1}`` at the current iterate); everything fixed for the step --
    the Laplacian matvec, ``dt``, the u/v block size -- rides in the context.
    """
    n = system.state.u.shape[0]
    laplacian = _laplacianMatvec(system, config, schemeConfig)

    def preconditioner(v, state, context):
        s = get_reference_state(state)
        u_block = v[:n]
        denom = 1.0 + context['dt'] * s.damping
        v_block = (v[n:] + context['dt'] * s.c ** 2 * laplacian(u_block)) / denom
        return torch.cat([u_block, v_block])

    return preconditioner, {'dt': dt, 'n': n, 'laplacian': laplacian}


@pytest.mark.parametrize('matvec', ['fd', 'jvp'])
def test_blockLaplacianPreconditionerCutsGMRESIterationsOnTheStageSolve(matvec):
    """At the linear-solver level, on the real SPH operator: the block
    lower-triangular Laplacian preconditioner must materially cut the GMRES
    iterations of the 2N backward-Euler stage solve, and the preconditioned
    and unpreconditioned solves must land on the same fixed point."""
    dt = 0.01
    system, config, _k = _buildStandingWaveSystem(nx=32, dim=1)
    schemeConfig = WaveEquationConfig()
    y0, step_fn = _backwardEulerStageStepFn(system, dt, config, schemeConfig)
    y_flat = flatten_integrated(y0)
    G_y = y_flat - flatten_integrated(step_fn(y0))
    mv = fd_matvec(step_fn, y0, y_flat, G_y) if matvec == 'fd' else jvp_matvec(step_fn, y0)

    preconditioner, context = _blockLowerTriangularWavePreconditioner(system, dt, config, schemeConfig)

    x_ref, it_ref = gmres(mv, -G_y, tol=1e-8, maxiter=y_flat.numel())
    x_prec, it_prec = gmres(mv, -G_y, tol=1e-8, maxiter=y_flat.numel(),
                            preconditioner=lambda v: preconditioner(v, system, context),
                            preconditioning='right')

    assert it_ref > 20, f'unpreconditioned stage solve should be stiff here, got {it_ref} iters'
    assert it_prec < 0.6 * it_ref, f'preconditioned {it_prec} iters not < 0.6*{it_ref}'
    torch.testing.assert_close(x_prec, x_ref, rtol=5e-3, atol=1e-4)


@pytest.mark.parametrize('matvec', ['fd', 'jvp'])
def test_jfnkWithPreconditionerMatchesReferenceAndReportsFewerIterations(matvec):
    """End to end through `getIntegrator('Backward Euler (implicit)')`: the
    preconditioned solve must agree with both the unpreconditioned JFNK solve
    and the hand-eliminated CG reference, and report fewer GMRES iterations
    in the `solver_diagnostics` a caller actually sees.
    """
    dt = 0.01
    schemeConfig = WaveEquationConfig()
    scheme = getIntegrator('Backward Euler (implicit)')

    cgSystem, config, _k = _buildStandingWaveSystem(nx=32, dim=1)
    cgResult = implicitBackwardEulerStep(cgSystem, dt, config, schemeConfig, tol=1e-10)

    refSystem, _c2, _k2 = _buildStandingWaveSystem(nx=32, dim=1)
    ref = scheme(refSystem, dt, f_wave_equation, config, schemeConfig,
                 solver=JFNKSolver(matvec=matvec, tol=1e-8, max_iterations=15))
    refState = get_reference_state(ref.state)

    precSystem, _c3, _k3 = _buildStandingWaveSystem(nx=32, dim=1)
    preconditioner, context = _blockLowerTriangularWavePreconditioner(precSystem, dt, config, schemeConfig)
    prec = scheme(precSystem, dt, f_wave_equation, config, schemeConfig,
                  solver=JFNKSolver(matvec=matvec, tol=1e-8, max_iterations=15,
                                    preconditioner=preconditioner, preconditioning='right',
                                    preconditioner_context=context))
    precState = get_reference_state(prec.state)

    torch.testing.assert_close(precState.u, refState.u, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(precState.v, refState.v, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(precState.u, cgResult.state.u, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(precState.v, cgResult.state.v, rtol=1e-4, atol=1e-5)

    it_ref = ref.solver_diagnostics[0].gmres_iterations
    it_prec = prec.solver_diagnostics[0].gmres_iterations
    assert it_ref > 20, f'unpreconditioned stage solve should be stiff here, got {it_ref} iters'
    assert it_prec < 0.6 * it_ref, f'preconditioned {it_prec} iters not < 0.6*{it_ref}'


# --- Cheap bonus: the driver is generic, so JFNK works on other DIRK -------
# tableaus too, for free. ----------------------------------------------------

@pytest.mark.parametrize('schemeName', ['Implicit Midpoint', 'SDIRK2'])
@pytest.mark.parametrize('matvec', ['fd', 'jvp'])
def test_jfnkWorksWithOtherDIRKTableausOnTheWaveEquation(schemeName, matvec):
    dt = 0.01
    system, config, _k = _buildStandingWaveSystem(nx=32, dim=1)
    schemeConfig = WaveEquationConfig()
    scheme = getIntegrator(schemeName)
    solver = JFNKSolver(matvec=matvec, tol=1e-8, max_iterations=15)
    result = scheme(system, dt, f_wave_equation, config, schemeConfig, solver=solver)
    st = get_reference_state(result.state)
    assert torch.isfinite(st.u).all() and torch.isfinite(st.v).all()
    assert st.u.abs().max().item() < 2.0  # bounded, no blow-up
