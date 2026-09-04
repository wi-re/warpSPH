"""End-to-end tests for the opt-in Krylov pressure solvers in the DFSPH
incompressible solve (BiCGStab / GMRES / CG / BiCG / MINRES), plus the Phase-0
operator probe and the relaxed-Jacobi regression guard. See
``docs/historic_plans/INCOMPRESSIBLE_SOLVER_PLAN.md``.

One divergence-free TGV case is built once (session scope) and shared. The
pressure operator is matrix-free, so the tests that need the operator matrix
assemble it densely on the small case (``_assembleA``): this both verifies the
matvec closure and gives a reference for the symmetry / spectrum probe.

Findings the probe encodes (measured, fp32, this operator):
  * symmetric to ~1e-6 (so BiCG's ``A^T = A`` placeholder is exact);
  * negative-semi-definite with a near-zero gauge mode (so CG/BiCG sign-flip);
  * not diagonally dominant (so the damped relaxed-Jacobi can diverge here --
    the unchanged default is still exercised and fingerprinted, not required to
    converge).
"""

import math

import pytest
from types import SimpleNamespace

import torch

from warpSPH.cases.tgv import tgvCase
from warpSPH.runner import buildContext, CaseSpec
from warpSPH.configurations import (
    IncompressibleSPHConfig, PressureSolverType, JacobiRelaxationMode,
    RelaxedJacobiSolverConfig,
    incompressibleConfigToDict, dictToIncompressibleSPHConfig,
)
from warpSPH.modules.incompressible.divergenceFree import solveDivergenceFree
from warpSPH.modules.incompressible.incompressible import solveIncompressible
from warpSPH.modules.incompressible.krylov import (
    buildIISPHMatvec, buildIISPHPrecond, solvePressureKrylov,
)
from warpSPH.modules.shifting.minres import minresSolve
from warpSPH.modules.incompressible.wp_alpha import computeAlpha
from warpSPH.modules.momentum.incompressible import computeMomentumIncompressible
from warpSPH.modules.pressure.iisph import computePressureAccelIISPH
from warpSPH.modules.incompressible.drift import computePressureShiftIISPH
from warpSPH.modules.density import computeDensities
from warpSPHCore import buildVerletList, SupportScheme

NX = 32
SEED = 0
VELOCITY_SCALE = 0.05


def _buildCase(nx=NX, velocityScale=VELOCITY_SCALE, seed=SEED,
               kernel='Wendland2', n_h=4.0, device=None, dim=2):
    # kernel / n_h (support radius in cell units -> target neighbor count) /
    # dim default to the tgv case's own defaults; the probe script
    # (scripts/probe_relaxedJacobiOmega.py) sweeps them.
    spec = (CaseSpec(caseName='tgv', scheme='divergenceFree',
                     params=dict(tgvCase.params)).merged(**tgvCase.defaults)
            .merged(nx=nx, kernel=kernel, n_h=n_h, dim=dim))
    if device is not None:
        spec = spec.merged(device=device)
    ctx = buildContext(tgvCase, spec)
    tgvCase.configureScheme(ctx)
    system = tgvCase.buildSystem(ctx)
    state = system.state
    config = ctx.config
    schemeConfig = ctx.schemeConfig
    adjacency = buildVerletList(state, config.domain, verletScale=config.verletScale,
                                supportMode=SupportScheme.SuperSymmetric,
                                priorNeighborhood=None, verbose=False)
    system.adjacency = adjacency
    state.densities = computeDensities(state, config, schemeConfig, adjacency)
    state.masses = state.masses / state.densities.mean() * 1.0
    state.densities = computeDensities(state, config, schemeConfig, adjacency)
    torch.manual_seed(seed)
    state.velocities = velocityScale * torch.randn_like(state.velocities)
    state.pressures = torch.zeros_like(state.densities)
    return SimpleNamespace(state=state, config=config, schemeConfig=schemeConfig,
                           adjacency=adjacency, dt=config.dt)


@pytest.fixture(scope='session')
def incompCase():
    return _buildCase()


def _assembleA(case):
    """Dense form of the matrix-free IISPH operator (column j = A e_j)."""
    N = case.state.densities.shape[0]
    matvec = buildIISPHMatvec(case.state, case.config, case.schemeConfig,
                              case.adjacency, case.dt)
    A = torch.empty(N, N, device=case.state.densities.device,
                    dtype=case.state.densities.dtype)
    col = torch.zeros(N, device=case.state.densities.device,
                      dtype=case.state.densities.dtype)
    for j in range(N):
        col.zero_()
        col[j] = 1.0
        A[:, j] = matvec(col)
    return A


def _sourceTerm(case, dvdt=None):
    if dvdt is None:
        dvdt = torch.zeros_like(case.state.velocities)
    predicted = case.state.velocities + case.dt * dvdt
    return -computeMomentumIncompressible(
        case.state, case.config, case.schemeConfig, case.adjacency, predicted)


def _run(case, solverType, maxIter=200, rtol=1e-5, variant='divergenceFree'):
    scfg = (case.schemeConfig.solverConfig.divergenceFreeSolver
            if variant == 'divergenceFree'
            else case.schemeConfig.solverConfig.pressureSolver)
    scfg.solverType = solverType
    scfg.maxIterations = maxIter
    scfg.rtol = rtol
    scfg.atol = 0.0
    scfg.restart = 30
    scfg.tolerance = 1e-9
    scfg.relaxationFactor = 0.5
    scfg.minIterations = 0
    case.state.pressures.zero_()
    dvdt = torch.zeros_like(case.state.velocities)
    fn = solveDivergenceFree if variant == 'divergenceFree' else solveIncompressible
    a_p, pressure, errors, pressures = fn(
        case.state, case.config, case.schemeConfig, case.adjacency, dvdt, case.dt)
    return a_p, pressure, errors


def test_pressureSolverTypeAndDefault():
    names = [e.name for e in PressureSolverType]
    assert names == ['relaxedJacobi', 'cg', 'bicg', 'bicgStab', 'gmres', 'minres']
    # The default must stay relaxedJacobi so the historical path is unchanged.
    assert RelaxedJacobiSolverConfig().solverType is PressureSolverType.relaxedJacobi
    assert IncompressibleSPHConfig().solverConfig.divergenceFreeSolver.solverType is \
        PressureSolverType.relaxedJacobi
    assert IncompressibleSPHConfig().solverConfig.pressureSolver.solverType is \
        PressureSolverType.relaxedJacobi


def test_solverConfigRoundTrip():
    cfg = IncompressibleSPHConfig()
    cfg.solverConfig.divergenceFreeSolver.solverType = PressureSolverType.gmres
    cfg.solverConfig.divergenceFreeSolver.rtol = 1e-6
    cfg.solverConfig.divergenceFreeSolver.atol = 1e-9
    cfg.solverConfig.divergenceFreeSolver.restart = 50
    cfg.solverConfig.pressureSolver.solverType = PressureSolverType.cg
    d = incompressibleConfigToDict(cfg)
    assert d['solverConfig']['divergenceFreeSolver']['solverType'] == 'gmres'
    assert d['solverConfig']['divergenceFreeSolver']['restart'] == 50
    rt = dictToIncompressibleSPHConfig(d)
    assert rt.solverConfig.divergenceFreeSolver.solverType is PressureSolverType.gmres
    assert rt.solverConfig.divergenceFreeSolver.rtol == pytest.approx(1e-6)
    assert rt.solverConfig.divergenceFreeSolver.atol == pytest.approx(1e-9)
    assert rt.solverConfig.divergenceFreeSolver.restart == 50
    assert rt.solverConfig.pressureSolver.solverType is PressureSolverType.cg
    # a dict missing the new keys falls back to the defaults
    d2 = incompressibleConfigToDict(IncompressibleSPHConfig())
    del d2['solverConfig']['divergenceFreeSolver']['solverType']
    rt2 = dictToIncompressibleSPHConfig(d2)
    assert rt2.solverConfig.divergenceFreeSolver.solverType is PressureSolverType.relaxedJacobi


def test_matvecMatchesJacobiInnerIteration(incompCase):
    """The matvec closure is exactly the relaxed-Jacobi inner iteration
    (``dt * shift(accel(p))``), for both the dt (div-free) and dt**2 (constant-
    density) scalings."""
    case = incompCase
    p = torch.randn_like(case.state.densities)
    a_p = computePressureAccelIISPH(case.state, p, case.config,
                                    supportScheme=SupportScheme.Scatter,
                                    adjacency=case.adjacency)
    for scale, dt in ((1, case.dt), (2, case.dt ** 2)):
        expected = dt * computePressureShiftIISPH(case.state, case.config, a_p,
                                                  supportScheme=SupportScheme.Scatter,
                                                  adjacency=case.adjacency)
        got = buildIISPHMatvec(case.state, case.config, case.schemeConfig,
                               case.adjacency, dt)(p)
        assert torch.allclose(got, expected, rtol=1e-5, atol=1e-8), f'scale={scale}'


def test_precondIsReciprocalDiagonal(incompCase):
    """The preconditioner is 1/(dt * computeAlpha) (clamped), and equals 1/Diag(A)
    against the densely-assembled operator."""
    case = incompCase
    precond = buildIISPHPrecond(case.state, case.config, case.schemeConfig,
                                case.adjacency, case.dt)
    D = case.dt * computeAlpha(case.state, case.config, case.schemeConfig,
                               case.adjacency, apparentVolumes=case.state.masses / case.state.densities)
    D = torch.clamp(D, max=-1e-6)
    assert torch.allclose(precond, 1.0 / D, rtol=1e-5, atol=1e-8)
    # the IISPH diagonal is the exact operator diagonal (relL2 ~ 1e-6)
    A = _assembleA(case)
    trueDiag = A.diag()
    rel = float((trueDiag.double() - D.double()).norm() / trueDiag.double().norm())
    assert rel < 1e-4, f'IISPH diagonal does not match the operator diagonal: {rel}'


def test_operatorIsSymmetricNegativeSemiDefinite(incompCase):
    """Phase-0 operator probe: the discrete IISPH operator is symmetric (to fp32)
    and negative-semi-definite with a near-zero gauge mode, and is NOT diagonally
    dominant (which is why the damped relaxed-Jacobi can diverge here)."""
    case = incompCase
    A = _assembleA(case).double()
    symm = float((A - A.T).norm() / A.norm())
    assert symm < 1e-4, f'operator is not symmetric: ||A-A^T||/||A||={symm}'
    eig = torch.linalg.eigvalsh((A + A.T) / 2.0)
    assert float(eig.max()) <= abs(float(eig.min())) * 1e-3 + 1e-12, \
        f'operator has a significant positive mode: {float(eig.max())}'
    assert float(eig.min()) < 0, 'expected a negative (definite) operator'
    # not diagonally dominant on every row
    diag = A.diag().abs()
    off = (A.norm(dim=1) ** 2 - diag ** 2).clamp(min=0).sqrt()
    assert bool((off > diag).any()), 'expected at least one non-diagonally-dominant row'


def _relResid(case, pressure, b=None):
    matvec = buildIISPHMatvec(case.state, case.config, case.schemeConfig,
                              case.adjacency, case.dt)
    if b is None:
        b = _sourceTerm(case)
    bn = float(torch.linalg.norm(b))
    return (float(torch.linalg.norm(b - matvec(pressure))) / bn) if bn > 0 else 0.0


def test_bicgStabReducesResidual(incompCase):
    _, pressure, errors = _run(incompCase, PressureSolverType.bicgStab)
    assert torch.isfinite(pressure).all()
    assert _relResid(incompCase, pressure) < 1e-2


def test_gmresReducesResidual(incompCase):
    _, pressure, errors = _run(incompCase, PressureSolverType.gmres)
    assert torch.isfinite(pressure).all()
    assert _relResid(incompCase, pressure) < 1e-2


def test_cgReducesResidual(incompCase):
    # CG is viable because the operator is symmetric; it converges more slowly on
    # the ill-conditioned/gauge-mode spectrum, so the bound is looser.
    _, pressure, errors = _run(incompCase, PressureSolverType.cg)
    assert torch.isfinite(pressure).all()
    assert _relResid(incompCase, pressure) < 1e-2


def test_bicgRuns(incompCase):
    # BiCG needs the adjoint (the placeholder A^T=A is exact here because the
    # operator is symmetric) but is the least robust of the four on the
    # indefinite/gauge-mode spectrum, so we only require that it runs to
    # completion and returns a finite iterate -- not that it converges.
    a_p, pressure, errors = _run(incompCase, PressureSolverType.bicg)
    assert torch.isfinite(pressure).all()
    assert torch.isfinite(a_p).all()


def test_minresReducesResidual(incompCase):
    # MINRES is the best measured method (symmetric; it handles the NSD operator
    # directly, so no sign flip). At the default 200-iteration budget its
    # residual drops well below the 1e-2 floor the other solvers use; measured
    # ~1e-3 on this state.
    _, pressure, errors = _run(incompCase, PressureSolverType.minres)
    assert torch.isfinite(pressure).all()
    assert _relResid(incompCase, pressure) < 2e-3


def _minresDenseLstsq(A, b, x0, maxiter, atol):
    """Reference MINRES: Lanczos + a dense per-step ``lstsq`` on the normal
    equation. ``M_j = [T_j; beta_j e_j^T]`` is (j+1) x j with T_j tridiagonal
    (diag alpha_i, off-diag beta_i) and the last row beta_j in column j-1; the
    RHS is beta_1 e_1 with beta_1 = ||r0||. Returns (x, status, estimates) with
    the same convention as ``minresSolve`` (estimates[0] = ||r0||, per-step
    recurrence residual, final status a 0-based iter or -13/-14)."""
    n = b.shape[0]
    kmax = max(1, min(int(maxiter), n))
    r = b - A @ x0
    beta1 = float(torch.linalg.norm(r))
    estimates = [beta1]
    if beta1 < atol:
        return x0.clone(), 0, estimates
    v = r / beta1
    vs = [v]                 # vs[i] = v_{i+1}
    alphas = []              # alphas[i] = alpha_{i+1}
    betas = [0.0]            # betas[k] = beta_k = ||w_k||; beta_0 = 0
    x_prev = x0
    for j in range(1, kmax + 1):
        w = A @ v - (betas[j - 1] * vs[j - 2] if j >= 2 else 0.0)
        alpha = float(v @ w)
        alphas.append(alpha)
        w = w - alpha * v
        beta = float(torch.linalg.norm(w))
        if not math.isfinite(beta) or beta < atol:
            return x_prev, -13, estimates
        betas.append(beta)
        if j < kmax:
            v = w / beta
            vs.append(v)
        M = torch.zeros(j + 1, j, dtype=A.dtype, device=A.device)
        for i in range(j):
            M[i, i] = alphas[i]
            M[i + 1, i] = betas[i + 1]      # subdiag, incl. last row beta_j
            if i < j - 1:
                M[i, i + 1] = betas[i + 1]  # superdiag beta_1..beta_{j-1}
        e = torch.zeros(j + 1, dtype=A.dtype, device=A.device)
        e[0] = beta1
        y, *_ = torch.linalg.lstsq(M, e)
        est = float(torch.linalg.norm(M @ y - e))
        estimates.append(est)
        x_prev = x0 + torch.stack([vs[i] * y[i] for i in range(j)], dim=0).sum(dim=0)
        if est < atol:
            return x_prev, j - 1, estimates
    return x_prev, -14, estimates


def test_minresGivensMatchesDenseLstsq():
    # Phase-6 mandate: the shipped Givens-LQ MINRES core must agree per-iterate
    # with a dense-``lstsq`` reference on a random SPD and a random NSD+gauge
    # 30x30 system (estimates to ~1e-10, residual estimate monotone
    # non-increasing, final iterate to ~1e-9). This is the line-by-line port
    # check for the Givens update, which must not be transcribed from memory.
    dt = torch.float64
    n = 30
    eye = torch.eye(n, dtype=dt)
    torch.manual_seed(0)
    R = torch.randn(n, n, dtype=dt)
    A_spd = R.T @ R + n * eye
    b_spd = torch.randn(n, dtype=dt)
    B = torch.randn(n, n, dtype=dt)
    A0 = -(B.T @ B + eye)
    P1 = eye / n
    A_nsd = (eye - P1) @ A0 @ (eye - P1)   # NSD with a constant gauge null space
    b_nsd = torch.randn(n, dtype=dt)
    b_nsd -= b_nsd.mean()
    x0 = torch.zeros(n, dtype=dt)
    rtol = 1e-10
    for name, A, b in (('SPD', A_spd, b_spd), ('NSD+gauge', A_nsd, b_nsd)):
        atol = rtol * float(torch.linalg.norm(b))
        xg, statusg, convg = minresSolve(lambda p: A @ p, b, x0, rtol=rtol, maxiter=n)
        xr, statusr, convr = _minresDenseLstsq(A, b, x0, maxiter=n, atol=atol)
        # minresSolve stamps the final verified true residual, so its recurrence
        # estimates are convg[:-1] and must line up with the dense estimates
        assert len(convg) - 1 == len(convr), \
            f'{name}: history lengths differ ({len(convg)} vs {len(convr)})'
        for eg, er in zip(convg[:-1], convr):
            assert abs(float(eg) - er) <= 1e-10 * max(abs(er), 1e-15), \
                f'{name}: Givens vs dense estimates diverged'
        for c1, c2 in zip(convg[:-2], convg[:-1]):
            assert float(c2) <= float(c1) + 1e-12 * float(convg[0]), \
                f'{name}: MINRES residual estimate not monotone'
        xdiff = float(torch.linalg.norm(xg - xr)) / max(float(torch.linalg.norm(xr)), 1.0)
        assert xdiff < 1e-9, f'{name}: Givens vs dense solution diverged: {xdiff}'


def test_krylovSolversAgree(incompCase):
    # BiCGStab and GMRES are the robust pair; their solutions should be close
    # (same operator, same preconditioner), a strong check that both are solving
    # the same A p = b. MINRES (symmetric, no sign flip) is compared too.
    _, pStab, _ = _run(incompCase, PressureSolverType.bicgStab)
    _, pGmres, _ = _run(incompCase, PressureSolverType.gmres)
    _, pMinres, _ = _run(incompCase, PressureSolverType.minres)
    # compare after removing the gauge (the pressure is defined up to a constant)
    ref = pStab - pStab.mean()
    scale = float(ref.norm())
    assert scale > 0
    for p, name in ((pGmres, 'gmres'), (pMinres, 'minres')):
        d = ref - (p - p.mean())
        assert float(d.norm()) / scale < 0.5, f'{name} disagrees with bicgStab'


def test_incompressibleVariantRuns(incompCase):
    # The constant-density variant (dt**2 operator, non-negative gauge) is wired
    # through the same dispatch; check it runs and yields a finite pressure.
    a_p, pressure, errors = _run(incompCase, PressureSolverType.bicgStab,
                                 variant='incompressible')
    assert torch.isfinite(pressure).all()
    assert float(pressure.min()) >= -1e-6, 'non-negative gauge should clamp at 0'


# --- relaxed-Jacobi regression guard -----------------------------------------
# A fixed 8-iteration relaxed-Jacobi run on the seeded state fingerprints the
# unchanged default path. The errors grow (the damped Jacobi diverges on this
# non-diagonally-dominant operator -- pre-existing behaviour); what the guard
# pins is that the *sequence* is stable, i.e. the code path was not altered by
# adding the Krylov branches.
_JACOBI_FP = {
    'errors': [0.046124004, 0.062533826, 0.10262572, 0.175545648,
               0.304131687, 0.532372236, 0.936860502, 1.656827211],
    'pmean': 5.722e-06,
}


def test_relaxedJacobiRegression(incompCase):
    case = incompCase
    scfg = case.schemeConfig.solverConfig.divergenceFreeSolver
    scfg.solverType = PressureSolverType.relaxedJacobi
    scfg.minIterations, scfg.maxIterations = 0, 8
    scfg.tolerance, scfg.relaxationFactor = 1e-9, 0.5
    case.state.pressures.zero_()
    dvdt = torch.zeros_like(case.state.velocities)
    _, pressure, errors, _ = solveDivergenceFree(
        case.state, case.config, case.schemeConfig, case.adjacency, dvdt, case.dt)
    assert len(errors) == 8
    for got, want in zip(errors, _JACOBI_FP['errors']):
        assert got == pytest.approx(want, rel=1e-4), f'error sequence changed: {errors}'
    assert float(pressure.mean()) == pytest.approx(_JACOBI_FP['pmean'], rel=1e-3, abs=1e-6)
    # determinism: a second run reproduces the same iterate
    case.state.pressures.zero_()
    _, pressure2, errors2, _ = solveDivergenceFree(
        case.state, case.config, case.schemeConfig, case.adjacency, dvdt, case.dt)
    assert torch.allclose(pressure, pressure2)
    assert errors == errors2


# --- opt-in fp64 Krylov bookkeeping ------------------------------------------
# The recurrence (x/r/p dot products, Givens/least-squares scalars) runs in
# fp64 while the SPH matvec stays at production fp32. On this state the fp32
# BiCGStab recurrence loses its shadow-system orthogonality (kappa(M^-1 A) is
# ~1e8, above the fp32 precision limit) and stagnates; the fp64 bookkeeping
# goes ~10x further at the same matvec cost. See the "BiCGStab deep-dive"
# section of docs/historic_plans/INCOMPRESSIBLE_SOLVER_PLAN.md.


def test_krylovFp64ConfigRoundTrip():
    cfg = IncompressibleSPHConfig()
    cfg.solverConfig.divergenceFreeSolver.krylovFp64 = True
    cfg.solverConfig.pressureSolver.krylovFp64 = True
    d = incompressibleConfigToDict(cfg)
    assert d['solverConfig']['divergenceFreeSolver']['krylovFp64'] is True
    assert d['solverConfig']['pressureSolver']['krylovFp64'] is True
    rt = dictToIncompressibleSPHConfig(d)
    assert rt.solverConfig.divergenceFreeSolver.krylovFp64 is True
    assert rt.solverConfig.pressureSolver.krylovFp64 is True
    # default is off; a dict missing the key falls back to the default
    assert IncompressibleSPHConfig().solverConfig.divergenceFreeSolver.krylovFp64 is False
    d2 = incompressibleConfigToDict(IncompressibleSPHConfig())
    del d2['solverConfig']['divergenceFreeSolver']['krylovFp64']
    del d2['solverConfig']['pressureSolver']['krylovFp64']
    rt2 = dictToIncompressibleSPHConfig(d2)
    assert rt2.solverConfig.divergenceFreeSolver.krylovFp64 is False


def test_krylovFp64DoesNotWorsenResidual(incompCase):
    case = incompCase
    scfg = case.schemeConfig.solverConfig.divergenceFreeSolver
    rel = {}
    try:
        for fp64 in (False, True):
            scfg.krylovFp64 = fp64
            case.state.pressures.zero_()
            _, pressure, _ = _run(case, PressureSolverType.bicgStab, maxIter=200)
            # the fp64 iterate must be cast back to the production dtype
            assert pressure.dtype == case.state.densities.dtype
            assert torch.isfinite(pressure).all()
            rel[fp64] = _relResid(case, pressure)
    finally:
        scfg.krylovFp64 = False
    # fp64 bookkeeping must not do worse than the fp32 recurrence on the same
    # budget (on this state it does ~10x better; the 1.5x margin keeps the
    # assertion robust across seeds/devices)
    assert rel[True] <= 1.5 * rel[False], f'fp64 {rel[True]} vs fp32 {rel[False]}'


# --- Optimal-step relaxed Jacobi (relaxationMode='optimal') ------------------
# The fixed-omega path converges only for omega < 2/rho(D^-1 A) (see the
# divergenceFree.py docstring); rho ~= 5.64 on this operator family (window
# omega < 0.355), so the historical omega=0.5 default diverges. `optimal`
# replaces omega with the exact per-step residual minimizer (same matvec
# cost, no stability window, monotonically decreasing residual). See the
# "Relaxed Jacobi: the omega stability window" section of
# docs/regression/incompressible_pressure_solver_choice.md.


def test_relaxationModeDefaultAndRoundTrip():
    # the default must stay fixed so the historical path is unchanged
    assert RelaxedJacobiSolverConfig().relaxationMode is JacobiRelaxationMode.fixed
    assert IncompressibleSPHConfig().solverConfig.divergenceFreeSolver.relaxationMode is \
        JacobiRelaxationMode.fixed
    assert IncompressibleSPHConfig().solverConfig.pressureSolver.relaxationMode is \
        JacobiRelaxationMode.fixed
    cfg = IncompressibleSPHConfig()
    cfg.solverConfig.divergenceFreeSolver.relaxationMode = JacobiRelaxationMode.optimal
    d = incompressibleConfigToDict(cfg)
    assert d['solverConfig']['divergenceFreeSolver']['relaxationMode'] == 'optimal'
    rt = dictToIncompressibleSPHConfig(d)
    assert rt.solverConfig.divergenceFreeSolver.relaxationMode is JacobiRelaxationMode.optimal
    # a dict missing the new key falls back to the default
    d2 = incompressibleConfigToDict(IncompressibleSPHConfig())
    del d2['solverConfig']['divergenceFreeSolver']['relaxationMode']
    rt2 = dictToIncompressibleSPHConfig(d2)
    assert rt2.solverConfig.divergenceFreeSolver.relaxationMode is JacobiRelaxationMode.fixed


def test_optimalStepJacobiMonotoneAndWindowFree(incompCase):
    """optimal step: monotonically decreasing residual over the full budget,
    and it works with relaxationFactor=0.5 -- outside the fixed stability
    window, where the fixed path diverges (that is the point of the mode)."""
    case = incompCase
    scfg = case.schemeConfig.solverConfig.divergenceFreeSolver
    scfg.solverType = PressureSolverType.relaxedJacobi
    scfg.relaxationMode = JacobiRelaxationMode.optimal
    scfg.minIterations, scfg.maxIterations = 0, 64
    scfg.tolerance, scfg.relaxationFactor = 1e-9, 0.5
    try:
        case.state.pressures.zero_()
        dvdt = torch.zeros_like(case.state.velocities)
        _, pressure, errors, _ = solveDivergenceFree(
            case.state, case.config, case.schemeConfig, case.adjacency, dvdt, case.dt)
        assert torch.isfinite(pressure).all()
        assert len(errors) == 64  # tol=1e-9 never met: full budget
        # monotone (tiny fp32 wiggle tolerated)
        for k in range(1, len(errors)):
            assert errors[k] <= errors[k - 1] * 1.01, f'non-monotone at {k}: {errors}'
        # substantial reduction over the budget
        assert errors[-1] < 0.5 * errors[0]
    finally:
        scfg.relaxationMode = JacobiRelaxationMode.fixed


def test_optimalStepAtLeastAsGoodAsInWindowFixed(incompCase):
    """optimal step must be at least as good as in-window fixed omega=0.3 at
    the same budget (it is the per-step residual minimizer; the 1.15x margin
    keeps the assertion robust across fp32 rounding)."""
    case = incompCase
    scfg = case.schemeConfig.solverConfig.divergenceFreeSolver
    scfg.solverType = PressureSolverType.relaxedJacobi
    scfg.minIterations, scfg.maxIterations = 0, 64
    scfg.tolerance, scfg.relaxationFactor = 1e-9, 0.3
    dvdt = torch.zeros_like(case.state.velocities)
    try:
        case.state.pressures.zero_()
        _, _, err_fixed, _ = solveDivergenceFree(
            case.state, case.config, case.schemeConfig, case.adjacency, dvdt, case.dt)
        scfg.relaxationMode = JacobiRelaxationMode.optimal
        case.state.pressures.zero_()
        _, _, err_opt, _ = solveDivergenceFree(
            case.state, case.config, case.schemeConfig, case.adjacency, dvdt, case.dt)
    finally:
        scfg.relaxationMode = JacobiRelaxationMode.fixed
    assert len(err_opt) == len(err_fixed) == 64
    assert err_opt[-1] <= 1.15 * err_fixed[-1], \
        f'optimal {err_opt[-1]} worse than in-window fixed-0.3 {err_fixed[-1]}'


def test_optimalStepRejectedForConstantDensitySolver(incompCase):
    """the constant-density solver clamps pressures non-negative, which breaks
    the exact residual recurrence the optimal step needs: it must refuse."""
    case = incompCase
    scfg = case.schemeConfig.solverConfig.pressureSolver
    scfg.solverType = PressureSolverType.relaxedJacobi
    scfg.relaxationMode = JacobiRelaxationMode.optimal
    dvdt = torch.zeros_like(case.state.velocities)
    try:
        with pytest.raises(ValueError, match='optimal'):
            solveIncompressible(case.state, case.config, case.schemeConfig,
                                case.adjacency, dvdt, case.dt)
    finally:
        scfg.relaxationMode = JacobiRelaxationMode.fixed