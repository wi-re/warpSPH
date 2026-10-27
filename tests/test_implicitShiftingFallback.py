"""Tests for the opt-in inner-solve fallback chain
(`modules/shifting/solverDriver.py::solveImplicitSystem`, controlled by
`ShiftProperties.implicitFallback`, and its `ShiftingScheme.dynamic` entry
point `modules/shifting/implicitShifting.computeDynamicImplicitShift`).

Two layers:
  - Dense driver-contract tests (no warp/SPH): the fallback must be a strict
    no-op unless the primary bails (status < 0), and must never return an
    iterate worse than the primary's (it keeps the best by stamped true
    residual). This is what keeps the legacy `ShiftingScheme.implicit` path
    (default `implicitFallback=none`) byte-identical.
  - Production-operator tests (warp): on the exact operator
    `computeImplicitShift` solves, the `krylov` fallback recovers the jitter=0
    BiCGStab rho-breakdown that the legacy path used to consume silently.

The end-to-end `ShiftingScheme.dynamic` dispatch (through
`wrapper.solveShifting`) is covered by
`test_implicitShifting.py::test_shiftingConvergesToUniformDensity`.
"""

import pytest
import torch

from warpSPH.modules.shifting.bicgstab import bicgstabSolve
from warpSPH.modules.shifting.solverDriver import solveImplicitSystem, runKrylov
from warpSPH.configurations.moduleConfigurations.shifting import (
    ShiftingImplicitSolver, ShiftingImplicitFallback)


# ---------------------------------------------------------------------------
# Dense driver-contract tests (no warp/SPH)
# ---------------------------------------------------------------------------

def _matvec(A):
    return lambda x: A @ x


def _randomSpd(n, seed=0, dtype=torch.float32):
    g = torch.Generator(device='cpu').manual_seed(seed)
    M = torch.randn(n, n, generator=g)
    return (M @ M.T + n * torch.eye(n)).to(dtype)


def _stamped(hist):
    return float(hist[-1]) if hist else float('inf')


def _denseArgs(maxiter):
    return dict(tol=0.0, rtol=1e-6, maxiter=maxiter, precond=None, threshold=None, dim=1)


def test_driverNoneIsByteIdenticalToPrimary():
    # implicitFallback=none must return exactly what the primary solver
    # returns -- this is what keeps the legacy users' behavior unchanged
    n = 120
    A = _randomSpd(n, seed=0)
    b = A @ torch.randn(n)
    args = _denseArgs(500)
    x0 = torch.zeros(n)
    xN, iN, cN = solveImplicitSystem(_matvec(A), b, x0, args, ShiftingImplicitSolver.bicgstab, 30, ShiftingImplicitFallback.none)
    xB, iB, cB = bicgstabSolve(_matvec(A), b, x0, **args)
    assert torch.equal(xN, xB) and iN == iB
    assert len(cN) == len(cB) and all(torch.equal(a, bb) for a, bb in zip(cN, cB))


def test_driverConvergedPrimaryIsNotRetried():
    # a converged primary (status >= 0) short-circuits: fallback=krylov must
    # return the primary's result unchanged, not run the other solver
    n = 120
    A = _randomSpd(n, seed=1)
    b = A @ torch.randn(n)
    args = _denseArgs(500)
    x0 = torch.zeros(n)
    xK, iK, _ = solveImplicitSystem(_matvec(A), b, x0, args, ShiftingImplicitSolver.bicgstab, 30, ShiftingImplicitFallback.krylov)
    xB, iB, _ = bicgstabSolve(_matvec(A), b, x0, **args)
    assert iB >= 0, 'precondition: the primary must converge here'
    assert iK == iB and torch.equal(xK, xB)


def test_driverKrylovFallbackNeverWorse():
    # force a primary budget-bailout (maxiter=3); the fallback retries the
    # other Krylov solver and keeps the better iterate, so its stamped
    # residual must be <= the primary's, and the iterate must stay finite
    n = 120
    A = _randomSpd(n, seed=2)
    b = A @ torch.randn(n)
    args = _denseArgs(3)
    x0 = torch.zeros(n)
    _xP, iP, cP = runKrylov(_matvec(A), b, x0, ShiftingImplicitSolver.bicgstab, args, 30)
    xF, _iF, cF = solveImplicitSystem(_matvec(A), b, x0, args, ShiftingImplicitSolver.bicgstab, 30, ShiftingImplicitFallback.krylov)
    assert iP < 0, 'precondition: the primary must bail here'
    assert _stamped(cF) <= _stamped(cP)
    assert torch.isfinite(xF).all()


def test_driverKrylovRichardsonNeverWorse():
    # the full krylov_richardson chain: the Richardson polish is warm-started
    # from the best Krylov iterate and only kept if it improves, so the result
    # is still never worse than the bailed-out primary
    n = 120
    A = _randomSpd(n, seed=3)
    b = A @ torch.randn(n)
    args = _denseArgs(3)
    x0 = torch.zeros(n)
    _xP, iP, cP = runKrylov(_matvec(A), b, x0, ShiftingImplicitSolver.bicgstab, args, 30)
    xR, _iR, cR = solveImplicitSystem(_matvec(A), b, x0, args, ShiftingImplicitSolver.bicgstab, 30, ShiftingImplicitFallback.krylov_richardson)
    assert iP < 0
    assert _stamped(cR) <= _stamped(cP)
    assert torch.isfinite(xR).all()


# ---------------------------------------------------------------------------
# Production-operator tests (warp): the fallback on the exact operator
# computeImplicitShift solves
# ---------------------------------------------------------------------------

def _jitteredLatticeState(nx, dim, L, jitter, seed, device, dtype):
    from warpSPH.utils import buildDomainDescription
    from warpSPH.configurations.simulationConfig import buildConfig
    from warpSPH.sample.regular import sampleRegularParticles
    from warpSPHCore import ParticleState
    domain = buildDomainDescription(l=L, dim=dim, periodic=True, device=device, dtype=dtype)
    config, _integrator = buildConfig(dim=dim, nx=nx, domain=domain, device=device,
                                      dtype=dtype, dx=L / nx, cflFactor=0.3)
    template = sampleRegularParticles(nx=nx, domain=domain, targetNeighbors=config.targetNeighbors)
    n = template.positions.shape[0]
    particleDx = template.masses.pow(1 / dim).mean().item()
    gen = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(template.positions.shape, generator=gen).to(device=device, dtype=dtype)
    positions = template.positions + jitter * particleDx * noise
    state = ParticleState(
        positions=positions, supports=template.supports, masses=template.masses,
        kinds=torch.zeros(n, device=device, dtype=torch.int32),
        densities=torch.ones(n, device=device, dtype=dtype))
    state.velocities = torch.zeros_like(state.positions)
    return state, config, domain


def _productionSystem(state, config, domain, schemeConfig):
    # Reuse the exact production code `computeImplicitShift` calls internally
    # (same matvec / RHS / diagonal / preconditioner ingredients), so the
    # measured inner residual is the real solver's, not a reimplementation.
    from warpSPHCore import SupportScheme, buildVerletList
    from warpSPH.modules.density import computeDensities
    from warpSPH.modules.shifting.wp_implicitShifting import computeShiftingPairTerms
    from warpSPH.modules.shifting.implicitShifting import (
        _buildSystem, _buildDiagBlock, _multiplyLaplacianBlock)
    dim = state.positions.shape[1]
    numParticles = state.positions.shape[0]
    rho0 = schemeConfig.fluid.restDensity
    adjacency = buildVerletList(state, domain, config.verletScale, SupportScheme.SuperSymmetric, None)
    state.densities = computeDensities(state, config, schemeConfig, adjacency)
    _K, J, H = computeShiftingPairTerms(state, domain, config.kernel, adjacency)
    i_all, j_all = adjacency.livePairs()[0], adjacency.livePairs()[1]
    Hw_all, B, x0 = _buildSystem(state, config, schemeConfig, domain, adjacency,
                                 i_all, j_all, J, H, rho0, dim, numParticles)
    diagBlock, i, j, Hw = _buildDiagBlock(schemeConfig.shiftProperties.implicitOperator,
                                          i_all, j_all, Hw_all, numParticles, dim,
                                          state.positions.device, state.positions.dtype)

    def matvec(x, diagBlock=diagBlock, i=i, j=j, Hw=Hw, numParticles=numParticles, dim=dim):
        return _multiplyLaplacianBlock(diagBlock, Hw, x, i, j, numParticles, dim)

    dx = config.dx.cpu().item() if isinstance(config.dx, torch.Tensor) else config.dx
    threshold = schemeConfig.shiftProperties.implicitSolverThreshold or dx / 2
    diag = torch.diagonal(diagBlock, dim1=-2, dim2=-1).flatten()
    precond = torch.where(diag.abs() > 1e-8, 1.0 / diag, torch.zeros_like(diag))
    if not torch.any(diag.abs() > 1e-8):
        precond = None
    return dict(matvec=matvec, B=B, x0=x0, precond=precond, threshold=threshold, dim=dim)


def test_krylovFallbackRecoversProductionBreakdown():
    """jitter=0 is the documented BiCGStab rho-breakdown regime (see
    `scripts/troubleshoot_implicitShiftingConvergence.py`): the legacy path
    consumed the bailed-out iterate as if it had converged. With
    `implicitFallback=krylov`, the driver retries GMRES and recovers the same
    system to a far smaller inner residual. Run on CPU: the breakdown is a
    deterministic function of (seed, nx, jitter) and was verified there."""
    import warp as wp
    wp.init()
    from warpSPH.configurations.weaklyCompressible import WeaklyCompressibleSPHConfig
    device = torch.device('cpu')
    dtype = torch.float32
    state, config, domain = _jitteredLatticeState(16, 2, 1.0, 0.0, 1234, device, dtype)
    schemeConfig = WeaklyCompressibleSPHConfig()
    schemeConfig.shiftProperties.active = True
    schemeConfig.shiftProperties.implicitSolver = ShiftingImplicitSolver.bicgstab
    schemeConfig.surfaceDetectionConfig.active = False
    sys = _productionSystem(state, config, domain, schemeConfig)
    args = dict(tol=schemeConfig.shiftProperties.implicitTolerance,
                rtol=schemeConfig.shiftProperties.implicitRelativeTolerance,
                maxiter=schemeConfig.shiftProperties.implicitMaxSolverIter,
                precond=sys['precond'], threshold=sys['threshold'], dim=sys['dim'])
    bnorm = torch.linalg.norm(sys['B']).item()
    _xP, iP, cP = runKrylov(sys['matvec'], sys['B'], sys['x0'], ShiftingImplicitSolver.bicgstab, args, 30)
    xF, _iF, cF = solveImplicitSystem(sys['matvec'], sys['B'], sys['x0'], args,
                                      ShiftingImplicitSolver.bicgstab, 30, ShiftingImplicitFallback.krylov)
    assert iP < 0, f'expected the primary BiCGStab to bail at jitter=0, got {iP}'
    # the fallback must recover the system: far smaller inner residual, finite
    assert _stamped(cF) < 0.1 * _stamped(cP), (
        f'fallback rel-resid {_stamped(cF) / bnorm:.2e} not far below the '
        f'primary\'s {_stamped(cP) / bnorm:.2e}')
    assert torch.isfinite(xF).all()


def test_legacyImplicitPathUnchangedByDefault():
    """With the default config (implicitFallback=none), the driver must
    produce exactly the primary solver's iterate -- i.e. the legacy
    `ShiftingScheme.implicit` path is byte-identical to before the fallback
    existed. Checked here at the driver level on the production operator."""
    import warp as wp
    wp.init()
    from warpSPH.configurations.weaklyCompressible import WeaklyCompressibleSPHConfig
    device = torch.device('cpu')
    dtype = torch.float32
    state, config, domain = _jitteredLatticeState(16, 2, 1.0, 0.1, 1234, device, dtype)
    schemeConfig = WeaklyCompressibleSPHConfig()
    schemeConfig.shiftProperties.active = True
    schemeConfig.surfaceDetectionConfig.active = False
    assert schemeConfig.shiftProperties.implicitFallback is ShiftingImplicitFallback.none
    sys = _productionSystem(state, config, domain, schemeConfig)
    args = dict(tol=schemeConfig.shiftProperties.implicitTolerance,
                rtol=schemeConfig.shiftProperties.implicitRelativeTolerance,
                maxiter=schemeConfig.shiftProperties.implicitMaxSolverIter,
                precond=sys['precond'], threshold=sys['threshold'], dim=sys['dim'])
    xD, iD, _ = solveImplicitSystem(sys['matvec'], sys['B'], sys['x0'], args,
                                    schemeConfig.shiftProperties.implicitSolver, 30,
                                    schemeConfig.shiftProperties.implicitFallback)
    xB, iB, _ = runKrylov(sys['matvec'], sys['B'], sys['x0'],
                          schemeConfig.shiftProperties.implicitSolver, args, 30)
    assert iD == iB and torch.equal(xD, xB)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))