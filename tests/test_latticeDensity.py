"""`warpSPHCore.util.latticeDensity` -- the closed-form perfect-lattice density.

The reference each route is graded against is the brute-force integer loop over
the lattice box, written out longhand here so the test does not share code with
the thing it is testing.
"""
import itertools
import math

import numpy as np
import pytest

from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.kernels.eval_kernel import eval_k, eval_C_d
from warpSPHCore.type_config import scalar_t

#: `L` is evaluated with the repo's own `eval_k`, which computes in the
#: configured working precision -- float32 under `tests/conftest.py`. The shell
#: sum is exact *as maths*; what it can be asserted to is what that precision
#: allows, so the tolerance follows it rather than being a fixed 1e-12 (which
#: passed only because both sides happened to round identically before
#: `latticeDensity` started casting its argument to `scalar_t` -- needed so it
#: works under a float64 configuration at all, where a raw Python float finds
#: no `eval_k` overload).
EXACT = 1e-12 if scalar_t(0).__class__.__name__ == 'float64' else 5e-7
from warpSPHCore.util import (epsteinZeta, jacobiR2, latticeDensity,
                              latticeDensityFactor,
                              latticeDensityIsStrictlyAbove1, shellCounts)

KERNELS = [KernelFunctions.CubicSpline, KernelFunctions.QuarticSpline,
           KernelFunctions.QuinticSpline, KernelFunctions.Wendland2,
           KernelFunctions.Wendland4, KernelFunctions.Wendland6]
#: Deliberately includes a non-integer ratio: the shell route has to be exact
#: there too, which is the whole reason it replaces an integer loop.
RATIOS = [2.0, 3.0, 3.7, 4.0, 5.0]


def _bruteForce(kernel, n_h, dim):
    """`L` by summing every lattice point in the enclosing box, longhand."""
    reach = int(math.ceil(n_h))
    total = 0.0
    for offset in itertools.product(range(-reach, reach + 1), repeat=dim):
        q = math.sqrt(sum(i * i for i in offset)) / n_h
        if q <= 1.0:
            total += float(eval_k(scalar_t(q), dim, kernel.value))
    return float(eval_C_d(dim, kernel.value)) * total / n_h ** dim


@pytest.mark.parametrize('kernel', KERNELS)
@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('n_h', RATIOS)
def test_shellSumIsExact(kernel, dim, n_h):
    assert latticeDensity(kernel, n_h, dim) == pytest.approx(
        _bruteForce(kernel, n_h, dim), abs=EXACT)


@pytest.mark.parametrize('kernel', KERNELS)
@pytest.mark.parametrize('dim', [1, 2, 3])
def test_fourierIdentityMatchesRealSpace(kernel, dim):
    """Poisson summation is an identity, not an approximation; the tolerance is
    the float64 cancellation in the quadrature for `what`, not the maths."""
    for n_h in RATIOS:
        assert latticeDensity(kernel, n_h, dim, 'fourier') == pytest.approx(
            latticeDensity(kernel, n_h, dim), abs=2e-5)


@pytest.mark.parametrize('kernel', [KernelFunctions.Wendland2,
                                    KernelFunctions.Wendland4,
                                    KernelFunctions.Wendland6])
@pytest.mark.parametrize('dim', [1, 2, 3])
def test_closedFormRecoversMostOfTheOffset(kernel, dim):
    """The sum-free form drops the oscillatory remainder, so it is graded on
    the fraction of the *offset* it recovers, not on absolute agreement.

    2D is far better than 1D/3D and that is structural, not tuning: the
    oscillatory term is one half-power below the algebraic one, and the gap
    between them widens with the dimension of the algebraic term. Measured over
    `n_h` in [3, 6] the worst residual/offset is 0.054 (2D), 0.272 (3D), 0.341
    (1D); the thresholds here are those with a little headroom.
    """
    tolerance = 0.10 if dim == 2 else 0.40
    for n_h in (4.0, 5.0):
        exact = latticeDensity(kernel, n_h, dim)
        closed = latticeDensity(kernel, n_h, dim, 'closed')
        assert abs(closed - exact) < tolerance * abs(exact - 1.0)


def test_closedFormRefusesSplines():
    with pytest.raises(KeyError):
        latticeDensity(KernelFunctions.CubicSpline, 4.0, 2, 'closed')


@pytest.mark.parametrize('jmax', [97, 2000])
def test_jacobiTwoSquareMatchesDirectCount(jmax):
    assert np.array_equal(jacobiR2(jmax), shellCounts(2, jmax))


def test_epsteinZetaAgainstKnownConstants():
    #: Z_1(1) = 2 zeta(2) = pi^2/3; Z_2(1) is the (divergent) 2D case avoided
    #: by only ever asking for sigma >= 3/2 here.
    assert epsteinZeta(1, 1.0) == pytest.approx(math.pi ** 2 / 3, rel=1e-12)
    assert epsteinZeta(1, 2.0) == pytest.approx(math.pi ** 4 / 45, rel=1e-12)
    #: Z_2(sigma) = 4 zeta(sigma) beta(sigma); at sigma = 2 that is
    #: 4 * zeta(2) * beta(2) = 4 * (pi^2/6) * Catalan.
    catalan = 0.915965594177219015
    assert epsteinZeta(2, 2.0) == pytest.approx(
        4 * (math.pi ** 2 / 6) * catalan, rel=1e-10)


@pytest.mark.parametrize('kernel,expected', [
    (KernelFunctions.Wendland2, True), (KernelFunctions.Wendland4, True),
    (KernelFunctions.Wendland6, True), (KernelFunctions.CubicSpline, False)])
def test_positiveDefinitePredicateMatchesMeasuredSign(kernel, expected):
    """Wendland kernels are positive definite, so `L > 1` at every ratio and no
    `h` solves `L = 1`; the splines cross 1 and do have roots."""
    assert latticeDensityIsStrictlyAbove1(kernel) is expected
    values = [latticeDensity(kernel, r / 10, 2) for r in range(20, 81)]
    assert all(v > 1.0 for v in values) is expected


def test_factorIsTheReciprocal():
    for kernel in KERNELS:
        assert (latticeDensityFactor(kernel, 4.0, 2)
                == pytest.approx(1.0 / latticeDensity(kernel, 4.0, 2), rel=1e-15))


# --------------------------------------------------------------------------- #
# The correction itself: does switching it on make a real operator read rho0?
# --------------------------------------------------------------------------- #
import torch
from warpSPHCore import (OperationProperties, ParticleState, WarpOperation,
                         warpOperation)
from warpSPHCore.util import n_h_to_nH, volumeToSupport
from warpSPH.utils import buildDomainDescription


def _perfectLattice(kernel, n_h, dim, calibrate, dx=0.05):
    """Summation density at the centre of a defect-free lattice, through the
    real operator -- not the closed form."""
    h = volumeToSupport(dx ** dim, n_h_to_nH(n_h, dim), dim)
    reach = int(math.ceil(n_h)) + 1
    axis = np.arange(-reach, reach + 1) * dx
    grid = np.stack(np.meshgrid(*([axis] * dim), indexing='ij'), -1).reshape(-1, dim)
    dtype = torch.get_default_dtype()

    def state(pos):
        n = len(pos)
        return ParticleState(positions=torch.tensor(pos, dtype=dtype),
                             supports=torch.full((n,), h, dtype=dtype),
                             masses=torch.full((n,), dx ** dim, dtype=dtype),
                             densities=torch.ones(n, dtype=dtype),
                             kinds=torch.zeros(n, dtype=torch.int32))

    return float(warpOperation(
        queryParticles=state(np.zeros((1, dim))),
        referenceParticles=state(grid),
        operationProperties=OperationProperties(
            operation=WarpOperation.Density, kernel=kernel,
            n_h=n_h, calibrateNormalization=calibrate),
        domain=buildDomainDescription(l=4 * h, dim=dim, periodic=False, dtype=dtype),
    )[0])


@pytest.mark.parametrize('kernel', [KernelFunctions.Wendland2,
                                    KernelFunctions.Wendland4,
                                    KernelFunctions.CubicSpline])
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('n_h', [3.0, 4.0, 4.7])
def test_calibratedLatticeReadsRestDensity(kernel, dim, n_h):
    """With the flag on, a perfect lattice measures exactly 1 -- including for
    CubicSpline at n_h = 4, where L < 1 and the correction goes the other way."""
    assert _perfectLattice(kernel, n_h, dim, True) == pytest.approx(1.0, abs=EXACT)


@pytest.mark.parametrize('kernel', [KernelFunctions.Wendland2,
                                    KernelFunctions.CubicSpline])
def test_uncalibratedIsTheClosedFormOffset(kernel):
    """With the flag off nothing changes, and what it measures is exactly the
    `L` this module computes -- the two halves of the claim in one test."""
    for n_h in (3.0, 4.0):
        assert _perfectLattice(kernel, n_h, 2, False) == pytest.approx(
            latticeDensity(kernel, n_h, 2), abs=EXACT)


def test_correctionScalesEveryDerivativeAlike():
    """A uniform rescale of W must rescale grad W by the SAME factor, or the
    operators stop being mutually consistent. Checked on the ratio rather than
    the values, so it is independent of what the gradient itself is."""
    from warpSPHCore import GradientScheme, SupportScheme
    dim, n_h, dx = 2, 4.0, 0.05
    h = volumeToSupport(dx ** dim, n_h_to_nH(n_h, dim), dim)
    axis = np.arange(-5, 6) * dx
    grid = np.stack(np.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, dim)
    dtype = torch.get_default_dtype()
    n = len(grid)
    pts = ParticleState(positions=torch.tensor(grid, dtype=dtype),
                        supports=torch.full((n,), h, dtype=dtype),
                        masses=torch.full((n,), dx ** dim, dtype=dtype),
                        densities=torch.ones(n, dtype=dtype),
                        kinds=torch.zeros(n, dtype=torch.int32))
    values = torch.tensor(grid[:, 0], dtype=dtype)          # a linear field
    domain = buildDomainDescription(l=4 * h, dim=dim, periodic=False, dtype=dtype)

    def run(operation, calibrate, **kw):
        return warpOperation(
            queryParticles=pts, referenceParticles=pts,
            operationProperties=OperationProperties(
                operation=operation, kernel=KernelFunctions.Wendland2,
                supportMode=SupportScheme.Gather,
                gradientMode=GradientScheme.Naive,
                n_h=n_h, calibrateNormalization=calibrate),
            domain=domain, **kw)

    expected = latticeDensityFactor(KernelFunctions.Wendland2, n_h, dim)
    rho = run(WarpOperation.Density, False), run(WarpOperation.Density, True)
    grad = (run(WarpOperation.Gradient, False, queryValues=values, referenceValues=values),
            run(WarpOperation.Gradient, True, queryValues=values, referenceValues=values))
    assert float(rho[1][0] / rho[0][0]) == pytest.approx(expected, rel=1e-5)
    # Componentwise: the y-component of grad(x) is identically zero, so a
    # per-particle mask still divides 0/0 and yields nan.
    base = grad[0].flatten()
    live = base.abs() > 1e-4
    assert int(live.sum()) > 50, 'not enough non-zero gradient components to test'
    ratio = grad[1].flatten()[live] / base[live]
    assert ratio.min().item() == pytest.approx(expected, rel=1e-5)
    assert ratio.max().item() == pytest.approx(expected, rel=1e-5)
