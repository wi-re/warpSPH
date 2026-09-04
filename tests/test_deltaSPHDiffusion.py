"""`modules/deltaSPH/wp_densityDelta.py`'s `psi_ij`: what the corrected
delta-SPH diffusion operator must annihilate.

Antuono et al. 2010/2012's correction to the Molteni-Colagrossi density
Laplacian exists to make the diffusive term a *bi*-Laplacian, so that it damps
high-frequency density noise without eating the smooth field underneath. Its
defining algebraic property is pair-by-pair cancellation on a field that is
linear in space: with `f = a.x`,

    (grad f_i + grad f_j) . gradW_ij  ==  2 a . gradW_ij
    2 (f_j - f_i) x_ij . gradW_ij / |x_ij|^2  ==  -2 a . gradW_ij

so `psi_ij . gradW_ij == 0` for every pair, exactly, at any resolution and on
any particle distribution. A quadratic field is annihilated too, up to
discretisation error, since a bi-Laplacian sees only fourth derivatives.

This is a regression test, not a discovery: through 2026-09-05 the gradient
term entered `psi` with the wrong sign, so the two terms *added* instead of
cancelling and `deltaSPH` was numerically `2 x densityOnly` on any smooth
field -- a second-order diffusion twice as strong as the uncorrected one,
rather than the fourth-order one it is meant to be. It never blew up (the sign
was still diffusive), which is exactly why a property test is the thing that
catches it. `scripts/probe_deltaSPHPsiProjection.py` has the full write-up and
checks the operator against an independent `O(N^2)` torch reference.

The same property is ACSPH's acceptance criterion for AC-2L (De Courcy et al.
2024 Eq. 33 == this operator with the pressure in place of the density): its
Sec. 4.1.1 separates AC-2L from AC-2 precisely by whether the operator can hold
a hydrostatic -- i.e. linear -- pressure gradient.
"""

import pytest
import torch

from warpSPHCore import (OperationDirection, OperationProperties, ParticleState,
                         SupportScheme, WarpOperation,
                         computeRenormalizationMatrices)
from warpSPH.configurations.simulationConfig import SimulationConfig
from warpSPH.enumTypes import DensityDiffusionScheme
from warpSPH.modules.deltaSPH import computeScalarFieldDiffusion
from warpSPH.modules.density.gradRhoL import computeGradRhoL
from warpSPH.utils.domain import buildDomainDescription

N_PER_SIDE = 20
JITTER = 0.05


@pytest.fixture(scope='module')
def lattice(runtime):
    """A jittered 2D lattice on the unit square. The jitter matters: on a
    perfect lattice several of these cancel by symmetry alone, which would let
    a broken operator pass."""
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    dx = 1.0 / N_PER_SIDE
    xs = (torch.arange(N_PER_SIDE, device=device, dtype=dtype) + 0.5) * dx
    gx, gy = torch.meshgrid(xs, xs, indexing='ij')
    positions = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1).contiguous()
    torch.manual_seed(0)
    positions = positions + JITTER * dx * torch.randn_like(positions)
    n = positions.shape[0]

    config = SimulationConfig(
        device=device, dtype=dtype, dim=2,
        domain=buildDomainDescription(l=4.0, dim=2, periodic=False,
                                      device=device, dtype=dtype))
    h = config.n_h * dx
    state = ParticleState(
        positions=positions,
        supports=torch.full((n,), h, device=device, dtype=dtype),
        masses=torch.full((n,), dx ** 2, device=device, dtype=dtype),
        densities=torch.ones(n, device=device, dtype=dtype),
        kinds=torch.zeros(n, dtype=torch.int32, device=device))
    # Kernel truncation at the walls is a boundary-condition question, not an
    # operator one; only rows with a full neighbourhood are graded.
    interior = ((positions > 1.5 * h) & (positions < 1.0 - 1.5 * h)).all(-1)
    _, _, L = computeRenormalizationMatrices(
        queryParticles=state,
        operationProperties=OperationProperties(
            kernel=config.kernel, operation=WarpOperation.Gradient,
            operationMode=OperationDirection.AllToAll,
            supportMode=SupportScheme.SuperSymmetric),
        domain=config.domain, returnEigVals=True)
    return state, config, L, interior


def diffusion(lattice, scheme, field):
    state, config, L, interior = lattice
    gradFieldL = computeGradRhoL(state, config, None, None, L, field=field)
    out = computeScalarFieldDiffusion(state, config, None, scheme,
                                      gradFieldL=gradFieldL, field=field)
    return out[interior]


def test_correctedOperatorAnnihilatesALinearField(lattice):
    """The exact property -- pair-by-pair cancellation, so this is machine
    precision, not a discretisation estimate."""
    state, _, _, _ = lattice
    a = torch.tensor([0.7, -1.3], device=state.positions.device,
                     dtype=state.positions.dtype)
    field = state.positions @ a + 2.0

    corrected = diffusion(lattice, DensityDiffusionScheme.deltaSPH, field)
    uncorrected = diffusion(lattice, DensityDiffusionScheme.densityOnly, field)

    scale = float(uncorrected.pow(2).mean().sqrt())
    assert scale > 1e-2, 'the uncorrected operator must NOT annihilate this'
    # 1e-3, not machine zero: the cancellation is exact in exact arithmetic,
    # but the individual pair terms are `O(|a| |gradW|)` and gradW carries a
    # `1/h^3`, so float32 leaves ~1e-4 relative after summing a neighbourhood.
    # The uncorrected operator is 4-5 orders above this.
    residual = float(corrected.pow(2).mean().sqrt())
    assert residual < 1e-3 * scale, (
        f'corrected rms {residual:.3e} vs uncorrected {scale:.3e}: the gradient '
        f'and difference terms are not cancelling')


def test_correctedOperatorNearlyAnnihilatesAQuadraticField(lattice):
    """A bi-Laplacian sees only fourth derivatives, so a quadratic survives
    only as discretisation error -- two orders down on the uncorrected
    operator, not machine zero."""
    state, _, _, _ = lattice
    field = (state.positions ** 2).sum(-1)

    corrected = diffusion(lattice, DensityDiffusionScheme.deltaSPH, field)
    uncorrected = diffusion(lattice, DensityDiffusionScheme.densityOnly, field)

    ratio = float(corrected.pow(2).mean().sqrt() / uncorrected.pow(2).mean().sqrt())
    assert ratio < 0.05, f'corrected/uncorrected rms = {ratio:.3g}'


def test_theGradientTermDoesNotSimplyDoubleTheDifferenceTerm(lattice):
    """The specific shape of the 2026-09-05 sign bug, pinned directly: with the
    signs wrong, `deltaOnly` came out *equal* to `densityOnly` (rather than its
    negative) and `deltaSPH` was exactly their sum."""
    state, _, _, _ = lattice
    a = torch.tensor([0.7, -1.3], device=state.positions.device,
                     dtype=state.positions.dtype)
    field = state.positions @ a + 2.0

    gradientTerm = diffusion(lattice, DensityDiffusionScheme.deltaOnly, field)
    differenceTerm = diffusion(lattice, DensityDiffusionScheme.densityOnly, field)

    scale = float(differenceTerm.abs().max())
    assert float((gradientTerm + differenceTerm).abs().max()) < 1e-3 * scale
    assert float((gradientTerm - differenceTerm).abs().max()) > 0.5 * scale


def test_theUnrenormalizedVariantCancelsToo(lattice):
    """`denormalized` is the same combination with the plain (un-`L`) gradient.
    The cancellation is a property of the *forms*, so it must hold there too --
    only to the accuracy of the uncorrected gradient, which is why this is a
    loose bound and the renormalized one is machine zero."""
    state, config, _, interior = lattice
    from warpSPHCore import GradientScheme, warpOperation
    a = torch.tensor([0.7, -1.3], device=state.positions.device,
                     dtype=state.positions.dtype)
    field = state.positions @ a + 2.0
    gradPlain = warpOperation(
        state,
        OperationProperties(kernel=config.kernel, operation=WarpOperation.Gradient,
                            supportMode=SupportScheme.SuperSymmetric,
                            operationMode=OperationDirection.AllToAll,
                            gradientMode=GradientScheme.Difference),
        queryValues=field, domain=config.domain)
    corrected = computeScalarFieldDiffusion(
        state, config, None, DensityDiffusionScheme.denormalized,
        gradField=gradPlain, field=field)[interior]
    uncorrected = diffusion(lattice, DensityDiffusionScheme.densityOnly, field)

    ratio = float(corrected.pow(2).mean().sqrt() / uncorrected.pow(2).mean().sqrt())
    assert ratio < 0.2, f'corrected/uncorrected rms = {ratio:.3g}'
