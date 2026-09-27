"""`SimulationConfig.densityCorrection` -- the warpSPH-side wiring of the
D&A (2012) eq. 18/19 self-term correction.

The maths and constants live in `warpSPHCore.util.densityCorrection`
(graded by the warpSPHCore test suite); this file covers the config
plumbing (`buildConfig`'s bool shorthand, the serialization round-trip)
and the `computeDensities` hook: off is bit-exact with the raw operator,
on is bit-exact with the util applied to the raw estimate, and on a
defect-free densest lattice (2D hexagonal / 3D FCC, the refit's own
lattice) the corrected estimate lands inside the refit band
(`results/eps_constants_multidim.json`, runtime N_H convention).

CPU build; the precision follows tests/conftest.py (float32).
"""
import math

import numpy as np
import pytest
import torch

from warpSPHCore import (DomainDescription, KernelFunctions, OperationProperties,
                         ParticleState, SupportScheme, WarpOperation,
                         buildVerletList, warpOperation)
from warpSPHCore.sampling.lattice import sampleDensestLattice
from warpSPHCore.util import applyDensityCorrection, densityCorrectionConstants

from warpSPH.configurations.simulationConfig import (DensityCorrection,
                                                     buildConfig,
                                                     configurationToDict,
                                                     dictToConfig)
from warpSPH.modules.density import computeDensities

WENDLANDS = [KernelFunctions.Wendland2, KernelFunctions.Wendland4,
             KernelFunctions.Wendland6]

_V = {2: math.pi, 3: 4.0 * math.pi / 3.0}

#: The refit's corrected-curve band: max |corr - 1| over the closed window
#: 40 <= N_H <= 400 (dense sweep, runtime N_H convention) plus headroom for
#: the warp launch's float32 accumulation noise (the warpSPHCore suite
#: grades the same bands on the pure-torch path).
BAND = {
    (2, KernelFunctions.Wendland2): 3.0e-4,
    (2, KernelFunctions.Wendland4): 7.5e-5,
    (2, KernelFunctions.Wendland6): 4.5e-5,
    (3, KernelFunctions.Wendland2): 6.0e-3,
    (3, KernelFunctions.Wendland4): 3.5e-3,
    (3, KernelFunctions.Wendland6): 1.4e-2,
}

_LATTICE = {}
_DOMAIN = {}
_STATE = {}
_ADJACENCY = {}


def _lattice(dim):
    """The refit's densest lattice at rho = 1 (2D hexagonal / 3D FCC)."""
    if dim not in _LATTICE:
        _LATTICE[dim] = sampleDensestLattice({2: 4096, 3: 4000}[dim], 1.0, dim)
    return _LATTICE[dim]


def _massOf(dim) -> float:
    lat = _lattice(dim)
    return float(np.prod(lat.box)) / lat.count


def _H(dim, N_H) -> float:
    """Support radius for a lattice with N_H neighbours at rho = 1."""
    return (N_H * _massOf(dim) / _V[dim]) ** (1.0 / dim)


def _domainFor(dim) -> DomainDescription:
    """A periodic domain that is exactly the lattice's own box, so the
    tiling is defect-free and every particle sees the refit's environment."""
    if dim not in _DOMAIN:
        lat = _lattice(dim)
        dtype = torch.get_default_dtype()
        _DOMAIN[dim] = DomainDescription(
            min=torch.zeros(dim, dtype=dtype),
            max=torch.tensor(lat.box, dtype=dtype),
            periodic=torch.ones(dim, dtype=torch.bool),
            dim=dim)
    return _DOMAIN[dim]


def _stateAndDomain(dim, H):
    """A defect-free periodic lattice at rho = 1 with uniform support H."""
    key = (dim, H)
    if key not in _STATE:
        lat = _lattice(dim)
        N = lat.count
        dtype = torch.get_default_dtype()
        state = ParticleState(
            positions=torch.tensor(lat.positions, dtype=dtype),
            supports=torch.full((N,), H, dtype=dtype),
            masses=torch.full((N,), _massOf(dim), dtype=dtype),
            densities=torch.ones(N, dtype=dtype),
            kinds=torch.zeros(N, dtype=torch.int32))
        _STATE[key] = (state, _domainFor(dim), _massOf(dim))
    return _STATE[key]


def _adjacency(dim, H):
    key = (dim, H)
    if key not in _ADJACENCY:
        state, domain, _ = _stateAndDomain(dim, H)
        _ADJACENCY[key] = buildVerletList(
            state, domain, supportMode=SupportScheme.Gather,
            priorNeighborhood=None, verbose=False)
    return _ADJACENCY[key]


def _buildConfig(dim, kernel, dc):
    _, domain, _ = _stateAndDomain(dim, _H(dim, 100.0))
    config, _ = buildConfig(
        domain=domain,
        dim=dim,
        kernel=kernel,
        device=torch.device('cpu'),
        dtype=torch.get_default_dtype(),
        densityCorrection=dc,
    )
    return config


# --------------------------------------------------------------------------- #
# The hook: off is the identity, on is the util on the raw estimate
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('kernel', WENDLANDS)
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('off', [None, False, DensityCorrection()])
def test_flagOffIsBitExactWithRawOperator(kernel, dim, off):
    """Every OFF spelling (absent, bool shorthand, explicit dataclass) leaves
    the operator's tensor untouched -- the default path must stay bit-exact."""
    H = _H(dim, 100.0)
    state, domain, _ = _stateAndDomain(dim, H)
    config = _buildConfig(dim, kernel, off)
    adj = _adjacency(dim, H)
    got = computeDensities(state, config, None, adj)
    ref = warpOperation(
        state,
        OperationProperties(
            kernel=kernel, operation=WarpOperation.Density,
            supportMode=SupportScheme.Gather,
            n_h=config.n_h,
            calibrateNormalization=config.calibrateNormalization),
        domain=domain, adjacency=adj)
    assert torch.equal(got, ref)


@pytest.mark.parametrize('kernel', WENDLANDS)
@pytest.mark.parametrize('dim', [2, 3])
def test_flagOnIsBitExactWithTheUtilOnTheRawEstimate(kernel, dim):
    """ON is exactly `applyDensityCorrection` on the finished raw estimate --
    checked bit-exactly against the same call, and not a no-op."""
    H = _H(dim, 100.0)
    state, domain, _ = _stateAndDomain(dim, H)
    adj = _adjacency(dim, H)
    raw = computeDensities(state, _buildConfig(dim, kernel, False), None, adj)
    corr = computeDensities(state, _buildConfig(dim, kernel, True), None, adj)
    manual = applyDensityCorrection(raw, state.masses, state.supports,
                                    kernel, dim)
    assert torch.equal(corr, manual)
    assert not torch.equal(corr, raw)


def test_overrideReachesTheHook():
    """The DensityCorrection dataclass's eps100 override reaches the util;
    the bool form and the bare dataclass agree."""
    dim, kernel = 3, KernelFunctions.Wendland2
    H = _H(dim, 100.0)
    state, domain, _ = _stateAndDomain(dim, H)
    adj = _adjacency(dim, H)
    raw = computeDensities(state, _buildConfig(dim, kernel, False), None, adj)

    corr_bool = computeDensities(state, _buildConfig(dim, kernel, True), None, adj)
    corr_data = computeDensities(
        state, _buildConfig(dim, kernel, DensityCorrection(enabled=True)), None, adj)
    assert torch.equal(corr_bool, corr_data)

    eps100, _, _ = densityCorrectionConstants(kernel, dim)
    over = computeDensities(
        state, _buildConfig(dim, kernel,
                            DensityCorrection(enabled=True,
                                              eps100=2.0 * eps100)), None, adj)
    manual = applyDensityCorrection(raw, state.masses, state.supports,
                                    kernel, dim, eps100=2.0 * eps100)
    assert torch.equal(over, manual)


def test_nonShippedKernelRaises():
    """A kernel without fitted constants fails loudly instead of silently
    running uncorrected (the table lookup raises, see the warpSPHCore
    docstring for why this kernel is not shipped)."""
    dim, kernel = 2, KernelFunctions.CubicSpline
    H = _H(dim, 100.0)
    state, domain, _ = _stateAndDomain(dim, H)
    config = _buildConfig(dim, kernel, True)
    with pytest.raises(KeyError):
        computeDensities(state, config, None, _adjacency(dim, H))


# --------------------------------------------------------------------------- #
# The correction, end to end: within the refit band on the densest lattice
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('kernel', WENDLANDS)
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('N_H', [40.0, 100.0, 400.0])
def test_correctedLatticeWithinRefitBand(kernel, dim, N_H):
    """Through the real operator: the corrected densest-lattice estimate sits
    inside the refit band, and at the window edge the RAW estimate is clearly
    outside it (the correction is not a no-op)."""
    H = _H(dim, N_H)
    state, domain, _ = _stateAndDomain(dim, H)
    adj = _adjacency(dim, H)
    raw = computeDensities(state, _buildConfig(dim, kernel, False), None, adj)
    corr = computeDensities(state, _buildConfig(dim, kernel, True), None, adj)
    band = BAND[(dim, kernel)]
    raw_dev = float((raw - 1.0).abs().max())
    corr_dev = float((corr - 1.0).abs().max())
    assert corr_dev <= band, \
        f"{kernel.name} {dim}D N_H={N_H}: |corr-1| = {corr_dev:.3e} > {band}"
    if N_H == 40.0:
        # The bias is monotone in N_H (decreasing), so the window edge is
        # the worst case for the raw deviation.
        assert raw_dev >= 2.0 * band, \
            f"{kernel.name} {dim}D N_H=40: raw {raw_dev:.3e} is not clearly " \
            f"outside the {band} band"
        assert corr_dev < raw_dev


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #

def test_configSerializationRoundTrip():
    """The nested dataclass round-trips through configurationToDict /
    dictToConfig, and a dict written before the field existed still loads
    (the dataclass default)."""
    config, _ = buildConfig(
        domain=_domainFor(2),
        dim=2,
        kernel=KernelFunctions.Wendland4,
        device=torch.device('cpu'),
        dtype=torch.get_default_dtype(),
        densityCorrection=DensityCorrection(enabled=True, eps100=0.05,
                                            alpha=1.25))
    d = configurationToDict(config)
    assert d['densityCorrection'] == {
        'enabled': True, 'eps100': 0.05, 'alpha': 1.25}
    back = dictToConfig(d)
    assert back.densityCorrection == \
        DensityCorrection(enabled=True, eps100=0.05, alpha=1.25)

    d.pop('densityCorrection')
    assert dictToConfig(d).densityCorrection == DensityCorrection()
