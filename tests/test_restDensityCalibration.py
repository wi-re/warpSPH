"""`calibrateRestDensity`'s residual check: on a regular, unjittered lattice the sampler's mass/cell ratio
is 1 to ~1e-6, so a larger deviation means the initial state was corrupted and `onResidual` decides what
happens ('raise' by default). Nothing exercised that path before (OPEN_PROBLEMS §6 item 1)."""

from __future__ import annotations

import warnings

import pytest

from warpSPH.cases.sloshingTank import sloshingTankCase
from warpSPH.cases.weaklyCompressible import calibrateRestDensity
from warpSPH.runner import run


@pytest.fixture(scope='module')
def lattice():
    res = run(sloshingTankCase, nx=60, nSteps=1, progress=False, quiet=True,
              params=dict(calibrateRestDensity=False))
    return res.ctx, res.state


@pytest.fixture
def fresh(lattice):
    """The lattice with masses and the normalisation flag restored after each test."""
    ctx, system = lattice
    masses = system.state.masses.clone()
    flag = ctx.config.calibrateNormalization
    ctx.config.calibrateNormalization = False
    yield ctx, system
    system.state.masses.copy_(masses)
    ctx.config.calibrateNormalization = flag


def _corrupt(system, factor):
    fluid = system.state.kinds == 0
    system.state.masses[fluid] *= factor


def test_a_clean_lattice_passes_and_enables_the_normalisation(fresh):
    ctx, system = fresh
    rho = calibrateRestDensity(ctx, system)
    assert abs(ctx.scratch['latticeDensitySplit']['massRatio'] - 1.0) < 1e-4
    assert ctx.config.calibrateNormalization
    assert abs(rho - 1.0) < 1e-2


def test_a_bulk_mass_error_raises_by_default(fresh):
    ctx, system = fresh
    _corrupt(system, 1.01)
    with pytest.raises(ValueError, match='massRatio'):
        calibrateRestDensity(ctx, system)
    assert not ctx.config.calibrateNormalization, 'a refused calibration must not have been applied'


def test_warn_reports_and_continues(fresh):
    ctx, system = fresh
    _corrupt(system, 1.01)
    with pytest.warns(UserWarning, match='massRatio'):
        calibrateRestDensity(ctx, system, onResidual='warn')
    assert ctx.config.calibrateNormalization


def test_ignore_is_silent(fresh):
    ctx, system = fresh
    _corrupt(system, 1.01)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        calibrateRestDensity(ctx, system, onResidual='ignore')
    assert ctx.config.calibrateNormalization


def test_the_tolerance_is_what_decides(fresh):
    ctx, system = fresh
    _corrupt(system, 1.01)
    calibrateRestDensity(ctx, system, tolerance=0.05)   # 1 % is inside a 5 % tolerance


def test_an_invalid_onResidual_is_rejected_once_there_is_a_residual(fresh):
    ctx, system = fresh
    _corrupt(system, 1.01)
    with pytest.raises(ValueError, match='raise/warn/ignore'):
        calibrateRestDensity(ctx, system, onResidual='whatever')
