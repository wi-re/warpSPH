"""Independent cubic-kernel sums, thermodynamic closure and variational force.

Float64 checks are rerun in a fresh process because Warp's precision is
chosen at first import. No full simulation or GPU timing is claimed here.
"""

from dataclasses import fields, replace
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from warpSPH.configurations import buildConfig
from warpSPH.configurations.pesph import PESPHConfig, dictToPESPHConfig, pesphConfigToDict
from warpSPH.enumTypes import AdaptiveSupportScheme, PESPHVariant
from warpSPH.modules.adaptiveSupport import evaluateOptimalSupport
from warpSPH.modules.pesph import computePressureForce, computePressureState, computePressureSums, pressureWeights
from warpSPH.systems.pesph import PressureEnergyState, PressureEntropyState, pesphStateType
from warpSPH.utils.domain import buildDomainDescription
from warpSPHCore import KernelFunctions, SupportScheme, buildVerletList
from warpSPHCore.type_config import get_torch_precision
from warpSPHIntegrators.state import integrated_field_names

DTYPE = get_torch_precision()
DOUBLE = DTYPE is torch.float64
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VARIANTS = tuple(PESPHVariant)
TOL = 2e-11 if DOUBLE else 4e-5


def setup(dim=2, nx=10, *, variant=PESPHVariant.PressureEntropy, periodic=True, uniform=False):
    g = torch.Generator().manual_seed(892)
    domain = buildDomainDescription(2.0, dim, periodic=periodic, device=DEV, dtype=DTYPE)
    config, _ = buildConfig(dim=dim, domain=domain, kernel=KernelFunctions.CubicSpline,
                           n_h=4.0, device=DEV, dtype=DTYPE, calibrateNormalization=False)
    axis = torch.arange(nx, dtype=DTYPE) * (2.0 / nx) - 1.0 + 1.0 / nx
    x = torch.stack([q.reshape(-1) for q in torch.meshgrid(*([axis] * dim), indexing='ij')], dim=1)
    n = len(x)
    if not uniform:
        x += (0.12 / nx) * (torch.rand(x.shape, generator=g, dtype=DTYPE) - 0.5)
    z = torch.zeros(n, device=DEV, dtype=DTYPE)
    m = torch.full((n,), (2.0 / nx) ** dim, dtype=DTYPE)
    if not uniform:
        m *= 0.5 + torch.rand(n, generator=g, dtype=DTYPE)
    h = torch.full((n,), 4.0 * (2.0 / nx), dtype=DTYPE)
    if not uniform:
        h *= 0.8 + 0.3 * torch.rand(n, generator=g, dtype=DTYPE)
    v = torch.randn(x.shape, generator=g, dtype=DTYPE) * 0.1
    A = torch.ones(n, dtype=DTYPE) if uniform else 0.5 + torch.rand(n, generator=g, dtype=DTYPE)
    u = A / 0.4
    state = pesphStateType(variant)(
        positions=x.to(DEV), velocities=v.to(DEV), supports=h.to(DEV), masses=m.to(DEV),
        densities=torch.ones_like(z), kinds=torch.zeros(n, dtype=torch.int32, device=DEV),
        materials=torch.zeros(n, dtype=torch.int32, device=DEV), UIDs=torch.arange(n, dtype=torch.int32, device=DEV),
        UIDcounter=n, internalEnergies=u.to(DEV), totalEnergies=(m*(u + 0.5*v.square().sum(-1))).to(DEV),
        entropies=A.to(DEV), pressures=torch.ones_like(z), soundspeeds=torch.ones_like(z),
        divergence=z.clone(), alpha0s=z.clone(), alphas=z.clone())
    scheme = PESPHConfig(variant=variant, dissipation=False, adaptiveSupportScheme=AdaptiveSupportScheme.NoScheme,
                         adaptiveSupportCorrections=False)
    return state, config, scheme


def adjacency(state, config):
    return buildVerletList(state, config.domain, verletScale=1.0, supportMode=SupportScheme.SuperSymmetric)


def cubic(state, config):
    """Independent dense kernel matrix: no core kernel or neighbour-list code."""
    x = state.positions
    delta = x[:, None] - x[None, :]
    length = config.domain.max - config.domain.min
    delta = torch.where(config.domain.periodic, delta - length * torch.round(delta/length), delta)
    r = delta.norm(dim=-1)
    h = state.supports[:, None]
    q = r / h
    dim = x.shape[1]
    C = {1: 8/3, 2: 80/(7*math.pi), 3: 16/math.pi}[dim]
    k = (1-q).clamp_min(0).pow(3) - 4*(0.5-q).clamp_min(0).pow(3)
    dk = -3*(1-q).clamp_min(0).pow(2) + 12*(0.5-q).clamp_min(0).pow(2)
    W = C*k/h.pow(dim)
    dW = -C*(dim*k + q*dk)/h.pow(dim+1)
    return W, dW


@pytest.mark.parametrize('dim,nx', [(1, 40), (2, 10), (3, 6)])
@pytest.mark.parametrize('variant', VARIANTS)
def test_sums_match_independent_reference_including_self(dim, nx, variant):
    state, cfg, scheme = setup(dim, nx, variant=variant)
    weights = pressureWeights(state, variant, scheme.gamma)
    sums = computePressureSums(state, cfg, weights, adjacency(state, cfg))
    W, dW = cubic(state, cfg)
    expected = (W.sum(1), W @ weights, dW.sum(1), dW @ weights, W @ state.masses)
    for f, ref in zip(fields(sums), expected):
        # dn/dh nearly cancels on a lattice: use the field's scale, not
        # pointwise relative error at its zero crossings (float64 stays strict).
        assert (getattr(sums, f.name)-ref).abs().max() < TOL*ref.abs().max().clamp_min(1)


@pytest.mark.parametrize('variant', VARIANTS)
def test_support_derivatives_finite_difference(variant):
    state, cfg, scheme = setup(variant=variant)
    weights = pressureWeights(state, variant, scheme.gamma)
    sums = computePressureSums(state, cfg, weights, adjacency(state, cfg))
    eps = 2e-5 if DOUBLE else 2e-3
    Wplus, _ = cubic(replace(state, supports=state.supports+eps), cfg)
    Wminus, _ = cubic(replace(state, supports=state.supports-eps), cfg)
    for actual, finite in ((sums.numberSupportDerivative, ((Wplus-Wminus)/(2*eps)).sum(1)),
                           (sums.weightedSupportDerivative, ((Wplus-Wminus)/(2*eps)) @ weights)):
        assert torch.allclose(actual, finite, rtol=2e-7 if DOUBLE else 4e-3,
                              atol=2e-7 if DOUBLE else 4e-3)


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('adaptive', [False, True])
@pytest.mark.parametrize('dim,nx', [(1, 40), (2, 10), (3, 6)])
def test_pair_conserves_momentum_and_energy(variant, adaptive, dim, nx):
    state, cfg, scheme = setup(dim, nx, variant=variant, periodic=False)
    scheme.adaptiveSupportCorrections = adaptive
    adj = adjacency(state, cfg)
    pressure = computePressureState(state, cfg, scheme, adj)
    acc, work = computePressureForce(state, pressure, cfg, adj)
    force = state.masses[:, None] * acc
    assert force.sum(0).abs().max() < TOL*force.abs().sum()
    if dim == 2:
        torque = state.positions[:, 0]*force[:, 1] - state.positions[:, 1]*force[:, 0]
        assert torque.sum().abs() < TOL*force.abs().sum()
    elif dim == 3:
        torque = torch.linalg.cross(state.positions, force).sum(0)
        assert torque.abs().max() < TOL*force.abs().sum()
    kineticRate = (force*state.velocities).sum()
    heatRate = (state.masses*work).sum()
    assert abs(kineticRate+heatRate) < TOL*((force*state.velocities).abs().sum() + (state.masses*work).abs().sum())


@pytest.mark.parametrize('adaptive', [False, True])
@pytest.mark.parametrize('dim,nx', [(1, 40), (2, 10)])
def test_entropy_force_is_gradient_of_thermal_energy(adaptive, dim, nx):
    state, cfg, scheme = setup(dim=dim, nx=nx)
    if adaptive:
        scheme = replace(scheme, adaptiveSupportScheme=AdaptiveSupportScheme.NumberDensity,
                         adaptiveSupportCorrections=True, adaptiveSupportIterations=40,
                         adaptiveSupportThreshold=1e-7 if DOUBLE else 1e-5)
    def close(s):
        if adaptive:
            _, s.supports, adj, *_ = evaluateOptimalSupport(s, cfg, scheme, SupportScheme.Gather)
        else:
            adj = adjacency(s, cfg)
        return computePressureState(s, cfg, scheme, adj), adj
    pressure, adj = close(state)
    acc, _ = computePressureForce(state, pressure, cfg, adj)
    eps = 2e-6 if DOUBLE else 4e-4
    for i in (0, 13, 27):
        axis = i % dim
        energies = []
        for sign in (-1, 1):
            perturbed = state.clone()
            perturbed.positions[i, axis] += sign*eps
            p, _ = close(perturbed)
            energies.append((perturbed.masses*p.internalEnergies).sum())
        force = -(energies[1]-energies[0])/(2*eps)
        assert torch.allclose(state.masses[i]*acc[i, axis], force,
                              rtol=3e-6 if DOUBLE else 2e-2,
                              atol=2e-7 if DOUBLE else 3e-3)


@pytest.mark.parametrize('variant', VARIANTS)
def test_force_matches_published_variant_coefficients(variant):
    state, cfg, scheme = setup(variant=variant)
    scheme.adaptiveSupportCorrections = True
    adj = adjacency(state, cfg)
    pressure = computePressureState(state, cfg, scheme, adj)
    acc, work = computePressureForce(state, pressure, cfg, adj)
    W, dW = cubic(state, cfg)
    n = W.sum(1)
    weights = pressureWeights(state, variant, scheme.gamma)
    y = W @ weights
    omega = 1+state.supports/(2*n)*dW.sum(1)
    b = state.supports/(2*n)*(dW@weights)/omega
    f = 1-b[:, None]/weights[None, :]
    if variant is PESPHVariant.PressureEnergy:
        u = state.totalEnergies/state.masses - 0.5*state.velocities.square().sum(-1)
        product = (scheme.gamma-1)**2 * u[:, None]*u[None, :]
        left = state.masses[None, :]*product*f/y[:, None]
        right = state.masses[None, :]*product*f.T/y[None, :]
    else:
        A = state.entropies
        product = (A[:, None]*A[None, :]).pow(1/scheme.gamma)
        P = y.pow(scheme.gamma)
        left = state.masses[None, :]*product*f*P[:, None].pow(1-2/scheme.gamma)
        right = state.masses[None, :]*product*f.T*P[None, :].pow(1-2/scheme.gamma)
    delta = state.positions[:, None] - state.positions[None, :]
    delta = delta - 2*torch.round(delta/2)
    r = delta.norm(dim=-1)
    q = r/state.supports[:, None]
    dk = -3*(1-q).clamp_min(0).square()+12*(.5-q).clamp_min(0).square()
    grad = (80/(7*math.pi))*dk[:, :, None]/state.supports[:, None, None].pow(3)
    grad = grad*delta/r.clamp_min(torch.finfo(r.dtype).eps)[:, :, None]
    refAcc = -(left[:, :, None]*grad-right[:, :, None]*grad.transpose(0, 1)).sum(1)
    refWork = (left*((state.velocities[:, None]-state.velocities[None, :])*grad).sum(-1)).sum(1)
    for actual, ref in ((acc, refAcc), (work, refWork)):
        assert (actual-ref).abs().max() < TOL*ref.abs().max().clamp_min(1)


def test_uniform_variants_agree_and_mass_density_is_independent():
    state, cfg, scheme = setup(dim=1, nx=40, uniform=True)
    adj = adjacency(state, cfg)
    pA = computePressureState(state, cfg, scheme, adj)
    pE = computePressureState(state, cfg, replace(scheme, variant=PESPHVariant.PressureEnergy), adj)
    assert torch.allclose(pA.pressures, torch.ones_like(state.masses), rtol=TOL, atol=TOL)
    assert torch.allclose(pA.pressures, pE.pressures, rtol=TOL, atol=TOL)
    for pressure in (pA, pE):
        acc, _ = computePressureForce(state, pressure, cfg, adj)
        assert acc.abs().max() < (2e-11 if DOUBLE else 2e-4), 'uniform pressure generated a startup force'
    state, cfg, scheme = setup()
    p = computePressureState(state, cfg, scheme, adjacency(state, cfg))
    assert (p.sums.massDensity - state.masses*p.sums.numberDensity).abs().max() > 1e-2
    assert (p.sums.massDensity - p.thermodynamicDensity).abs().max() > 1e-2
    assert torch.allclose(p.pressures, state.entropies*p.thermodynamicDensity.pow(scheme.gamma), rtol=TOL)


@pytest.mark.parametrize('variant', VARIANTS)
def test_authoritative_fields_and_config_roundtrip(variant):
    state, cfg, scheme = setup(variant=variant)
    expected = ('positions', 'velocities', 'totalEnergies' if variant is PESPHVariant.PressureEnergy else 'entropies')
    assert tuple(integrated_field_names(state)) == expected
    assert pesphConfigToDict(dictToPESPHConfig(pesphConfigToDict(scheme))) == pesphConfigToDict(scheme)
    weights = pressureWeights(state, variant, scheme.gamma)
    state.internalEnergies.fill_(999)
    assert torch.equal(weights, pressureWeights(state, variant, scheme.gamma))


@pytest.mark.parametrize('kwargs', [dict(gamma=1), dict(gamma=float('nan')), dict(backgroundPressure=1),
                                   dict(adaptiveSupportScheme=AdaptiveSupportScheme.Owen),
                                   dict(adaptiveSupportCorrections=False), dict(supportSolveOnGrid=True)])
def test_configuration_rejects_inconsistent_model(kwargs):
    with pytest.raises(ValueError, match='PESPH'):
        PESPHConfig(**kwargs).validate()


def test_invalid_thermodynamics_are_rejected():
    state, cfg, scheme = setup()
    # Invalid asynchronous device assertions terminate a CUDA context; test
    # the rejection on CPU so later GPU tests can continue in this process.
    for f in fields(state):
        value = getattr(state, f.name)
        if isinstance(value, torch.Tensor):
            setattr(state, f.name, value.cpu())
    state.entropies[0] = -1
    with pytest.raises(RuntimeError, match='positive'):
        pressureWeights(state, scheme.variant, scheme.gamma)
    state.totalEnergies[0] = -1
    with pytest.raises(RuntimeError, match='positive'):
        pressureWeights(state, PESPHVariant.PressureEnergy, scheme.gamma)


def test_automatic_pair_list_covers_both_supports():
    state, cfg, scheme = setup()
    pressure = computePressureState(state, cfg, scheme)
    explicit = computePressureForce(state, pressure, cfg, adjacency(state, cfg))
    automatic = computePressureForce(state, pressure, cfg)
    for actual, ref in zip(automatic, explicit):
        assert torch.allclose(actual, ref, rtol=TOL, atol=TOL)


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('integratorName', ['rungeKutta2', 'rungeKutta4', 'velocityVerlet'])
def test_conservative_stage_lifecycle(variant, integratorName):
    from warpSPHIntegrators.schemes import get_scheme, IntegrationSchemeType
    from warpSPH.modules.pesph.stage import refreshPESPHState
    from warpSPH.schemes.compressible.pesph import pesphConservativeStep
    from warpSPH.systems.pesph import PESPHSystem
    state, cfg, scheme = setup(dim=1, nx=40, variant=variant)
    system = PESPHSystem(state=state, domain=cfg.domain)
    refreshPESPHState(system, cfg, scheme)
    initialA = state.entropies.clone()
    initialEnergy = state.totalEnergies.sum().clone()
    initialMomentum = (state.masses[:, None]*state.velocities).sum(0)
    integrator = get_scheme(getattr(IntegrationSchemeType, integratorName))
    for _ in range(3):
        system = integrator.function(state=system, dt=1e-4, f=pesphConservativeStep,
                                     config=cfg, schemeConfig=scheme).state
    assert type(system) is PESPHSystem
    assert type(system.state) is pesphStateType(variant)
    current = system.state
    assert abs(current.totalEnergies.sum()-initialEnergy) < (3e-7 if DOUBLE else 2e-5)*abs(initialEnergy)
    assert torch.allclose((current.masses[:, None]*current.velocities).sum(0), initialMomentum,
                          rtol=TOL, atol=TOL)
    if variant is PESPHVariant.PressureEntropy:
        assert torch.equal(current.entropies, initialA)
    # Accepted-state closure must use final positions/E/A, not the last RK stage.
    pressure = computePressureState(current, cfg, scheme, system.adjacency)
    assert torch.allclose(current.pressures, pressure.pressures, rtol=TOL, atol=TOL)
    assert torch.allclose(current.internalEnergies, pressure.internalEnergies, rtol=TOL, atol=TOL)


@pytest.mark.skipif(not DOUBLE, reason='strict finite differences require float64')
@pytest.mark.parametrize('adaptive', [False, True])
def test_operator_autograd(adaptive):
    state, cfg, scheme = setup(dim=1, nx=8)
    state.supports.fill_(0.7)
    adj = adjacency(state, cfg)
    weights = pressureWeights(state, scheme.variant, scheme.gamma).detach().requires_grad_()
    x = state.positions.detach().requires_grad_()
    h = state.supports.detach().requires_grad_()
    velocity = state.velocities.detach().requires_grad_()
    def summed(x, h, weights):
        candidate = replace(state, positions=x, supports=h)
        sums = computePressureSums(candidate, cfg, weights, adj)
        return tuple(getattr(sums, f.name) for f in fields(sums))
    assert torch.autograd.gradcheck(summed, (x, h, weights), eps=1e-6, atol=1e-6, rtol=1e-5)
    def accelerated(x, h, weights, velocity):
        from warpSPH.modules.pesph import closePressure
        candidate = replace(state, positions=x, supports=h, velocities=velocity)
        sums = computePressureSums(candidate, cfg, weights, adj)
        pressure = closePressure(candidate, sums, weights, scheme.variant, scheme.gamma, adaptive=adaptive)
        return computePressureForce(candidate, pressure, cfg, adj)
    assert torch.autograd.gradcheck(accelerated, (x, h, weights, velocity), eps=1e-6, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(DOUBLE, reason='already running at float64')
def test_float64():
    env = dict(os.environ, warpSPHCore_PRECISION='float64')
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider',
                             '-k', 'not float64', __file__], cwd=Path(__file__).resolve().parents[1],
                            env=env, text=True, capture_output=True, timeout=600)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-2000:]


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('dim,nx', [(2, 6), (3, 5)])
@pytest.mark.parametrize('storeI', [False, True])
def test_chunked_pressure_passes_preserve_global_rows(monkeypatch, variant, dim, nx, storeI):
    from warpSPHCore.radiusSearch import buildChunkedVerletList, RowChunkPolicy
    state, cfg, scheme = setup(dim=dim, nx=nx, variant=variant)
    scheme.adaptiveSupportCorrections = True
    flat = adjacency(state, cfg)
    chunked = buildChunkedVerletList(state, cfg.domain,
        policy=RowChunkPolicy(edgeBudget=4096, maxRows=3, bytesPerEdge=8), verletScale=1.0,
        supportMode=SupportScheme.SuperSymmetric, storeI=storeI)
    assert len(chunked.chunks) > 1
    expected = computePressureState(state, cfg, scheme, flat)
    from warpSPHCore.autograd import PreparedLauncher
    from warpSPH.modules.pesph import operators
    launch = operators.launchOperator
    calls = []
    def observe(spec, ctx, **extras):
        result = launch(spec, ctx, **extras)
        if ctx.adjacency is chunked:
            calls.append((spec, ctx, extras, result))
        return result
    monkeypatch.setattr(operators, 'launchOperator', observe)
    actual = computePressureState(state, cfg, scheme, chunked)
    for f in fields(actual.sums):
        torch.testing.assert_close(getattr(actual.sums, f.name), getattr(expected.sums, f.name), rtol=0, atol=0)
    for a, b in zip(computePressureForce(state, actual, cfg, chunked),
                    computePressureForce(state, expected, cfg, flat)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(calls) == 2
    for spec, ctx, extras, expectedOutputs in calls:
        outputs = PreparedLauncher(ctx).launch(spec, **extras)
        for actualOutput, expectedOutput in zip(outputs, expectedOutputs):
            torch.testing.assert_close(actualOutput, expectedOutput, rtol=0, atol=0)
