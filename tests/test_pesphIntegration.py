"""PESPH dissipation budgets, runner contracts and faithful continuation."""

from dataclasses import fields, replace
import copy
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
import torch

from test_pesphOperators import setup, cubic, DOUBLE, TOL, VARIANTS
from warpSPH.configurations import BoundaryCondition, BoundaryConditionType, PESPHConfig
from warpSPH.configurations.moduleConfigurations.diffusionParameters import VelocityPairPolicy
from warpSPH.enumTypes import PESPHVariant, ViscositySwitch, AdaptiveSupportScheme
from warpSPH.modules.dissipation.rates import computeDissipationRates
from warpSPH.modules.pesph import computePressureForce
from warpSPH.modules.pesph.stage import refreshPESPHState
from warpSPH.schemes.compressible.pesph import pesphStep
from warpSPH.schemes import buildScheme
from warpSPH.systems.pesph import PESPHSystem, pesphStateType
from warpSPH.runner import RunConfiguration, configurationSchema


@pytest.mark.parametrize('policy', [VelocityPairPolicy.Raw, VelocityPairPolicy.Limited])
def test_existing_viscosity_and_conduction_balance_total_energy(policy):
    state, cfg, scheme = setup(variant=PESPHVariant.PressureEnergy)
    scheme.dissipation = True
    scheme.viscositySwitchParams.scheme = ViscositySwitch.NoneSwitch
    scheme.diffusionParams.velocityPairPolicy = policy.value
    state.alphas.fill_(1)
    state.alpha0s.fill_(1)
    system = PESPHSystem(state=state, domain=cfg.domain)
    p = refreshPESPHState(system, cfg, scheme)
    acc, _ = computePressureForce(state, p, cfg, system.adjacency)
    rates = computeDissipationRates(state, 1e-4, cfg, scheme, system.adjacency, acc)
    kinetic = (state.masses[:, None]*state.velocities*rates.acceleration).sum()
    heat = (state.masses*rates.viscousHeating).sum()
    assert heat > 1e-6, 'viscosity never engaged'
    assert abs(kinetic+heat) < TOL*(abs(kinetic)+abs(heat))
    conduction = state.masses*rates.conduction
    assert conduction.abs().sum() > 1e-6
    assert abs(conduction.sum()) < TOL*conduction.abs().sum()
    update, *_ = pesphStep(system, 1e-4, cfg, scheme)
    assert abs(update.dEdt.sum()) < TOL*update.dEdt.abs().sum()


def test_external_forcing_includes_work():
    state, cfg, scheme = setup(variant=PESPHVariant.PressureEnergy)
    system = PESPHSystem(state=state, domain=cfg.domain)
    base, *_ = pesphStep(system, 1e-4, cfg, scheme)
    acceleration = torch.tensor([.2, -.4], device=state.positions.device, dtype=state.positions.dtype)
    bc = BoundaryCondition(
        BoundaryConditionType.forcing,
        sdf=lambda x: (-torch.ones_like(x[:, 0]), torch.zeros_like(x)),
        forcingFunctions=[lambda s, *args: s.masses[:, None]*acceleration])
    scheme.boundaryConditions = [bc]
    forced, *_ = pesphStep(system, 1e-4, cfg, scheme)
    expected = state.masses*(state.velocities*acceleration).sum(-1)
    assert torch.allclose(forced.dvdt-base.dvdt, acceleration.expand_as(state.velocities), atol=TOL, rtol=TOL)
    assert torch.allclose(forced.dEdt-base.dEdt, expected, atol=TOL, rtol=TOL)


def test_local_entropy_heating_is_guarded_for_measured_energy_defect():
    state, cfg, scheme = setup()
    W, _ = cubic(state, cfg)
    A = state.entropies.detach().requires_grad_()
    y = W@(state.masses*A.pow(1/scheme.gamma))
    rho = y/A.pow(1/scheme.gamma)
    u = A*rho.pow(scheme.gamma-1)/(scheme.gamma-1)
    sensitivity, = torch.autograd.grad((state.masses*u).sum(), A)
    heating = torch.zeros_like(A)
    heating[13] = 1
    local = (scheme.gamma-1)*rho.detach().pow(1-scheme.gamma)*heating
    expected = (state.masses*heating).sum()
    defect = abs((sensitivity*local).sum()-expected)/expected
    assert defect > 1e-3
    with pytest.raises(ValueError, match='dissipation=False'):
        replace(scheme, dissipation=True).validate()
    reference = replace(scheme, dissipation=True, entropyHeating='SwiftLocal')
    reference.validate()
    reference.viscositySwitchParams.scheme = ViscositySwitch.NoneSwitch
    state.alphas.fill_(1)
    state.alpha0s.fill_(1)
    system = PESPHSystem(state=state, domain=cfg.domain)
    p = refreshPESPHState(system, cfg, reference)
    acc, _ = computePressureForce(state, p, cfg, system.adjacency)
    q = computeDissipationRates(state, 1e-4, cfg, reference, system.adjacency, acc).heating
    update, *_ = pesphStep(system, 1e-4, cfg, reference)
    assert torch.allclose(update.dAdt, (scheme.gamma-1)*p.thermodynamicDensity.pow(1-scheme.gamma)*q,
                          atol=TOL, rtol=TOL)


@pytest.mark.parametrize('variant', VARIANTS)
def test_driven_fluid_band_translates_prescribed_thermal_fields(variant):
    from warpSPH.modules.pesph.boundaries import applyPESPHDirichlet
    state, cfg, scheme = setup(variant=variant)
    system = PESPHSystem(state=state, domain=cfg.domain)
    p = refreshPESPHState(system, cfg, scheme)
    old = (state.totalEnergies if variant is PESPHVariant.PressureEnergy else state.entropies).clone()
    mask = state.positions[:, 0] < 0
    bc = BoundaryCondition(BoundaryConditionType.dynamic,
        sdf=lambda x: (x[:, 0], torch.zeros_like(x)),
        dirichletFunctions={
            'densities': lambda s, *args: torch.full_like(s.masses, 2.),
            'internalEnergies': lambda s, *args: torch.full_like(s.masses, 3.),
            'velocities': lambda s, *args: torch.full_like(s.velocities, .2)})
    scheme.boundaryConditions = [bc]
    assert applyPESPHDirichlet(system, 1e-4, cfg, scheme, p)
    actual = state.totalEnergies if variant is PESPHVariant.PressureEnergy else state.entropies
    assert torch.equal(actual[~mask], old[~mask])
    refreshPESPHState(system, cfg, scheme, solveSupport=False)
    assert torch.allclose(state.internalEnergies[mask], torch.full_like(state.masses[mask],3.),
                          atol=TOL, rtol=TOL)


@pytest.mark.parametrize('variant', VARIANTS)
def test_runner_selects_state_after_config_and_preserves_support_defaults(variant):
    from warpSPH.cases.compressible.sod import sodCase
    config = RunConfiguration('sod', solver={'scheme': 'PESPH', 'variant': variant.name,
                                           'dissipation': variant is PESPHVariant.PressureEnergy})
    ctx = config.prepare(sodCase)
    assert ctx.SimulationState is pesphStateType(variant)
    assert ctx.schemeConfig.adaptiveSupportScheme is AdaptiveSupportScheme.NumberDensity
    assert ctx.schemeConfig.adaptiveSupportCorrections
    assert len(tuple(ctx.bundle)) == 7
    schema = configurationSchema(sodCase, 'PESPH')
    assert 'variant' in schema['properties']['solver']['properties']


@pytest.mark.parametrize('variant', VARIANTS)
def test_initialization_preserves_mass_and_balances_unequal_mass_contact(variant):
    from warpSPH.modules.pesph.initialization import initializePESPHState
    from warpSPHCore import buildVerletList, SupportScheme
    state, cfg, scheme = setup(dim=1, nx=40, uniform=True, variant=variant)
    state.masses[:20] *= 3
    masses = state.masses.clone()
    foreign = buildVerletList(state, cfg.domain, supportMode=SupportScheme.Gather)
    system = PESPHSystem(state=state, domain=cfg.domain, adjacency=foreign)
    p = initializePESPHState(system, cfg, scheme)
    acc, _ = computePressureForce(state, p, cfg, system.adjacency)
    assert torch.equal(state.masses, masses)
    assert system.adjacency is not foreign
    assert torch.allclose(state.pressures, torch.ones_like(state.pressures), rtol=TOL, atol=TOL)
    assert acc.abs().max() < (1e-10 if DOUBLE else 2e-4)
    W, _ = cubic(state, cfg)
    assert torch.allclose(state.densities, W@state.masses, rtol=TOL, atol=TOL)


def test_fixed_timestep_is_capped_by_signal_speed():
    from warpSPH.modules.timestep.pesph import pesphTimestep
    state, cfg, scheme = setup(dim=1, nx=40)
    system = PESPHSystem(state=state, domain=cfg.domain)
    refreshPESPHState(system, cfg, scheme)
    cfg.adaptiveDt = False
    ctx = SimpleNamespace(config=cfg, schemeConfig=scheme, spec=SimpleNamespace(dt=1.0))
    bound = pesphTimestep(ctx, system)
    assert 0 < bound < 1
    state.velocities *= 100
    faster = pesphTimestep(ctx, system)
    assert faster < bound
    cfg.adaptiveDt = True
    state.previousTimestep = float(bound)/10
    assert pesphTimestep(ctx, system) <= state.previousTimestep*cfg.dtGrowthFactor


def runOptions(tmp_path, variant, *, steps=3, **extra):
    return dict(scheme='PESPH', nx=48, nSteps=steps, adaptiveDt=False, dt=1e-4,
                progress=False, quiet=True, plot=False, video=False, exportRoot=str(tmp_path),
                schemeOverrides={'variant': variant.name,
                                 'dissipation': variant is PESPHVariant.PressureEnergy}, **extra)


@pytest.mark.parametrize('variant', VARIANTS)
def test_uniform_case_has_no_startup_kick(tmp_path, variant):
    from warpSPH.cases.compressible.linearWave import linearWaveCase
    from warpSPH.runner import run
    result = run(linearWaveCase, **runOptions(tmp_path, variant, steps=2), params={'A': 0.0})
    assert not result.diverged
    assert result.state.state.velocities.abs().max() < (1e-10 if DOUBLE else 2e-5)
    assert (result.series('kineticEnergy') < (1e-18 if DOUBLE else 1e-9)).all()
    assert float(result.state.state.masses.sum()) == pytest.approx(1.0, rel=TOL)


def assertSameState(left, right, *, skip=(), exact=False):
    assert type(left) is type(right)
    for f in fields(left):
        if f.name in skip:
            continue
        a, b = getattr(left, f.name), getattr(right, f.name)
        if isinstance(a, torch.Tensor):
            if isinstance(b, torch.Tensor):
                assert (torch.equal(a, b) if exact else torch.allclose(a, b, rtol=TOL, atol=TOL)), f.name
            else:
                assert a.numel() == 1 and float(a) == pytest.approx(b), f.name
        elif isinstance(b, torch.Tensor):
            assert b.numel() == 1 and float(b) == pytest.approx(a), f.name
        else:
            assert a == b, f.name


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('storage', ['states', 'trajectory'])
@pytest.mark.parametrize('switch', ['CullenDehnen2010', 'Rosswog2020'])
def test_restart_preserves_authoritative_fields_and_continuation(tmp_path, variant, storage, switch):
    from warpSPH.cases.compressible.sod import sodCase
    from warpSPH.runner import run
    from warpSPH.io import importSimulationSystem, loadTrajectory, loadTrajectoryFrame
    options = runOptions(tmp_path, variant)
    options['schemeOverrides']['viscositySwitchParams'] = {'scheme': switch}
    split = run(sodCase, **options, store=True, storeMode=storage, storeInterval=1, exportInterval=1e-4)
    assert not split.diverged
    cfg, scheme = split.ctx.config, split.ctx.schemeConfig
    if storage == 'states':
        paths = sorted((Path(split.exportPath)/'trajectory').glob('*.h5'))
        resumed, _, _, _ = importSimulationSystem(str(paths[-1]), cfg.device, cfg.dtype)
    else:
        handle, meta = loadTrajectory(split.exportPath, cfg.device)
        try:
            resumed, _ = loadTrajectoryFrame(handle, meta, -1, schemeConfig=scheme)
            other = PESPHVariant.PressureEntropy if variant is PESPHVariant.PressureEnergy else PESPHVariant.PressureEnergy
            with pytest.raises(ValueError, match='variant'):
                loadTrajectoryFrame(handle, meta, -1, schemeConfig=replace(scheme, variant=other, dissipation=False))
        finally:
            handle.close()
    assertSameState(split.state.state, resumed.state, exact=True)
    assert float(split.state.t) == pytest.approx(float(resumed.t), abs=1e-12)
    # A restart rebuilds the neighbour cache. Control for that independently:
    # float32 summation-order noise can be amplified by Rosswog's time
    # difference of entropy, even without any I/O. All histories must restore
    # exactly above; float64 checks their continued values as well. Float32
    # checks the physical trajectory and records the switch limitation in
    # PESPH_PLAN, rather than pretending cache rebuilds are bitwise stable.
    rebuilt = copy.deepcopy(split.state)
    rebuilt.adjacency = None
    # Continue both copies under the same ordinary integrator and stored dt.
    for _ in range(2):
        split.state = split.ctx.integrator.function(state=split.state, dt=1e-4, f=pesphStep,
                                                    config=cfg, schemeConfig=scheme).state
        resumed = split.ctx.integrator.function(state=resumed, dt=1e-4, f=pesphStep,
                                                config=cfg, schemeConfig=scheme).state
        rebuilt = split.ctx.integrator.function(state=rebuilt, dt=1e-4, f=pesphStep,
                                                config=cfg, schemeConfig=scheme).state
    skip = ('alphas', 'alpha0s', 'entropyRates') if not DOUBLE and switch == 'Rosswog2020' else ()
    assertSameState(rebuilt.state, resumed.state, skip=skip)
    assertSameState(split.state.state, resumed.state, skip=skip)


def test_full_state_loader_restores_scalar_switch_history(tmp_path):
    from warpSPH.io.hdf5 import dumpState, loadState
    state, _, _ = setup()
    state.entropiesPrevTime = 0.125
    state.previousTimestep = 0.007
    with h5py.File(tmp_path/'state.h5', 'w') as f:
        dumpState(state, f.create_group('state'))
        restored = loadState(f['state'], state.positions.device, type(state))
    assert restored.entropiesPrevTime == .125
    assert restored.previousTimestep == .007
    assert restored.thermodynamicVariable == 'entropy'


def test_trajectory_preserves_uninitialized_and_scalar_history(tmp_path):
    from warpSPH.io.export import writeFrame
    from warpSPH.io.importIO import _frameValue
    state, cfg, scheme = setup()
    system = PESPHSystem(state=state, domain=cfg.domain)
    names = ('previousTimestep', 'entropiesPrevTime')
    with h5py.File(tmp_path/'history.h5', 'w') as f:
        groups = tuple(f.create_group(name) for name in
                       ('positions', 'velocities', 'densities', 'times', 'rigidBodies', *names))
        writeFrame(groups, 0, system, None, cfg, scheme, extraFields=names)
        state.previousTimestep, state.entropiesPrevTime = .007, .125
        writeFrame(groups, 1, system, None, cfg, scheme, extraFields=names)
        for name, expected in zip(names, (.007, .125)):
            assert _frameValue(f[name], 'frame_00000', cfg.device) is None
            assert float(_frameValue(f[name], 'frame_00001', cfg.device)) == pytest.approx(expected)


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('dim,nx', [(2, 6), (3, 5)])
@pytest.mark.parametrize('support', ['NoScheme', 'NumberDensity'])
def test_chunked_initialization_and_native_steps_preserve_fields(variant, dim, nx, support):
    """Tiny repeated parity controls intentionally omit video; no timing claim."""
    from warpSPH.modules.pesph.initialization import initializePESPHState
    from warpSPHCore.radiusSearch import buildChunkedVerletList, ChunkedAdjacency, RowChunkPolicy
    from warpSPHCore import SupportScheme
    from warpSPHIntegrators.schemes import get_scheme, IntegrationSchemeType
    initial, cfg, scheme = setup(dim=dim, nx=nx, variant=variant, uniform=True)
    initial.velocities *= .05
    scheme = replace(scheme, adaptiveSupportScheme=getattr(AdaptiveSupportScheme, support),
        adaptiveSupportCorrections=support == 'NumberDensity', adaptiveSupportIterations=60,
        dissipation=variant is PESPHVariant.PressureEnergy)
    systems = []
    for chunked in (False, True):
        state = initial.clone()
        # Start from a Gather list, as samplers do. Initialization must rebuild
        # union support while preserving the selected resident representation.
        adj = buildChunkedVerletList(state, cfg.domain,
            policy=RowChunkPolicy(edgeBudget=4096, maxRows=3, bytesPerEdge=8), verletScale=cfg.verletScale,
            supportMode=SupportScheme.Gather) if chunked else None
        system = PESPHSystem(state=state, domain=cfg.domain, adjacency=adj)
        initializePESPHState(system, cfg, scheme)
        assert isinstance(system.adjacency, ChunkedAdjacency) is chunked
        if chunked:
            assert system.adjacency.mode is SupportScheme.SuperSymmetric
        integrator = get_scheme(IntegrationSchemeType.rungeKutta2)
        for _ in range(2):
            system = integrator.function(state=system, dt=1e-5, f=pesphStep, config=cfg, schemeConfig=scheme).state
            assert isinstance(system.adjacency, ChunkedAdjacency) is chunked
        systems.append(system)
    assertSameState(systems[0].state, systems[1].state, exact=True)
    assert float(systems[0].t) == float(systems[1].t)


@pytest.mark.parametrize('context', ['materials', 'boundary', 'source', 'reverse-ad', 'forward-ad', 'graph'])
def test_chunked_pesph_rejects_unqualified_contexts(context):
    from warpSPH.modules.pesph.chunked import checkPESPHChunks
    from warpSPHCore.radiusSearch import buildChunkedVerletList, RowChunkPolicy
    from warpSPHCore import SupportScheme
    state, cfg, scheme = setup(dim=2, nx=6, uniform=True)
    adj = buildChunkedVerletList(state, cfg.domain,
        policy=RowChunkPolicy(edgeBudget=4096, maxRows=3, bytesPerEdge=8),
        supportMode=SupportScheme.SuperSymmetric)
    system = PESPHSystem(state=state, domain=cfg.domain, adjacency=adj)
    if context == 'materials':
        scheme.materials = [object()]
    elif context == 'boundary':
        scheme.boundaryConditions = [object()]
    elif context == 'source':
        system.sourceTerms = object()
    elif context == 'reverse-ad':
        state.velocities.requires_grad_()
    elif context == 'graph':
        cfg.compute.execution = 'graph'
    if context == 'forward-ad':
        with torch.autograd.forward_ad.dual_level():
            state.velocities = torch.autograd.forward_ad.make_dual(state.velocities, torch.ones_like(state.velocities))
            with pytest.raises(NotImplementedError, match='differentiation'):
                checkPESPHChunks(system, cfg, scheme)
    else:
        with pytest.raises(NotImplementedError, match='differentiation' if context == 'reverse-ad' else 'eager periodic'):
            checkPESPHChunks(system, cfg, scheme)
