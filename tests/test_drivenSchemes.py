"""Source coupling controls, separate from long turbulent-flow qualification."""
from dataclasses import replace
import math
import os
from pathlib import Path
from types import SimpleNamespace
import h5py
import pytest
import torch
from warpSPH.runner import RunConfiguration
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as case
from warpSPH.modules.eos import refreshEOS

SCHEMES = ['CRKSPH', 'CompSPH', 'Monaghan', 'GSPH', 'InutsukaGSPH']


def configuration(scheme, steps=20, **params):
    return RunConfiguration('drivenTurbulence', simulation=dict(device=os.environ.get('WARPSPH_TEST_DEVICE', 'cpu'), nx=12,
        nSteps=steps, maxDt=.002, plot=False, video=False, store=False, show=False,
        quiet=True, progress=False), params=params, solver=dict(scheme=scheme))


def initialEnergy(ctx, system):
    s = system.state
    ctx.scratch['initialEnergy'] = (s.masses.double()*(s.internalEnergies.double()+.5*s.velocities.double().square().sum(-1))).sum()
    refreshEOS(s, ctx.schemeConfig)
    s.totalEnergies = s.Q[:, -1]
    ctx.config.dt = case.timestep(ctx, system)


@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('acceleration', [0., 3.])
@pytest.mark.parametrize('tableau', ['rungeKutta2', 'rungeKutta4'])
def test_uniform_acceleration_and_bath_follow_analytic_solution(scheme, acceleration, tableau):
    initial = {}
    def build(ctx):
        system = case.buildSystem(ctx)
        s = system.state
        u0 = system.sourceTerms.targetInternalEnergy
        s.internalEnergies.fill_(1.4*u0)
        system.sourceTerms.acceleration = lambda pos, time: pos.new_tensor([acceleration, 0.]).expand_as(pos)
        system.sourceTerms.removeMeanAcceleration = False
        initial.update(u=1.4*u0, target=u0)
        initialEnergy(ctx, system)
        return system
    cfg = configuration(scheme, 40)
    cfg.simulation['integrationScheme'] = tableau
    result = cfg.run(replace(case, buildSystem=build))
    t, s = float(result.state.t), result.state.state
    expectedU = initial['target']+(initial['u']-initial['target'])*math.exp(-t/.2)
    torch.testing.assert_close(s.internalEnergies, torch.full_like(s.internalEnergies, expectedU), rtol=2e-5, atol=2e-6)
    expectedV = s.velocities.new_tensor([acceleration*t, 0.]).expand_as(s.velocities)
    torch.testing.assert_close(s.velocities, expectedV, rtol=1e-4, atol=2e-6)
    mass = float(s.masses.double().sum())
    assert abs(float(s.sourceEnergyBudget[0])-.5*mass*(acceleration*t)**2) < 1e-6
    assert abs(float(s.sourceEnergyBudget[1])-mass*(float(s.internalEnergies.double().mean())-initial['u'])) < 2e-6
    assert not result.diverged
    assert abs(result.trajectory[-1]['relativeEnergyBudgetResidual']) < 3e-6


@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('thermal', ['relaxation','adiabatic'])
def test_sources_use_fresh_native_midpoint_stages_and_keep_raw_cold_states_positive(scheme, thermal):
    observed = []
    def build(ctx):
        system = case.buildSystem(ctx)
        native = ctx.stepFunction
        def checked(system, dt, config, schemeConfig, verbose=False, stageIndex=None):
            update, *rest = native(system, dt, config, schemeConfig, verbose=verbose, stageIndex=stageIndex)
            observed.append((stageIndex, float(system.t), float(system.state.internalEnergies.min()), float(system.state.pressures.min())))
            return (update, *rest)
        ctx.bundle = replace(ctx.bundle, stepFunction=checked)
        return system
    cfg = configuration(scheme, 30, accelerationRMS=6., soundSpeed=.01, thermalModel=thermal)
    result = cfg.run(replace(case, buildSystem=build))
    assert not result.diverged and result.nSteps == 30
    assert len(observed) >= 60 and {r[0] for r in observed} == {0, 1}
    assert all(u > 0 and p > 0 for _, _, u, p in observed)
    for a, b in zip(observed[::2], observed[1::2]):
        assert a[0] == 0 and b[0] == 1 and b[1] > a[1]
    assert result.trajectory[-1]['t'] > 0


@pytest.mark.parametrize('scheme', SCHEMES)
def test_native_checkpoint_preserves_forcing_budget_and_continuation(scheme, tmp_path):
    full = configuration(scheme, 60, accelerationRMS=6.).run()
    firstCfg = configuration(scheme, 30, accelerationRMS=6.)
    firstCfg.simulation.update(store=True, storeMode='states', exportRoot=str(tmp_path/'first'))
    first = firstCfg.run()
    path = Path(first.exportPath)/'trajectory/finalState.h5'
    cfg = configuration(scheme, 30, accelerationRMS=6.)
    cfg.simulation['resumeFrom'] = str(path)
    resumed = cfg.run()
    assert not resumed.diverged
    assert float(full.state.t) == float(resumed.state.t)
    assert resumed.trajectory[0]['step'] == 29
    for name in ['positions','velocities','internalEnergies','supports','densities','sourceEnergyBudget']:
        torch.testing.assert_close(getattr(full.state.state,name),getattr(resumed.state.state,name),rtol=0,atol=0)
    for t in [.01, .151, 1.2]:
        torch.testing.assert_close(full.state.sourceTerms.acceleration.coefficients(t),
                                   resumed.state.sourceTerms.acceleration.coefficients(t),rtol=0,atol=0)


@pytest.mark.parametrize('scheme', SCHEMES)
def test_trajectory_exports_conserved_view_and_complete_driving(scheme, tmp_path):
    cfg = configuration(scheme, 5)
    cfg.simulation.update(store=True, storeMode='trajectory', storeInterval=1,
                          exportRoot=str(tmp_path/scheme))
    result = cfg.run()
    with h5py.File(Path(result.exportPath)/'trajectory.h5') as f:
        key = sorted(f['Q'], key=lambda k: int(k.split('_')[-1]))[-1]
        q = torch.from_numpy(f['Q'][key][:])
        torch.testing.assert_close(q, result.state.state.Q, rtol=0, atol=0)
        assert 'sourceEnergyBudget' in f
        assert 'frameData' in f


def test_compatible_work_uses_the_total_half_kick_and_keeps_heat_separate():
    from warpSPH.modules.forcing.primitive import finishSources
    dtype = torch.float64
    mass = torch.tensor([1., 2.], dtype=dtype)
    v = torch.tensor([[1., -.3],[-.5, .7]], dtype=dtype)
    hydro = torch.tensor([[.2,-.1],[-.1,.05]], dtype=dtype)
    body = torch.tensor([[.4,.2],[.1,-.6]], dtype=dtype)
    heat = torch.tensor([-.3,.2], dtype=dtype)
    dt = .07
    half = v+.5*dt*(hydro+body)
    u = torch.ones(2,dtype=dtype)
    # Only hydro pair work is the compatible internal-energy increment.
    hydroWork = -(mass[:,None]*half*hydro).sum()
    s = SimpleNamespace(masses=mass,velocities=v+dt*(hydro+body),internalEnergies=u+dt*hydroWork/mass.sum(),
        densities=torch.ones_like(mass),materials=torch.zeros(2,dtype=torch.int32))
    original = SimpleNamespace(state=SimpleNamespace(masses=mass,velocities=v,internalEnergies=u,
        sourceEnergyBudget=torch.zeros(2,dtype=dtype)))
    update = SimpleNamespace(dvdt=hydro+body, sourceAcceleration=body,sourceHeat=heat,
        sourceEnergyRates=torch.stack(((mass[:,None]*v*body).sum(),(mass*heat).sum())))
    system = SimpleNamespace(state=s,sourceTerms=object())
    finishSources(system,original,[update],[1.],dt,SimpleNamespace(gamma=5/3,materials=None),compatible=True)
    delta = (mass*(s.internalEnergies+.5*s.velocities.square().sum(-1)-u-.5*v.square().sum(-1))).sum()
    torch.testing.assert_close(delta,s.sourceEnergyBudget.sum(),rtol=1e-13,atol=1e-14)


@pytest.mark.parametrize('scheme', ['CRKSPH','CompSPH'])
def test_compatible_replay_commits_sources_once(scheme, monkeypatch):
    import warpSPH.modules.compSPH.pairWork as pairWork
    import warpSPH.schemes.compressible.crkSPH as crk
    import warpSPH.schemes.compressible.compSPH as comp
    normal = configuration(scheme, 20, accelerationRMS=6.).run()
    original = pairWork.accumulateStage
    count = [0]
    def spoiled(pw, stageIndex, adjacency, state, energyScheme):
        original(pw, stageIndex, adjacency, state, energyScheme)
        if stageIndex == 1 and pw.halfVelocity is None:
            pw.consistent, pw.work = False, None
            count[0] += 1
    monkeypatch.setattr(crk, 'accumulateStage', spoiled)
    monkeypatch.setattr(comp, 'accumulateStage', spoiled)
    # Native recomputation bypasses accumulateStage and writes W directly.
    # Spoil that path too, otherwise the default route never exercises replay.
    import warpSPH.modules.crk.work as crkWork
    import warpSPH.modules.compSPH.work as compWork
    def spoilShared(module, name):
        accumulate = getattr(module, name)
        def wrapped(system, state, update, config, schemeConfig, adjacency, correction, stageIndex, dt):
            accumulate(system, state, update, config, schemeConfig, adjacency, correction, stageIndex, dt)
            pw = system.pairWork
            if stageIndex == 1 and pw.halfVelocity is None:
                pw.consistent, pw.work = False, None
                count[0] += 1
        monkeypatch.setattr(module, name, wrapped)
    spoilShared(crkWork, 'accumulateCRKWork')
    spoilShared(compWork, 'accumulateCompWork')
    replay = configuration(scheme, 20, accelerationRMS=6.).run()
    assert count[0] == 20 and not replay.diverged
    double = normal.state.state.masses.dtype == torch.float64
    tolerance, absolute = (5e-12,2e-14) if double else (3e-5,2e-8)
    for name in ['velocities','internalEnergies','sourceEnergyBudget']:
        torch.testing.assert_close(getattr(normal.state.state,name),getattr(replay.state.state,name),rtol=tolerance,atol=absolute)
    torch.testing.assert_close(normal.state.sourceTerms.acceleration.generator.get_state(),
                               replay.state.sourceTerms.acceleration.generator.get_state(),rtol=0,atol=0)


@pytest.mark.parametrize('scheme', ['CRKSPH','Monaghan'])
def test_checkpoint_refuses_missing_native_neighbor_history(scheme, tmp_path):
    cfg = configuration(scheme, 4)
    cfg.simulation.update(store=True,storeMode='states',exportRoot=str(tmp_path/'first'))
    first = cfg.run()
    path = Path(first.exportPath)/'trajectory/finalState.h5'
    with h5py.File(path, 'a') as f:
        del f.attrs['checkpointAdjacencyVersion']
    resume = configuration(scheme, 2)
    resume.simulation['resumeFrom'] = str(path)
    with pytest.raises(ValueError,match='missing native neighbor history'):
        resume.run()


@pytest.mark.parametrize('variant,thermal', [('PressureEnergy', 'relaxation'), ('PressureEntropy', 'adiabatic')])
@pytest.mark.parametrize('tableau', ['rungeKutta2', 'rungeKutta4'])
def test_pesph_uniform_body_source_matches_analytic_solution(variant, thermal, tableau):
    """Tiny repeated analytic controls intentionally omit video."""
    def build(ctx):
        system = case.buildSystem(ctx)
        # Initialize from prescribed pressure through the native PESPH path.
        system.state.pressures *= 1.4
        system.sourceTerms.acceleration = lambda x, t: x.new_tensor([.3, 0.]).expand_as(x)
        system.sourceTerms.removeMeanAcceleration = False
        return system
    cfg = configuration('PESPH', 12, accelerationRMS=0., thermalModel=thermal)
    cfg.solver.update(variant=variant, dissipation=variant == 'PressureEnergy')
    cfg.simulation.update(integrationScheme=tableau)
    result = cfg.run(replace(case, buildSystem=build))
    assert not result.diverged
    s, t = result.state.state, float(result.state.t)
    mass = result.trajectory[0]['mass']
    initialU = result.trajectory[0]['thermalEnergy']/mass
    target = result.state.sourceTerms.targetInternalEnergy
    expectedU = target+(initialU-target)*math.exp(-t/.2) if thermal == 'relaxation' else initialU
    torch.testing.assert_close(s.internalEnergies, torch.full_like(s.internalEnergies, expectedU), rtol=3e-5, atol=2e-6)
    torch.testing.assert_close(s.velocities, s.velocities.new_tensor([.3*t, 0.]).expand_as(s.velocities), rtol=3e-4, atol=2e-6)
    assert abs(float(s.sourceEnergyBudget[0])-.5*mass*(.3*t)**2) < 1e-7
    assert abs(float(s.sourceEnergyBudget[1])-mass*(expectedU-initialU)) < 3e-6
    assert abs(result.trajectory[-1]['relativeEnergyBudgetResidual']) < 5e-6
    assert result.state.sourceTerms is not None
    assert s.thermodynamicVariable == ('energy' if variant == 'PressureEnergy' else 'entropy')


@pytest.mark.parametrize('variant,thermal', [('PressureEnergy', 'relaxation'), ('PressureEntropy', 'adiabatic')])
@pytest.mark.parametrize('dim', [2, 3])
def test_pesph_driven_chunked_fields_and_stage_clocks_match_flat(variant, thermal, dim):
    from warpSPHCore.radiusSearch import buildChunkedVerletList, RowChunkPolicy, ChunkedAdjacency
    from warpSPHCore import SupportScheme
    records = []
    def solve(chunked):
        def build(ctx):
            from warpSPH.cases.compressible.drivenTurbulence import buildSPHSystem
            factory = (lambda state, cfg: buildChunkedVerletList(state, cfg.domain,
                policy=RowChunkPolicy(edgeBudget=4096, maxRows=4, bytesPerEdge=8),
                supportMode=SupportScheme.Gather, verletScale=cfg.verletScale)) if chunked else None
            system = buildSPHSystem(ctx, neighborhoodFactory=factory)
            native = ctx.stepFunction
            def observed(system, *args, **kwargs):
                records.append((chunked, float(system.t)))
                return native(system, *args, **kwargs)
            ctx.bundle = replace(ctx.bundle, stepFunction=observed)
            return system
        cfg = configuration('PESPH', 4, thermalModel=thermal, forcingInterval=.000025)
        cfg.solver.update(variant=variant, dissipation=variant == 'PressureEnergy')
        cfg.simulation.update(nx=6, dim=dim, maxDt=.0002, integrationScheme='rungeKutta4', storeMode='trajectory')
        return cfg.run(replace(case, buildSystem=build))
    flat, chunked = solve(False), solve(True)
    assert not flat.diverged and not chunked.diverged
    assert isinstance(chunked.state.adjacency, ChunkedAdjacency)
    assert [t for isChunk,t in records if not isChunk] == [t for isChunk,t in records if isChunk]
    assert len(records) == 32
    for name, value in vars(flat.state.state).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(getattr(chunked.state.state, name), value, rtol=0, atol=0, msg=name)
    assert float(chunked.state.t) == float(flat.state.t)
    assert float(chunked.ctx.config.dt) <= .000025*(1+1e-6)
    assert float(chunked.state.t) <= 4*.000025*(1+1e-6)


def test_pressure_entropy_driving_keeps_thermal_heating_guard():
    with pytest.raises(ValueError, match='adiabatic'):
        RunConfiguration('drivenTurbulence', solver=dict(scheme='PESPH', variant='PressureEntropy')).resolve()
