"""Material EOS in complete SPH stages, compatible energy, graphs and saved runs."""
from dataclasses import replace

import pytest
import torch

from warpSPH.runner import run, RunConfiguration
from warpSPH.cases.compressible.richtmyerMeshkov import richtmyerMeshkovCase

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA SPH')


@pytest.mark.parametrize('scheme', ['CRKSPH', 'CompSPH', 'Monaghan'])
@pytest.mark.parametrize('switch', ['NoneSwitch', 'CullenDehnen2010'])
def test_material_step_uses_local_eos(scheme, switch):
    result = run(richtmyerMeshkovCase, scheme=scheme, nx=16, nSteps=4, tLimit=1.,
                 plot=False, quiet=True, progress=False, execution='eager', params={'viscositySwitch': switch})
    s = result.state.state
    gamma = torch.tensor([1.4, 5/3], device=s.densities.device)[s.materials.long()]
    torch.testing.assert_close(s.pressures, (gamma - 1) * s.densities * s.internalEnergies)
    torch.testing.assert_close(s.soundspeeds**2, gamma * s.pressures / s.densities)
    assert not result.diverged
    assert s.internalEnergies.min() > 0


def test_cross_material_energy_partition_is_complementary():
    ctx = RunConfiguration('richtmyerMeshkov', simulation={'nx': 16, 'tLimit': 1.}).prepare(richtmyerMeshkovCase)
    ctx.schemeConfig.compatibleEnergy = False  # retain stage pair arrays for inspection
    system = richtmyerMeshkovCase.buildSystem(ctx)
    from warpSPH.schemes.compressible.crkSPH import crkSPH_step
    _, adjacency, s = crkSPH_step(system, ctx.config.dt, ctx.config, ctx.schemeConfig)
    i, j = adjacency.livePairs()
    cross = s.materials[i.long()] != s.materials[j.long()]
    assert cross.any()
    assert torch.equal(s.f_ij[:i.numel()][cross], torch.full_like(s.f_ij[:i.numel()][cross], .5))
    assert (s.masses[i[cross].long()] != s.masses[j[cross].long()]).all()
    # Equal partition is complementary in both orientations, independent of masses.
    assert torch.isfinite(s.f_ij[:i.numel()]).all()


def test_identical_eos_labels_retain_crk_entropy_weighting():
    from warpSPH.configurations import MaterialConfig, EOSConfig
    from warpSPH.modules.compSPH.operators import computeCompSPHBalanceTerm
    from warpSPH.schemes.compressible.crkSPH import crkSPH_step
    ctx = RunConfiguration('richtmyerMeshkov', simulation={'nx': 16, 'tLimit': 1.}).prepare(richtmyerMeshkovCase)
    ctx.schemeConfig.materials[1] = MaterialConfig('sameGas', EOSConfig(gamma=1.4))
    ctx.schemeConfig.compatibleEnergy = False
    system = richtmyerMeshkovCase.buildSystem(ctx)
    update, adjacency, state = crkSPH_step(system, ctx.config.dt, ctx.config, ctx.schemeConfig)
    explicit = state.f_ij.clone()
    ctx.schemeConfig.materials = None
    scalar = computeCompSPHBalanceTerm(state, ctx.config, ctx.schemeConfig, adjacency,
        state.velocities + .5 * ctx.config.dt * update.dvdt, ctx.config.dt)
    i, j = adjacency.livePairs()
    torch.testing.assert_close(explicit[:i.numel()], scalar[:i.numel()], rtol=0, atol=0)


@pytest.mark.parametrize('pipeline', [False, True])
@pytest.mark.parametrize('densityPolicy', ['legacy', 'sameMaterial', 'massReplacement'])
def test_material_graph_matches_eager_grid_and_replays(monkeypatch, pipeline, densityPolicy):
    import warpSPH.utils.cudaGraph as cudaGraph
    original = richtmyerMeshkovCase.configureScheme
    graphs = []
    constructor = cudaGraph.GraphedIntegratorStep.__init__

    def remember(self, *args, **kwargs):
        constructor(self, *args, **kwargs)
        graphs.append(self)
    monkeypatch.setattr(cudaGraph.GraphedIntegratorStep, '__init__', remember)

    def configure(ctx):
        original(ctx)
        ctx.schemeConfig.crkDensityPolicy = densityPolicy
        ctx.schemeConfig.supportSolveOnGrid = True
        ctx.schemeConfig.inPlaceAdjacency = True

    case = replace(richtmyerMeshkovCase, configureScheme=configure)
    options = dict(nx=16, nSteps=160, tLimit=1.5, plot=False, quiet=True, progress=False, pipelineOutputs=pipeline)
    eager = run(case, execution='eager', **options)
    graphed = run(case, execution='graph', **options)
    [graph] = graphs
    assert graph.disabled is None and graph.replays > 50
    for name in ('positions', 'velocities', 'densities', 'internalEnergies', 'pressures', 'soundspeeds', 'materials'):
        torch.testing.assert_close(getattr(eager.state.state, name), getattr(graphed.state.state, name), rtol=0, atol=0)
    assert abs(graphed.series('relativeEnergyDrift')[-1]) < 1e-5


@pytest.mark.parametrize('densityPolicy', ['legacy', 'sameMaterial', 'massReplacement'])
def test_material_trajectory_roundtrip(tmp_path, densityPolicy):
    from warpSPH.configurations import dictToCRKSPHConfig
    from warpSPH.io.importIO import importConfigs, loadTrajectory, loadTrajectoryFrame
    def sources(ctx):
        system = richtmyerMeshkovCase.buildSystem(ctx)
        s = system.state
        s.emitters = (torch.arange(s.masses.numel(), device=s.masses.device, dtype=torch.int32) % 3) + 17
        s.bodyIDs = torch.full_like(s.materials, -1)
        return system
    case = replace(richtmyerMeshkovCase, buildSystem=sources)
    result = run(case, nx=16, nSteps=3, tLimit=1., execution='eager',
        schemeOverrides={'crkDensityPolicy': densityPolicy},
        quiet=True, progress=False, store=True, storeMode='trajectory', storeInterval=1,
        plot=False, exportRoot=str(tmp_path))
    _, config = importConfigs(tmp_path / result.exportPath / 'config.json', dictToCRKSPHConfig)
    assert config.crkDensityPolicy == densityPolicy
    f, meta = loadTrajectory(result.exportPath, 'cuda', extraFields=richtmyerMeshkovCase.extraFields)
    try:
        system, t = loadTrajectoryFrame(f, meta, -1, schemeConfig=config)
        assert system.state.materials.dtype == torch.int32
        for name in ('materials', 'emitters', 'bodyIDs', 'internalEnergies', 'pressures', 'soundspeeds'):
            torch.testing.assert_close(getattr(system.state, name), getattr(result.state.state, name))
    finally:
        f.close()


@pytest.mark.parametrize('policy', ['legacy', 'sameMaterial', 'massReplacement'])
def test_emitter_relabeling_leaves_material_dynamics_unchanged(policy):
    from warpSPH.schemes.compressible.crkSPH import crkSPH_step
    ctx = RunConfiguration('richtmyerMeshkov', simulation={'nx': 16, 'tLimit': 1.},
        solver={'crkDensityPolicy': policy}).prepare(richtmyerMeshkovCase)
    system = richtmyerMeshkovCase.buildSystem(ctx)
    source = system.state.clone()
    u0, _, s0 = crkSPH_step(system, ctx.config.dt, ctx.config, ctx.schemeConfig)
    source.emitters = torch.arange(source.masses.numel(), device=source.masses.device, dtype=torch.int32)
    system.state = source
    u1, _, s1 = crkSPH_step(system, ctx.config.dt, ctx.config, ctx.schemeConfig)
    for name in ('dxdt', 'dvdt', 'dudt'):
        torch.testing.assert_close(getattr(u0, name), getattr(u1, name), rtol=0, atol=0)
    for name in ('densities', 'pressures', 'soundspeeds'):
        torch.testing.assert_close(getattr(s0, name), getattr(s1, name), rtol=0, atol=0)


@pytest.mark.parametrize('changeKind', ['eos', 'density'])
def test_graph_recaptures_changed_material_table(monkeypatch, changeKind):
    import warpSPH.utils.cudaGraph as cudaGraph
    from warpSPH.configurations import MaterialConfig, EOSConfig
    graphs = []
    constructor = cudaGraph.GraphedIntegratorStep.__init__

    def remember(self, *args, **kwargs):
        constructor(self, *args, **kwargs)
        graphs.append(self)
    monkeypatch.setattr(cudaGraph.GraphedIntegratorStep, '__init__', remember)

    def configure(ctx):
        richtmyerMeshkovCase.configureScheme(ctx)
        ctx.schemeConfig.supportSolveOnGrid = True
        ctx.schemeConfig.inPlaceAdjacency = True

    def change(ctx, state, step):
        if step == 7:
            if changeKind == 'eos':
                ctx.schemeConfig.materials[1] = MaterialConfig('changedGas', EOSConfig(gamma=1.5))
            else:
                ctx.schemeConfig.crkDensityPolicy = 'sameMaterial'

    case = replace(richtmyerMeshkovCase, configureScheme=configure, postStep=change)
    opts = dict(nx=16, nSteps=20, tLimit=1., quiet=True, progress=False, plot=False, pipelineOutputs=False)
    eager = run(case, execution='eager', **opts)
    graphed = run(case, execution='graph', **opts)
    [graph] = graphs
    assert graph.disabled is None and graph.captures >= 2
    assert graph._materialOwner is graphed.ctx.schemeConfig._materialRuntime
    for name in ('positions', 'velocities', 'internalEnergies', 'pressures'):
        torch.testing.assert_close(getattr(eager.state.state, name), getattr(graphed.state.state, name), rtol=0, atol=0)
