"""`SimulationConfig.compute` (`configurations/compute.py`): the settings that say how a run executes, in one place, with one precedence rule
(scheme / spec / solver value > this config > shipped default)."""

import types

import pytest
import torch

from warpSPH.configurations import (CHECK_SCHEDULES, EXECUTION_MODES, GRAPH_MAX_PARTICLES, ComputeConfig, SimulationConfig, buildConfig,
                                    configurationToDict, dictToConfig, graphParticleLimit, requestedExecution, resolveCheckSchedule,
                                    resolveExecution, resolveKrylovCheckEvery)
from warpSPH.enumTypes import CompressibleSPHScheme, WeaklyCompressibleSPHScheme

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
SCHEME = WeaklyCompressibleSPHScheme.sun2017DeltaSPH


def _cfg(**compute):
    return types.SimpleNamespace(compute=ComputeConfig(**compute))


def test_defaultsAreTheHistoricalBehaviour():
    c = ComputeConfig()
    assert (c.execution, c.graphMaxParticles, c.pipelineOutputs, c.iterationChecks, c.krylovCheckEvery) == ('eager', None, True, 'adaptive', None)
    assert SimulationConfig().compute == c
    assert resolveKrylovCheckEvery(_cfg()) == 8


@pytest.mark.parametrize('bad', [dict(execution='gpu'), dict(iterationChecks='sometimes'), dict(krylovCheckEvery=0), dict(graphMaxParticles=-1)])
def test_invalidValuesRaise(bad):
    with pytest.raises(ValueError):
        ComputeConfig(**bad)


def test_roundTripAndOldFiles():
    config, _ = buildConfig(compute={'execution': 'auto', 'graphMaxParticles': 123, 'iterationChecks': 'every', 'krylovCheckEvery': 3,
                                     'pipelineOutputs': False})
    d = configurationToDict(config)
    assert d['compute'] == {'execution': 'auto', 'graphMaxParticles': 123, 'pipelineOutputs': False, 'iterationChecks': 'every', 'krylovCheckEvery': 3, 'neighborCollection': 'auto', 'neighborQueryOrder': 'auto', 'neighborGridBuild': 'scheme'}
    back = dictToConfig(d)
    assert back.compute == config.compute
    del d['compute']                                                    # a file written before the field existed
    assert dictToConfig(d).compute == ComputeConfig()
    assert dictToConfig(configurationToDict(buildConfig(densityCorrection=True)[0])).densityCorrection.enabled     # the other nested config still round-trips


def test_precedence_schemeThenSpecThenConfig():
    cfg = _cfg(execution='auto')
    scheme = types.SimpleNamespace(stepMode=None)
    spec = types.SimpleNamespace(cudaGraph=None)
    assert requestedExecution(cfg, scheme, spec) == 'auto'                 # the config
    spec.cudaGraph = True
    assert requestedExecution(cfg, scheme, spec) == 'graph'                # the legacy spec flag wins over the config
    spec.cudaGraph = False
    assert requestedExecution(cfg, scheme, spec) == 'eager'
    scheme.stepMode = 'graph'
    assert requestedExecution(cfg, scheme, spec) == 'graph'                # the scheme's own switch wins over both
    scheme.stepMode = 'turbo'
    with pytest.raises(ValueError, match='stepMode must be one of'):
        requestedExecution(cfg, scheme, spec)
    assert requestedExecution(cfg) == 'auto'                               # no scheme config / spec at all


def test_autoUsesTheTableTheOverrideAndTheSchemeOverride():
    n = GRAPH_MAX_PARTICLES[(SCHEME, 2)]
    cfg = _cfg(execution='auto')
    assert resolveExecution(cfg, SCHEME, None, None, n - 1, 2) == 'graph'
    assert resolveExecution(cfg, SCHEME, None, None, n, 2) == 'eager'
    assert resolveExecution(cfg, SCHEME, None, None, 10, 3) == 'eager'     # no entry for (scheme, 3D): never auto-graph
    assert resolveExecution(_cfg(execution='auto', graphMaxParticles=50), SCHEME, None, None, 49, 2) == 'graph'
    assert resolveExecution(_cfg(execution='auto', graphMaxParticles=50), SCHEME, None, None, 50, 2) == 'eager'
    scheme = types.SimpleNamespace(stepMode=None, stepModeMaxParticles=7)
    assert graphParticleLimit(_cfg(graphMaxParticles=50), SCHEME, 2, scheme) == 7
    assert resolveExecution(_cfg(execution='eager'), SCHEME, None, None, 1, 2) == 'eager'
    assert resolveExecution(_cfg(execution='graph'), SCHEME, None, None, 10 ** 9, 2) == 'graph'
    assert GRAPH_MAX_PARTICLES[(CompressibleSPHScheme.CompSPH, 3)] == 0


def test_checkScheduleResolution():
    solver = types.SimpleNamespace(convergenceCheckSchedule='auto')
    assert resolveCheckSchedule(solver, _cfg()) == 'adaptive'
    assert resolveCheckSchedule(solver, _cfg(iterationChecks='every')) == 'every'
    solver.convergenceCheckSchedule = 'adaptive'
    assert resolveCheckSchedule(solver, _cfg(iterationChecks='every')) == 'adaptive'     # the solver's own value wins
    assert resolveCheckSchedule(types.SimpleNamespace(), _cfg(iterationChecks='every')) == 'every'
    assert resolveCheckSchedule(solver, types.SimpleNamespace()) == 'adaptive'           # a config without compute
    solver.convergenceCheckSchedule = 'whenever'
    with pytest.raises(ValueError):
        resolveCheckSchedule(solver, _cfg())
    assert set(CHECK_SCHEDULES) == {'every', 'adaptive'} and set(EXECUTION_MODES) == {'eager', 'graph', 'auto'}


def test_krylovIntervalReachesThePressureSolver(monkeypatch):
    import warpSPH.modules.incompressible.krylov as krylov
    seen = {}

    def fake(matvec, b, x0, **kw):
        seen.update(kw)
        return b, 0, [torch.zeros(())]
    monkeypatch.setattr(krylov, 'cgSolve', fake)
    monkeypatch.setattr(krylov, 'buildIISPHMatvec', lambda *a, **k: (lambda p: p))
    monkeypatch.setattr(krylov, 'buildIISPHPrecond', lambda *a, **k: torch.ones(4))
    monkeypatch.setattr(krylov, 'computePressureAccelIISPH', lambda **k: torch.zeros(4, 2))
    monkeypatch.setattr(krylov, 'resolveBoundaryOperatorTerms', lambda *a, **k: types.SimpleNamespace(operatorMovesBoundary=True))
    particles = types.SimpleNamespace(kinds=torch.zeros(4, dtype=torch.int32), pressures=torch.zeros(4), densities=torch.ones(4))
    solverCfg = types.SimpleNamespace(solverType=krylov.PressureSolverType.cg, rtol=1e-5, atol=0.0, maxIterations=5, restart=30, krylovFp64=False)
    schemeConfig = types.SimpleNamespace(solverConfig=types.SimpleNamespace())
    krylov.solvePressureKrylov(particles, _cfg(krylovCheckEvery=3), schemeConfig, None, torch.ones(4), 1.0, solverCfg, gauge=None)
    assert seen['checkEvery'] == 3


def _dambreak(**spec):
    from warpSPH.cases.freeSurface.dambreak import dambreakCase
    from warpSPH.runner import run
    H, L = 0.6, 1.0
    params = dict(W=5.366 * H, fillRatio=H / L, fluidWidth=2.0 * H / (5.366 * H), gravityMagnitude=9.81, pressureProbeHeights=[0.16, 0.584],
                  pressureProbeDiscRadius=0.045, referenceVelocity=1.95 * (9.81 * H) ** 0.5, machTarget=1.95 / 40.0)
    return run(dambreakCase, scheme='sun2017DeltaSPH', L=L, nx=12, nSteps=3, params=params, quiet=True, store=False, progress=False, plot=False, **spec)


@cuda
@pytest.mark.parametrize('spec,expected', [
    (dict(cudaGraph=False), 'eager'),                                     # the case defaults cudaGraph on; an explicit False wins
    (dict(cudaGraph=None, execution='graph'), 'graph'),
    (dict(cudaGraph=None, execution='auto', graphMaxParticles=10 ** 6), 'graph'),
    (dict(cudaGraph=None, execution='auto', graphMaxParticles=10), 'eager'),
    (dict(cudaGraph=None, execution='eager'), 'eager'),
])
def test_runnerPicksTheExecutionFromTheComputeConfig(spec, expected):
    result = _dambreak(**spec)
    assert result.ctx.scratch['execution'] == expected
    assert bool(getattr(result.ctx.schemeConfig, 'cudaGraph', False)) == (expected == 'graph')
    assert (result.ctx.scratch.get('stepGraph') is not None) == (expected == 'graph')


@pytest.mark.parametrize('bad', [dict(neighborCollection='bad'), dict(neighborQueryOrder='bad'), dict(neighborGridBuild='bad')])
def test_invalidNeighborChoicesRaise(bad):
    with pytest.raises(ValueError, match='compute.neighbor'):
        ComputeConfig(**bad)


def test_neighborOptionsRoundTripAndFactorySelection():
    from warpSPH.configurations import neighborSearchOptions
    config, _ = buildConfig(compute={'neighborCollection': 'fused', 'neighborQueryOrder': 'input', 'neighborGridBuild': 'eager'})
    assert dictToConfig(configurationToDict(config)).compute == config.compute
    factory = lambda: None
    assert neighborSearchOptions(config, gridFactory=factory) == dict(collection='fused', queryOrder='input', gridFactory=None)
    assert neighborSearchOptions(types.SimpleNamespace(), gridFactory=factory) == dict(collection='auto', queryOrder='auto', gridFactory=factory)
    assert neighborSearchOptions(None) == dict(collection='auto', queryOrder='auto')
    config.compute.neighborQueryOrder = 'bad'
    with pytest.raises(ValueError, match='neighborQueryOrder'):
        neighborSearchOptions(config)


def test_neighborChoicesReachARealSchemeCall(monkeypatch):
    import importlib
    module = importlib.import_module('warpSPH.schemes.weaklyCompressible.deltaSPH')
    seen = {}
    class StopAfterSearch(Exception):
        pass
    def search(*args, **kwargs):
        seen.update(kwargs)
        raise StopAfterSearch
    monkeypatch.setattr(module, 'buildVerletList', search)
    config = _cfg(neighborCollection='fused', neighborQueryOrder='input')
    config.domain = None
    config.verletScale = 1.2
    system = types.SimpleNamespace(state=object(), adjacency=None)
    with pytest.raises(StopAfterSearch):
        module.deltaSPH_step(system, .01, config, types.SimpleNamespace())
    assert seen['collection'] == 'fused' and seen['queryOrder'] == 'input'
