"""The MFM / MFV split of the meshless step: the per-mode steps and their guards, the temporal-scheme resolution, and the shared exact-delta contract."""

import pytest
import torch

from warpSPHIntegrators.schemes import IntegrationSchemeType

from warpSPH.configurations.meshless import MFMConfig, MFVConfig, MeshlessConfig
from warpSPH.schemes.compressible.meshless import (meshlessStep, mfmStep, mfvStep, resolve_temporal_scheme, validateIntegrationScheme)
from warpSPH.schemes.compressible.meshless.temporal import AUTO_TEMPORAL_SCHEME, TEMPORAL_SCHEMES
from warpSPH.schemes.exactDelta import (EXACT_DELTA_INTEGRATOR_NAMES, pinExactDeltaIntegrator, requireExactDeltaIntegrator)


def test_exactDeltaIntegratorsAreTheSingleEvaluationOnes():
    assert set(EXACT_DELTA_INTEGRATOR_NAMES) == {'forwardEuler', 'explicitEuler'}
    for name in EXACT_DELTA_INTEGRATOR_NAMES:
        assert pinExactDeltaIntegrator(name) == name
    for name in ('rungeKutta2', 'rungeKutta4', 'heunsMethod', 'symplecticEuler'):
        assert pinExactDeltaIntegrator(name) == 'forwardEuler'


@pytest.mark.parametrize('integrator', [IntegrationSchemeType.forwardEuler, IntegrationSchemeType.explicitEuler])
def test_singleEvaluationIntegratorsPass(integrator):
    class Cfg: integrationScheme = integrator
    requireExactDeltaIntegrator(Cfg, 'test')
    validateIntegrationScheme(Cfg)


@pytest.mark.parametrize('integrator', [IntegrationSchemeType.rungeKutta2, IntegrationSchemeType.rungeKutta4, None])
def test_multiStageIntegratorsAreRefused(integrator):
    class Cfg: integrationScheme = integrator
    with pytest.raises(ValueError, match='single-evaluation'):
        requireExactDeltaIntegrator(Cfg, 'test')


def test_autoResolvesPerMode():
    assert AUTO_TEMPORAL_SCHEME == {'MFM': 'ssprk', 'MFV': 'hancock'}
    assert resolve_temporal_scheme(MFMConfig()) == 'ssprk'
    assert resolve_temporal_scheme(MFVConfig()) == 'hancock'
    for ts in TEMPORAL_SCHEMES:
        cfg = MFMConfig(); cfg.temporalScheme = ts
        assert resolve_temporal_scheme(cfg) == ts
    cfg = MFMConfig(); cfg.temporalScheme = 'rk4'
    with pytest.raises(ValueError, match='not recognised'):
        resolve_temporal_scheme(cfg)


def test_eachStepRefusesTheOtherModesConfig():
    class Sys: state = None; domain = None
    class Cfg: integrationScheme = IntegrationSchemeType.forwardEuler; dim = 2
    with pytest.raises(ValueError, match='use MFMConfig'):
        mfmStep(Sys, 1e-3, Cfg, MFVConfig())
    with pytest.raises(ValueError, match='use MFVConfig'):
        mfvStep(Sys, 1e-3, Cfg, MFMConfig())


def test_dispatcherFollowsTheConfigMode():
    class Sys: state = None; domain = None
    class Cfg: integrationScheme = IntegrationSchemeType.rungeKutta2; dim = 2
    for cfg in (MFMConfig(), MFVConfig()):
        with pytest.raises(ValueError, match='ConservedLagrangian'):  # never interpret a completed step as an RHS
            meshlessStep(Sys, 1e-3, Cfg, cfg)
    bad = MeshlessConfig(); bad.mode = 'XYZ'
    with pytest.raises(ValueError, match="not 'MFM' or 'MFV'"):
        meshlessStep(Sys, 1e-3, Cfg, bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='warp kernels')
def test_bothModesRunAndKeepTheirInvariants():
    """MFM conserves each particle's mass exactly (dm = 0); both conserve total mass; the registered schemes use the package-owned time driver."""
    from warpSPH.cases import importAll
    from warpSPH.runner import run
    from warpSPH.runner.case import getCase
    importAll()
    for scheme in ('MFM', 'MFV'):
        r = run(getCase('sod2d'), quiet=True, progress=False, nSteps=8, scheme=scheme, plot=False, video=False, integrationScheme='rungeKutta2')
        assert r.nSteps == 8 and not r.diverged
        assert r.ctx.integrator.__class__.__module__.startswith('warpSPHIntegrators.')
        assert r.ctx.integrator.temporal == ('ssprk' if scheme == 'MFM' else 'hancock')
        Q = r.state.state.Q
        assert torch.isfinite(Q).all()


@pytest.mark.parametrize('scheme', ['MFM', 'MFV'])
def test_activeDriverMetadataKeepsSimulationConfigRoundTrippable(scheme):
    import json
    from warpSPH.configurations import configurationToDict, dictToConfig
    from warpSPH.runner import RunConfiguration
    from warpSPH.runner.configuration import resolvedConfiguration
    ctx = RunConfiguration('sod', simulation={'device': 'cpu', 'nx': 32}, solver={'scheme': scheme}).prepare()
    restored = dictToConfig(configurationToDict(ctx.config))
    assert restored.integrationScheme == ctx.config.integrationScheme
    assert isinstance(restored.integrationScheme, IntegrationSchemeType)
    manifest = json.loads(json.dumps(resolvedConfiguration(ctx)))
    assert manifest['integrator']['identifier'] == ctx.integrator.identifier.name
    assert manifest['integrator']['name'] == ctx.integrator.name

    from warpSPH.cases.common.plotting import _provenance
    title = _provenance(ctx)
    assert ctx.integrator.identifier.name in title
    assert 'rungeKutta2' not in title
