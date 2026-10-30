"""Tiny eager A/B trajectories: identical fresh stages, sources and row arithmetic.

Video is intentionally omitted for these bounded two-step parity fixtures.
The case hook explicitly seeds resident chunks; no runner default is changed.
"""
from dataclasses import replace
import importlib
import os

import pytest
import torch

from warpSPH.runner import RunConfiguration
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as case
from warpSPH.enumTypes import AdaptiveSupportScheme
from warpSPHCore import OperationProperties, KernelFunctions, setNeighborLanes
from warpSPHCore.radiusSearch import buildChunkedVerletList, ChunkedAdjacency, RowChunkPolicy
from warpSPH.modules.adaptiveSupport.wp_omega import computeOmegaWarp
from warpSPH.modules.adaptiveSupport.wp_psi0 import computePsi0Warp


@pytest.mark.parametrize('dim,tableau,support,gradH,reconstruction', [
    (2, 'rungeKutta2', 'NoScheme', False, 'raw'),
    (2, 'rungeKutta4', 'Monaghan', True, 'raw'),
    (3, 'rungeKutta2', 'Owen', False, 'raw'),
    (2, 'rungeKutta2', 'NoScheme', True, 'velocity'),
    (2, 'rungeKutta2', 'NoScheme', False, 'states'),
])
def test_eager_monaghan_preserves_full_step_fields(monkeypatch, dim, tableau, support, gradH, reconstruction):
    rhs = []
    device = os.environ.get('WARPSPH_TEST_DEVICE', 'cpu')
    def run(chunked):
        def build(ctx):
            system = case.buildSystem(ctx)
            ctx.schemeConfig.adaptiveSupportScheme = getattr(AdaptiveSupportScheme, support)
            ctx.schemeConfig.adaptiveSupportCorrections = gradH
            ctx.schemeConfig.adaptiveSupportIterations = 4
            from warpSPH.configurations.moduleConfigurations.diffusionParameters import VelocityPairPolicy, ViscosityTerms
            if reconstruction == 'velocity':
                ctx.schemeConfig.diffusionParams.velocityPairPolicy = VelocityPairPolicy.MonotoneLimited.value
                ctx.schemeConfig.diffusionParams.balsaraPairLimiter = True
            elif reconstruction == 'states':
                ctx.schemeConfig.diffusionParams.viscosityTerm = ViscosityTerms.RiemannDissipation.value
                ctx.schemeConfig.diffusionParams.riemannReconstruction = True
            x = system.state.positions
            system.state.velocities[:, 0] = .12 * torch.sin(6.28 * x[:, 1])
            system.state.velocities[:, 1] = .09 * torch.cos(6.28 * x[:, 0])
            system.state.internalEnergies *= 1 + .06 * torch.cos(6.28 * x[:, 0])
            from warpSPH.modules.eos import refreshEOS
            refreshEOS(system.state, ctx.schemeConfig)
            s = system.state
            ctx.scratch['initialEnergy'] = (s.masses.double() *
                (s.internalEnergies.double() + .5 * s.velocities.double().square().sum(-1))).sum()
            if chunked:
                system.adjacency = buildChunkedVerletList(s, ctx.config.domain,
                    policy=RowChunkPolicy(edgeBudget=2048, maxRows=5), verletScale=ctx.config.verletScale)
                # Standalone support operators exercise their tuple and scalar outputs.
                props = OperationProperties(kernel=KernelFunctions.Wendland2)
                for op in (computeOmegaWarp, computePsi0Warp):
                    a, b = op(s, props, ctx.config.domain, adjacency=system.adjacency), op(s, props, ctx.config.domain)
                    if isinstance(a, torch.Tensor):
                        a, b = (a,), (b,)
                    for av, bv in zip(a, b):
                        # Grid and CSR can visit rows in different orders; compare the
                        # actual step against the exact same flat CSR order below.
                        torch.testing.assert_close(av, bv, rtol=3e-6, atol=3e-6)
                def forbidden(*a, **kw):
                    raise AssertionError('chunked step rebuilt a monolithic neighbor list')
                module = importlib.import_module('warpSPHCore.radiusSearch.verlet.build')
                monkeypatch.setattr(module, 'radiusSearchCompactHashMap_', forbidden)
            original = ctx.bundle.stepFunction
            def counted(*a, **kw):
                rhs.append(chunked)
                return original(*a, **kw)
            ctx.bundle = replace(ctx.bundle, stepFunction=counted)
            return system
        cfg = RunConfiguration('drivenTurbulence', simulation=dict(device=device, nx=8, dim=dim,
            nSteps=2, maxDt=.0002, plot=False, video=False, store=False, show=False,
            quiet=True, progress=True, velocityAlarmPlotInterval=1, stallProgress=1e-3, storeMode='trajectory', integrationScheme=tableau),
            params=dict(accelerationRMS=.1), solver=dict(scheme='Monaghan'))
        return cfg.run(replace(case, buildSystem=build))
    setNeighborLanes(1)
    try:
        flat = run(False)
        chunked = run(True)
    finally:
        setNeighborLanes(None)
    assert not flat.diverged and not chunked.diverged and flat.nSteps == chunked.nSteps == 2
    assert flat.state.t == chunked.state.t
    assert rhs.count(False) == rhs.count(True) == (8 if tableau == "rungeKutta4" else 4)
    for name, value in vars(flat.state.state).items():
        other = getattr(chunked.state.state, name)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, other, rtol=0, atol=0, msg=lambda m: name + ': ' + m)
    from warpSPH.caseUtils.drivenTurbulence import adjacencyCheckpoint
    with pytest.raises(NotImplementedError, match='checkpoint export'):
        adjacencyCheckpoint(chunked.state)
    assert isinstance(chunked.state.adjacency, ChunkedAdjacency)
    assert len(chunked.state.adjacency.chunks) > 1
    assert all(c.metadata.capacity <= 2048 for c in chunked.state.adjacency)
    assert min(chunked.trajectory[-1][k] for k in ('minDensity', 'minPressure', 'minEnergy')) > 0
    assert abs(chunked.trajectory[-1]['relativeEnergyBudgetResidual']) < 2e-6
