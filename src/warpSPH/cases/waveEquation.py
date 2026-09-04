"""Scalar wave equation on SPH particles, 1D/2D/3D.

The registered `Case` for the non-fluid demo scheme in
`schemes/waveEquation.py`: `d2u/dt2 = c^2 laplacian(u)` on a static particle
lattice, with a couple of Gaussian-ish sources and (in 2D) an optional
circular obstacle of reduced wave speed, absorbed at the domain edge by the
cosine border-damping profile (`caseUtils/waveEquation/damping.py`) instead
of a hard wall.

`shape_generation.py`'s SDF-based source/obstacle placement is 2D-only, so
`--dim 1`/`--dim 3` fall back to a single smooth point source
(`sample/waveSystem.py`'s `sampleSmoothPointSourceWaveSystem`) instead of
going through it; see `docs/historic_plans/WAVE_EQUATION_PLAN.md` for why -- that path also
doubles as the differentiable one, since it contributes to `u` as a direct
function of a `position`/`magnitude` tensor rather than through the SDF id
grid's non-differentiable step function. Positions are static -- the
adjacency built once in `buildSystem` is reused every step by
`f_wave_equation` -- so this is an explicit, non-moving-neighbourhood case;
implicit time integration and forward-mode AD are follow-up work, not wired
up here.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from ..caseUtils.waveEquation.gencase import genInitial
from ..caseUtils.waveEquation.shape_generation import \
    populate_source_obstacle_grids_structured
from ..configurations import WaveBoundary, WaveCaseConfig, WaveShapeSpec, WaveSource
from ..runner import Case, RunContext, caseMain, registerCase
from ..sample.bySamplingScheme import sampleParticles
from ..sample.waveSystem import finalizeWaveSystemSetup, sampleSmoothPointSourceWaveSystem
from ..systems.waveSystem import computeDt
from .plotting import Field, ProfileAxis, particlePlot, profilePlot
from warpSPHCore import OperationProperties, WarpOperation, warpOperation

__all__ = ['waveEquationCase']


def _buildCaseConfig(ctx: RunContext) -> WaveCaseConfig:
    radius = ctx.param('sourceRadius')
    sources = [WaveSource(
        shapeSpec=WaveShapeSpec(kind='sphere', position=tuple(ctx.param('source1Position')),
                                params={'radius': radius}),
        magnitude=ctx.param('amplitude'),
    )]
    if ctx.param('secondSource'):
        sources.append(WaveSource(
            shapeSpec=WaveShapeSpec(kind='sphere', position=tuple(ctx.param('source2Position')),
                                    params={'radius': radius}),
            magnitude=-ctx.param('amplitude'),
        ))

    obstacles: List[WaveBoundary] = []
    if ctx.param('obstacleEnabled'):
        obstacles.append(WaveBoundary(
            shapeSpec=WaveShapeSpec(kind='sphere', position=tuple(ctx.param('obstaclePosition')),
                                    params={'radius': ctx.param('obstacleRadius')}),
            speed=ctx.param('obstacleSpeed'),
        ))

    return WaveCaseConfig(
        name=ctx.spec.caseName,
        domainBox=ctx.param('domainBox'),
        domainDamping=ctx.param('domainDamping'),
        smoothICs=ctx.param('smoothICs'),
        defaultSpeed=ctx.param('c'),
        defaultBoundarySpeed=ctx.param('boundarySpeed'),
        defaultObstacleSpeed=ctx.param('obstacleSpeed'),
        defaultAmplitude=ctx.param('amplitude'),
        sources=sources,
        obstacles=obstacles,
    )


def buildSystem(ctx: RunContext):
    caseConfig = _buildCaseConfig(ctx)

    if ctx.spec.dim == 2:
        particles = sampleParticles(ctx.spec.nx, ctx.config)
        u, v, cGrid, dampGrid, uSourceGrid, cSourceGrid = genInitial(
            particles, ctx.config,
            domainBox=caseConfig.domainBox,
            domainDamping=caseConfig.domainDamping,
        )
        uSourceGrid, cSourceGrid, sourceMagnitudes, obstacleSpeeds = \
            populate_source_obstacle_grids_structured(
                particles, ctx.config, caseConfig, uSourceGrid, cSourceGrid)

        waveSystem, _dt = finalizeWaveSystemSetup(
            particles, u, v, cGrid, dampGrid, uSourceGrid, cSourceGrid,
            sourceMagnitudes, obstacleSpeeds, ctx.config, caseConfig,
        )
    else:
        # `shape_generation` is 2D-only; 1D/3D get a single smooth point
        # source at the domain centre instead of going through the SDF shape
        # machinery.
        position = torch.zeros(ctx.spec.dim, device=ctx.config.device, dtype=ctx.config.dtype)
        magnitude = torch.tensor(float(ctx.param('amplitude')), device=ctx.config.device, dtype=ctx.config.dtype)
        waveSystem = sampleSmoothPointSourceWaveSystem(
            ctx.spec.nx, ctx.config, caseConfig,
            position=position, magnitude=magnitude, radius=ctx.param('sourceRadius'),
        )
        obstacleSpeeds = []

    # `initialConditions` needs both to derive `dt` from the same CFL
    # condition `finalizeWaveSystemSetup` would otherwise use internally.
    ctx.scratch['caseConfig'] = caseConfig
    ctx.scratch['obstacleSpeeds'] = obstacleSpeeds

    return waveSystem


def configureScheme(ctx: RunContext) -> None:
    """Carry the generic `--kernel`/`--supportMode`/`--gradientMode`/
    `--laplacianMode` flags onto the wave scheme's own config: `f_wave_equation`
    reads its operator settings from `schemeConfig`, not the generic `config`,
    the way every other scheme's own numerics live on `schemeConfig`."""
    schemeConfig = ctx.schemeConfig
    schemeConfig.kernel = ctx.config.kernel
    schemeConfig.supportMode = ctx.config.supportMode
    schemeConfig.gradientMode = ctx.config.gradientMode
    schemeConfig.laplacianMode = ctx.config.laplacianMode


def initialConditions(ctx: RunContext, system) -> None:
    # Positions and wave speed are both static, so the CFL-derived dt is exact
    # for the whole run -- no `timestep` hook needed.
    if ctx.spec.dt is None:
        caseConfig = ctx.scratch['caseConfig']
        obstacleSpeeds = ctx.scratch['obstacleSpeeds']
        ctx.config.dt = computeDt(system, ctx.config, caseConfig, None,
                                  obstacleSpeeds, verbose=not ctx.spec.quiet)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    """Discrete wave energy (kinetic + gradient-based potential term), plus
    u/v extrema -- what `tests/test_physics.py` asserts an energy-drift bound
    on."""
    particles = state.state
    kinetic = 0.5 * (particles.masses * particles.v ** 2).sum()

    grad_u = warpOperation(
        particles, queryValues=particles.u,
        domain=state.domain, adjacency=state.adjacency,
        operationProperties=OperationProperties(
            operation=WarpOperation.Gradient,
            kernel=ctx.schemeConfig.kernel,
            supportMode=ctx.schemeConfig.supportMode,
            gradientMode=ctx.schemeConfig.gradientMode,
        ),
    )
    potential = 0.5 * (particles.masses * particles.c ** 2 * (grad_u ** 2).sum(dim=-1)).sum()

    return {
        'kineticEnergy': kinetic.detach().cpu().item(),
        'potentialEnergy': potential.detach().cpu().item(),
        'totalEnergy': (kinetic + potential).detach().cpu().item(),
        'uMax': particles.u.abs().max().detach().cpu().item(),
        'vMax': particles.v.abs().max().detach().cpu().item(),
    }


# `particlePlot` (vispy/matplotlib scatter or grid, by `Field`) is what every
# 2D/3D fluid case uses, but its underlying `warpSPHPlotting.visualize` is not
# meant for 1D particle sets -- `dim==1` cases (sod, noh, kidder, ...) all use
# `profilePlot` (a plain matplotlib scatter against `x`) instead. This case
# runs in both regimes, so `setupPlot`/`updatePlot` dispatch on `ctx.spec.dim`
# rather than picking one.
_setupFieldPlot, _updateFieldPlot = particlePlot([
    Field('u', 'u', colorMap='RdBu', colorMapKind='diverging'),
    Field('v', 'v', colorMap='RdBu', colorMapKind='diverging'),
    Field('c', 'wave speed'),
])

_setupProfilePlot, _updateProfilePlot, _drawProfilePlot = profilePlot([
    ProfileAxis('u', 'u'),
    ProfileAxis('v', 'v'),
    ProfileAxis('c', 'wave speed'),
], shape=(1, 3))


def setupPlot(ctx: RunContext, state):
    if ctx.spec.dim == 1:
        return _setupProfilePlot(ctx, state)
    return _setupFieldPlot(ctx, state)


def updatePlot(ctx: RunContext, state, handle, step: int) -> None:
    if ctx.spec.dim == 1:
        _updateProfilePlot(ctx, state, handle, step)
    else:
        _updateFieldPlot(ctx, state, handle, step)


waveEquationCase = registerCase(Case(
    name='waveEquation',
    scheme='waveEquation',
    description='Scalar wave equation on SPH particles (1D/2D/3D), non-fluid demo scheme.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    defaults=dict(
        caseName='waveEquation',
        dim=2,
        nx=64,
        L=2.0,
        n_h=4.0,
        periodic=True,
        kernel='Wendland2',
        supportMode='SuperSymmetric',
        gradientMode='Difference',
        laplacianMode='Brookshaw',
        integrationScheme='rungeKutta4',
        samplingScheme='regular',
        dt=None,
        adaptiveDt=True,
        # `computeDt`'s CFL number is acoustic (dt ~ h/c); the strong border
        # damping this case relies on for absorption (`dampingStrength=48` in
        # `borderDamping_strong`) is a stiff term under explicit RK4, whose
        # own stability limit (`dt < ~2.8/dampingStrength`) is the tighter
        # constraint at the default resolution -- 0.3 blows up within a few
        # tens of steps, 0.1 stays bounded for the whole default run.
        cflFactor=0.1,
        tLimit=2.0,
        plotInterval=10,
        storeInterval=50,
    ),
    params=dict(
        c=1.0,
        amplitude=10.0,
        sourceRadius=0.15,
        source1Position=[-0.4, 0.0],
        source2Position=[0.4, 0.0],
        secondSource=True,
        obstacleEnabled=True,
        obstaclePosition=[0.0, 0.35],
        obstacleRadius=0.2,
        obstacleSpeed=0.5,
        boundarySpeed=0.01,
        domainDamping=True,
        domainBox=False,
        smoothICs=False,
    ),
))


if __name__ == '__main__':
    caseMain(waveEquationCase)
