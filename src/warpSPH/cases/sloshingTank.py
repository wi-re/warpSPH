"""Laterally-excited sloshing tank -- SPHERIC Test Case 10 (2D free surface).

A shallow layer of water (depth ``h = 0.093 m``) in a rigid rectangular tank
(``0.900 x 0.508 m``) that is **rolled** harmonically about the centre of its
floor (amplitude ~+-4 deg, period ~1.7 s, near the first sloshing mode). The
figure of merit is the pressure history at wall **Sensor 1**, on the left wall
at the still-water line ``(-0.45, 0.093)``; the measured signal and the
prescribed motion are in ``examples/sloshingTank/SPHERIC_TestCase10/``.

Ported from diffSPH's ``examples/weaklyCompressible/16_SloshingTank.ipynb``. See
``examples/sloshingTank/PLAN.md`` for the full write-up.

Tank-fixed frame
----------------
Rather than move the walls, the run is solved in the frame that rolls with the
tank and **gravity is rotated** by the roll angle each step::

    g_dir(t) = R(-theta(t)) . (0, -1) = (-sin theta, -cos theta)

``modules/gravity/directional.py`` re-reads ``gravityConfig.direction`` every
step, so the ``postStep`` hook below just rewrites it from a spline of the
tabulated roll angle. This drops the non-inertial Euler and centrifugal terms
(small at this amplitude/period, and dropped by the diffSPH reference too); an
``inertialForces`` extension can add them later.

Runs under two schemes
----------------------
- ``deltaSPH`` (weakly compressible, the default): sound speed and ``dt`` are
  fixed together from ``targetDt``; the sensor pressure is the nearest boundary
  particle's density mapped through the Tait/linear EOS, scaled by
  ``rho0Physical`` (the scheme runs at ``restDensity = 1``).
- ``divergenceFree`` (incompressible DFSPH): pass
  ``--scheme divergenceFree --integrationScheme semiImplicitEuler
  --kernel Wendland2 --cflFactor 0.2 --dt 1e-3`` (the ``run_sloshingTank.py``
  ``--scheme dfsph`` preset does this). The sensor pressure is the solver's
  carried ``pressures`` at the sensor particle, scaled by ``rho0Physical``.
  Note ``divergenceFree`` is known to struggle with quiescent
  free-surface-under-gravity states (``DFSPH_IMPROVEMENT_PLAN.md`` Part 23,
  ``hydrostaticColumn``) -- a divergence or heavy over-dissipation here is a
  result to report, not necessarily a wiring bug.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict

import numpy as np
import torch

from ..configurations.moduleConfigurations.gravity import GravityType
from ..configurations.moduleConfigurations.shifting import ShiftingProjectionScheme
from ..configurations.region import BCType
from ..enumTypes import IncompressibleSPHScheme, WeaklyCompressibleSPHScheme
from ..runner import Case, RunContext, caseMain, registerCase
from ..utils import buildDomainDescription
from ..regions import sampleDomainSDF
from .kolmogorovIncompressible import kolmogorovIncompressibleTimestep
from .plotting import Field, particlePlot
from .weaklyCompressible import (boundaryRegion, buildRegionSystem, fluidRegion,
                                 setupTimestep, shapeSdf)

__all__ = ['sloshingTankCase']

#: ``.../examples/sloshingTank/SPHERIC_TestCase10`` -- three parents up from
#: ``src/warpSPH/cases/`` is the repo root.
_SPHERIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    'examples', 'sloshingTank', 'SPHERIC_TestCase10')
_DEFAULT_ROLL_FILE = os.path.join(_SPHERIC_DIR, 'data_files', 'lateral_water_1x.txt')

_TAIT_GAMMA = 7.0


# -- roll history -----------------------------------------------------------

def loadRollHistory(path: str) -> Dict[str, np.ndarray]:
    """Read the SPHERIC roll table -> ``{t, theta(rad), pressure(mbar)}``.

    Columns (tab separated): ``Time[s]  Pressure[mbar]
    Position_smooth_splines[deg]  Velocity[deg/s]  Acceleration[deg/s2]
    Position_original[deg]``.
    """
    raw = np.genfromtxt(path, delimiter='\t', skip_header=1)
    return {
        't': raw[:, 0].astype(np.float64),
        'pressureMbar': raw[:, 1].astype(np.float64),
        'theta': np.radians(raw[:, 2].astype(np.float64)),
        'omega': np.radians(raw[:, 3].astype(np.float64)),
    }


def _rollFilePath(ctx: RunContext) -> str:
    path = ctx.param('rollDataFile') or _DEFAULT_ROLL_FILE
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'sloshingTank: roll-history file not found: {path!r}. Pass '
            f'--rollDataFile or check examples/sloshingTank/SPHERIC_TestCase10/.')
    return path


def _rollAngle(ctx: RunContext, t: float) -> float:
    roll = ctx.scratch['rollHistory']
    return float(np.interp(t + ctx.param('rollStartTime'), roll['t'], roll['theta']))


def _applyRollGravity(ctx: RunContext, t: float) -> None:
    """Rotate the (down) gravity vector into the tank frame at time ``t``."""
    theta = _rollAngle(ctx, t)
    ctx.schemeConfig.gravityConfig.direction = torch.tensor(
        [-math.sin(theta), -math.cos(theta)], device=ctx.device, dtype=ctx.dtype)
    ctx.scratch['rollAngle'] = theta


# -- geometry -------------------------------------------------------------------

def configureScheme(ctx: RunContext) -> None:
    L = ctx.spec.L                       # tank internal width B
    dx = L / ctx.spec.nx
    band = ctx.param('band')
    halfW = 0.5 * L
    tankH = ctx.param('tankHeight')

    # Periodic wrapper, widened by `band` particle layers; the walls are cut
    # from the interior rectangle, which is asymmetric in y (floor at 0).
    span = max(L, tankH) + 4.0 * band * dx
    domain = buildDomainDescription(span, ctx.spec.dim, True, ctx.device, ctx.dtype)
    domain.min = torch.tensor([-halfW - band * dx, -band * dx],
                              device=ctx.device, dtype=ctx.dtype)
    domain.max = torch.tensor([halfW + band * dx, tankH + band * dx],
                              device=ctx.device, dtype=ctx.dtype)
    domain.periodic[:] = True

    interior = buildDomainDescription(L, ctx.spec.dim, False, ctx.device, ctx.dtype)
    interior.min = torch.tensor([-halfW, 0.0], device=ctx.device, dtype=ctx.dtype)
    interior.max = torch.tensor([halfW, tankH], device=ctx.device, dtype=ctx.dtype)

    ctx.config.domain = domain
    ctx.config.dx = dx
    ctx.config.nx = ctx.spec.nx + 2 * band
    ctx.scratch['interiorDomain'] = interior

    sc = ctx.schemeConfig
    sc.surfaceDetectionConfig.active = True
    sc.gravityConfig.active = True
    sc.gravityConfig.type = GravityType.Directional
    sc.gravityConfig.magnitude = ctx.param('gravityMagnitude')
    sc.gravityConfig.direction = [0.0, -1.0]
    sc.bandwith = L / ctx.param('bandWidth') / dx

    if ctx.scheme is IncompressibleSPHScheme.divergenceFree:
        sc.diffusionParams.inviscid = False
        sc.diffusionParams.viscidNu = ctx.param('nu')
        if hasattr(sc, 'shiftProperties'):
            sc.shiftProperties.active = ctx.param('shifting', False)
        if hasattr(sc, 'xsphFilterScale'):
            sc.xsphFilterScale = ctx.param('xsphScale', 0.0)
    else:                                # deltaSPH -- weakly compressible
        sc.diffusionParams.inviscid = ctx.param('inviscid')
        sc.diffusionParams.inviscidAlpha = ctx.param(
            'alpha', sc.diffusionParams.inviscidAlpha)
        if not ctx.param('inviscid'):
            sc.diffusionParams.viscidNu = ctx.param('nu')
        # The δ⁺-SPH particle shift near the free surface -- the reason this
        # case survives the wall slam at all (WCSPH_SHIFTING_PLAN.md). On by
        # default now that `surfaceNormal` is the projection default; the knobs
        # are for the shift A/B (`--shift-projection`, `--no-shift`).
        if hasattr(sc, 'shiftProperties'):
            sc.shiftProperties.active = ctx.param('shifting', True)
            proj = ctx.param('shiftProjection', None)
            if proj:
                sc.shiftProperties.projectionScheme = ShiftingProjectionScheme[proj]
            # Sun 2019 Eq. (9) continuity δu-terms -- the volume-consistency
            # fix for the free-surface de-densification (WCSPH_SHIFTING_PLAN
            # §2d). Off by default; opt in for the sloshing quantitative pass.
            sc.shiftProperties.correctdrhodt = ctx.param('correctdrhodt', False)

    ctx.scratch['rollHistory'] = loadRollHistory(_rollFilePath(ctx))


def buildSystem(ctx: RunContext):
    interior = ctx.scratch['interiorDomain']
    halfW = 0.5 * ctx.spec.L
    fill = ctx.param('fillDepth')

    fluidSdf = shapeSdf('box', args=[[halfW, 0.5 * fill]], offset=[0.0, 0.5 * fill])
    wallSdf = lambda x: sampleDomainSDF(x, interior, invert=False)

    regions = [
        fluidRegion(ctx, fluidSdf),
        boundaryRegion(ctx, wallSdf, kind=BCType[ctx.param('wallBC')]),
    ]
    return buildRegionSystem(ctx, regions)


def initialConditions(ctx: RunContext, system) -> None:
    particles = system.state
    particles.velocities[:] = 0.0
    if getattr(particles, 'pressures', None) is not None:
        particles.pressures[:] = 0.0

    if ctx.scheme is IncompressibleSPHScheme.divergenceFree:
        if ctx.config.dt is None:
            ctx.config.dt = ctx.spec.dt if ctx.spec.dt is not None else 1e-3
    else:
        setupTimestep(ctx, system)       # fixes fluid.fixedSoundSpeed and config.dt

    ctx.scratch['sensorIndex'] = None
    _applyRollGravity(ctx, 0.0)


def postStep(ctx: RunContext, state, step: int) -> None:
    t = state.t.item() if torch.is_tensor(state.t) else float(state.t)
    _applyRollGravity(ctx, t)


def sloshingTimestep(ctx: RunContext, state) -> float:
    if ctx.scheme is IncompressibleSPHScheme.divergenceFree:
        return kolmogorovIncompressibleTimestep(ctx, state)
    return ctx.config.dt


# -- sensor / diagnostics -----------------------------------------------------

def _locateSensor(ctx: RunContext, particles) -> int:
    sensor = torch.as_tensor(ctx.param('sensorPos'), device=particles.positions.device,
                             dtype=particles.positions.dtype)
    boundary = particles.kinds == 1
    allIdx = torch.arange(particles.positions.shape[0], device=particles.positions.device)
    bIdx = allIdx[boundary]
    dist = torch.linalg.norm(particles.positions[boundary] - sensor, dim=-1)
    idx = int(bIdx[torch.argmin(dist)].item())
    ctx.scratch['sensorActualPos'] = particles.positions[idx].detach().cpu().tolist()
    ctx.scratch['sensorGap'] = float(dist.min().item())
    return idx


def _probePressure(ctx: RunContext, particles, quantity: torch.Tensor):
    """Gaussian Shepard smoothing of ``quantity`` over fluid particles within
    ``probeRadius`` of the sensor -- a spray-robust cross-check on the
    single-particle reading."""
    sensor = torch.as_tensor(ctx.param('sensorPos'), device=particles.positions.device,
                             dtype=particles.positions.dtype)
    fluid = particles.kinds == 0
    r = torch.linalg.norm(particles.positions[fluid] - sensor, dim=-1)
    radius = ctx.param('probeRadius', 3.0 * ctx.config.dx)
    near = r < radius
    if int(near.sum()) < 3:
        return None
    w = torch.exp(-(r[near] / (0.5 * radius)) ** 2)
    return float((w * quantity[fluid][near]).sum() / w.sum())


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    particles = state.state
    fluid = particles.kinds == 0
    vel = particles.velocities[fluid]
    rho = particles.densities[fluid]
    rho0Phys = ctx.param('rho0Physical')
    rho0 = ctx.schemeConfig.fluid.restDensity

    d = {
        'maxVelocity': torch.linalg.norm(vel, dim=-1).max().item(),
        'kineticEnergy': (0.5 * particles.masses[fluid] * (vel ** 2).sum(-1)).sum().item(),
        'maxDensity': rho.max().item(),
        'minDensity': rho.min().item(),
        'rollAngleDeg': math.degrees(ctx.scratch.get('rollAngle', 0.0)),
    }

    if ctx.scratch.get('sensorIndex') is None:
        ctx.scratch['sensorIndex'] = _locateSensor(ctx, particles)
    idx = ctx.scratch['sensorIndex']

    sensorRho = float(particles.densities[idx])
    ratio = sensorRho / rho0
    d['sensorRho'] = sensorRho
    d['sensorDensityRatio'] = ratio

    if ctx.scheme is IncompressibleSPHScheme.divergenceFree:
        # DFSPH now persists both projection pressures (schemes/dfsph.py):
        #   pressures   -> constant-density / particle-shift solve pressure
        #   soundspeeds -> divergence-free projection pressure
        # Both are populated on fluid rows only, so the wall sensor particle
        # reads 0 -- take the sensor value from a fluid-particle probe instead.
        pCD = getattr(particles, 'pressures', None)
        pDF = getattr(particles, 'soundspeeds', None)
        probe = _probePressure(ctx, particles, pCD * rho0Phys) if pCD is not None else None
        probeDF = _probePressure(ctx, particles, pDF * rho0Phys) if pDF is not None else None
        if probe is not None:
            d['sensorPressureCD'] = probe
        if probeDF is not None:
            d['sensorPressureDF'] = probeDF
        d['sensorPressure'] = probe if probe is not None else 0.0
    else:
        cs = float(ctx.schemeConfig.fluid.fixedSoundSpeed)
        scale = rho0Phys * cs * cs
        d['sensorPressureLinear'] = scale * (ratio - 1.0)
        d['sensorPressureTait'] = scale / _TAIT_GAMMA * (ratio ** _TAIT_GAMMA - 1.0)
        d['sensorPressure'] = d['sensorPressureTait']
        probe = _probePressure(
            ctx, particles,
            scale / _TAIT_GAMMA * ((particles.densities / rho0) ** _TAIT_GAMMA - 1.0))
    if probe is not None:
        d['sensorPressureProbe'] = probe
    return d


#: The two panels the run plots. Exported so the notebook can feed them
#: straight to `buildFieldPlotter` (the live-updating path in Jupyter).
SLOSHING_FIELDS = [
    Field('velocities', 'velocity magnitude', colorMap='viridis', mapping='L2Norm',
          boundary='Visualize'),
    Field('densities', 'density', colorMap='RdBu', colorMapKind='diverging',
          flip=True, midPoint=1.0, boundary='Visualize'),
]

setupPlot, updatePlot = particlePlot(SLOSHING_FIELDS, figsize=(13, 5))


def extraData(ctx: RunContext, state) -> Dict[str, Any]:
    return {k: v for k, v in ctx.spec.params.items()
            if not isinstance(v, (list, dict))}


sloshingTankCase = registerCase(Case(
    name='sloshingTank',
    scheme='deltaSPH',
    description='Laterally-excited sloshing tank, SPHERIC Test Case 10 (2D free '
                'surface); gravity rolled in the tank frame, wall Sensor 1 pressure.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    postStep=postStep,
    timestep=sloshingTimestep,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=extraData,
    defaults=dict(
        caseName='16-sloshingTank',
        dim=2,
        nx=150,
        L=0.9,                           # tank internal width B
        n_h=4.0,
        kernel='Wendland4',
        integrationScheme='rungeKutta2',
        supportMode='KernelMeanSymmetric',
        gradientMode='Difference',
        laplacianMode='Brookshaw',
        samplingScheme='regular',
        periodic=True,
        tLimit=7.0,
        dt=None,
        adaptiveDt=True,
        cflFactor=0.3,
        minDt=1e-8,
        maxDt=2e-3,
        storeMode='trajectory',
        exportInterval=0.01,
        plotInterval=50,
        storeInterval=500,
    ),
    params=dict(
        tankHeight=0.508,
        fillDepth=0.093,
        band=5,
        bandWidth=16.0,
        wallBC='freeSlip',
        gravityMagnitude=9.81,
        rho0Physical=1000.0,
        targetDt=2.0e-4,
        # roll excitation
        rollDataFile='',                 # '' -> the bundled lateral_water_1x.txt
        rollStartTime=0.0,
        # sensor 1: left wall at the still-water line
        sensorPos=[-0.45, 0.093],
        probeRadius=0.02,
        # dissipation
        inviscid=True,
        alpha=0.02,
        nu=1.0e-6,
        # divergenceFree only
        shifting=False,
        xsphScale=0.0,
        markerSize=4,
    ),
))


if __name__ == '__main__':
    caseMain(sloshingTankCase)
