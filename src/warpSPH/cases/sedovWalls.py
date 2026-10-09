"""Sedov-Taylor blast in a box with solid walls (1D, 2D or 3D), compressible.

COMPRESSIBLE_WALLS_PLAN.md: the blast of `sedov`, but the gas fills
[-1, 1]^d and `wallLayers` rows of wall (`kinds == 1`) surround it, so the
shock reflects off the walls (and, in 2D/3D, the corners) instead of leaving.
`goalRadius` sets the end time as in `sedov` -- the time the free shock would
reach that radius -- so a value above 1 runs past the first wall impact.

The periodic `sedov` on [-1, 1]^d is the exact reference: its initial state is
even in every coordinate, so each periodic image is a mirror image and the
box faces act as reflecting walls. `scripts/compare_wallMirror.py --case sedov`
runs both and compares them.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..caseUtils import buildSedov
from ..runner import Case, RunContext, caseMain, registerCase
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, configureCompressible,
                           paramExtraData)
from .compressibleWalls import WALL_PARAMS
from .plotting import Field, particlePlot
from .sedov import goalTime, sedovSolution
from .sedov import setupPlot as sedovProfileSetup
from .sedov import updatePlot as sedovProfileUpdate

__all__ = ['sedovWallsCase']


def _extent(ctx: RunContext) -> float:
    """Domain edge length: the [-1, 1] box plus the wall layers on both sides."""
    return 2.0 * (ctx.spec.nx + 2 * ctx.param('wallLayers')) / ctx.spec.nx


def configureScheme(ctx: RunContext) -> None:
    if ctx.spec.nx % 2 == 0 and ctx.param('initialization') != 'quadrant':
        # the 'hat'/'singular' blast sits on the particle at the origin: odd nx
        ctx.spec = ctx.spec.merged(nx=ctx.spec.nx + 1)
    L = _extent(ctx)
    domain = ctx.config.domain
    domain.min = torch.full_like(domain.min, -0.5 * L)
    domain.max = torch.full_like(domain.max, 0.5 * L)
    configureCompressible(ctx)


def _appendWalls(state, nx: int, nw: int, dim: int):
    """Continue the reference's cell-centred lattice (x = -1 + (i + 1/2) dx) by
    `nw` cells beyond each face of [-1, 1]^d and append those points as wall
    rows, each a copy of an ambient fluid row (the corner one) apart from its
    position, kind and UID. Every per-row tensor field is extended."""
    dx = 2.0 / nx
    axis = (torch.arange(-nw, nx + nw, device=state.positions.device, dtype=state.positions.dtype) + 0.5) * dx - 1.0
    grid = torch.stack(torch.meshgrid(*([axis] * dim), indexing='ij'), dim=-1).reshape(-1, dim)
    walls = grid[(grid.abs() > 1.0).any(dim=-1)]
    n, m = state.positions.shape[0], walls.shape[0]
    template = int(torch.argmax(state.positions.abs().sum(-1)))
    for name, value in vars(state).items():
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == n:
            setattr(state, name, torch.cat([value, value[template:template + 1].expand(m, *value.shape[1:])]))
    state.positions = torch.cat([state.positions[:n], walls])
    state.kinds = torch.cat([state.kinds[:n], torch.ones(m, dtype=state.kinds.dtype, device=state.kinds.device)])
    # wall normals: the gradient of the box SDF (axis-aligned on a face, diagonal past a corner)
    q = walls.abs() - 1.0
    outside = q.clamp_min(0.0) * torch.sign(walls)
    normals = outside / outside.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    state.wallNormals = torch.cat([torch.zeros(n, dim, dtype=walls.dtype, device=walls.device), normals])
    state.UIDs = torch.arange(n + m, dtype=state.UIDs.dtype, device=state.UIDs.device)
    state.UIDcounter = n + m
    return state


def buildSystem(ctx: RunContext):
    """The periodic `sedov` initial state, exactly (same sampler, same blast),
    with the wall rows appended outside [-1, 1]^d."""
    solution = sedovSolution(ctx)
    ctx.scratch['solution'] = solution
    ctx.spec = ctx.spec.merged(tLimit=goalTime(ctx, solution))
    nx, nw = ctx.spec.nx, ctx.param('wallLayers')
    system = buildSedov(
        ctx.SimulationSystem, ctx.SimulationState,
        config=ctx.config,
        nx=nx, dim=ctx.spec.dim, domainExtent=2.0,
        periodicDomain=True,
        rho0=ctx.param('rho0'), E0=ctx.param('E0'),
        initialization=ctx.param('initialization'),
        gamma=ctx.param('gamma'), kernel=ctx.config.kernel,
        targetNeighbors=ctx.config.targetNeighbors,
        dtype=ctx.config.dtype, device=ctx.config.device)
    _appendWalls(system.state, nx, nw, ctx.spec.dim)
    ctx.config.dx = 2.0 / nx
    return system


SEDOV_WALL_FIELDS = [
    Field('densities', 'Density', colorMap='viridis', gridResolution=512, boundary='Hide'),
    Field('pressures', 'Pressure', colorMap='inferno', gridResolution=512, boundary='Hide'),
]
_fieldSetup, _fieldUpdate = particlePlot(SEDOV_WALL_FIELDS, figsize=(11, 5))


def setupPlot(ctx: RunContext, state):
    if ctx.spec.dim == 1:
        return sedovProfileSetup(ctx, state)
    return _fieldSetup(ctx, state)


def updatePlot(ctx: RunContext, state, plot, step: int) -> None:
    if ctx.spec.dim == 1:
        return sedovProfileUpdate(ctx, state, plot, step)
    return _fieldUpdate(ctx, state, plot, step)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    s = state.state
    fluid = s.kinds == 0
    out = compressibleDiagnostics(ctx, state)
    out['penetrating'] = (s.positions[fluid].abs() > 1.0).any(dim=-1).sum().item()
    out['maxVelocity'] = torch.linalg.norm(s.velocities[fluid], dim=-1).max().item()
    out['maxDensity'] = s.densities[fluid].max().item()
    out['minInternalEnergy'] = s.internalEnergies[fluid].min().item()
    return out


sedovWallsCase = registerCase(Case(
    name='sedovWalls',
    scheme='CRKSPH',
    description='Sedov-Taylor blast in a box with solid walls (1D, 2D or 3D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    extraFields=('internalEnergies', 'supports'),
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='sedovWalls',
        dim=2,
        nx=101,
        L=2.0,
        periodic=False,
        tLimit=1.0,
        plotInterval=10,
        storeInterval=500,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        E0=1.0,
        goalRadius=1.4,
        initialization='hat',
        **WALL_PARAMS,
    ),
))


if __name__ == '__main__':
    caseMain(sedovWallsCase)
