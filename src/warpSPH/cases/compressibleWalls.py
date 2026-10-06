"""What the compressible-wall cases share: a 1D lattice with `wallLayers`
wall particles (`kinds == 1`) on each side, the state sampled from a function
of position, and the wall-side diagnostics (COMPRESSIBLE_WALLS_PLAN.md).
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch

from ..configurations import CompressibleSPHConfig
from ..enumTypes import AdaptiveSupportScheme
from ..modules import evaluateOptimalSupport, idealGasEOS
from ..modules.timestep.compressible import computeTimestep
from ..runner import RunContext
from ..utils.support import volumeToSupport
from warpSPHCore import SupportScheme

__all__ = ['WALL_PARAMS', 'buildWalledSystem', 'fluidEdges', 'nearWallRows', 'fluidEnergies',
           'boxSDF', 'circleSDF', 'walledDomainBounds', 'buildWalledBox']

#: `wallLatticeSupport`: 'auto' (the scheme's default), 'on' or 'off'; `wallSlip`:
#: 'freeSlip' or 'noSlip' -- see `CompressibleSPHConfig` and modules/compressibleWall
WALL_PARAMS = dict(wallLayers=16, wallProbeLayers=3, wallLatticeSupport='auto', wallSlip='freeSlip')


def buildWalledSystem(ctx: RunContext, stateAt: Callable[[torch.Tensor], Tuple[torch.Tensor, ...]],
                      wallVelocity: Tuple[float, float] = (0.0, 0.0), gaps: Tuple[int, int] = (0, 0)):
    """`stateAt(x) -> (rho, p, v)` for fluid positions `x` (shape (N,)); wall
    rows take rho and p at the nearest fluid position, with the same mass
    per spacing, so the fluid next to a wall sees a complete neighbourhood.

    The wall rows' velocity is `wallVelocity` (left, right), not the fluid's:
    it is the velocity they are prescribed to move with for the whole run
    (modules/compressibleWall), so a fixed wall must start at zero. `gaps`
    (left, right) leaves that many lattice cells empty outside each wall, room
    for a wall that moves outwards (a withdrawing piston)."""
    config, schemeConfig = ctx.config, ctx.schemeConfig
    nx, nw = ctx.spec.nx, ctx.param('wallLayers')
    dx = ctx.spec.L / nx

    x = (torch.arange(gaps[0], nx - gaps[1], device=config.device, dtype=config.dtype) + 0.5) * dx - 0.5 * ctx.spec.L
    nx = x.numel()
    kinds = torch.zeros(nx, dtype=torch.int32, device=config.device)
    kinds[:nw] = 1
    kinds[nx - nw:] = 1
    rho, p, v = stateAt(x.clamp(x[nw], x[nx - nw - 1]))
    v = v.clone()
    v[:nw] = wallVelocity[0]
    v[nx - nw:] = wallVelocity[1]
    normals = torch.zeros(nx, 1, device=config.device, dtype=config.dtype)
    normals[:nw] = -1.0
    normals[nx - nw:] = 1.0

    pos = x.unsqueeze(-1)
    A, u, P, c = idealGasEOS(A=None, u=None, P=p, rho=rho, gamma=schemeConfig.gamma)
    ones = torch.ones_like(x)
    masses = rho * dx
    state = ctx.SimulationState(
        positions=pos, velocities=v.unsqueeze(-1).clone(),
        supports=ones * volumeToSupport(dx, config.targetNeighbors, 1),
        masses=masses, densities=rho.clone(),
        kinds=kinds, materials=torch.zeros_like(kinds),
        UIDs=torch.arange(nx, device=config.device, dtype=torch.int32), UIDcounter=nx,
        internalEnergies=u, totalEnergies=(u + 0.5 * v ** 2) * masses,
        entropies=A, pressures=P, soundspeeds=c,
        alphas=ones.clone(), alpha0s=ones.clone(), divergence=torch.zeros_like(x),
        wallNormals=normals,
    )

    adapt = CompressibleSPHConfig(adaptiveSupportIterations=16, adaptiveSupportThreshold=1e-3,
                                  adaptiveSupportScheme=AdaptiveSupportScheme.NoScheme)
    rhoOpt, h, *_ = evaluateOptimalSupport(state, config, supportScheme=SupportScheme.Gather, compParams=adapt)
    state.supports, state.densities = h, rhoOpt

    system = ctx.SimulationSystem(state=state, adjacency=None, domain=config.domain)
    config.dx = dx
    config.dt = computeTimestep(system, config, schemeConfig, dt=config.dt)
    return system


def fluidEdges(ctx: RunContext, s) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left and right fluid extents (positions of the outermost fluid rows)."""
    xf = s.positions[s.kinds == 0, 0]
    return xf.min(), xf.max()


def nearWallRows(ctx: RunContext, s) -> torch.Tensor:
    """Wall rows within `wallProbeLayers` spacings of the fluid; deeper ones see no fluid."""
    lo, hi = fluidEdges(ctx, s)
    reach = ctx.param('wallProbeLayers') * ctx.spec.L / ctx.spec.nx
    x = s.positions[:, 0]
    return (s.kinds != 0) & (x >= lo - reach) & (x <= hi + reach)


def fluidEnergies(s):
    """Kinetic, thermal and total energy of the fluid rows only (walls hold no energy)."""
    fluid = s.kinds == 0
    m, v = s.masses[fluid], s.velocities[fluid]
    kinetic = 0.5 * (m * (v ** 2).sum(-1)).sum()
    thermal = (m * s.internalEnergies[fluid]).sum()
    return dict(kineticEnergy=kinetic, thermalEnergy=thermal, totalEnergy=kinetic + thermal)


def boxSDF(x: torch.Tensor, lo, hi) -> torch.Tensor:
    """Signed distance to the box [lo, hi] (negative inside), x: (N, d)."""
    lo = torch.as_tensor(lo, dtype=x.dtype, device=x.device)
    hi = torch.as_tensor(hi, dtype=x.dtype, device=x.device)
    q = torch.maximum(lo - x, x - hi)
    outside = torch.linalg.norm(q.clamp_min(0.0), dim=-1)
    inside = q.max(dim=-1).values.clamp_max(0.0)
    return outside + inside


def circleSDF(x: torch.Tensor, centre, radius: float) -> torch.Tensor:
    c = torch.as_tensor(centre, dtype=x.dtype, device=x.device)
    return torch.linalg.norm(x - c, dim=-1) - radius


def walledDomainBounds(ctx: RunContext, lo, hi, extendHi=None):
    """The run's domain for a walled box [lo, hi]: the box padded by the wall layers
    plus one cell (and reaching `extendHi`, per axis, if given: room for solid rows
    kept beyond the box). Call from `configureScheme`; sets `ctx.config.domain`."""
    dx = (hi[0] - lo[0]) / ctx.spec.nx
    pad = (ctx.param('wallLayers') + 1) * dx
    top = hi if extendHi is None else [max(a, b) for a, b in zip(hi, extendHi)]
    domain = ctx.config.domain
    domain.min = torch.tensor([v - pad for v in lo], dtype=domain.min.dtype, device=domain.min.device)
    domain.max = torch.tensor([v + pad for v in top], dtype=domain.max.dtype, device=domain.max.device)
    return dx


def buildWalledBox(ctx: RunContext, lo, hi, stateAt, solidSDF=None, wallVelocity=None,
                   extendHi=None, keepSolid=None):
    """A cell-centred lattice over the box [lo, hi] (any dim; `nx` cells along x,
    the same spacing on the other axes) and `wallLayers` cells beyond it.

    Rows inside the box and outside `solidSDF(x) < 0` are fluid; rows outside the
    fluid region but within `wallLayers` spacings of it are walls (`kinds == 1`);
    the rest are dropped (a solid body's interior beyond the wall depth). The
    fluid region's signed distance is `max(box, -solid)`.

    `stateAt(x) -> (rho, p, v)` gives every row's state (x: (N, d), v: (N, d));
    walls take rho and p from it at their own position, and their velocity from
    `wallVelocity(x) -> (N, d)` (zero by default): the velocity they move with for
    the whole run.

    A body that moves *into* the box needs rows that start outside it:
    `keepSolid(x) -> bool` keeps those lattice points too (as walls, whatever
    their depth), on a lattice reaching `extendHi`; their normals come from
    the body alone (`-grad solidSDF`), since the box's walls are not theirs."""
    config, schemeConfig = ctx.config, ctx.schemeConfig
    dim, nw = len(lo), ctx.param('wallLayers')
    dx = (hi[0] - lo[0]) / ctx.spec.nx
    axes = []
    top = hi if extendHi is None else [max(a, b) for a, b in zip(hi, extendHi)]
    for k in range(dim):
        n = int(round((top[k] - lo[k]) / dx))
        axes.append((torch.arange(-nw, n + nw, device=config.device, dtype=config.dtype) + 0.5) * dx + lo[k])
    x = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(-1, dim)
    def fluidSDF(y):
        d = boxSDF(y, lo, hi)
        return d if solidSDF is None else torch.maximum(d, -solidSDF(y))

    d = fluidSDF(x)
    kept = keepSolid(x) if keepSolid is not None else torch.zeros_like(d, dtype=torch.bool)
    keep = (d < nw * dx) | kept
    x, d, kept = x[keep], d[keep], kept[keep]
    wall = (d > 0) | kept
    kinds = wall.to(torch.int32)
    # wall normals (fluid -> wall) from the geometry: the SDF's central-difference gradient
    # (of the body alone for the kept body rows)
    eps = 1e-3 * dx

    def gradient(f):
        return torch.stack([(f(x + eps * e) - f(x - eps * e)) / (2 * eps)
                            for e in torch.eye(dim, device=x.device, dtype=x.dtype)], dim=-1)

    grad = gradient(fluidSDF)
    if keepSolid is not None:
        grad = torch.where(kept.unsqueeze(-1), -gradient(solidSDF), grad)
    normals = grad / grad.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    normals = torch.where(wall.unsqueeze(-1) & (grad.norm(dim=-1, keepdim=True) > 0.5), normals,
                          torch.zeros_like(normals))

    rho, p, v = stateAt(x)
    v = v.clone()
    if wallVelocity is not None:
        v[wall] = wallVelocity(x[wall]).to(v.dtype)
    else:
        v[wall] = 0.0
    n = x.shape[0]
    A, u, P, c = idealGasEOS(A=None, u=None, P=p, rho=rho, gamma=schemeConfig.gamma)
    ones = torch.ones(n, device=config.device, dtype=config.dtype)
    masses = rho * dx ** dim
    state = ctx.SimulationState(
        positions=x, velocities=v,
        supports=ones * volumeToSupport(dx ** dim, config.targetNeighbors, dim),
        masses=masses, densities=rho.clone(),
        kinds=kinds, materials=torch.zeros_like(kinds),
        UIDs=torch.arange(n, device=config.device, dtype=torch.int32), UIDcounter=n,
        internalEnergies=u, totalEnergies=(u + 0.5 * (v ** 2).sum(-1)) * masses,
        entropies=A, pressures=P, soundspeeds=c,
        alphas=ones.clone(), alpha0s=ones.clone(), divergence=torch.zeros(n, device=config.device, dtype=config.dtype),
        wallNormals=normals,
    )
    adapt = CompressibleSPHConfig(adaptiveSupportIterations=16, adaptiveSupportThreshold=1e-3,
                                  adaptiveSupportScheme=AdaptiveSupportScheme.NoScheme)
    rhoOpt, h, *_ = evaluateOptimalSupport(state, config, supportScheme=SupportScheme.Gather, compParams=adapt)
    state.supports, state.densities = h, rhoOpt

    system = ctx.SimulationSystem(state=state, adjacency=None, domain=config.domain)
    config.dx = dx
    config.dt = computeTimestep(system, config, schemeConfig, dt=config.dt)
    return system
