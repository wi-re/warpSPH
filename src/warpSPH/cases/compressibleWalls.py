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

__all__ = ['WALL_PARAMS', 'buildWalledSystem', 'fluidEdges', 'nearWallRows', 'fluidEnergies']

WALL_PARAMS = dict(wallLayers=16, wallProbeLayers=3)


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
