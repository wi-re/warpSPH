from typing import Optional

import torch

from warpSPHCore import (OperationDirection, OperationProperties, SupportScheme,
                         WarpOperation, warpOperation)

from ..mdbc._util import stateHasBoundaryParticles
from ...utils.support import nH_to_n_h

__all__ = ['CompressibleWall', 'beginCompressibleWall', 'wallRiemannState']

_EPS = 1e-12


def _wallVolumes(currentState, config) -> torch.Tensor:
    """`m / rho` of every row at the first call (the initial state), cached on `config`.

    Re-derived when the row count changes (a new run reusing the config)."""
    volumes = getattr(config, '_wallVolumes', None)
    if volumes is None or volumes.shape != currentState.masses.shape:
        volumes = currentState.masses / currentState.densities
        config._wallVolumes = volumes
    return volumes


def wallRiemannState(rho, p, approach, gamma):
    """Star state (rho*, p*) of the Riemann problem between a gas (rho, p) moving
    at `approach` towards a wall and its mirror image: the gas the wall stops.

    The star velocity is the wall's, so the one-sided wave function satisfies
    `f(p*) = approach`. `approach > 0` is a shock,
    `f = (p* - p) sqrt(A / (p* + B))`, A = 2 / ((g+1) rho), B = (g-1)/(g+1) p,
    solved in closed form; `approach < 0` a rarefaction,
    `p* = p (1 + (g-1)/2 approach / c)^(2g/(g-1))`, isentropic, floored at
    vacuum; `approach = 0` returns (rho, p)."""
    g = gamma
    # a gas at zero pressure (Sedov's ambient) reads slightly negative after round-off;
    # the wall must not turn that into a NaN (sqrt of a negative in the shock branch)
    p = p.clamp_min(0.0)
    rhoSafe, pSafe = rho.clamp_min(_EPS), p.clamp_min(_EPS)

    un = approach.clamp_min(0.0)
    a = 2.0 / ((g + 1.0) * rhoSafe)
    b = (g - 1.0) / (g + 1.0) * p
    un2 = un * un
    jump = (un2 + torch.sqrt((un2 * un2 + 4.0 * a * un2 * (p + b)).clamp_min(0.0))) / (2.0 * a)
    pShock = p + jump
    rhoShock = rho * ((g + 1.0) * pShock + (g - 1.0) * pSafe) / ((g - 1.0) * pShock + (g + 1.0) * pSafe)

    c = torch.sqrt(g * pSafe / rhoSafe)
    base = (1.0 + 0.5 * (g - 1.0) * approach.clamp_max(0.0) / c).clamp_min(0.0)
    ratio = base ** (2.0 * g / (g - 1.0))
    pFan = p * ratio
    rhoFan = rho * ratio ** (1.0 / g)

    shock = approach >= 0.0
    return torch.where(shock, rhoShock, rhoFan), torch.where(shock, pShock, pFan)


class CompressibleWall:
    """One right-hand-side evaluation's view of the wall rows (`kinds == 1`).

    Made by `beginCompressibleWall` at the start of the evaluation, before
    anything overwrites the wall rows: it keeps their stored velocity as the
    prescribed wall velocity (`target`). Wall rows are never accelerated
    (`finishUpdate` zeroes their `dvdt`), so the stored velocity stays the one
    the case gave them, and their positions advance with it: a fixed wall has
    zero velocity, a piston its speed.

    `apply` overwrites the wall rows' density, internal energy, mass, support
    and velocity; the caller redoes the fluid density sum after it (the wall
    masses changed) and applies it once more.
    """

    def __init__(self, currentState, config):
        self.wall = currentState.kinds == 1
        self.solid = currentState.kinds != 0
        self.target = currentState.velocities.clone()
        self.volumes = _wallVolumes(currentState, config)

    def apply(self, currentState, config, schemeConfig, adjacency, latticeSupport: bool = False) -> None:
        # `latticeSupport` is the scheme's default; `schemeConfig.wallLatticeSupport` overrides it
        """The wall state is the Shepard gather of the fluid rows; with
        `schemeConfig.wallRiemannState` it is then replaced by the star state of
        the mirrored Riemann problem for the fluid's approach speed `u_n` against
        the wall (`wallRiemannState`). The wall normal is the state's
        `wallNormals` where the case gave one, else the direction from the fluid
        centroid to the wall row (no geometry needed, but diagonal wherever walls
        meet; the centroid gather ignores periodic wrapping). Wall rows take the fluid's smoothing
        length and a mass `rho_wall * V` (V fixed at the first call), so the
        fluid's density sum sees a wall as compressed as the gas it holds.

        Wall velocity (`schemeConfig.wallSlip`): 'freeSlip' keeps the prescribed
        normal component and takes the fluid's tangential one, so the pair terms
        see no tangential shear against the wall; 'noSlip' is the prescribed
        velocity. Identical in 1D."""
        wall = self.wall
        gamma = schemeConfig.gamma
        props = OperationProperties(
            kernel=config.kernel,
            operation=WarpOperation.Interpolate,
            supportMode=SupportScheme.Gather,
            operationMode=OperationDirection.FluidToBoundary,
        )

        def gather(values):
            return warpOperation(currentState, props, domain=config.domain,
                                 adjacency=adjacency, queryValues=values)

        shepard = gather(torch.ones_like(currentState.densities))
        ok = wall & (shepard > _EPS)
        safe = torch.where(ok, shepard, torch.ones_like(shepard))
        rho = gather(currentState.densities) / safe
        u = gather(currentState.internalEnergies) / safe
        vFluid = gather(currentState.velocities) / safe.unsqueeze(-1)

        centroid = gather(currentState.positions) / safe.unsqueeze(-1)
        normal = currentState.positions - centroid
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        given = getattr(currentState, 'wallNormals', None)
        if given is not None:
            # the case's geometry: where a wall meets another wall (a corner, a piston
            # sliding along a floor) the fluid fills only part of a wall row's
            # neighbourhood and the centroid direction is diagonal -- a floor row next
            # to a piston then reads the fluid streaming away as a rarefaction and
            # sucks it into the seam
            known = given.norm(dim=-1, keepdim=True) > 0.5
            normal = torch.where(known, given, normal)
        relative = vFluid - self.target
        approach = (relative * normal).sum(-1)

        if getattr(schemeConfig, 'wallRiemannState', True):
            p = (gamma - 1.0) * rho * u
            rho, pStar = wallRiemannState(rho, p, approach, gamma)
            u = pStar / ((gamma - 1.0) * rho.clamp_min(_EPS))

        currentState.densities = torch.where(ok, rho, currentState.densities)
        currentState.internalEnergies = torch.where(ok, u, currentState.internalEnergies)

        # wall mass follows the wall density at a fixed volume, so the fluid's density
        # sum sees a wall as compressed as the gas it holds (a fixed mass reads as rest density)
        currentState.masses = torch.where(ok, rho * self.volumes, currentState.masses)

        # wall rows take the fluid's smoothing length (the support solve gives the
        # outer ones a huge h -- no fluid to count -- which reaches far into the fluid).
        # `latticeSupport`: never less than their own lattice's. CRK needs it: a wall
        # row with h below its spacing under-samples its own neighbourhood, its moment
        # matrix goes singular and the run blows up at the first shock impact. It costs
        # the summation-density schemes their near-wall rows (Monaghan's outermost row
        # -9% -> -32% in p), so they leave it off. Rows no fluid reaches keep the
        # support solve's h, so they stay out of the fluid's reach.
        h = currentState.supports
        hWall = gather(h) / safe
        configured = getattr(schemeConfig, 'wallLatticeSupport', None)
        if configured is not None:
            latticeSupport = configured
        if latticeSupport:
            dim = currentState.positions.shape[-1]
            hWall = torch.maximum(hWall, nH_to_n_h(config.targetNeighbors, dim) * self.volumes ** (1.0 / dim))
        currentState.supports = torch.where(ok, hWall, h)

        velocity = self.target
        if getattr(schemeConfig, 'wallSlip', 'freeSlip') == 'freeSlip':
            tangential = relative - approach.unsqueeze(-1) * normal
            velocity = torch.where(ok.unsqueeze(-1), self.target + tangential, self.target)
        currentState.velocities = torch.where(wall.unsqueeze(-1), velocity, currentState.velocities)

    def balanceFractions(self, f_ij, adjacency, currentState):
        """Compatible-energy split for pairs with a wall row: the whole pair work
        goes to the fluid row. The wall's share would otherwise be lost (its state
        is overwritten every evaluation), so a fixed wall would pump energy; with
        it the fluid gains exactly the wall's power `F . v_wall`."""
        if f_ij is None:
            return f_ij
        i, j = adjacency.i.long(), adjacency.j.long()
        toWall = (currentState.kinds[i] == 0) & self.solid[j]
        return torch.where(toWall, torch.ones_like(f_ij), f_ij)

    def finishUpdate(self, update) -> None:
        """Wall rows move with their prescribed velocity and are never accelerated
        or heated: `dxdt = target`, every other rate zero."""
        column = self.solid.unsqueeze(-1)
        if getattr(update, 'dxdt', None) is not None:
            update.dxdt = torch.where(column, self.target, update.dxdt)
        if getattr(update, 'dvdt', None) is not None:
            update.dvdt = torch.where(column, torch.zeros_like(update.dvdt), update.dvdt)
        for name in ('dudt', 'dEdt', 'drhodt'):
            field = getattr(update, name, None)
            if field is not None:
                setattr(update, name, torch.where(self.solid, torch.zeros_like(field), field))


def beginCompressibleWall(currentState, config) -> Optional[CompressibleWall]:
    """A `CompressibleWall` for this evaluation, or None when there are no wall rows."""
    if not stateHasBoundaryParticles(currentState, config):
        return None
    return CompressibleWall(currentState, config)
