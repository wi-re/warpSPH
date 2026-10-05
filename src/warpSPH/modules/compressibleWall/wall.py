import torch

from warpSPHCore import (OperationDirection, OperationProperties, SupportScheme,
                         WarpOperation, warpOperation)

from ..mdbc._util import stateHasBoundaryParticles

__all__ = ['applyCompressibleWall', 'zeroWallUpdate']

_EPS = 1e-12


def _wallVolumes(currentState, config) -> torch.Tensor:
    """`m / rho` of every row at the first call (the initial state), cached on `config`."""
    volumes = getattr(config, '_wallVolumes', None)
    if volumes is None:
        volumes = currentState.masses / currentState.densities
        config._wallVolumes = volumes
    return volumes


def applyCompressibleWall(currentState, config, adjacency, gamma=None, wallVelocity=None) -> bool:
    """Overwrite the wall rows' density, internal energy and velocity in place.

    The wall state is the Shepard gather of the fluid rows, then (given `gamma`)
    the state behind the shock that stops the fluid's approach speed `u_n`
    against the wall: with the gathered state (rho, p) it satisfies
    `u_n = (p* - p) sqrt(2 / ((g+1) rho) / (p* + (g-1)/(g+1) p))`, and `u_n = 0`
    leaves the gathered state unchanged. The wall normal is the direction from
    the fluid centroid to the wall row, so no geometry is needed; the centroid
    gather ignores periodic wrapping. Wall rows also get the fluid's smoothing
    length and a mass `rho_wall * V` (V fixed at the first call), so the caller
    must redo the fluid density sum after this and apply it again. Returns
    whether there were wall rows. `wallVelocity` is a (dim,) tensor or sequence,
    zero by default (fixed wall).
    """
    if not stateHasBoundaryParticles(currentState, config):
        return False
    wall = currentState.kinds == 1
    volumes = _wallVolumes(currentState, config)
    props = OperationProperties(
        kernel=config.kernel,
        operation=WarpOperation.Interpolate,
        supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.FluidToBoundary,
    )

    def gather(values):
        return warpOperation(currentState, props, domain=config.domain,
                             adjacency=adjacency, queryValues=values)

    v = currentState.velocities
    if wallVelocity is None:
        target = torch.zeros_like(v)
    else:
        target = torch.as_tensor(wallVelocity, dtype=v.dtype, device=v.device).expand_as(v)

    shepard = gather(torch.ones_like(currentState.densities))
    ok = wall & (shepard > _EPS)
    safe = torch.where(ok, shepard, torch.ones_like(shepard))
    rho = gather(currentState.densities) / safe
    u = gather(currentState.internalEnergies) / safe

    if gamma is not None:
        p = (gamma - 1.0) * rho * u
        centroid = gather(currentState.positions) / safe.unsqueeze(-1)
        normal = currentState.positions - centroid
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        approach = ((gather(v) / safe.unsqueeze(-1) - target) * normal).sum(-1).clamp_min(0.0)
        a = 2.0 / ((gamma + 1.0) * rho.clamp_min(_EPS))
        b = (gamma - 1.0) / (gamma + 1.0) * p
        un2 = approach * approach
        jump = (un2 + torch.sqrt(un2 * un2 + 4.0 * a * un2 * (p + b))) / (2.0 * a)
        pStar = p + jump
        pSafe = p.clamp_min(_EPS)
        rho = rho * ((gamma + 1.0) * pStar + (gamma - 1.0) * pSafe) / ((gamma - 1.0) * pStar + (gamma + 1.0) * pSafe)
        u = pStar / ((gamma - 1.0) * rho.clamp_min(_EPS))

    currentState.densities = torch.where(ok, rho, currentState.densities)
    currentState.internalEnergies = torch.where(ok, u, currentState.internalEnergies)

    # wall mass follows the wall density at a fixed volume, so the fluid's density
    # sum sees a wall as compressed as the gas it holds (a fixed mass reads as rest density)
    currentState.masses = torch.where(ok, rho * volumes, currentState.masses)

    # wall rows take the fluid's smoothing length: the support solve gives the
    # outer ones a huge h (no fluid to count), which then reaches far into the fluid
    h = currentState.supports
    hFluid = gather(h) / safe
    hDefault = h[currentState.kinds == 0].median()
    currentState.supports = torch.where(wall, torch.where(ok, hFluid, hDefault), h)

    currentState.velocities = torch.where(wall.unsqueeze(-1), target, v)
    return True


def zeroWallUpdate(update, currentState) -> None:
    """Zero every rate of the non-fluid rows: the wall moves only by prescription."""
    solid = currentState.kinds != 0
    column = solid.unsqueeze(-1)
    for name in ('dxdt', 'dvdt'):
        field = getattr(update, name, None)
        if field is not None:
            setattr(update, name, torch.where(column, torch.zeros_like(field), field))
    for name in ('dudt', 'dEdt', 'drhodt'):
        field = getattr(update, name, None)
        if field is not None:
            setattr(update, name, torch.where(solid, torch.zeros_like(field), field))
