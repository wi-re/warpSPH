"""Translate existing driven fluid-band BCs to authoritative PESPH E/A.

Kidder and Rayleigh-Taylor prescribe rho/u/v in fluid buffer bands. Reuse
their dispatcher, then enforce the authoritative energy or coupled entropy
constraint. Prescribed u takes priority over the auxiliary density input: the
pressure-volume density remains an outcome of the discrete closure.
These are driven open-system bands, not a solid-wall pressure closure.
"""

import torch

from ...enumTypes import PESPHVariant
from ...math import getPeriodicPositions
from ..boundaryConditions import enforceDirichlet


def applyPESPHDirichlet(system, dt, config, scheme, pressure):
    conditions = [bc for bc in scheme.boundaryConditions if bc.dirichletFunctions]
    if not conditions:
        return False
    state = system.state
    thermal = torch.zeros_like(state.masses, dtype=torch.bool)
    velocity = thermal.clone()
    energy = thermal.clone()
    prescribedRho = pressure.thermodynamicDensity.clone()
    for bc in conditions:
        names = set(bc.dirichletFunctions)
        if names - {'velocities', 'densities', 'internalEnergies'}:
            raise ValueError('PESPH driven bands support prescribed velocity, density and internal energy only')
        mask = bc.sdf(getPeriodicPositions(state.positions, config.domain))[0] < 0
        if names & {'densities', 'internalEnergies'}:
            thermal |= mask
        if 'internalEnergies' in names:
            energy |= mask
        if 'velocities' in names:
            velocity |= mask
    enforceDirichlet(system, system.t, dt, config, scheme)
    for bc in conditions:
        if 'densities' in bc.dirichletFunctions:
            mask = bc.sdf(getPeriodicPositions(state.positions, config.domain))[0] < 0
            prescribedRho[mask] = state.densities[mask]
    if scheme.variant is PESPHVariant.PressureEnergy:
        target = state.masses*(state.internalEnergies + .5*state.velocities.square().sum(-1))
        state.totalEnergies = torch.where(thermal | velocity, target, state.totalEnergies)
    else:
        target = (scheme.gamma-1)*state.internalEnergies*prescribedRho.pow(1-scheme.gamma)
        # Preserve the historical density-only entropy prescription. An
        # explicit u prescription instead solves the nonlocal closure, without
        # changing entropy on unconstrained rows.
        state.entropies = torch.where(thermal & ~energy, target, state.entropies)
        from .thermalBoundary import matchBoundaryInternalEnergy
        matchBoundaryInternalEnergy(state, config, scheme, state.internalEnergies.clone(), energy, system.adjacency)
    return True
