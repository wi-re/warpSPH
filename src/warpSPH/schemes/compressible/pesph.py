"""Shared pressure-volume RHS: dissipative energy SPH and adiabatic entropy SPH."""

import torch
from warpSPHCore.profiling import record_function

from ...enumTypes import PESPHVariant
from ...modules.pesph import computePressureForce
from ...modules.pesph.stage import refreshPESPHState
from ...systems.pesph import PESPHSystemUpdate
from ...modules.boundaryConditions import computeForcing, enforceUpdates
from ...modules.pesph.boundaries import applyPESPHDirichlet
from ...modules.dissipation.rates import computeDissipationRates


def pesphConservativeStep(system, dt, config, schemeConfig, verbose=False):
    if schemeConfig.boundaryConditions:
        raise ValueError('The conservative PESPH prototype does not implement boundary conditions or forcing')
    return _step(system, dt, config, schemeConfig, dissipative=False)


def pesphStep(system, dt, config, schemeConfig, verbose=False):
    return _step(system, dt, config, schemeConfig, dissipative=schemeConfig.dissipation)


def _step(system, dt, config, schemeConfig, *, dissipative):
    for bc in schemeConfig.boundaryConditions:
        if set(bc.updateFunctions or {}) - {'dxdt', 'dvdt'}:
            raise ValueError('PESPH driven bands support position/velocity derivative overrides only')
    with record_function('[PESPH] support and thermodynamics'):
        pressure = refreshPESPHState(system, config, schemeConfig)
        if applyPESPHDirichlet(system, dt, config, schemeConfig, pressure):
            pressure = refreshPESPHState(system, config, schemeConfig, solveSupport=False)
    with record_function('[PESPH] pressure pair'):
        acceleration, thermalWork = computePressureForce(system.state, pressure, config, system.adjacency)
    state = system.state
    heating = torch.zeros_like(thermalWork)
    if dissipative:
        with record_function('[PESPH] dissipation'):
            rates = computeDissipationRates(state, dt, config, schemeConfig, system.adjacency, acceleration)
        acceleration = acceleration + rates.acceleration
        heating = rates.heating
    with record_function('[PESPH] forcing and energy'):
        force = computeForcing(system, dt, system.t, config, schemeConfig)
        acceleration = acceleration + force/state.masses[:, None]
        # Include viscous heat, conductivity, switch heat AND external work.
        energyRate = state.masses * ((state.velocities*acceleration).sum(-1) + thermalWork + heating)
    entropyRate = None
    if schemeConfig.variant is PESPHVariant.PressureEntropy:
        from ...modules.pesph.entropyHeating import entropyHeatingRate
        entropyRate = entropyHeatingRate(state, pressure, heating, config, schemeConfig, system.adjacency)
    update = PESPHSystemUpdate(dxdt=state.velocities.clone(), dvdt=acceleration.clone(),
                              dEdt=energyRate, dAdt=entropyRate)
    enforceUpdates(update, system, dt, system.t, config, schemeConfig)
    update.dEdt = update.dEdt + state.masses*(state.velocities*(update.dvdt-acceleration)).sum(-1)
    return update, system.adjacency, state
