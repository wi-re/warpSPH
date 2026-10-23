"""Thermodynamic state contracts for a shared PESPH implementation.

Only E or A is integrated. Internal energy and the other thermodynamic
variable are derived. The scheme bundle selects the active state after
configuration, so the integrator advances exactly one thermal variable.
"""

from dataclasses import dataclass

import torch
from warpSPHIntegrators.state import BaseIntegrationSystem, constant, integrated, reference_state, tagged
from warpSPHIntegrators.state.fields import update_component

from .compressibleMonaghan import CompressibleState, CompressibleSystem
from ..enumTypes import PESPHVariant


@dataclass
class PESPHState(CompressibleState):
    internalEnergies: torch.Tensor = constant(tags=('internalEnergy',))
    totalEnergies: torch.Tensor = constant(tags=('energy',), default=None)
    thermodynamicVariable: str = constant(default='energy')
    previousTimestep: float = constant(default=None)


@dataclass
class PressureEnergyState(PESPHState):
    totalEnergies: torch.Tensor = integrated('dEdt', tags=('energy',), default=None)


@dataclass
class PressureEntropyState(PESPHState):
    entropies: torch.Tensor = integrated('dAdt', tags=('entropy',), default=None)
    thermodynamicVariable: str = constant(default='entropy')


def pesphStateType(variant):
    if variant is PESPHVariant.PressureEnergy:
        return PressureEnergyState
    if variant is PESPHVariant.PressureEntropy:
        return PressureEntropyState
    raise ValueError(f'Unknown PESPH variant {variant!r}')


@dataclass
class PESPHSystemUpdate:
    dxdt: torch.Tensor = tagged(tags=('position_derivative',))
    dvdt: torch.Tensor = tagged(tags=('velocity_derivative',))
    dEdt: torch.Tensor = tagged(tags=('totalEnergy_derivative',), default=None)
    dAdt: torch.Tensor = tagged(tags=('entropy_derivative',), default=None)


@dataclass
class PESPHSystem(CompressibleSystem):
    """Reuse position/velocity updates, selecting only the active thermal field."""
    state: PESPHState = reference_state(tags=('physics_state',))

    def initializeNewState(self, *args, **kwargs):
        return type(self)(state=self.state.initializeNewState(), adjacency=self.adjacency,
                          domain=self.domain, t=self.t)

    def apply_quantity_update(self, update, spec, **kwargs):
        if isinstance(self.state, PressureEntropyState):
            return update_component(self, update, spec, 'entropy', 'entropy_derivative')
        return update_component(self, update, spec, 'energy', 'totalEnergy_derivative')

    def finalize(self, initialState, dt, returnValues, updateValues, weights=(), *args, **kwargs):
        config = kwargs.get('config', args[0] if args else None)
        schemeConfig = kwargs.get('schemeConfig', args[1] if len(args) > 1 else None)
        if config is None or schemeConfig is None:
            raise ValueError('PESPH finalize requires config and schemeConfig')
        self.adjacency = returnValues[-1][0]
        last = returnValues[-1][1]
        self.state.supports = last.supports.clone()
        for name in ('alphas', 'alpha0s', 'divergence'):
            value = getattr(last, name)
            setattr(self.state, name, None if value is None else value.clone())
        from ..modules.pesph.stage import refreshPESPHState
        pressure = refreshPESPHState(self, config, schemeConfig)
        from ..modules.pesph.boundaries import applyPESPHDirichlet
        if applyPESPHDirichlet(self, dt, config, schemeConfig, pressure):
            refreshPESPHState(self, config, schemeConfig, solveSupport=False)
        if schemeConfig.dissipation:
            from ..modules.shockCapturing.wrapper import advanceViscositySwitchStep
            advanceViscositySwitchStep(self, initialState, returnValues, schemeConfig, config)
        self.state.previousTimestep = dt
        # Bypass the Monaghan finalizer: it overwrites integrated A and applies
        # a mass-density EOS. Derived values above use the accepted q/E/A.
        return BaseIntegrationSystem.finalize(self, initialState, dt, returnValues,
                                               updateValues, weights, *args, **kwargs)
