"""Body sources in PESPH's authoritative energy/entropy variables.

Energy SPH integrates the existing m*v.a plus bath heat. Entropy SPH permits
adiabatic body acceleration only: A is unchanged and total energy is derived.
No local/nonlocal thermal entropy prescription is selected by this adapter.
"""
from ...enumTypes import PESPHVariant
from ..forcing.primitive import sourceFields
from ..meshless.sources import ConservedSources


def validatePESPHSource(system, scheme):
    source = system.sourceTerms
    if not isinstance(source, ConservedSources):
        raise NotImplementedError('PESPH body sources require ConservedSources.')
    budget = getattr(system.state, 'sourceEnergyBudget', None)
    if budget is None or budget.shape != (2,):
        raise ValueError('PESPH body sources require a two-component sourceEnergyBudget state field.')
    if scheme.variant is PESPHVariant.PressureEntropy and source.coolingTime is not None:
        raise NotImplementedError('Driven pressure-entropy PESPH requires adiabatic thermalModel; bath entropy allocation remains unqualified.')


def applyPESPHSource(system, update, scheme):
    source = system.sourceTerms
    if source is None:
        return update
    from .chunked import pesphInputs
    pesphInputs(system.adjacency, lambda: validatePESPHSource(system, scheme))
    momentum, work, heat, rates = sourceFields(system)
    update.dvdt = update.dvdt + momentum/system.state.masses[:, None]
    update.dEdt = update.dEdt + work + heat
    update.sourceEnergyRates = rates
    # Adiabatic entropy driving has no dA/dt contribution. Its existing hydro
    # and explicitly selected dissipative entropy rates remain authoritative.
    return update


def finishPESPHSource(system, initial, updates, weights, dt):
    if system.sourceTerms is None:
        return
    if len(updates) != len(weights) or any(u.sourceEnergyRates is None for u in updates):
        raise RuntimeError('PESPH body sources require fresh rates for every accepted Butcher stage')
    exchange = sum(w*u.sourceEnergyRates for w, u in zip(weights, updates))
    system.state.sourceEnergyBudget = initial.state.sourceEnergyBudget + dt*exchange
    # Derived thermodynamics have already been refreshed by the native PESPH
    # finalizer. Never replace E/A or apply a mass-density EOS here.
