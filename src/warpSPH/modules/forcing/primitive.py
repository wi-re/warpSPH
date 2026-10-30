"""Fixed-mass body sources, retaining the native hydro and Butcher stages.

Compatible hydro cancels its own discrete pair work with the *total* half-kick
velocity. The uncancelled work is the body's half-kick work, not an arbitrary
RK quadrature of instantaneous power. Bath heat is additional internal energy.
"""
import torch


def applySources(system, update):
    source, s = system.sourceTerms, system.state
    from ...distributed import processGroup, validateSourceClock
    communication = processGroup(system)
    if communication is not None:
        validateSourceClock(system, communication)
    momentum, work, heat = source.fields(s.masses, s.positions, s.velocities,
                                         s.internalEnergies, system.t)
    if communication is not None:
        communication.assertReplicated('source:fields', dict(momentum=momentum, work=work, heat=heat))
    update.sourceAcceleration = momentum/s.masses[:, None]
    update.sourceHeat = heat/s.masses
    update.sourceEnergyRates = (torch.stack((work.double().sum(), heat.double().sum())) if communication is None else
        torch.stack((communication.sumOwned(work, system.adjacency, tag='source:work'),
                     communication.sumOwned(heat, system.adjacency, tag='source:heat'))))
    update.dvdt = update.dvdt + update.sourceAcceleration
    # Replace, don't mutate: CRK's retained native energy weights must continue
    # to see hydro dudt, not thermostat heating/cooling.
    update.dudt = update.dudt + update.sourceHeat
    update.dEdt = update.dEdt + work + heat
    return update


def finishSources(system, initial, updates, weights, dt, scheme, *, compatible=False):
    if system.sourceTerms is None:
        return
    if len(updates) != len(weights) or any(u.sourceEnergyRates is None for u in updates):
        raise RuntimeError('body sources require fresh rates for every accepted Butcher stage')
    s, s0 = system.state, initial.state
    exchange = sum(w*u.sourceEnergyRates for w, u in zip(weights, updates))
    if compatible:
        heat = sum(w*u.sourceHeat for w, u in zip(weights, updates))
        s.internalEnergies = s.internalEnergies + dt*heat
        # Match pairWork.energyIncrement's operation order for its half-kick.
        totalAcceleration = torch.zeros_like(s0.velocities)
        externalAcceleration = torch.zeros_like(s0.velocities)
        for w, u in zip(weights, updates):
            totalAcceleration += w*u.dvdt
            externalAcceleration += w*u.sourceAcceleration
        half = s0.velocities + .5*totalAcceleration*dt
        forceContributions = s0.masses.double()[:, None]*half.double()*externalAcceleration.double()
        from ...distributed import processGroup
        communication = processGroup(system)
        forceWork = (forceContributions.sum() if communication is None else
            communication.sumOwned(forceContributions, system.adjacency, tag='source:accepted-work'))
        del forceContributions
        exchange = torch.stack((forceWork, exchange[1]))
    s.sourceEnergyBudget = s0.sourceEnergyBudget + dt*exchange
    # Public thermodynamics describe accepted u, using the native density policy.
    from ..eos import refreshEOS
    refreshEOS(s, scheme)
    s.totalEnergies = s.masses*(s.internalEnergies+.5*s.velocities.square().sum(-1))
