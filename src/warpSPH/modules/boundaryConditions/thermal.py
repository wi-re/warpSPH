"""Accepted prescribed-u states and their discrete projection exchange.

These are driven fluid constraints, not an impermeable conducting-wall model.
The two budget entries count thermal and kinetic energy changed by the accepted
projection only. They do not include unresolved boundary forces, continuous
source work, or a physical heat flux. RHS applications never advance the budget.
"""
import torch


def hasThermalDirichlet(scheme):
    return any('internalEnergies' in (bc.dirichletFunctions or {})
               for bc in getattr(scheme, 'boundaryConditions', ()) or ())


def caloricEnergy(state):
    mass = state.masses.double()
    return torch.stack(((mass*state.internalEnergies.double()).sum(),
        .5*(mass*state.velocities.double().square().sum(-1)).sum()))


def caloricConserved(Q):
    """Thermal/kinetic energy from authoritative conserved fields."""
    q = Q.double()
    kinetic = .5*(q[:,1:-1].square().sum(-1)/q[:,0]).sum()
    return torch.stack((q[:,-1].sum()-kinetic, kinetic))


def finishThermalDirichlet(system, dt, config, scheme, *, account=True, pressure=None):
    """Enforce the accepted state after successful hydro/source finalization.

    Initial conditions call with account=False, defining the prescribed initial
    state rather than counting its construction as heat supplied during a run.
    Replace the budget tensor: a stage/replay must not mutate its parent's copy.
    """
    if not hasThermalDirichlet(scheme):
        return False
    from ...systems.pesph import PESPHState
    state = system.state
    if isinstance(state, PESPHState):
        from ..pesph.stage import refreshPESPHState
        from ..pesph.boundaries import applyPESPHDirichlet
        if pressure is None:
            pressure = refreshPESPHState(system, config, scheme, solveSupport=False)
        before = caloricEnergy(state)
        applyPESPHDirichlet(system, dt, config, scheme, pressure)
        refreshPESPHState(system, config, scheme, solveSupport=False)
    else:
        from .bcs import enforceDirichlet
        from ..eos import refreshEOS
        before = caloricEnergy(state)
        enforceDirichlet(system, system.t, dt, config, scheme)
        refreshEOS(state, scheme)
        state.totalEnergies = state.masses*(state.internalEnergies+.5*state.velocities.square().sum(-1))
    previous = state.boundaryProjectionBudget
    if account:
        exchange = caloricEnergy(state)-before
        state.boundaryProjectionBudget = (torch.zeros_like(exchange) if previous is None else previous)+exchange
    else:
        state.boundaryProjectionBudget = torch.zeros_like(before)
    return True
