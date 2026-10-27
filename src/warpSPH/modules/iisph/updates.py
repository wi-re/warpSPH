"""Normal derivative updates for the summation-density IISPH scheme."""

import torch
from ...systems.incompressible import IncompressibleSystemUpdate


def assembleIISPHUpdate(state, acceleration, pressureAcceleration):
    """Return velocity and acceleration rates; density is summed each evaluation."""
    return IncompressibleSystemUpdate(
        dxdt=state.velocities.clone(),
        dvdt=acceleration + pressureAcceleration,
        drhodt=torch.zeros_like(state.densities),
        passive=torch.zeros_like(state.densities, dtype=torch.bool))


def maskIISPHUpdate(update, kinds):
    """Keep static samples fixed after applying the update boundary conditions."""
    fluid = kinds == 0
    update.dxdt = torch.where(fluid[:, None], update.dxdt, 0)
    update.dvdt = torch.where(fluid[:, None], update.dvdt, 0)
    update.drhodt = torch.where(fluid, update.drhodt, 0)
    return update
