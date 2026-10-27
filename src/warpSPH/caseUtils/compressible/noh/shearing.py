"""Shearing Noh implosion initial state, used by `warpSPH.cases.compressible.shearingNoh`.

Layers a transverse shear onto the Noh setup: the x-velocity is a sign step at
`x=0` (converging left/right halves, as in the plain Noh case) while the
y-velocity is a `cos(2*pi*x)` wave of amplitude `vs`, so the converging shock
is crossed by a periodic shear it is not aligned with. Density/pressure come
from `setupBasicCompressibleInitialState` unchanged.
"""

from ....sample.compressible import setupBasicCompressibleInitialState
import torch
import math

__all__ = ['sampleShearingNoh']


def sampleShearingNoh(vs, nx, config, schemeConfig, extraData, SimulationState, SimulationSystem):

    compressibleSystem = setupBasicCompressibleInitialState(nx, config, schemeConfig, SimulationState, SimulationSystem)

    v_initial_y = vs * torch.cos(2 * math.pi * compressibleSystem.state.positions[:,0])
    v_initial_x = torch.where(compressibleSystem.state.positions[:,0] < 0, 1.0, -1.0)

    v_initial = torch.stack([v_initial_x, v_initial_y], dim=1)

    compressibleSystem.state.velocities = v_initial

    return compressibleSystem