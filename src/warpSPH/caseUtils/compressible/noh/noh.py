"""Noh implosion initial state, used by `warpSPH.cases.compressible.noh`.

Sets every particle's velocity to a unit vector pointing at the origin
(`-normalize(positions)`), so uniform cold gas converges radially from rest
density/pressure; all other state comes from
`setupBasicCompressibleInitialState` unchanged.
"""

from ....sample.compressible import setupBasicCompressibleInitialState
import torch

__all__ = ['sampleNoh1D']


def sampleNoh1D(nx, config, schemeConfig, SimulationState, SimulationSystem):
    particleSystem = setupBasicCompressibleInitialState(nx, config, schemeConfig, SimulationState, SimulationSystem)
    v_initial = - torch.nn.functional.normalize(particleSystem.state.positions, dim = -1)
    particleSystem.state.velocities = v_initial
    return particleSystem