"""Dispatch layer selecting an adaptive-support scheme per ``compParams.adaptiveSupportScheme``.

Routes to the Owen (lookup-table) or Monaghan (Newton-iteration) solver, or
returns the particle state's current density/support unchanged for
``NoScheme``.
"""

from .optimalSupportOwen import evaluateOptimalSupportOwen
from .optimalSupportMonaghan import evaluateOptimalSupportMonaghan
from ...systems.baseState import BaseState
from ...configurations import SimulationConfig, CompressibleSPHConfig
import numpy as np
import torch
from warpSPHCore import *
from typing import Optional, Union
from ...enumTypes import AdaptiveSupportScheme
from ...utils.support import nH_to_n_h
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
__all__ = ['evaluateOptimalSupport']


def _clampSupportsToVolume(state, h, densities, config):
    """Upper-bound each fluid row's h at the target spacing ``n_h_target * (m/rho)^(1/d)``.

    The adaptive-support relaxations are one-way ratchets: they grow h while a
    kernel is under-sampled but never shrink a well-sampled one, so h can freeze
    over-large after a transient (the outermost rows of a shock reflection sit at
    ~1.8x their neighbour's h despite the same ``m/rho``). Clamping to the
    spacing restores ``h ~ spacing`` and is a no-op for density-tracking h, so it
    cannot block a particle from growing a large support where ``m/rho`` is large.
    Corrects a derived quantity on existing rows only -- no insert/remove/move.
    """
    dim = config.domain.dim
    n_h_target = nH_to_n_h(config.targetNeighbors, dim)
    volume = state.masses / densities.clamp_min(1e-12)
    bound = n_h_target * volume ** (1.0 / dim)
    fluid = state.kinds == 0
    return torch.where(fluid, torch.minimum(h, bound), h)


def _clampEnabled(state, config) -> bool:
    mode = getattr(config, 'supportVolumeClamp', 'walls')
    if mode == 'always':
        return True
    if mode == 'off':
        return False
    if mode == 'walls':
        # read from this state, not `stateHasBoundaryParticles`' per-config cache: case builders
        # solve the supports before the wall rows are appended
        return bool((state.kinds != 0).any())
    raise ValueError(f"supportVolumeClamp must be 'always', 'walls' or 'off', got {mode!r}")


def evaluateOptimalSupport(
        particleState: BaseState,
        config: SimulationConfig,
        compParams: CompressibleSPHConfig,
        supportScheme: SupportScheme = SupportScheme.Scatter,
        adjacency: Optional[AdjacencyList] = None,
):
    with record_function("[warpSPH] - evaluateOptimalSupport"):
        if compParams.adaptiveSupportScheme == AdaptiveSupportScheme.Owen:
            result = evaluateOptimalSupportOwen(
                particles = particleState,
                config = config,
                compConfig = compParams,
                kernel_ = config.kernel,
                adjacency = adjacency,
                supportScheme = supportScheme,
                verbose = False
            )
        elif compParams.adaptiveSupportScheme == AdaptiveSupportScheme.Monaghan:
            result = evaluateOptimalSupportMonaghan(
                particleState = particleState,
                config = config,
                compParams = compParams,
                supportScheme = supportScheme,
                adjacency = adjacency
            )
        elif compParams.adaptiveSupportScheme == AdaptiveSupportScheme.NumberDensity:
            result = evaluateOptimalSupportMonaghan(
                particleState = particleState,
                config = config,
                compParams = compParams,
                supportScheme = supportScheme,
                adjacency = adjacency,
                numberDensity = True
            )
        elif compParams.adaptiveSupportScheme == AdaptiveSupportScheme.NoScheme:

            # print('Adaptive support scheme set to NoneSupport, skipping optimal support evaluation')
            return particleState.densities, particleState.supports, adjacency, None, None
        else:
            raise ValueError(f"Unsupported adaptive support scheme: {compParams.adaptiveSupportScheme}")

        densities, h, adjacency, *rest = result
        if _clampEnabled(particleState, config):
            h = _clampSupportsToVolume(particleState, h, densities, config)
        return (densities, h, adjacency) + tuple(rest)