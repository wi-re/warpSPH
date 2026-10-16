"""Dispatch layer selecting an adaptive-support scheme per ``compParams.adaptiveSupportScheme``.

Routes to the Owen (lookup-table) or Monaghan (Newton-iteration) solver, or
returns the particle state's current density/support unchanged for
``NoScheme``.
"""

from warpSPH.configurations.compute import neighborSearchOptions

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


def evaluateOptimalSupport(
        particleState: BaseState,
        config: SimulationConfig,
        compParams: CompressibleSPHConfig,
        supportScheme: SupportScheme = SupportScheme.Scatter,
        adjacency: Optional[AdjacencyList] = None,
        validateOnce: bool = False,
        rebuildSlack: float = 0.0,
        computeDensity: bool = True,
        grid = None,
        gridRadius: Optional[float] = None,
        info: Optional[dict] = None,
):
    """`validateOnce` / `rebuildSlack`: the Owen solver validates the neighbour list once per solve instead of once
    per iteration (see `evaluateOptimalSupportOwen`); the other support schemes ignore them. `grid` / `gridRadius` / `info` (Owen
    only): run on a fixed-shape hash-map grid and look at no neighbour list at all, see `evaluateOptimalSupportOwen`.
    `computeDensity=False` (Owen only) skips the closing summation density: the returned densities are then the particles' current ones --
    for a caller that computes its own density next. (The wall cases' support volume clamp is no longer applied here:
    `modules/compressibleWall/supportClamp.py`, called by the step after the list.)"""
    with record_function("[warpSPH] - evaluateOptimalSupport"):
        if compParams.adaptiveSupportScheme == AdaptiveSupportScheme.Owen:
            result = evaluateOptimalSupportOwen(
                particles = particleState,
                config = config,
                compConfig = compParams,
                kernel_ = config.kernel,
                adjacency = adjacency,
                supportScheme = supportScheme,
                verbose = False,
                validateOnce = validateOnce,
                rebuildSlack = rebuildSlack,
                computeDensity = computeDensity,
                grid = grid, gridRadius = gridRadius, info = info,
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
            if validateOnce:
                # the contract of `validateOnce`: the returned list is valid for the returned supports and the current
                # positions, whichever scheme produced the supports (here: the ones the particles already have)
                adjacency = buildVerletList(
                    particleState,
                    domain=config.domain,
                    verletScale=config.verletScale,
                    supportMode=SupportScheme.SuperSymmetric,
                    priorNeighborhood=adjacency,
                    verbose=False,
                    rebuildSlack=rebuildSlack,
                    **neighborSearchOptions(config))
            return particleState.densities, particleState.supports, adjacency, None, None
        else:
            raise ValueError(f"Unsupported adaptive support scheme: {compParams.adaptiveSupportScheme}")

        return tuple(result)
