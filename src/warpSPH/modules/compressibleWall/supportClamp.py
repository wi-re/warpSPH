"""The wall cases' support volume clamp, as a step of its own (it used to hide in `evaluateOptimalSupport`).

The adaptive-support relaxations are one-way ratchets: they grow h while a kernel is under-sampled but never shrink
a well-sampled one, so h can freeze over-large after a transient (the outermost rows of a shock reflection sit at
~1.8x their neighbour's h despite the same m/rho). Clamping each fluid row's h to the target spacing
``n_h_target * (m / rho)^(1/d)`` restores h ~ spacing and is a no-op for density-tracking h (OPEN_PROBLEMS §20).

It only ever REDUCES h, so a neighbour list built on the unclamped supports is valid for the clamped ones: call it
after the support solve and the list, before the density the step keeps. `SimulationConfig.supportVolumeClamp`:
``'walls'`` (default; only where the state has wall rows -- walls are experimental, no wall row, no clamp), ``'always'``
or ``'off'``.
"""

import torch

from warpSPHCore import OperationProperties, SupportScheme, warpOperation
from warpSPHCore.enumTypes import WarpOperation
from warpSPHCore.profiling import record_function

from ...utils.support import nH_to_n_h

__all__ = ['clampSupportsToVolume']


def clampSupportsToVolume(state, config, adjacency, wall, *, returnDensity: bool = False):
    """The clamped supports (``state.supports`` itself when the clamp does not apply); with ``returnDensity`` also the
    summation density at the CLAMPED supports, for a caller that wants to overwrite its density after the change.

    The bound uses the summation density at the supports the state has now (what the solver's closing density used to
    provide), summed over `adjacency`."""
    mode = getattr(config, 'supportVolumeClamp', 'walls')
    if mode not in ('always', 'walls', 'off'):
        raise ValueError(f"supportVolumeClamp must be 'always', 'walls' or 'off', got {mode!r}")
    applies = mode == 'always' or (mode == 'walls' and wall is not None)
    if not applies:
        return (state.supports, None) if returnDensity else state.supports
    with record_function("[warpSPH] - clampSupportsToVolume"):
        props = OperationProperties(kernel=config.kernel, operation=WarpOperation.Density, supportMode=SupportScheme.Gather)
        density = warpOperation(state, props, domain=config.domain, adjacency=adjacency)
        dim = config.domain.dim
        bound = nH_to_n_h(config.targetNeighbors, dim) * (state.masses / density.clamp_min(1e-12)) ** (1.0 / dim)
        clamped = torch.where(state.kinds == 0, torch.minimum(state.supports, bound), state.supports)
        if not returnDensity:
            return clamped
        original = state.supports
        state.supports = clamped
        try:
            return clamped, warpOperation(state, props, domain=config.domain, adjacency=adjacency)
        finally:
            state.supports = original
