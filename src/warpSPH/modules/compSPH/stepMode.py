"""One switch for how a CompSPH / CRKSPH step runs: `CompSPHConfig.stepMode`.

* ``'eager'`` (default): the support solve and the pair list work on the Verlet list, every check is read when it is made. Fewest
  moving parts, and the fastest once the step is limited by the GPU (large N, 3D).
* ``'graph'``: the whole step is replayed from one CUDA graph. This needs the pieces that make a step free of host reads, and the
  mode switches them on together: the support solve on the fixed-shape grid, the pair-list check as deferred device flags (one host
  read per replay, a failing flag repeats the step eagerly), the pair-list rebuild in place between replays. Faster when the step is
  limited by the host (small N), slower otherwise: the grid solve costs more kernel time than the list solve.
* ``'auto'``: ``'graph'`` below a particle count that depends on the scheme and the dimension (`GRAPH_MAX_PARTICLES`), else ``'eager'``.

A ``'graph'`` request that cannot be honoured (no CUDA, stored trajectories, solid wall rows, a viscosity switch with a step-boundary
host time, Dirichlet boundary conditions) runs ``'eager'``, with a warning for an explicit request.
"""

import warnings
from typing import Dict, Optional, Tuple

from ...configurations.compute import EXECUTION_MODES, GRAPH_MAX_PARTICLES, graphParticleLimit, requestedExecution, resolveExecution
from ...enumTypes import CompressibleSPHScheme

__all__ = ['GRAPH_MAX_PARTICLES', 'STEP_MODES', 'resolveStepMode']

STEP_MODES = EXECUTION_MODES
# The 'auto' particle limits (`GRAPH_MAX_PARTICLES`) now live with the rest of the compute settings in `configurations/compute.py`.

def _blocker(ctx, runningState, spec) -> Optional[str]:
    """Why this run cannot replay the step from a graph, or None."""
    from ..compressibleWall.wall import beginCompressibleWall
    from ..shockCapturing.wrapper import STEP_HOOKS
    if not runningState.state.positions.is_cuda:
        return 'the particles are not on a CUDA device'
    if getattr(spec, 'store', False):
        return 'trajectory / state storing reads per-stage data a replayed step does not rebuild'
    switch = getattr(ctx.schemeConfig, 'viscositySwitchParams', None)
    if switch is not None and STEP_HOOKS.get(switch.scheme) is not None:
        return 'the viscosity switch advances with a host step time'
    for bc in getattr(ctx.schemeConfig, 'boundaryConditions', None) or []:
        if getattr(bc, 'dirichletFunctions', None):
            return 'Dirichlet boundary conditions select rows with host-sized masks'
    if beginCompressibleWall(runningState.state, ctx.config) is not None:
        return 'solid wall rows (their pair arrays follow the live edge count)'
    return None


def resolveStepMode(ctx, runningState, spec) -> str:
    """``'eager'`` or ``'graph'`` for this run (see the module doc); in graph mode the switches it needs are turned on."""
    schemeConfig = ctx.schemeConfig
    if ctx.scheme not in (CompressibleSPHScheme.CompSPH, CompressibleSPHScheme.CRKSPH):
        return 'eager'
    requested = requestedExecution(ctx.config, schemeConfig, spec)         # the scheme's stepMode, else the spec's cudaGraph, else config.compute.execution
    n, dim = runningState.state.positions.shape
    if resolveExecution(ctx.config, ctx.scheme, schemeConfig, spec, n, dim) == 'eager':
        return 'eager'
    reason = _blocker(ctx, runningState, spec)
    if reason is not None:
        if requested == 'graph':
            warnings.warn(f"[warpSPH] stepMode='graph' runs eagerly: {reason}", stacklevel=2)
        return 'eager'
    schemeConfig.supportSolveOnGrid = True
    schemeConfig.inPlaceAdjacency = True
    return 'graph'
