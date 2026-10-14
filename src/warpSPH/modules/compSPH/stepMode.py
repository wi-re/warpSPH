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

from ...enumTypes import CompressibleSPHScheme

__all__ = ['GRAPH_MAX_PARTICLES', 'STEP_MODES', 'resolveStepMode']

STEP_MODES = ('eager', 'graph', 'auto')

#: 'auto' replays the step from a graph below this many particles. ROUGH, CONSERVATIVE defaults, hardware dependent (the 2D limit has since moved up with the
#: multi-lane kernels; docs/compsph_step/README.md 2.8): not a result. First estimates from `sod` / `sod2d` / `sod3d` on one GPU (RTX PRO 6000
#: Blackwell, steps 100-1100, `docs/compsph_step/README.md` 2.7): the crossover is where the step stops being limited by the host.
#: 1D: tie at 160k; 2D: tie at 5-7k; 3D: the graph step is slower at every size measured (>= 3k) because the grid solve visits
#: 3^3 cells, so 'auto' never picks it. A slower GPU or a bigger host overhead moves them; `CompSPHConfig.stepModeMaxParticles` overrides.
GRAPH_MAX_PARTICLES: Dict[Tuple[CompressibleSPHScheme, int], int] = {
    (CompressibleSPHScheme.CompSPH, 1): 100_000,
    (CompressibleSPHScheme.CompSPH, 2): 5_000,
    (CompressibleSPHScheme.CompSPH, 3): 0,
    # CRKSPH (docs/compsph_step/README.md 2.9, sod / sod2d / sod3d, steps 40-340): 1D tie near 125k; 2D graph 13 % ahead at 20k, 8 % behind at 30k
    # (crossover ~25k); 3D slower with the graph at 10k, as for CompSPH. Rounded down.
    (CompressibleSPHScheme.CRKSPH, 1): 100_000,
    (CompressibleSPHScheme.CRKSPH, 2): 20_000,
    (CompressibleSPHScheme.CRKSPH, 3): 0,
}


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
    requested = getattr(schemeConfig, 'stepMode', 'eager')
    if requested not in STEP_MODES:
        raise ValueError(f"stepMode must be one of {STEP_MODES}, not {requested!r}")
    if requested == 'eager':
        return 'eager'
    if requested == 'auto':
        n, dim = runningState.state.positions.shape
        limit = getattr(schemeConfig, 'stepModeMaxParticles', None)
        if limit is None:
            limit = GRAPH_MAX_PARTICLES.get((ctx.scheme, dim), 0)
        if n >= limit:
            return 'eager'
    reason = _blocker(ctx, runningState, spec)
    if reason is not None:
        if requested == 'graph':
            warnings.warn(f"[warpSPH] stepMode='graph' runs eagerly: {reason}", stacklevel=2)
        return 'eager'
    schemeConfig.supportSolveOnGrid = True
    schemeConfig.inPlaceAdjacency = True
    return 'graph'
