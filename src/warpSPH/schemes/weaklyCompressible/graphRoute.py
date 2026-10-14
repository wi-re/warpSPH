"""Whether (and how) the delta-SPH right-hand side is replayed from a CUDA graph.

With `schemeConfig.cudaGraph` the right-hand side (everything after the adjacency update) is captured once per Verlet-list generation and
replayed (`utils/cudaGraph.py:GraphedStateFunction`): bitwise the same result without the per-op host cost that dominates small problems.
`rhsIsGraphable` is the policy: a graph bakes in every Python scalar it saw at capture, so anything that depends on `t` / `dt` or keeps
Python-side per-step state must run eagerly.
"""

__all__ = ['rhsIsGraphable', 'graphedRHS', 'liveConfigKey']


def rhsIsGraphable(schemeConfig, stageIndex) -> bool:
    """Whether `_deltaSPH_rhs` is a pure function of the state here, i.e.
    safe to capture: a graph bakes in every Python scalar it saw at capture,
    so anything depending on `t`/`dt` (Dirichlet/forcing/update BC hooks,
    the 'derivative' no-penetration shift's `/ config.dt`) must run eagerly,
    as must frozen diffusion under a stage-indexed integrator: its later
    stages read the tensors stage 0 left in `schemeConfig._frozenDiffusionCache`,
    Python-side state a replay does not refresh. (`stageIndex is None`, e.g.
    symplectic Euler, recomputes the diffusion every call and never reads it.)"""
    if getattr(schemeConfig, 'mdbcNoPenShiftMode', 'derivative') == 'derivative':
        return False
    # analytic walls: capturable when the support is the host constant fixed at initialisation (no device-to-host read inside the capture)
    if getattr(schemeConfig, 'boundaryProvider', None) is not None:
        if not getattr(schemeConfig, '_analyticSupport', None):
            return False
        # a mixed scene's particle bodies are moved by `updateBodyParticlesWCSPH` (host reads), not capturable
        bodies = getattr(schemeConfig, 'rigidBodies', None) or []
        if any(getattr(rb, 'representation', None) is None for rb in bodies):
            return False
        # a moving analytic body (prescribed or driven by the fluid) is advanced in place by every execution of the step, and the graph's validation executes it more than once: only static bodies
        # (evaluated once, at the first call)
        static = getattr(schemeConfig, '_analyticBodiesStatic', None)
        if static is None:
            # a prescribed spin of AXISYMMETRIC bodies about their own centres (`schemeConfig.analyticSpinAxisymmetric`, e.g. the inner cylinder of Taylor-Couette) changes no wall term between steps: its pose is irrelevant, only its angular velocity enters
            static = not any(rb.dynamic or (float(rb.angularVelocity) != 0.0 and not getattr(schemeConfig, 'analyticSpinAxisymmetric', False)) or float(rb.linearVelocity.abs().max()) != 0.0 for rb in bodies)
            schemeConfig._analyticBodiesStatic = static
        if not static:
            return False
    # the implicit shifting solves its linear system with host-synchronising Krylov solvers: a capture of the step fails and leaves the CUDA context in an error state
    sp = getattr(schemeConfig, 'shiftProperties', None)
    if sp is not None and sp.active and getattr(sp.scheme, 'name', None) in ('implicit', 'dynamic'):
        return False
    if getattr(schemeConfig, 'freezeDiffusionAcrossStages', False) and stageIndex is not None:
        return False
    # both keep Python-side per-step state (the stage-0 mask, flip counters)
    if getattr(schemeConfig, 'freezeSurfaceMaskAcrossStages', False) or \
            getattr(schemeConfig, 'surfaceMaskDiagnostics', False):
        return False
    for bc in getattr(schemeConfig, 'boundaryConditions', None) or []:
        if bc.dirichletFunctions or bc.forcingFunctions or bc.updateFunctions:
            return False
    return True


def graphedRHS(schemeConfig, rhs):
    g = getattr(schemeConfig, '_rhsGraph', None)
    if g is None:
        from ...utils.cudaGraph import GraphedStateFunction
        g = GraphedStateFunction(rhs, 'deltaSPH right-hand side')
        schemeConfig._rhsGraph = g
    return g


def liveConfigKey(schemeConfig):
    """What a captured right-hand side reads from the scheme config by ADDRESS: the directional gravity vector, if it is a tensor.

    A graph replays the memory it captured. A case that REPLACES such a tensor between steps (rather than updating it in place) would be silently
    replayed with the captured, stale (or reused) data; keying the capture on the tensor's address turns that into a recapture, which is slow but right.
    Cases that vary the gravity direction (sloshingTank) update the tensor in place and never recapture."""
    direction = getattr(getattr(schemeConfig, 'gravityConfig', None), 'direction', None)
    return direction.data_ptr() if hasattr(direction, 'data_ptr') else None
