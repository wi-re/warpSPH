"""`ComputeConfig`: how a run uses the machine, in one place (`SimulationConfig.compute`).

Physics lives in the scheme config; *how it is computed* -- eager or replayed from a CUDA graph, how often an iterative solver reads its stopping
test back to the host -- is the same decision for every scheme, depends on the problem size and the hardware, and used to be spread over a case-spec
flag (`cudaGraph`), a per-scheme switch (`CompSPHConfig.stepMode`), a per-solver field (`convergenceCheckSchedule`) and function defaults
(`checkEvery`). They resolve here, with one rule for the precedence:

    an explicit scheme / spec / solver value  >  this config  >  the shipped default

so existing cases and tests keep working (`stepMode`, `CaseSpec.cudaGraph` and a solver's `convergenceCheckSchedule` default to "inherit").

* `execution`: ``'eager'`` (default), ``'graph'`` (replay the step, or the scheme's right-hand side, from a CUDA graph where the scheme supports it) or
  ``'auto'``: graph below a particle count (`GRAPH_MAX_PARTICLES`, per scheme and dimension; `graphMaxParticles` overrides), eager above. A graph
  request a run cannot honour (no CUDA, stored trajectories, host-sized boundary conditions, ...) runs eagerly, with a warning for an explicit one.
  The crossover is where the step stops being limited by the host: below it the replay is faster, above it the eager path is, because the
  graphed variants (fixed-shape grid support solve, deferred checks) cost kernel time.
* `pipelineOutputs`: with a whole-step graph, run step n's diagnostics / frame while step n+1 replays.
* `iterationChecks`: ``'adaptive'`` (default) or ``'every'`` -- when a relaxed-Jacobi style loop (`warpSPHCore.linalg.CheckSchedule`) reads its
  convergence test back: after every iteration, or at checkpoints learned from the previous solve of the same kind (a solve may run past its first
  converged iteration by up to the gap to the next checkpoint: more converged, never less).
* `krylovCheckEvery`: how many iterations a Krylov solver (`warpSPHCore.linalg.krylov`) runs between host reads of its "still running" flag; None
  takes `DEFAULT_KRYLOV_CHECK_EVERY`. The solver freezes its iterate from the iteration a stopping test fires, so the interval costs wasted work
  (at most interval - 1 iterations), never accuracy.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ..enumTypes import CompressibleSPHScheme, WeaklyCompressibleSPHScheme

__all__ = ['ComputeConfig', 'EXECUTION_MODES', 'CHECK_SCHEDULES', 'DEFAULT_KRYLOV_CHECK_EVERY', 'GRAPH_MAX_PARTICLES', 'graphParticleLimit',
           'requestedExecution', 'resolveExecution', 'resolveCheckSchedule', 'resolveKrylovCheckEvery']

EXECUTION_MODES = ('eager', 'graph', 'auto')
CHECK_SCHEDULES = ('every', 'adaptive')
DEFAULT_KRYLOV_CHECK_EVERY = 8

#: 'auto' replays the step from a graph below this many particles. ROUGH, CONSERVATIVE defaults, hardware dependent: not a result. Measured on one GPU
#: (RTX PRO 6000 Blackwell), the crossover being where the step stops being limited by the host.
#: CompSPH (`docs/compsph_step/README.md` 2.7): 1D tie at 160k; 2D tie at 5-7k (moved up since with the multi-lane kernels); 3D the graph step is
#: slower at every size measured (>= 3k: the grid solve visits 3^3 cells), so 'auto' never picks it. CRKSPH (2.9): 1D tie near 125k; 2D graph 13 % ahead at
#: 20k, 8 % behind at 30k; 3D never. Delta-SPH family (`sun2017DeltaSPH`, `dambreak`, 2D, 2026-10-14): graph 3.4x faster at 6k particles, 2.0x at
#: 16k, 1.5x at 42k, 1.1x at 92k, tie at 200k -> 100k. A slower GPU or a bigger host overhead moves them. Entries that are absent run eagerly.
GRAPH_MAX_PARTICLES: Dict[Tuple[Any, int], int] = {
    (CompressibleSPHScheme.CompSPH, 1): 100_000,
    (CompressibleSPHScheme.CompSPH, 2): 5_000,
    (CompressibleSPHScheme.CompSPH, 3): 0,
    (CompressibleSPHScheme.CRKSPH, 1): 100_000,
    (CompressibleSPHScheme.CRKSPH, 2): 20_000,
    (CompressibleSPHScheme.CRKSPH, 3): 0,
    (WeaklyCompressibleSPHScheme.deltaSPH, 2): 100_000,
    (WeaklyCompressibleSPHScheme.sun2017DeltaSPH, 2): 100_000,
}


@dataclass
class ComputeConfig:
    execution: str = field(default='eager', metadata={'description': "how a step runs: 'eager', 'graph' (CUDA-graph replay where the scheme supports it) or 'auto' (graph below graphMaxParticles / the per-scheme table, eager above)"})
    graphMaxParticles: Optional[int] = field(default=None, metadata={'description': "execution='auto': graph below this many particles (None = the table GRAPH_MAX_PARTICLES for the scheme and dimension)"})
    pipelineOutputs: bool = field(default=True, metadata={'description': "with a whole-step graph: run step n's diagnostics and frame while step n+1 replays"})
    iterationChecks: str = field(default='adaptive', metadata={'description': "relaxed-Jacobi style loops read their convergence test back after every iteration ('every') or at checkpoints learned from the previous solve ('adaptive')"})
    krylovCheckEvery: Optional[int] = field(default=None, metadata={'description': 'Krylov solvers read their still-running flag every this many iterations (None = DEFAULT_KRYLOV_CHECK_EVERY)'})

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if self.execution not in EXECUTION_MODES:
            raise ValueError(f"compute.execution must be one of {EXECUTION_MODES}, not {self.execution!r}")
        if self.iterationChecks not in CHECK_SCHEDULES:
            raise ValueError(f"compute.iterationChecks must be one of {CHECK_SCHEDULES}, not {self.iterationChecks!r}")
        if self.krylovCheckEvery is not None and int(self.krylovCheckEvery) < 1:
            raise ValueError(f'compute.krylovCheckEvery must be >= 1, not {self.krylovCheckEvery!r}')
        if self.graphMaxParticles is not None and int(self.graphMaxParticles) < 0:
            raise ValueError(f'compute.graphMaxParticles must be >= 0, not {self.graphMaxParticles!r}')


def _compute(config) -> ComputeConfig:
    return getattr(config, 'compute', None) or ComputeConfig()


def graphParticleLimit(config, scheme, dim: int, schemeConfig=None) -> int:
    """Particle count below which 'auto' picks the graph: the scheme config's `stepModeMaxParticles`, else `compute.graphMaxParticles`, else the table."""
    override = getattr(schemeConfig, 'stepModeMaxParticles', None) if schemeConfig is not None else None
    if override is None:
        override = _compute(config).graphMaxParticles
    if override is not None:
        return int(override)
    return GRAPH_MAX_PARTICLES.get((scheme, dim), 0)


def requestedExecution(config, schemeConfig=None, spec=None) -> str:
    """The execution mode asked for, before size and blockers: the scheme's `stepMode`, else the spec's legacy `cudaGraph` flag, else `compute.execution`."""
    mode = getattr(schemeConfig, 'stepMode', None) if schemeConfig is not None else None
    if mode is not None:
        if mode not in EXECUTION_MODES:
            raise ValueError(f"stepMode must be one of {EXECUTION_MODES}, not {mode!r}")
        return mode
    flag = getattr(spec, 'cudaGraph', None) if spec is not None else None
    if flag is not None:
        return 'graph' if flag else 'eager'
    return _compute(config).execution


def resolveExecution(config, scheme, schemeConfig, spec, nParticles: int, dim: int) -> str:
    """``'eager'`` or ``'graph'`` for this run, from the request and (for 'auto') the particle count. Whether the scheme *can* graph this
    configuration is the scheme's own check on top of this."""
    request = requestedExecution(config, schemeConfig, spec)
    if request == 'auto':
        return 'graph' if nParticles < graphParticleLimit(config, scheme, dim, schemeConfig) else 'eager'
    return request


def resolveCheckSchedule(solverCfg, config) -> str:
    """'every' or 'adaptive' for a relaxed-Jacobi style loop: the solver config's own `convergenceCheckSchedule` unless it is 'auto' / absent."""
    mode = getattr(solverCfg, 'convergenceCheckSchedule', None)
    if mode is None or mode == 'auto':
        return _compute(config).iterationChecks
    if mode not in CHECK_SCHEDULES:
        raise ValueError(f"convergenceCheckSchedule must be 'auto' or one of {CHECK_SCHEDULES}, not {mode!r}")
    return mode


def resolveKrylovCheckEvery(config) -> int:
    value = _compute(config).krylovCheckEvery
    return DEFAULT_KRYLOV_CHECK_EVERY if value is None else int(value)
