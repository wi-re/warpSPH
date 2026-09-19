"""SPH particle density estimation.

Computes particle densities via plain SPH kernel summation -- the base
density estimator most schemes build on before layering grad-h or
gradient-renormalization corrections. The default is gather support because
it matches the Cullen-Dehnen switch's E.1 density estimate (per the CRK
paper); callers with a different historical density mode pass `supportMode`
explicitly (the Monaghan scheme passes `config.supportMode`, whose default
is SuperSymmetric). This is the single scheme-level density entry point --
a density correction that should apply to all schemes (the D&A eq.-18
self-term correction, `SimulationConfig.densityCorrection`, is wired here;
see the warpSPHCore replication's renorm_eps_design_note.md) hooks here
rather than at each scheme call site.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *




from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *

__all__ = ['computeDensities']


def computeDensities(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], supportMode: Optional[SupportScheme] = None) -> torch.Tensor:
    with record_function("[warpSPH] - computeDensities"):
        densities = warpOperation(
            currentState,
            OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Density,
                # default: gather -- cullen switch E.1 in the CRK paper uses
                # gather for density estimation; pass supportMode explicitly
                # for a different historical mode
                supportMode = supportMode if supportMode is not None else SupportScheme.Gather,
                # Lattice-normalisation correction (LATTICE_DENSITY_PLAN.md).
                # Off unless the config asks for it. The summation density is
                # the operator the offset is defined on, so this is where it is
                # wired first; the other operators still default to off.
                n_h = getattr(config, 'n_h', None),
                calibrateNormalization = getattr(config, 'calibrateNormalization', False),
            ),
            domain = config.domain,
            adjacency = adjacency,
        )
        # The D&A (2012) eq. 18/19 self-term correction: a per-particle
        # remap of the finished raw estimate (eps depends on N_H, which
        # depends on the raw density), applied after the operator, so the
        # warp kernel itself is unchanged. Off unless the config asks for
        # it; the bool form is the runner's shorthand, the
        # DensityCorrection dataclass carries the optional eps100/alpha
        # overrides. `getattr(dc, 'eps100', None)` is None for the bool
        # form, which selects the warpSPHCore-shipped constants.
        dc = getattr(config, 'densityCorrection', None)
        if isinstance(dc, bool) or getattr(dc, 'enabled', False):
            densities = applyDensityCorrection(
                densities,
                currentState.masses,
                currentState.supports,
                config.kernel,
                config.dim,
                eps100 = getattr(dc, 'eps100', None),
                alpha = getattr(dc, 'alpha', None),
            )
        return densities