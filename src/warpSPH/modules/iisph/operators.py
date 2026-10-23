"""IISPH kernel geometry, using the Core neighbor lists and precision."""

import torch
import warp as wp
from warpSPHCore.type_config import scalar_t
from .wp_operators import kernels


class IISPHPairs:
    """Kernel geometry over the Core's row-ordered live neighbor list."""

    def __init__(self, state, config, adjacency, reference=None):
        reference = state if reference is None else reference
        self.n = len(state.positions)
        self.device = state.positions.device
        self.kernels = kernels(state.positions.shape[1])
        self.j = adjacency.livePairs()[1].to(torch.int64).contiguous()
        self.starts = adjacency.edgeOffsets.to(torch.int64).contiguous()
        self.counts = adjacency.numNeighbors.to(torch.int64).contiguous()
        self.gradient = torch.empty((len(self.j), state.positions.shape[1]),
                                    dtype=state.positions.dtype, device=self.device)
        self.weights = torch.empty_like(self.gradient[:, 0]).contiguous()
        self.norm = 1.0
        if config.calibrateNormalization:
            from warpSPHCore.util.latticeDensity import latticeDensityFactor
            self.norm = latticeDensityFactor(config.kernel, float(config.n_h), state.positions.shape[1])
        self.geometryParameters = [self.kernels['V'](*(config.domain.max-config.domain.min).tolist()),
                                   self.kernels['I'](*[int(v) for v in config.domain.periodic.tolist()]),
                                   config.kernel.value, scalar_t(self.norm)]
        self.launch('geometry', [state.positions, reference.positions, state.supports,
                    adjacency.livePairs()[0].to(torch.int64).contiguous(), self.j,
                    *self.geometryParameters, self.gradient, self.weights], len(self.j))

    def launch(self, name, args, n=None):
        values = []
        for value in args:
            if isinstance(value, torch.Tensor):
                value = value.contiguous()
                dtype = self.kernels['V'] if value.ndim == 2 else None
                values.append(wp.from_torch(value, dtype=dtype))
            else:
                values.append(value)
        wp.launch(self.kernels[name], dim=self.n if n is None else n,
                  inputs=values, device=str(self.device))

    def base(self):
        return [self.j, self.starts, self.counts, self.gradient]
