"""Scatter-add reduction (`scatter_sum`) and its shape-broadcasting helper
(`broadcast`), vendored from PyTorch Geometric so warpSPH does not need it
as a dependency. Used to accumulate per-pair contributions back onto
per-particle tensors (e.g. `modules/compSPH/multistep.py`'s energy update).
"""

import torch
from typing import Optional, Union

__all__ = ['scatter_sum', 'segment_sum', 'broadcast']

# ------ Beginning of scatter functionality ------ #
# Scatter summation functionality based on pytorch geometric scatter functionality
# This is included here to make the code independent of pytorch geometric for portability
# Note that pytorch geometric is licensed under an MIT licenses for the PyG Team <team@pyg.org>
# @torch.jit.script
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

from warpSPHCore.profiling import record_function
# @torch.jit.script
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    with record_function("scatter_sum"):
        index = broadcast(index, src, dim)
        if out is None:
            size = list(src.size())
            if dim_size is not None:
                size[dim] = dim_size
            elif index.numel() == 0:
                size[dim] = 0
            else:
                size[dim] = int(index.max()) + 1
            out = torch.zeros(size, dtype=src.dtype, device=src.device)
            return out.scatter_add_(dim, index, src)
        else:
            return out.scatter_add_(dim, index, src)
# ------ End of scatter functionality ------ #

def segment_sum(src: torch.Tensor, numNeighbors: torch.Tensor) -> torch.Tensor:
    """Row sums of per-pair values over a row-sorted (CSR) adjacency list.

    Run-to-run deterministic replacement for `scatter_sum(src, adjacency.i, ...)`:
    `scatter_add_` uses CUDA atomics, so the float summation order (and the
    last bits of the result) changes between runs. `torch.segment_reduce` sums
    each row in a fixed order. Requires `src` ordered by row, with
    `numNeighbors[r]` consecutive entries for row r (the `AdjacencyList`
    contract: `i` is sorted and `numNeighbors` is its run-length encoding).
    Unlike `scatter_sum` this needs no `dim_size` and does no host sync.
    """
    with record_function("segment_sum"):
        return torch.segment_reduce(src, 'sum', lengths=numNeighbors, axis=0, unsafe=True)
