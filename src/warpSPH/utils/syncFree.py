"""Host-sync-free replacements for the boolean-mask idioms used on the mDBC
ghost/boundary particle subsets.

``x[kinds == 2]`` (and ``out[idx] = y[mask]``) sizes its result by a count
that lives on the GPU, so every such expression forces a device->host sync
(``nonzero``), and makes the code impossible to capture in a CUDA graph. The
helpers here keep every array full length (one row per particle) and express
the subset by index remapping instead:

* a *gather* for ghost rows reads through :func:`ghostSourceIndex`
  (non-ghost rows read themselves, harmlessly);
* a *scatter* from ghost rows onto their boundary particles goes through
  :func:`scatterRows`, which sends every non-ghost row to a trash slot one past
  the end.

Row values for the subset are computed exactly as before (same elementwise
arithmetic), so results are bitwise identical to the masked form.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

__all__ = ['deviceConstant', 'ghostSourceIndex', 'ghostTargetIndex', 'scatterRows', 'cachedKindIndex']

_CONSTANTS: Dict[Tuple, torch.Tensor] = {}


def deviceConstant(values: Sequence, dtype: torch.dtype, device) -> torch.Tensor:
    """A cached device tensor holding ``values``.

    ``torch.tensor(pythonList, device='cuda')`` is a synchronous pageable
    host->device copy on every call (and illegal inside CUDA-graph capture);
    configuration constants (gravity direction, BC type tables) only need it
    once. The returned tensor is shared -- never modify it in place.
    """
    key = (tuple(float(v) if isinstance(v, float) else v for v in values), dtype, str(device))
    t = _CONSTANTS.get(key)
    if t is None:
        t = torch.tensor(list(values), dtype=dtype, device=device)
        _CONSTANTS[key] = t
    return t


def ghostSourceIndex(kinds: torch.Tensor, ghostIndices: torch.Tensor, kind: int = 2) -> torch.Tensor:
    """Per-row gather index: ``ghostIndices[i]`` for rows of ``kind``, ``i``
    otherwise. ``values[ghostSourceIndex(...)]`` equals
    ``values[ghostIndices[mask]]`` on the masked rows, with no sync."""
    rows = torch.arange(kinds.shape[0], device=kinds.device, dtype=ghostIndices.dtype)
    return torch.where(kinds == kind, ghostIndices, rows)


def ghostTargetIndex(kinds: torch.Tensor, ghostIndices: torch.Tensor, kind: int = 2) -> torch.Tensor:
    """Per-row scatter target for :func:`scatterRows`: ``ghostIndices[i]`` for
    rows of ``kind``, the trash slot ``N`` otherwise."""
    trash = torch.full_like(ghostIndices, kinds.shape[0])
    return torch.where(kinds == kind, ghostIndices, trash)


def scatterRows(base: torch.Tensor, target: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """``out = base.clone(); out[target[i]] = values[i]`` for every row whose
    target is not the trash slot ``N`` (see :func:`ghostTargetIndex`).

    Equivalent to ``out[ghostIndices[mask]] = values[mask]`` when the real
    targets are unique (true for the mDBC ghost->boundary map), without the
    mask-induced sync."""
    n = base.shape[0]
    padded = torch.cat([base, base.new_zeros((1,) + tuple(base.shape[1:]))], dim=0)
    padded[target] = values
    return padded[:n]


def cachedKindIndex(kinds: torch.Tensor, kind: int, config) -> torch.Tensor:
    """``(kinds == kind).nonzero()`` computed once per run and cached on
    `config`. `kinds` is a run-constant field (assigned at particle generation,
    never mutated -- the same assumption `mdbc._util.stateHasBoundaryParticles`
    makes), so after the first call gathers of that particle subset need no
    host sync. The row order is ascending, exactly as boolean-mask indexing."""
    cache = getattr(config, '_kindIndexCache', None)
    if cache is None:
        cache = {}
        config._kindIndexCache = cache
    key = (kind, str(kinds.device), int(kinds.shape[0]))
    idx = cache.get(key)
    if idx is None:
        idx = (kinds == kind).nonzero().squeeze(1)
        cache[key] = idx
    return idx


def sortedQuantiles(x: torch.Tensor, qs, ignoreNan: bool = False, sortedX: Optional[torch.Tensor] = None):
    """`[torch.quantile(x, q) for q in qs]` (or `nanquantile` with
    `ignoreNan`) from ONE sort, bitwise: the same float32 rank arithmetic
    (`q * (n - 1)`, NaN handling), gathers and `lerp` as ATen's
    `quantile_compute`, with the `q` values as a cached device constant (so
    no host sync, unlike a tensor-`q` `torch.quantile`, which range-checks
    `q` on the device). `sortedX`: `x` already sorted ascending (e.g. a
    positive rescaling of a sorted array, which stays sorted). 1-D `x`."""
    s = torch.sort(x)[0] if sortedX is None else sortedX
    q = deviceConstant(list(qs), s.dtype, s.device)
    return list(quantilesFromSorted(s, q, ignoreNan).unbind(-1))


def quantilesFromSorted(s: torch.Tensor, q: torch.Tensor, ignoreNan: bool = False) -> torch.Tensor:
    """The quantiles `q` (a device tensor, `s`' dtype) of the ascending 1-D
    `s`: ATen's `quantile_compute` after its sort, op for op (pure torch, so
    it can sit inside a `compileGlue` function)."""
    if ignoreNan:
        ranks = q * (s.isnan().logical_not().sum(-1, keepdim=True) - 1)
        ranks = ranks.masked_fill(ranks < 0, 0)
    else:
        last = s.shape[-1] - 1
        ranks = torch.masked_fill(q * last, s.isnan().any(-1, keepdim=True), last)
    below = ranks.to(torch.long)
    weights = ranks - below
    above = ranks.ceil().to(torch.long)
    return s.gather(-1, below).lerp_(s.gather(-1, above), weights)
