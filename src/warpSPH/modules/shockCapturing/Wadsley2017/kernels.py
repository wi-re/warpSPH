"""Warp kernels of the Wadsley et al. (2017) shock detector: no torch ops, no host reads.

The detector (`Wadsley2017.wadsleyDetector`) used to take four host syncs per call: `torch.linalg.pinv` (the
M^-1 that corrects the velocity Jacobian), `torch.bincount` and two `torch.segment_reduce(lengths=...)` calls
(the neighbour average of D/|T|, Eq. 29), plus ~100 elementwise torch ops around them. Here:

* the pseudo-inverse in ``V @ pinv(M)`` (`reconstruction.computeVelocityJacobian`, corrected) is ``linalg.applyPinv``.
* ``wadsleyPointKernel`` -- per particle: the pressure-gradient normal ``n``, ``D`` (Eq. 24), ``|T|`` and ``D / |T|``.
* ``wadsleyNeighbourMeanKernel`` -- per CSR row: ``R`` (Eq. 29), the ``1 - (r / H)^4`` weighted mean of ``D/|T|``
  over the row, one thread per particle summing its row in order, so the result is deterministic (no atomics).

Forward only: the detector's outputs are detached by the caller. Launched with ``wp.launch`` like every other
operation, so they follow the current stream (graph captures included) and need no synchronisation.
"""

import torch
import warp as wp
from warp.types import matrix

from warpSPHCore import scalar_t

from ....linalg import loadPadded3x3 as _load

__all__ = ['wadsleyPoint', 'wadsleyNeighbourMean']

@wp.kernel
def wadsleyPointKernel(V: wp.array2d(dtype=scalar_t), gradP: wp.array2d(dtype=scalar_t), dim: int, tiny: scalar_t,
                       D: wp.array(dtype=scalar_t), Tnorm: wp.array(dtype=scalar_t), ratio: wp.array(dtype=scalar_t),
                       normal: wp.array2d(dtype=scalar_t)):
    """Eqs. (22)-(24) per particle: n = gradP / |gradP| (0 for a uniform P), D, |T|_F with T = (V + V^T) / 2, D / |T|."""
    i = wp.tid()
    Vi = _load(V, i, dim)
    g2 = scalar_t(0.0)
    for a in range(3):
        if a < dim:
            g2 = g2 + gradP[i, a] * gradP[i, a]
    gnorm = wp.sqrt(g2)
    n = wp.vector(scalar_t(0.0), scalar_t(0.0), scalar_t(0.0))
    if gnorm > scalar_t(0.0):
        inv = scalar_t(1.0) / wp.max(gnorm, tiny)
        for a in range(3):
            if a < dim:
                n[a] = gradP[i, a] * inv
    div = scalar_t(0.0)
    for a in range(3):
        if a < dim:
            div = div + Vi[a, a]
    Di = div
    if dim > 1:
        dvdn = scalar_t(0.0)
        for a in range(3):
            for b in range(3):
                if a < dim and b < dim:
                    dvdn = dvdn + n[a] * Vi[a, b] * n[b]
        d = scalar_t(dim)
        Di = d / (d - scalar_t(1.0)) * (dvdn + wp.max(-div, scalar_t(0.0)) / d)
    t2 = scalar_t(0.0)
    for a in range(3):
        for b in range(3):
            if a < dim and b < dim:
                tab = scalar_t(0.5) * (Vi[a, b] + Vi[b, a])
                t2 = t2 + tab * tab
    tn = wp.sqrt(t2)
    D[i] = Di
    Tnorm[i] = tn
    r = scalar_t(0.0)
    if tn > scalar_t(0.0):
        r = Di / wp.max(tn, tiny)
    ratio[i] = r
    for a in range(3):
        if a < dim:
            normal[i, a] = n[a]


@wp.kernel
def wadsleyNeighbourMeanKernel(x: wp.array2d(dtype=scalar_t), H: wp.array(dtype=scalar_t), m: wp.array(dtype=scalar_t),
                               ratio: wp.array(dtype=scalar_t), edgeOffsets: wp.array(dtype=wp.int32),
                               numNeighbors: wp.array(dtype=wp.int32), neighbors: wp.array(dtype=wp.int64),
                               domainMin: wp.array(dtype=scalar_t), domainMax: wp.array(dtype=scalar_t),
                               periodic: wp.array(dtype=wp.bool), dim: int, tiny: scalar_t,
                               R: wp.array(dtype=scalar_t)):
    """Eq. (29): R_i = sum_j w_ij (D_j / |T|_j) / sum_j w_ij over i's row, w_ij = (1 - (r_ij / H_i)^4) m_j for r_ij < H_i, clamped to [-1, 1]."""
    i = wp.tid()
    begin = wp.int32(edgeOffsets[i])
    count = wp.int32(numNeighbors[i])
    num = scalar_t(0.0)
    den = scalar_t(0.0)
    for k in range(count):
        j = wp.int32(neighbors[begin + k])
        r2 = scalar_t(0.0)
        for a in range(3):
            if a < dim:
                dx = x[i, a] - x[j, a]
                if periodic[a]:
                    L = domainMax[a] - domainMin[a]
                    dx = dx - L * wp.rint(dx / L)
                r2 = r2 + dx * dx
        q = wp.sqrt(r2) / H[i]
        if q < scalar_t(1.0):
            q2 = q * q
            w = (scalar_t(1.0) - q2 * q2) * m[j]
            num = num + w * ratio[j]
            den = den + w
    r = scalar_t(0.0)
    if den > scalar_t(0.0):
        r = num / wp.max(den, tiny)
    R[i] = wp.clamp(r, scalar_t(-1.0), scalar_t(1.0))


# ------------------------------------------------------------------------------------------------ torch entry points

def _w(t: torch.Tensor):
    return wp.from_torch(t.contiguous())


def wadsleyPoint(V: torch.Tensor, gradP: torch.Tensor):
    """`(D, |T|, D/|T|, n)` from the corrected velocity Jacobian `(N, dim, dim)` and the pressure gradient `(N, dim)`."""
    N, dim, _ = V.shape
    D, Tnorm, ratio = (torch.empty(N, dtype=V.dtype, device=V.device) for _ in range(3))
    normal = torch.empty(N, dim, dtype=V.dtype, device=V.device)
    wp.launch(wadsleyPointKernel, dim=N, device=wp.device_from_torch(V.device),
              inputs=[_w(V.reshape(N, dim * dim)), _w(gradP), dim, torch.finfo(V.dtype).tiny, _w(D), _w(Tnorm), _w(ratio),
                      _w(normal)])
    return D, Tnorm, ratio, normal


def wadsleyNeighbourMean(state, ratio: torch.Tensor, adjacency, domain) -> torch.Tensor:
    """R (Eq. 29) for every particle from its CSR row of `adjacency`."""
    N, dim = state.positions.shape
    R = torch.empty(N, dtype=ratio.dtype, device=ratio.device)
    # the list's own dtypes (int32 row arrays, int64 columns): converting per call would copy
    as32 = lambda t: t if t.dtype == torch.int32 else t.to(torch.int32)
    as64 = lambda t: t if t.dtype == torch.int64 else t.to(torch.int64)
    wp.launch(wadsleyNeighbourMeanKernel, dim=N, device=wp.device_from_torch(ratio.device),
              inputs=[_w(state.positions), _w(state.supports), _w(state.masses), _w(ratio),
                      _w(as32(adjacency.edgeOffsets)), _w(as32(adjacency.numNeighbors)), _w(as64(adjacency.paddedJ)),
                      _w(domain.min.to(ratio.dtype)), _w(domain.max.to(ratio.dtype)), _w(domain.periodic),
                      dim, torch.finfo(ratio.dtype).tiny, _w(R)])
    return R
