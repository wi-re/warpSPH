"""Moving-least-squares (Liu & Liu) interpolation and boundary extrapolation.

`interpolateLiuLiu` fits a local linear field (value + gradient) at each query
point from `computeLiuMatricesWarp`'s moment matrix/vector and solves it with
a pseudo-inverse, falling back to zero for points with fewer than
`neighbor_threshold` neighbors OR whose moment matrix `A_g` is ill-conditioned
(`|det(A_g)| < determinantThreshold`, mirroring DualSPHysics'
`determlimit ~ 1e-3` gate in `JSphCpu_mdbc.cpp` -- see
`DELTASPH_VALIDATION_PLAN.md` Part 3). The combined mask is returned as
`wellConditioned` so callers can build the same 0th-order/Shepard/rest-value
fallback ladder DualSPHysics uses, instead of re-deriving their own
neighbour-count-only heuristic (a bare count missed the case of a thin, fast
front sliding over a dry bed, whose fluid neighbours can be numerous but all
sit in one shallow, nearly-coplanar band).

`determinantThreshold` defaults to `None`, resolved per-call from
`_DETERMINANT_THRESHOLDS[config.kernel]` -- DualSPHysics' `1e-3` is specific
to *their* Wendland C2 kernel; an A/B on the Marrone dam break
(`scripts/probe_deltaSPHMarrone.py --kernel Wendland2`) showed the same
literal value is too loose for Wendland C4 at matched support radius (it let
a coplanar, thin-sheet boundary stencil through as "well-conditioned",
producing an explosive `P_b`), and `scripts/
probe_mdbcKernelDeterminantScale.py` derived a C4-specific value from the
ratio of `det(A_g)` between the two kernels on matched synthetic geometry.
Passing `determinantThreshold` explicitly still overrides the lookup; an
unlisted kernel falls back to the DualSPHysics constant with no calibration
guarantee -- calibrate it with that probe before trusting mDBC/wall-pressure
extrapolation on a new kernel choice.

`liuExtend`/`liuMirror` reuse that fit to
extrapolate a field across a signed-distance boundary: for points within two
supports of (and behind) the boundary, they mirror the query position through
the surface, evaluate the local fit there, and blend down to a Shepard
(0th-order) estimate — or `defaultValue` when there are no neighbors at all.
`liuMirror` differs from `liuExtend` only in dropping the gradient-based
linear correction term, i.e. it is a pure reflection/Shepard extrapolation.
Both recurse over trailing dimensions for multi-component fields.
"""

import torch
from .wp_mat import computeLiuMatricesWarp

from warpSPHCore import *
from typing import Any, Optional
from ...configurations.simulationConfig import SimulationConfig
from torch.profiler import record_function

__all__ = ['interpolateLiuLiu', 'liuExtend', 'liuMirror']

#: Per-kernel `|det(A_g)|` acceptance floor for the mDBC/wall-pressure MLS fit
#: (`DELTASPH_VALIDATION_PLAN.md` Part 3). `Wendland2` is DualSPHysics' own
#: `determlimit` (`JSphCpu_mdbc.cpp`); `Wendland4` is derived from it via
#: `scripts/probe_mdbcKernelDeterminantScale.py` (C4/C2 `det(A_g)` ratio ~1.79
#: on matched synthetic geometry, at this repo's `n_h = 4.0` support). Any
#: other kernel falls back to the C2 value uncalibrated -- run that probe for
#: a new kernel rather than trusting this default on it.
_DETERMINANT_THRESHOLDS = {
    KernelFunctions.Wendland2: 1e-3,
    KernelFunctions.Wendland4: 1.8e-3,
}
_DEFAULT_DETERMINANT_THRESHOLD = 1e-3


def interpolateLiuLiu(
    queryPositions: torch.Tensor,
    referenceParticles: Any,
    referenceQuantities: torch.Tensor,
    config: SimulationConfig,
    adjacency: AdjacencyList = None,
    neighbor_threshold: int = 4,
    direction: OperationDirection = OperationDirection.AllToAll,
    supportScale: float = 1.0,
    determinantThreshold: Optional[float] = None,
):
    if determinantThreshold is None:
        determinantThreshold = _DETERMINANT_THRESHOLDS.get(
            config.kernel, _DEFAULT_DETERMINANT_THRESHOLD)
    with record_function("[warpSPH] - interpolateLiuLiu"):
        h = referenceParticles.supports.clone()
        referenceParticles.supports = h * supportScale

        _, b, A_g, neighCounts = computeLiuMatricesWarp(
            queryPositions = queryPositions,
            referenceParticles = referenceParticles,
            referenceQuantities = referenceQuantities,
            operationProperties = OperationProperties(
                kernel = config.kernel,
                supportMode = SupportScheme.Scatter,
                operationMode = direction,
            ),
            domain = config.domain,
            adjacency = adjacency
        )
        referenceParticles.supports = h

        determinant = torch.linalg.det(A_g)
        wellConditioned = torch.logical_and(
            neighCounts > neighbor_threshold,
            torch.abs(determinant) >= determinantThreshold,
        )

        A_g_inv = torch.zeros_like(A_g)
        A_g_inv[wellConditioned] = torch.linalg.pinv(A_g[wellConditioned])

        res = torch.matmul(A_g_inv, b.unsqueeze(2))[:,:,0]

        return res[:,0], res[:,1:], neighCounts, A_g, b, wellConditioned


def liuExtend(
        q: torch.Tensor,
        sim_config: Any,
        scheme_config: Any,
        particles: Any,
        distances: torch.Tensor,
        normals: torch.Tensor,
        current_time: torch.Tensor,
        dt: torch.Tensor,
        neighborThreshold: int = 4,
        defaultValue: float = 0.0,
):
    if len(q.shape) > 1:
        q_flat = q.view(q.shape[0], -1)
        defaults = torch.full((q_flat.shape[1],), defaultValue, device=q.device, dtype=q.dtype) if isinstance(defaultValue, (int, float)) else defaultValue.flatten()
        extended_q = torch.zeros_like(q_flat)
        for i in range(q_flat.shape[1]):
            extended_q[:, i] = liuExtend(
                q = q_flat[:, i],
                sim_config = sim_config,
                scheme_config = scheme_config,
                particles = particles,
                distances = distances,
                normals = normals,
                current_time = current_time,
                dt = dt,
                neighborThreshold = neighborThreshold,
                defaultValue = defaults[i] if defaults.ndim > 0 else defaults.item(),
            )
        return extended_q.view(q.shape)
    


    mask = torch.logical_and(torch.abs(distances) < particles.supports * 2.0, distances < 0)
    masked_positions = particles.positions[mask]
    relPos = 2 * distances[mask].unsqueeze(1) * normals[mask]
    queryPositions = masked_positions - relPos

    shep, b, A_g, neighCounts = computeLiuMatricesWarp(
        queryPositions = queryPositions,
        referenceParticles = particles,
        referenceQuantities = q,
        operationProperties = OperationProperties(
                kernel = sim_config.kernel,
                supportMode = SupportScheme.Scatter,
                operationMode = OperationDirection.FluidToBoundary,
        ),
        domain = sim_config.domain,
    )
    # print(shep)

    A_g_inv = torch.zeros_like(A_g)
    A_g_inv[neighCounts > 4] = torch.linalg.pinv(A_g[neighCounts > 4])

    res = torch.matmul(A_g_inv, b.unsqueeze(2))[:,:,0]

    extended_q = torch.zeros_like(masked_positions[:,0])

    b_scalar = b[:,0]
    b_grad = b[:,1:]

    res_scalar = res[:,0]
    res_grad = res[:,1:]

    # print(res_scalar, res_grad)

    # directions = torch.nn.functional.normalize(normals[mask], dim=1)
    dot = torch.einsum('ij,ij->i', relPos, res_grad)
    projected_q = res_scalar + dot #* distances[mask]

    shepValue = b[:,0] / (shep + 1e-8)
    shepProjection = shepValue + torch.einsum('ij,ij->i', relPos, b_grad + 1e-8)
    # defaultValue = torch.zeros_like(shepValue)

    projected_q[neighCounts < neighborThreshold] = shepProjection[neighCounts < neighborThreshold]
    projected_q[neighCounts == 0] = defaultValue

#     projected_q = shepProjection

    q_new = q.clone()
    q_new[mask] = projected_q
    return q_new#, queryPositions, shepValue, neighCounts

def liuMirror(
        q: torch.Tensor,
        sim_config: Any,
        scheme_config: Any,
        particles: Any,
        distances: torch.Tensor,
        normals: torch.Tensor,
        current_time: torch.Tensor,
        dt: torch.Tensor,
        neighborThreshold: int = 4,
        defaultValue: float = 0.0,
):
    if len(q.shape) > 1:
        q_flat = q.view(q.shape[0], -1)
        defaults = torch.full((q_flat.shape[1],), defaultValue, device=q.device, dtype=q.dtype) if isinstance(defaultValue, (int, float)) else defaultValue.flatten()
        extended_q = torch.zeros_like(q_flat)
        for i in range(q_flat.shape[1]):
            extended_q[:, i] = liuMirror(
                q = q_flat[:, i],
                sim_config = sim_config,
                scheme_config = scheme_config,
                particles = particles,
                distances = distances,
                normals = normals,
                current_time = current_time,
                dt = dt,
                neighborThreshold = neighborThreshold,
                defaultValue = defaults[i] if defaults.ndim > 0 else defaults.item(),
            )
        return extended_q.view(q.shape)
    mask = torch.logical_and(torch.abs(distances) < particles.supports * 2.0, distances < 0)
    masked_positions = particles.positions[mask]
    relPos = 2 * distances[mask].unsqueeze(1) * normals[mask]
    queryPositions = masked_positions - relPos

    shep, b, A_g, neighCounts = computeLiuMatricesWarp(
        queryPositions = queryPositions,
        referenceParticles = particles,
        referenceQuantities = q,
        operationProperties = OperationProperties(
                kernel = sim_config.kernel,
                supportMode = SupportScheme.Scatter,
                operationMode = OperationDirection.FluidToBoundary,
        ),
        domain = sim_config.domain,
    )
    # print(shep)

    A_g_inv = torch.zeros_like(A_g)
    A_g_inv[neighCounts > 4] = torch.linalg.pinv(A_g[neighCounts > 4])

    res = torch.matmul(A_g_inv, b.unsqueeze(2))[:,:,0]

    extended_q = torch.zeros_like(masked_positions[:,0])

    b_scalar = b[:,0]
    b_grad = b[:,1:]

    res_scalar = res[:,0]
    res_grad = res[:,1:]

    # print(res_scalar, res_grad)

    # directions = torch.nn.functional.normalize(normals[mask], dim=1)
    dot = torch.einsum('ij,ij->i', relPos, res_grad)
    projected_q = res_scalar# + dot #* distances[mask]

    shepValue = b[:,0] / (shep + 1e-8)
    shepProjection = shepValue# + torch.einsum('ij,ij->i', relPos, b_grad + 1e-8)
    # defaultValue = torch.zeros_like(shepValue)

    projected_q[neighCounts < neighborThreshold] = shepProjection[neighCounts < neighborThreshold]
    projected_q[neighCounts == 0] = defaultValue

#     projected_q = shepProjection

    q_new = q.clone()
    q_new[mask] = projected_q
    return q_new#, queryPositions, shepValue, neighCounts