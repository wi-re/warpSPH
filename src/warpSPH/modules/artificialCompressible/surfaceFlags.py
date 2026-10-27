"""Free-surface flags the ACSPH loop reads: the raw edge set, lambda, the
renormalisation state and the diffusion taper (authors' D3)."""

from dataclasses import dataclass
from typing import Any, Optional

import torch

from ..surfaceDetection import detectFreeSurface


def diffusionTaper(lam, lambdaZero=0.1, lambdaOne=0.4):
    """`C2_interp(lambda, F0, F1)` of the authors' `common_functions_cuda.cuh`:
    with `q = clamp((lambda - F1)/(F0 - F1), 0, 1)`, `(1 - q)^4 (4q + 1)` -- 1 at
    `lambda >= F1`, 0 at `lambda <= F0`, C2 in between."""
    q = ((lam - lambdaOne) / (lambdaZero - lambdaOne)).clamp(0.0, 1.0)
    return (1.0 - q) ** 4 * (4.0 * q + 1.0)



@dataclass
class SurfaceFlags:
    """What one surface detection hands the dual-time loop."""
    rawEdge: torch.Tensor            # undilated edge set: the authors' `is_surf == 1`
    dilated: torch.Tensor            # the dilated set `V` (JST region, contact masks)
    normals: Any
    lambdas: torch.Tensor            # renormalisation eigenvalue (1 where detection is inactive)
    renormalization: Any
    taper: Optional[torch.Tensor]    # `diffusionTaper(lambdas)` or None when the taper is off


def detectSurface(state, config, schemeConfig, adjacency, store=True) -> SurfaceFlags:
    """Free-surface detection at `state`'s positions. With `store` the indicators,
    normals and lambdas are written onto the state (once per real step, the frozen
    set); without, nothing is touched (the per-iteration refresh, D2).

    Frozen for the whole real step by default: the paper re-evaluates the
    *diffusion* every dual-time iteration but says nothing about re-detecting the
    surface (`acParams.redetectSurfacePerIteration` does)."""
    fs, fsm, normals, renormalization, lam = detectFreeSurface(
        state, config, schemeConfig, schemeConfig.surfaceDetectionConfig,
        adjacency, returnNormals=True)
    if lam is None:      # surface detection inactive: nothing is an edge row
        lam = torch.ones_like(state.densities)
    acParams = schemeConfig.acParams
    taper = (diffusionTaper(lam, acParams.diffusionTaperLambdaZero, acParams.diffusionTaperLambdaOne)
             .to(state.densities.dtype) if acParams.diffusionLambdaTaper else None)
    if store:
        state.surfaceIndicators = (fsm > 0.5).to(torch.int32)
        state.surfaceNormals = normals
        state.surfaceLambdas = lam
    return SurfaceFlags(fs > 0.5, fsm > 0.5, normals, lam, renormalization, taper)
