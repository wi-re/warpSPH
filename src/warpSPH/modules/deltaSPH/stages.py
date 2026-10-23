"""Delta spatial stage data, scoped to one integrator step.

Indexed stages retain the existing frozen-diffusion policy. Unindexed callers
still recompute each evaluation; this cleanup does not change that method.
Direct legacy callers retain their configuration-owned adapter.
"""
from dataclasses import dataclass
from warpSPHCore.profiling import record_function
from ...enumTypes import DensityDiffusionScheme
from ..density import computeGradRho, computeGradRhoL
from . import computeDensityDiffusion, computeVelocityDiffusion


@dataclass
class DeltaStages:
    diffusion: object = None
    surfaceStart: object = None
    previousSurfaceStart: object = None
    surfaceStats: object = None


def diffusionRates(currentState, config, schemeConfig, adjacency, renormalizationState_, wall, stageIndex, *, owner=None):
    freezeDiffusion = getattr(schemeConfig, 'freezeDiffusionAcrossStages', False)
    isFirstStage = stageIndex is None or stageIndex == 0
    cached = (owner.diffusion if owner is not None else getattr(schemeConfig, '_frozenDiffusionCache', None)) if freezeDiffusion else None
    diffusionTerm = schemeConfig.diffusionParams.densityDiffusionTerm

    if freezeDiffusion and not isFirstStage and cached is not None:
        with record_function("[warpSPH] - [deltaSPH - 09-12] reuse frozen diffusion"):
            drhodt_diss, dvdt_diss = cached
    else:
        with record_function("[warpSPH] - [deltaSPH - 09] gradRho"):
            if diffusionTerm == DensityDiffusionScheme.denormalized or diffusionTerm == DensityDiffusionScheme.denormalizedOnly:
                gradRho = computeGradRho(currentState, config, schemeConfig, adjacency)
            else:
                gradRho = None
        with record_function("[warpSPH] - [deltaSPH - 10] gradRhoL"):
            if diffusionTerm == DensityDiffusionScheme.deltaSPH or diffusionTerm == DensityDiffusionScheme.deltaOnly:
                gradRhoL = computeGradRhoL(currentState, config, schemeConfig, adjacency, L = renormalizationState_)
            else:
                gradRhoL = None
        with record_function("[warpSPH] - [deltaSPH - 11] density diffusion"):
            drhodt_diss = computeDensityDiffusion(currentState, config, schemeConfig, adjacency, gradRho, gradRhoL)
        with record_function("[warpSPH] - [deltaSPH - 12] velocity diffusion"):
            dvdt_diss = computeVelocityDiffusion(currentState, config, schemeConfig, adjacency,
                                                 wall=wall)      # the wall's share is inside the (possibly frozen) tuple
        if freezeDiffusion:
            if owner is not None:
                owner.diffusion = (drhodt_diss, dvdt_diss)
            else:
                schemeConfig._frozenDiffusionCache = (drhodt_diss, dvdt_diss)

    return drhodt_diss, dvdt_diss
