"""`WeaklyCompressibleDiffusionParams`: viscosity (inviscid/viscid) and
density-diffusion settings for delta-SPH, embedded as `.diffusionParams` on
`WeaklyCompressibleSPHConfig`/`IncompressibleSPHConfig`. Distinct from the
compressible-scheme `DiffusionParameters` `wp.struct` in `diffusionParameters.py`
-- unrelated dataclass, different field set, sharing only the `diffusionParams`
attribute name on the respective scheme configs. `densityDiffusionTerm` selects
a `DensityDiffusionScheme` member (defined in `..enumTypes`, re-exported here
via the `enumTypes import *`).
"""

__all__ = ['WeaklyCompressibleDiffusionParams', 'buildDefaultDiffusionParamsWeaklyCompressibleSPH', 'wcDiffusionParamsToDict', 'dictToWCDiffusionParams']

from ...enumTypes import *
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass, field
import os
import torch
from enum import Enum



@dataclass
class WeaklyCompressibleDiffusionParams():
    inviscid : bool = field(default=True, metadata={"description": "Whether to use inviscid diffusion parameters"})
    inviscidAlpha : float = field(default=0.01, metadata={"description": "Alpha value for inviscid diffusion"})

    viscidNu : float = field(default=1e-3, metadata={"description": "Kinematic viscosity for viscous diffusion"})

    densityDelta: float = field(default=0.1, metadata={"description": "Density diffusion coefficient for delta-SPH"})
    densityDiffusionTerm: DensityDiffusionScheme = field(default=DensityDiffusionScheme.fourtakas2019, metadata={'description': 'Density diffusion term to use'})

def buildDefaultDiffusionParamsWeaklyCompressibleSPH() -> WeaklyCompressibleDiffusionParams:
    # DIAGNOSTIC ONLY: `WARPSPH_DEFAULT_DDT`, unset by default (every existing
    # case/script that doesn't set it is unaffected), lets a batch driver
    # (e.g. scratchpad/run_overnight_batch_2026-09-17.sh's gallery leg) force
    # a different densityDiffusionTerm across every example in one run,
    # without editing each example wrapper individually -- unlike
    # `integrationScheme`, this field is not a generic CaseSpec knob
    # `caseMain`/`buildArgumentParser` already exposes per-example, so there
    # is no CLI flag to forward here.
    #
    # `fourtakas2019` (DualSPHysics' own DDT_DDT2) is the default since
    # WCSPH_DEFAULT_CLOSEOUT_PLAN.md item E -- DELTASPH_VALIDATION_PLAN.md
    # Part 8.17/8.18: the tightest same-codebase match to diffSPH's own
    # dambreak band, structurally simpler than the Antuono bi-Laplacian (no
    # covariance/renormalization dependency), and run clean across Marrone
    # 3.1/3.4 at multiple resolutions/durations with no divergence.
    ddt = DensityDiffusionScheme.fourtakas2019
    override = os.environ.get('WARPSPH_DEFAULT_DDT')
    if override:
        ddt = DensityDiffusionScheme[override]
    return WeaklyCompressibleDiffusionParams(
        inviscid=True,
        inviscidAlpha=0.01,
        viscidNu=1e-3,
        densityDelta=0.1,
        densityDiffusionTerm=ddt
    )


def wcDiffusionParamsToDict(diffusionParams: WeaklyCompressibleDiffusionParams) -> Dict[str, Any]:
    return {
        'inviscid': diffusionParams.inviscid,
        'inviscidAlpha': diffusionParams.inviscidAlpha,
        'viscidNu': diffusionParams.viscidNu,
        'densityDelta': diffusionParams.densityDelta,
        'densityDiffusionTerm': diffusionParams.densityDiffusionTerm.name if isinstance(diffusionParams.densityDiffusionTerm, Enum) else diffusionParams.densityDiffusionTerm
    }
def dictToWCDiffusionParams(diffusionParamsDict: Dict[str, Any]) -> WeaklyCompressibleDiffusionParams:
    return WeaklyCompressibleDiffusionParams(
        inviscid=diffusionParamsDict.get('inviscid', True),
        inviscidAlpha=diffusionParamsDict.get('inviscidAlpha', 0.01),
        viscidNu=diffusionParamsDict.get('viscidNu', 1e-3),
        densityDelta=diffusionParamsDict.get('densityDelta', 0.1),
        densityDiffusionTerm=DensityDiffusionScheme[diffusionParamsDict.get('densityDiffusionTerm', 'deltaSPH')] if isinstance(diffusionParamsDict.get('densityDiffusionTerm'), str) else diffusionParamsDict.get('densityDiffusionTerm', DensityDiffusionScheme.deltaSPH)
    )
