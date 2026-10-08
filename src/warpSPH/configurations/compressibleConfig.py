"""`CompressibleSPHConfig`, the base scheme config for compressible SPH: EOS
(`gamma`, `rho0`, `backgroundPressure`), adaptive-support, `diffusionParams`,
`viscositySwitchParams`, and `boundaryConditions`. Used directly by the plain
Monaghan scheme (`schemes/monaghan.py`) and subclassed by `CompSPHConfig` and
`CRKSPHConfig`; read as `schemeConfig`/`compParams` throughout
`modules/{shockCapturing,adaptiveSupport,timestep,boundaryConditions}` and
`sample/`. `compressibleConfigToDict`/`dictToCompressibleConfig` are the dict
round-trip pair registered for all three schemes in `schemes/builder.py`.
"""

__all__ = ['CompressibleSPHConfig', 'compressibleConfigToDict', 'dictToCompressibleConfig']

# from ..system import CompressibleSystem, CompressibleSystemUpdate
# from ..config import SimulationConfig
import torch

# from ..modules import *
from warpSPHCore import *

from dataclasses import dataclass, field
from typing import Optional
from ..enumTypes import AdaptiveSupportScheme, ViscositySwitch

from .moduleConfigurations.diffusionParameters import DiffusionParameters, buildDefaultDiffusionParamsCompressibleSPH, diffusionParamsToDict, dictToDiffusionParams
from .moduleConfigurations.viscositySwitchParameters import ViscositySwitchConfig, viscositySwitchConfigToDict, dictToViscositySwitchConfig




from .moduleConfigurations.boundaryConditions import BoundaryCondition, BoundaryConditionType, boundaryConditionToDict, dictToBoundaryCondition
from typing import List

def buildDefaultViscositySwitchConfigCompressibleSPH() -> ViscositySwitchConfig:
    """The Monaghan host's default switch (2026-10-08, user): Rosswog (2020)'s entropy trigger at the paper's
    alpha_0 = 0, alpha_max = 1 -- with the limited reconstruction the best Kelvin-Helmholtz growth of the AV_PLAN
    sweep (nx 256: A(1.5) 0.151, above CRKSPH's 0.145). A case that names a switch gets a plain
    `ViscositySwitchConfig` for it, as before (`cases/compressible.configureCompressible`)."""
    config = ViscositySwitchConfig()
    config.scheme = ViscositySwitch.Rosswog2020
    config.alpha_min = 0.0
    config.alpha_max = 1.0
    return config


@dataclass
class CompressibleSPHConfig:
    gamma: float = field(default=1.4, metadata={'description': 'Adiabatic index'})
    backgroundPressure: float = field(default=0.0, metadata={'description': 'Background pressure to prevent tensile instability'})

    rho0: float = field(default=1.0, metadata={'description': 'Reference density'})

    adaptiveSupportScheme: AdaptiveSupportScheme = field(default=AdaptiveSupportScheme.Monaghan, metadata={'description': 'Adaptive support scheme to use'})
    adaptiveSupportIterations: int = field(default=1, metadata={'description': 'Number of iterations for adaptive support scheme'})
    adaptiveSupportThreshold: float = field(default=1e-3, metadata={'description': 'Threshold for adaptive support scheme'})
    adaptiveSupportCorrections: bool = field(default=True, metadata={'description': 'Whether to apply corrections in the adaptive support scheme (grad-H terms)'})
    pressureFormulation: str = field(default='meanKernel', metadata={'description': "Monaghan scheme pressure force + pdV work: 'meanKernel' (symmetric gradient with the mean kernel (grad W(h_i) + grad W(h_j))/2, du/dt from a velocity divergence) or 'perSide' (Price 2012 Eqs. 43-45: each side with its own h, the conjugate du/dt; with Omega when adaptiveSupportCorrections) -- SUPPORT_SOLVER_PLAN option C"})
    owenTable: str = field(default='lattice', metadata={'description': "Owen psi_H lookup table: 'lattice' (default since 2026-10-06: the lattice sum the runtime measures, exact on lattices in 1D/2D/3D) or 'shell' (the old continuum shell approximation, exact only in 1D; h 2 % / 0.9 % too large on a 2D / 3D lattice) -- SUPPORT_SOLVER_PLAN step 1a"})


    diffusionParams: DiffusionParameters = field(default_factory=buildDefaultDiffusionParamsCompressibleSPH)
    viscositySwitchParams: ViscositySwitchConfig = field(default_factory=buildDefaultViscositySwitchConfigCompressibleSPH)

    schemeName: str = field(default='Compressible SPH', metadata={'description': 'Name of the compressible SPH scheme to use'})

    boundaryConditions: List[BoundaryCondition] = field(default_factory=list, metadata={'description': 'List of boundary conditions to apply in the simulation'})

    # Solid walls (`kinds == 1` rows, modules/compressibleWall, COMPRESSIBLE_WALLS_PLAN.md)
    wallSlip: str = field(default='freeSlip', metadata={'description': "Wall-row velocity seen by the pair terms: 'freeSlip' (prescribed normal, fluid tangential) or 'noSlip' (prescribed)"})
    wallRiemannState: bool = field(default=True, metadata={'description': 'Wall state = star state of the mirrored Riemann problem (shock on approach, rarefaction on recession) instead of the plain Shepard gather'})
    wallLatticeSupport: Optional[bool] = field(default=None, metadata={'description': "Floor the wall rows' h at their own lattice spacing's (None: the scheme's default -- on for CRKSPH, whose moment matrices need it, off otherwise)"})

from typing import Dict, Any


def compressibleConfigToDict(config: CompressibleSPHConfig) -> Dict[str, Any]:
    return {
        'gamma': config.gamma,
        'backgroundPressure': config.backgroundPressure,
        'rho0': config.rho0,
        'adaptiveSupportScheme': config.adaptiveSupportScheme.name,
        'adaptiveSupportIterations': config.adaptiveSupportIterations,
        'adaptiveSupportThreshold': config.adaptiveSupportThreshold,
        'adaptiveSupportCorrections': config.adaptiveSupportCorrections,
        'owenTable': config.owenTable,
        'pressureFormulation': config.pressureFormulation,
        'diffusionParams': diffusionParamsToDict(config.diffusionParams),
        'viscositySwitchParams': viscositySwitchConfigToDict(config.viscositySwitchParams),
        'schemeName': config.schemeName,
        'boundaryConditions': [boundaryConditionToDict(bc) for bc in config.boundaryConditions],
        'wallSlip': config.wallSlip,
        'wallRiemannState': config.wallRiemannState,
        'wallLatticeSupport': config.wallLatticeSupport,
    }

def dictToCompressibleConfig(configDict: Dict[str, Any]) -> CompressibleSPHConfig:
    config = CompressibleSPHConfig()
    config.gamma = configDict['gamma']
    config.backgroundPressure = configDict['backgroundPressure']
    config.rho0 = configDict['rho0']
    config.adaptiveSupportScheme = AdaptiveSupportScheme[configDict['adaptiveSupportScheme']] if isinstance(configDict['adaptiveSupportScheme'], str) else configDict['adaptiveSupportScheme']
    config.adaptiveSupportIterations = configDict['adaptiveSupportIterations']
    config.adaptiveSupportThreshold = configDict['adaptiveSupportThreshold']
    config.adaptiveSupportCorrections = configDict['adaptiveSupportCorrections']
    config.owenTable = configDict.get('owenTable', 'shell')  # stored configs from before the field ran the shell table
    config.pressureFormulation = configDict.get('pressureFormulation', 'meanKernel')
    config.diffusionParams = dictToDiffusionParams(configDict['diffusionParams'])
    config.viscositySwitchParams = dictToViscositySwitchConfig(configDict['viscositySwitchParams'])
    config.schemeName = configDict['schemeName']
    config.boundaryConditions = [dictToBoundaryCondition(bcDict) for bcDict in configDict['boundaryConditions']]
    config.wallSlip = configDict.get('wallSlip', config.wallSlip)
    config.wallRiemannState = configDict.get('wallRiemannState', config.wallRiemannState)
    config.wallLatticeSupport = configDict.get('wallLatticeSupport', config.wallLatticeSupport)
    
    return config