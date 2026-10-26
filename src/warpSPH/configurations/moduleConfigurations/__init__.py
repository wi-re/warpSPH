"""Sub-config dataclasses/enums shared across the scheme configs in
`warpSPH.configurations` (e.g. `DiffusionParameters`, `ShiftProperties`,
`SurfaceDetectionConfig`) -- each embedded as a field on one or more of
`CompressibleSPHConfig`, `CompSPHConfig`, `CRKSPHConfig`,
`WeaklyCompressibleSPHConfig`, `IncompressibleSPHConfig`. Re-exported wholesale
from `warpSPH.configurations` (`__all__` there extends this module's).
"""

__all__ = []

from .iisph import IISPHConfig
__all__.append('IISPHConfig')
from .dfsph import DFSPHConfig, DFSPHProjectionConfig
__all__.extend(['DFSPHConfig', 'DFSPHProjectionConfig'])

from .boundaryConditions import BoundaryCondition, BoundaryConditionType, boundaryConditionToDict, dictToBoundaryCondition, BCType
__all__.extend(['BoundaryCondition', 'BoundaryConditionType', 'boundaryConditionToDict', 'dictToBoundaryCondition', 'BCType'])

from .diffusionParameters import DiffusionParameters, buildDefaultDiffusionParamsCompressibleSPH, diffusionParamsToDict, dictToDiffusionParams
__all__.extend(['DiffusionParameters', 'buildDefaultDiffusionParamsCompressibleSPH', 'diffusionParamsToDict', 'dictToDiffusionParams'])

from .fluidProperties import fluidProperties, buildDefaultFluidProperties
__all__.extend(['fluidProperties', 'buildDefaultFluidProperties'])

from .shifting import ShiftProperties, buildDefaultShiftProperties, ShiftingScheme, ShiftingProjectionScheme
__all__.extend(['ShiftProperties', 'buildDefaultShiftProperties', 'ShiftingScheme', 'ShiftingProjectionScheme'])

from .surfaceDetection import SurfaceDetectionConfig, buildDefaultSurfaceDetectionConfig, SurfaceDetectionScheme, NormalSource
__all__.extend(['SurfaceDetectionConfig', 'buildDefaultSurfaceDetectionConfig', 'SurfaceDetectionScheme', 'NormalSource'])

from .viscositySwitchParameters import ViscositySwitchConfig, viscositySwitchConfigToDict, dictToViscositySwitchConfig
__all__.extend(['ViscositySwitchConfig', 'viscositySwitchConfigToDict', 'dictToViscositySwitchConfig'])

from .weaklyCompressibleDiffusionParams import WeaklyCompressibleDiffusionParams, buildDefaultDiffusionParamsWeaklyCompressibleSPH, wcDiffusionParamsToDict, dictToWCDiffusionParams
__all__.extend(['WeaklyCompressibleDiffusionParams', 'buildDefaultDiffusionParamsWeaklyCompressibleSPH', 'wcDiffusionParamsToDict', 'dictToWCDiffusionParams'])

from .gravity import GravityType, gravityConfiguration, buildDefaultGravityConfiguration, gravityConfigurationToDict, dictToGravityConfiguration
__all__.extend(['GravityType', 'gravityConfiguration', 'buildDefaultGravityConfiguration', 'gravityConfigurationToDict', 'dictToGravityConfiguration'])

from .solver import PressureSolverType, JacobiRelaxationMode, JacobiConvergenceCriterion, BoundaryPressureMode, ShiftPressureGauge, ShiftApplication, BoundaryOperatorTerms, DensityEvolution, resolveDensityEvolution, resolveBoundaryOperatorTerms, RelaxedJacobiSolverConfig, buildDefaultPSConfig, buildDefaultDFConfig, IncompressibleSolverConfig, buildDefaultIncompressibleSolverConfig
__all__.extend(['PressureSolverType', 'JacobiRelaxationMode', 'JacobiConvergenceCriterion', 'BoundaryPressureMode', 'ShiftPressureGauge', 'ShiftApplication', 'BoundaryOperatorTerms', 'DensityEvolution', 'resolveDensityEvolution', 'resolveBoundaryOperatorTerms', 'RelaxedJacobiSolverConfig', 'buildDefaultPSConfig', 'buildDefaultDFConfig', 'IncompressibleSolverConfig', 'buildDefaultIncompressibleSolverConfig'])
