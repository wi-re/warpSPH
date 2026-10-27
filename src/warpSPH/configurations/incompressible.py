"""Shared physical configuration and independent native IISPH/DFSPH policies."""

__all__ = ['IncompressibleSPHConfig', 'incompressibleConfigToDict', 'dictToIncompressibleSPHConfig']

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List
import torch
from .moduleConfigurations import *
from .moduleConfigurations.boundaryConditions import boundaryConditionToDict, dictToBoundaryCondition
from .moduleConfigurations.shifting import shiftPropertiesToDict, dictToShiftProperties
from .moduleConfigurations.iisph import IISPHConfig
from .moduleConfigurations.dfsph import DFSPHConfig, dictToDFSPHConfig
from .moduleConfigurations.viscositySwitchParameters import ViscositySwitchConfig, viscositySwitchConfigToDict, dictToViscositySwitchConfig
from .region import ParticleRegion
from .rigidBody import RigidBody
from ..enumTypes import *

@dataclass
class IncompressibleSPHConfig:
    fluid: fluidProperties = field(default_factory=buildDefaultFluidProperties, metadata={"description": "Fluid properties for the weakly compressible SPH simulation"})

    adaptiveSupportScheme: AdaptiveSupportScheme = field(default=AdaptiveSupportScheme.NoScheme, metadata={'description': 'Adaptive support scheme to use'})
    adaptiveSupportIterations: int = field(default=1, metadata={'description': 'Number of iterations for adaptive support scheme'})
    adaptiveSupportThreshold: float = field(default=1e-3, metadata={'description': 'Threshold for adaptive support scheme'})
    adaptiveSupportCorrections: bool = field(default=True, metadata={'description': 'Whether to apply corrections in the adaptive support scheme (grad-H terms)'})


    diffusionParams: WeaklyCompressibleDiffusionParams = field(default_factory=buildDefaultDiffusionParamsWeaklyCompressibleSPH, metadata={'description': 'Diffusion parameters for the weakly compressible SPH simulation'})

    viscositySwitchParams: ViscositySwitchConfig = field(default_factory=ViscositySwitchConfig)

    schemeName: str = field(default='Incompressible SPH', metadata={'description': 'Name of the native projection scheme'})

    boundaryConditions: List[BoundaryCondition] = field(default_factory=list, metadata={'description': 'List of boundary conditions to apply in the simulation'})

    dt_viscosityConstraint: bool = field(default=True, metadata={'description': 'Whether to apply viscosity constraint in timestep computation'})
    dt_accelerationConstraint: bool = field(default=True, metadata={'description': 'Whether to apply acceleration constraint in timestep computation'})
    dt_acousticConstraint: bool = field(default=True, metadata={'description': 'Whether to apply acoustic constraint in timestep computation'})
    pressureForceTerm: PressureForceScheme = field(default=PressureForceScheme.Antuono, metadata={'description': 'Pressure force term to use'})

    shiftProperties: ShiftProperties = field(default_factory=buildDefaultShiftProperties, metadata={'description': 'Properties for the delta-SPH shift'})

    regions: List[ParticleRegion] = field(default_factory=list, metadata={'description': 'List of particle regions in the simulation'})
    rigidBodies: List[RigidBody] = field(default_factory=list, metadata={'description': 'List of rigid bodies in the simulation'})

    surfaceDetectionConfig: SurfaceDetectionConfig = field(default_factory=buildDefaultSurfaceDetectionConfig, metadata={'description': 'Configuration for surface detection module'})

    gravityConfig: gravityConfiguration = field(default_factory=buildDefaultGravityConfiguration, metadata={'description': 'Configuration for gravity module'})

    bandwith: float = field(default=10.0, metadata={'description': 'Bandwith for the divergence-free noise sampling module'})

    iisph: IISPHConfig = field(default_factory=IISPHConfig, metadata={'description': 'Native IISPH policy; independent of the composed DFSPH/shifting solvers'})
    dfsph: DFSPHConfig = field(default_factory=DFSPHConfig, metadata={'description': 'Native DFSPH density/divergence policies; static boundary settings are in iisph'})

    def validate(self, simulationConfig=None):
        if hasattr(self, 'solverConfig'):
            raise ValueError('solverConfig is retired; configure native iisph/dfsph policies')
        self.iisph.validate()
        self.dfsph.validate()


@dataclass
class PredictorDFSPHConfig(IncompressibleSPHConfig):
    dfsph: DFSPHConfig = field(default_factory=lambda: DFSPHConfig(ordering='np_df_cd'))


@dataclass
class ShiftDFSPHConfig(IncompressibleSPHConfig):
    dfsph: DFSPHConfig = field(default_factory=lambda: DFSPHConfig(ordering='np_df_shift'))


def incompressibleConfigToDict(config: IncompressibleSPHConfig) -> Dict[str, Any]:
    return {
        'eosType': config.fluid.eosType.name,
        'restDensity': config.fluid.restDensity,
        'polytropicExponent': config.fluid.polytropicExponent,
        'kappa': config.fluid.kappa,
        'gas_constant': config.fluid.gas_constant,
        'molarMass': config.fluid.molarMass,
        'fixedSoundSpeed': config.fluid.fixedSoundSpeed if not isinstance(config.fluid.fixedSoundSpeed, torch.Tensor) else config.fluid.fixedSoundSpeed.detach().cpu().item(),

        'adaptiveSupportScheme': config.adaptiveSupportScheme.name,
        'adaptiveSupportIterations': config.adaptiveSupportIterations,
        'adaptiveSupportThreshold': config.adaptiveSupportThreshold,
        'adaptiveSupportCorrections': config.adaptiveSupportCorrections,
        'diffusionParams': wcDiffusionParamsToDict(config.diffusionParams),
        'viscositySwitchParams': viscositySwitchConfigToDict(config.viscositySwitchParams),
        'schemeName': config.schemeName,
        'boundaryConditions': [boundaryConditionToDict(bc) for bc in config.boundaryConditions],
        'dt_viscosityConstraint': config.dt_viscosityConstraint,
        'dt_accelerationConstraint': config.dt_accelerationConstraint,
        'dt_acousticConstraint': config.dt_acousticConstraint,
        'iisph': asdict(config.iisph),
        'dfsph': asdict(config.dfsph),
        'bandwith': config.bandwith,

        'pressureForceTerm': config.pressureForceTerm.name,
        'shiftProperties': shiftPropertiesToDict(config.shiftProperties),
        'regions': [region.toDict() for region in config.regions],
        'rigidBodies': [body.toDict() for body in config.rigidBodies],
        'surfaceDetectionConfig': {
            'active': config.surfaceDetectionConfig.active,
            'colorFieldThreshold': config.surfaceDetectionConfig.colorFieldThreshold,
            'colorFieldGradThreshold': config.surfaceDetectionConfig.colorFieldGradThreshold,
            'barecascoThreshold': config.surfaceDetectionConfig.barecascoThreshold,
            'expansionIterations': config.surfaceDetectionConfig.expansionIterations,
            'flagIsolated': config.surfaceDetectionConfig.flagIsolated,
            'scheme': config.surfaceDetectionConfig.scheme.name,
            'normalSource': config.surfaceDetectionConfig.normalSource.name,
        },
        'gravityConfig': gravityConfigurationToDict(config.gravityConfig),

    }

def dictToIncompressibleSPHConfig(configDict: Dict[str, Any]) -> IncompressibleSPHConfig:
    config = IncompressibleSPHConfig()
    config.fluid.eosType = EquationOfState[configDict['eosType']] if isinstance(configDict['eosType'], str) else configDict['eosType']
    config.fluid.restDensity = configDict['restDensity']
    config.fluid.polytropicExponent = configDict['polytropicExponent']
    config.fluid.kappa = configDict['kappa']
    config.fluid.gas_constant = configDict['gas_constant']
    config.fluid.molarMass = configDict['molarMass']
    config.fluid.fixedSoundSpeed = configDict['fixedSoundSpeed']
    config.adaptiveSupportScheme = AdaptiveSupportScheme[configDict['adaptiveSupportScheme']] if isinstance(configDict['adaptiveSupportScheme'], str) else configDict['adaptiveSupportScheme']
    config.adaptiveSupportIterations = configDict['adaptiveSupportIterations']
    config.adaptiveSupportThreshold = configDict['adaptiveSupportThreshold']
    config.adaptiveSupportCorrections = configDict['adaptiveSupportCorrections']
    config.diffusionParams = dictToWCDiffusionParams(configDict['diffusionParams'])
    config.viscositySwitchParams = dictToViscositySwitchConfig(configDict['viscositySwitchParams'])
    config.schemeName = configDict['schemeName']
    config.boundaryConditions = [dictToBoundaryCondition(bcDict) for bcDict in configDict['boundaryConditions']]
    config.dt_viscosityConstraint = configDict['dt_viscosityConstraint']
    config.dt_accelerationConstraint = configDict['dt_accelerationConstraint']
    config.dt_acousticConstraint = configDict['dt_acousticConstraint']
    config.iisph = IISPHConfig(**configDict.get('iisph', {}))
    config.iisph.validate()
    config.dfsph = dictToDFSPHConfig(configDict.get('dfsph', {}))
    # config.densityDiffusionTerm = DensityDiffusionScheme[configDict['densityDiffusionTerm']] if isinstance(configDict['densityDiffusionTerm'], str) else configDict['densityDiffusionTerm']
    config.pressureForceTerm = PressureForceScheme[configDict['pressureForceTerm']] if isinstance(configDict['pressureForceTerm'], str) else configDict['pressureForceTerm']
    config.bandwith = configDict.get('bandwith', 10.0)
    config.shiftProperties = dictToShiftProperties(configDict.get('shiftProperties', {}))
    config.regions = [ParticleRegion.fromDict(value) for value in configDict.get('regions', [])]
    config.rigidBodies = [RigidBody.fromDict(value) for value in configDict.get('rigidBodies', [])]
    surfaceConfigDict = configDict.get('surfaceDetectionConfig')
    if surfaceConfigDict is not None:
        config.surfaceDetectionConfig = SurfaceDetectionConfig(
            active=surfaceConfigDict.get('active', buildDefaultSurfaceDetectionConfig().active),
            colorFieldThreshold=surfaceConfigDict.get('colorFieldThreshold', buildDefaultSurfaceDetectionConfig().colorFieldThreshold),
            colorFieldGradThreshold=surfaceConfigDict.get('colorFieldGradThreshold', buildDefaultSurfaceDetectionConfig().colorFieldGradThreshold),
            barecascoThreshold=surfaceConfigDict.get('barecascoThreshold', buildDefaultSurfaceDetectionConfig().barecascoThreshold),
            expansionIterations=surfaceConfigDict.get('expansionIterations', buildDefaultSurfaceDetectionConfig().expansionIterations),
            flagIsolated=bool(surfaceConfigDict.get('flagIsolated', True)),
            scheme=SurfaceDetectionScheme[surfaceConfigDict.get('scheme', buildDefaultSurfaceDetectionConfig().scheme.name)] if isinstance(surfaceConfigDict.get('scheme', buildDefaultSurfaceDetectionConfig().scheme.name), str) else surfaceConfigDict.get('scheme', buildDefaultSurfaceDetectionConfig().scheme),
            normalSource=NormalSource[surfaceConfigDict.get('normalSource', buildDefaultSurfaceDetectionConfig().normalSource.name)] if isinstance(surfaceConfigDict.get('normalSource', buildDefaultSurfaceDetectionConfig().normalSource.name), str) else surfaceConfigDict.get('normalSource', buildDefaultSurfaceDetectionConfig().normalSource),
        )
    config.gravityConfig = dictToGravityConfiguration(configDict['gravityConfig']) if configDict.get('gravityConfig') is not None else buildDefaultGravityConfiguration()
    if configDict.get('solverConfig'):
        if 'iisph' not in configDict or 'dfsph' not in configDict:
            raise ValueError('Legacy incompressible solverConfig is retired; rebuild the configuration '
                             'with native iisph/dfsph policies. See docs/incompressible/MIGRATION.md.')
        import warnings
        warnings.warn('Ignoring retired solverConfig in a native configuration; only iisph/dfsph '
                      'policies control native projections.', DeprecationWarning, stacklevel=2)
    return config
