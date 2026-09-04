"""`IncompressibleSPHConfig`, the scheme config for the DFSPH divergence-free
incompressible solver (`schemes/divergenceFree.py`, `modules/incompressible/{incompressible,
divergenceFree}.py`), registered in the incompressible `SchemeBundle` in
`schemes/builder.py`. Bundles fluid properties, adaptive support, diffusion,
viscosity switch, boundary conditions, delta-SPH shifting, `regions`/
`rigidBodies`, surface detection, gravity, and `solverConfig` (the relaxed-Jacobi
pressure/divergence-free solver settings). Note: unlike its
`WeaklyCompressibleSPHConfig` sibling, `incompressibleConfigToDict`/
`dictToIncompressibleSPHConfig` do not round-trip `regions` or `rigidBodies` at
all -- they're silently dropped on export and left at the dataclass default
(empty list) on import.
"""

__all__ = ['IncompressibleSPHConfig', 'incompressibleConfigToDict', 'dictToIncompressibleSPHConfig']

# from ..system import CompressibleSystem, CompressibleSystemUpdate
# from ..config import SimulationConfig
import torch

# from ..modules import *
from warpSPHCore import *

from dataclasses import dataclass, field
from typing import Optional
from ..enumTypes import AdaptiveSupportScheme, ViscositySwitch, EquationOfState

from .moduleConfigurations.diffusionParameters import DiffusionParameters, buildDefaultDiffusionParamsCompressibleSPH, diffusionParamsToDict, dictToDiffusionParams
from .moduleConfigurations.viscositySwitchParameters import ViscositySwitchConfig, viscositySwitchConfigToDict, dictToViscositySwitchConfig




from .moduleConfigurations.boundaryConditions import BoundaryCondition, BoundaryConditionType, boundaryConditionToDict, dictToBoundaryCondition
from typing import List


from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from .region import RegionType, ParticleRegion
from .rigidBody import RigidBody

from .moduleConfigurations.surfaceDetection import SurfaceDetectionConfig, buildDefaultSurfaceDetectionConfig
from .moduleConfigurations import *
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

    schemeName: str = field(default='Compressible SPH', metadata={'description': 'Name of the compressible SPH scheme to use'})

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

    solverConfig : IncompressibleSolverConfig = field(default_factory=buildDefaultIncompressibleSolverConfig, metadata={'description': 'Configuration for the incompressible solver'})

from typing import Dict, Any


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
        'bandwith': config.bandwith,

        'pressureForceTerm': config.pressureForceTerm.name,
        'shiftProperties': {
            'iterations': config.shiftProperties.iterations,
            'CFL': config.shiftProperties.CFL,
            'computeMach': config.shiftProperties.computeMach,
            'maxC': config.shiftProperties.maxC,
            'active': config.shiftProperties.active,
            'scheme': config.shiftProperties.scheme.name,
            'projectionScheme': config.shiftProperties.projectionScheme.name,
            'summationDensity': config.shiftProperties.summationDensity,
            'surfaceScaling': config.shiftProperties.surfaceScaling,
            'threshold': config.shiftProperties.threshold,
            'projectQuantities': config.shiftProperties.projectQuantities,
        },
        'surfaceDetectionConfig': {
            'active': config.surfaceDetectionConfig.active,
            'colorFieldThreshold': config.surfaceDetectionConfig.colorFieldThreshold,
            'colorFieldGradThreshold': config.surfaceDetectionConfig.colorFieldGradThreshold,
            'barecascoThreshold': config.surfaceDetectionConfig.barecascoThreshold,
            'expansionIterations': config.surfaceDetectionConfig.expansionIterations,
            'scheme': config.surfaceDetectionConfig.scheme.name,
            'normalSource': config.surfaceDetectionConfig.normalSource.name,
        },
        'gravityConfig': gravityConfigurationToDict(config.gravityConfig),
        'solverConfig': {
            'pressureSolver': {
                'minIterations': config.solverConfig.pressureSolver.minIterations,
                'maxIterations': config.solverConfig.pressureSolver.maxIterations,
                'tolerance': config.solverConfig.pressureSolver.tolerance,
                'relaxationFactor': config.solverConfig.pressureSolver.relaxationFactor,
                'relaxationMode': config.solverConfig.pressureSolver.relaxationMode.name,
                'solverType': config.solverConfig.pressureSolver.solverType.name,
                'rtol': config.solverConfig.pressureSolver.rtol,
                'atol': config.solverConfig.pressureSolver.atol,
                'restart': config.solverConfig.pressureSolver.restart,
                'krylovFp64': config.solverConfig.pressureSolver.krylovFp64,
                'boundaryOperatorTerms': config.solverConfig.pressureSolver.boundaryOperatorTerms.name,
                'convergenceCriterion': config.solverConfig.pressureSolver.convergenceCriterion.name,
            },
            'divergenceFreeSolver': {
                'minIterations': config.solverConfig.divergenceFreeSolver.minIterations,
                'maxIterations': config.solverConfig.divergenceFreeSolver.maxIterations,
                'tolerance': config.solverConfig.divergenceFreeSolver.tolerance,
                'relaxationFactor': config.solverConfig.divergenceFreeSolver.relaxationFactor,
                'relaxationMode': config.solverConfig.divergenceFreeSolver.relaxationMode.name,
                'solverType': config.solverConfig.divergenceFreeSolver.solverType.name,
                'rtol': config.solverConfig.divergenceFreeSolver.rtol,
                'atol': config.solverConfig.divergenceFreeSolver.atol,
                'restart': config.solverConfig.divergenceFreeSolver.restart,
                'krylovFp64': config.solverConfig.divergenceFreeSolver.krylovFp64,
                'boundaryOperatorTerms': config.solverConfig.divergenceFreeSolver.boundaryOperatorTerms.name,
                'convergenceCriterion': config.solverConfig.divergenceFreeSolver.convergenceCriterion.name,
            },
            'integrateRho': config.solverConfig.integrateRho,
            'densityEvolution': config.solverConfig.densityEvolution.name,
            # None = no bundle-level override; each solver's own setting stands.
            'boundaryOperatorTerms': (config.solverConfig.boundaryOperatorTerms.name
                                     if config.solverConfig.boundaryOperatorTerms is not None else None),
            'forceShiftPressureGauge': config.solverConfig.forceShiftPressureGauge,
            'boundaryPressureMode': config.solverConfig.boundaryPressureMode.name,
            'mdbcNoPenetrationShift': config.solverConfig.mdbcNoPenetrationShift,
            'warmStartConstantDensity': config.solverConfig.warmStartConstantDensity,
            'akinciBoundaryVolume': config.solverConfig.akinciBoundaryVolume,
            'akinciBoundaryVolumeScale': config.solverConfig.akinciBoundaryVolumeScale,
            'shiftPressureGauge': config.solverConfig.shiftPressureGauge.name,
            'shiftApplication': config.solverConfig.shiftApplication.name,
        }
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
    # config.densityDiffusionTerm = DensityDiffusionScheme[configDict['densityDiffusionTerm']] if isinstance(configDict['densityDiffusionTerm'], str) else configDict['densityDiffusionTerm']
    config.pressureForceTerm = PressureForceScheme[configDict['pressureForceTerm']] if isinstance(configDict['pressureForceTerm'], str) else configDict['pressureForceTerm']
    config.bandwith = configDict.get('bandwith', 10.0)
    shiftPropsDict = configDict.get('shiftProperties', {})
    config.shiftProperties = ShiftProperties(
        iterations=shiftPropsDict.get('iterations', 1),
        CFL=shiftPropsDict.get('CFL', 0.3),
        computeMach=shiftPropsDict.get('computeMach', False),
        maxC=shiftPropsDict.get('maxC', 0.3),
        active=shiftPropsDict.get('active', True),
        scheme=ShiftingScheme[shiftPropsDict.get('scheme', ShiftingScheme.deltaSPH.name)] if isinstance(shiftPropsDict.get('scheme', ShiftingScheme.deltaSPH.name), str) else shiftPropsDict.get('scheme', ShiftingScheme.deltaSPH),
        projectionScheme=ShiftingProjectionScheme[shiftPropsDict.get('projectionScheme', ShiftingProjectionScheme.dot.name)] if isinstance(shiftPropsDict.get('projectionScheme', ShiftingProjectionScheme.dot.name), str) else shiftPropsDict.get('projectionScheme', ShiftingProjectionScheme.dot),
        summationDensity=shiftPropsDict.get('summationDensity', False),
        surfaceScaling=shiftPropsDict.get('surfaceScaling', 0.1),
        threshold=shiftPropsDict.get('threshold', 0.5),
        projectQuantities=shiftPropsDict.get('projectQuantities', False),
    )
    surfaceConfigDict = configDict.get('surfaceDetectionConfig')
    if surfaceConfigDict is not None:
        config.surfaceDetectionConfig = SurfaceDetectionConfig(
            active=surfaceConfigDict.get('active', buildDefaultSurfaceDetectionConfig().active),
            colorFieldThreshold=surfaceConfigDict.get('colorFieldThreshold', buildDefaultSurfaceDetectionConfig().colorFieldThreshold),
            colorFieldGradThreshold=surfaceConfigDict.get('colorFieldGradThreshold', buildDefaultSurfaceDetectionConfig().colorFieldGradThreshold),
            barecascoThreshold=surfaceConfigDict.get('barecascoThreshold', buildDefaultSurfaceDetectionConfig().barecascoThreshold),
            expansionIterations=surfaceConfigDict.get('expansionIterations', buildDefaultSurfaceDetectionConfig().expansionIterations),
            scheme=SurfaceDetectionScheme[surfaceConfigDict.get('scheme', buildDefaultSurfaceDetectionConfig().scheme.name)] if isinstance(surfaceConfigDict.get('scheme', buildDefaultSurfaceDetectionConfig().scheme.name), str) else surfaceConfigDict.get('scheme', buildDefaultSurfaceDetectionConfig().scheme),
            normalSource=NormalSource[surfaceConfigDict.get('normalSource', buildDefaultSurfaceDetectionConfig().normalSource.name)] if isinstance(surfaceConfigDict.get('normalSource', buildDefaultSurfaceDetectionConfig().normalSource.name), str) else surfaceConfigDict.get('normalSource', buildDefaultSurfaceDetectionConfig().normalSource),
        )
    config.gravityConfig = dictToGravityConfiguration(configDict['gravityConfig']) if configDict.get('gravityConfig') is not None else buildDefaultGravityConfiguration()
    solverConfigDict = configDict.get('solverConfig', {})
    psDict = solverConfigDict.get('pressureSolver', {})
    dfDict = solverConfigDict.get('divergenceFreeSolver', {})

    def _solverType(d):
        v = d.get('solverType', PressureSolverType.relaxedJacobi)
        return PressureSolverType[v] if isinstance(v, str) else v

    def _relaxationMode(d):
        v = d.get('relaxationMode', JacobiRelaxationMode.fixed)
        return JacobiRelaxationMode[v] if isinstance(v, str) else v

    def _boundaryPressureMode(d):
        v = d.get('boundaryPressureMode', BoundaryPressureMode.mdbcDensity)
        return BoundaryPressureMode[v] if isinstance(v, str) else v

    def _shiftApplication(d):
        v = d.get('shiftApplication',
                  buildDefaultIncompressibleSolverConfig().shiftApplication)
        return ShiftApplication[v] if isinstance(v, str) else v

    def _densityEvolution(d):
        v = d.get('densityEvolution',
                  buildDefaultIncompressibleSolverConfig().densityEvolution)
        return DensityEvolution[v] if isinstance(v, str) else v

    def _convergenceCriterion(d, default):
        v = d.get('convergenceCriterion', default)
        return JacobiConvergenceCriterion[v] if isinstance(v, str) else v

    def _boundaryOperatorTerms(d, default):
        v = d.get('boundaryOperatorTerms', default)
        return BoundaryOperatorTerms[v] if isinstance(v, str) else v

    def _shiftPressureGauge(d):
        v = d.get('shiftPressureGauge',
                  buildDefaultIncompressibleSolverConfig().shiftPressureGauge)
        return ShiftPressureGauge[v] if isinstance(v, str) else v

    config.solverConfig = IncompressibleSolverConfig(
        pressureSolver=RelaxedJacobiSolverConfig(
            minIterations=psDict.get('minIterations', buildDefaultPSConfig().minIterations),
            maxIterations=psDict.get('maxIterations', buildDefaultPSConfig().maxIterations),
            tolerance=psDict.get('tolerance', buildDefaultPSConfig().tolerance),
            relaxationFactor=psDict.get('relaxationFactor', buildDefaultPSConfig().relaxationFactor),
            relaxationMode=_relaxationMode(psDict),
            solverType=_solverType(psDict),
            rtol=psDict.get('rtol', buildDefaultPSConfig().rtol),
            atol=psDict.get('atol', buildDefaultPSConfig().atol),
            restart=psDict.get('restart', buildDefaultPSConfig().restart),
            krylovFp64=psDict.get('krylovFp64', buildDefaultPSConfig().krylovFp64),
            boundaryOperatorTerms=_boundaryOperatorTerms(psDict, buildDefaultPSConfig().boundaryOperatorTerms),
            convergenceCriterion=_convergenceCriterion(psDict, buildDefaultPSConfig().convergenceCriterion),
        ),
        divergenceFreeSolver=RelaxedJacobiSolverConfig(
            minIterations=dfDict.get('minIterations', buildDefaultDFConfig().minIterations),
            maxIterations=dfDict.get('maxIterations', buildDefaultDFConfig().maxIterations),
            tolerance=dfDict.get('tolerance', buildDefaultDFConfig().tolerance),
            relaxationFactor=dfDict.get('relaxationFactor', buildDefaultDFConfig().relaxationFactor),
            relaxationMode=_relaxationMode(dfDict),
            solverType=_solverType(dfDict),
            rtol=dfDict.get('rtol', buildDefaultDFConfig().rtol),
            atol=dfDict.get('atol', buildDefaultDFConfig().atol),
            restart=dfDict.get('restart', buildDefaultDFConfig().restart),
            krylovFp64=dfDict.get('krylovFp64', buildDefaultDFConfig().krylovFp64),
            boundaryOperatorTerms=_boundaryOperatorTerms(dfDict, buildDefaultDFConfig().boundaryOperatorTerms),
            convergenceCriterion=_convergenceCriterion(dfDict, buildDefaultDFConfig().convergenceCriterion),
        ),
        integrateRho=solverConfigDict.get('integrateRho', buildDefaultIncompressibleSolverConfig().integrateRho),
        densityEvolution=_densityEvolution(solverConfigDict),
        boundaryOperatorTerms=_boundaryOperatorTerms(
            solverConfigDict, buildDefaultIncompressibleSolverConfig().boundaryOperatorTerms),
        forceShiftPressureGauge=solverConfigDict.get(
            'forceShiftPressureGauge', buildDefaultIncompressibleSolverConfig().forceShiftPressureGauge),
        boundaryPressureMode=_boundaryPressureMode(solverConfigDict),
        mdbcNoPenetrationShift=solverConfigDict.get(
            'mdbcNoPenetrationShift', buildDefaultIncompressibleSolverConfig().mdbcNoPenetrationShift),
        warmStartConstantDensity=solverConfigDict.get(
            'warmStartConstantDensity', buildDefaultIncompressibleSolverConfig().warmStartConstantDensity),
        akinciBoundaryVolume=solverConfigDict.get(
            'akinciBoundaryVolume', buildDefaultIncompressibleSolverConfig().akinciBoundaryVolume),
        akinciBoundaryVolumeScale=solverConfigDict.get(
            'akinciBoundaryVolumeScale', buildDefaultIncompressibleSolverConfig().akinciBoundaryVolumeScale),
        shiftPressureGauge=_shiftPressureGauge(solverConfigDict),
        shiftApplication=_shiftApplication(solverConfigDict),
    )
    


    return config