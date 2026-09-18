"""`WeaklyCompressibleSPHConfig`, the scheme config for delta-SPH / weakly
compressible SPH (`schemes/deltaSPH.py`, `modules/timestep/weaklyCompressible.py`,
`modules/mdbc/velocity.py`), registered in the delta-SPH `SchemeBundle` in
`schemes/builder.py`. Same field set as `IncompressibleSPHConfig` minus
`solverConfig` (no pressure solver here): fluid properties, adaptive support,
diffusion, viscosity switch, boundary conditions, delta-SPH shifting,
`regions`/`rigidBodies`, surface detection, and gravity. Unlike
`IncompressibleSPHConfig`'s round-trip pair, `weaklyCompressibleConfigToDict`/
`dictToWeaklyCompressibleConfig` do serialize `regions` and `rigidBodies`.
"""

__all__ = ['WeaklyCompressibleSPHConfig', 'Sun2017DeltaSPHConfig', 'weaklyCompressibleConfigToDict', 'dictToWeaklyCompressibleConfig']

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
class WeaklyCompressibleSPHConfig:
    fluid: fluidProperties = field(default_factory=buildDefaultFluidProperties, metadata={"description": "Fluid properties for the weakly compressible SPH simulation"})

    adaptiveSupportScheme: AdaptiveSupportScheme = field(default=AdaptiveSupportScheme.NoScheme, metadata={'description': 'Adaptive support scheme to use'})
    adaptiveSupportIterations: int = field(default=1, metadata={'description': 'Number of iterations for adaptive support scheme'})
    adaptiveSupportThreshold: float = field(default=1e-3, metadata={'description': 'Threshold for adaptive support scheme'})
    adaptiveSupportCorrections: bool = field(default=True, metadata={'description': 'Whether to apply corrections in the adaptive support scheme (grad-H terms)'})


    diffusionParams: WeaklyCompressibleDiffusionParams = field(default_factory=buildDefaultDiffusionParamsWeaklyCompressibleSPH, metadata={'description': 'Diffusion parameters for the weakly compressible SPH simulation'})

    # Sun et al. 2017 Sec. 2 (Antuono/Jameson technique): the delta-SPH
    # diffusive terms (density diffusion + artificial viscosity) are evaluated
    # ONCE per real step, at the committed t^n state, and held fixed across
    # every RK4 sub-stage rather than recomputed fresh at each stage's
    # (intermediate, possibly wildly different during a violent event) state.
    # False (default) leaves every existing case's behaviour unchanged --
    # `schemes/deltaSPH.py` recomputes fresh every stage as it always has.
    # `DELTASPH_VALIDATION_PLAN.md` Part 1 flagged this as a known deviation
    # from Sun 2017; Part 5.1's Marrone dam-break investigation is the first
    # case to actually exercise the flag. Only meaningful with a multi-stage
    # RK integrator (`integrationScheme=rungeKutta4` etc.) -- a no-op under
    # any single-evaluation scheme (forwardEuler/semiImplicitEuler), since
    # there is only one stage to freeze against.
    freezeDiffusionAcrossStages: bool = field(default=False, metadata={'description': "Freeze delta-SPH's diffusive terms across RK sub-stages (Sun et al. 2017 Sec. 2), re-evaluating once per real step instead of once per stage"})

    viscositySwitchParams: ViscositySwitchConfig = field(default_factory=ViscositySwitchConfig)

    schemeName: str = field(default='Compressible SPH', metadata={'description': 'Name of the compressible SPH scheme to use'})

    boundaryConditions: List[BoundaryCondition] = field(default_factory=list, metadata={'description': 'List of boundary conditions to apply in the simulation'})

    dt_viscosityConstraint: bool = field(default=True, metadata={'description': 'Whether to apply viscosity constraint in timestep computation'})
    dt_accelerationConstraint: bool = field(default=True, metadata={'description': 'Whether to apply acceleration constraint in timestep computation'})
    dt_acousticConstraint: bool = field(default=True, metadata={'description': 'Whether to apply acoustic constraint in timestep computation'})
    pressureForceTerm: PressureForceScheme = field(default=PressureForceScheme.Antuono, metadata={'description': 'Pressure force term to use'})
    #: Apply the same kernel-gradient renormalization matrix `deltaSPH.py`
    #: already computes for `gradRhoL` (`detectFreeSurface`'s per-particle
    #: covariance fit) to the pressure-force gradient too, but only where a
    #: per-particle kernel-completeness (Shepard-sum) gate says the local
    #: neighbourhood is complete AND the particle is not flagged free-surface
    #: -- everywhere else (including every free-surface particle) the raw,
    #: unrenormalized gradient is used, unchanged from the `False` behaviour.
    #: False (default) leaves every existing case unchanged.
    #:
    #: `DELTASPH_VALIDATION_PLAN.md` 5.14/5.15/5.17/5.19-5.22: the Antuono
    #: (`sun2018` Eq. 9) pressure switch's symmetric branch is a real,
    #: nonzero force at any truncated/disordered kernel support even for a
    #: uniform pressure field, because the raw `Sum_j V_j grad_i W_ij` is
    #: only ~0 for a complete, regular neighbourhood -- exactly what a
    #: renormalized gradient is meant to restore. Neither PST, the DDT
    #: renormalization, nor either dissipation term fixed it (5.20-5.22).
    #: 5.23's first attempt applied the renormalization unconditionally and
    #: made the peak 8.8x WORSE -- `Li` comes from a covariance fit that is
    #: itself most ill-conditioned exactly in the sparse/disordered
    #: neighbourhoods where the artifact lives, so an ungated renormalization
    #: amplifies noise there rather than correcting it. 5.32 traced what
    #: DualSPHysics' own production advanced-shifting extension does instead
    #: -- gate on kernel-sum completeness (`poup1>0.95`) and bulk
    #: classification, falling back to the raw gradient everywhere else --
    #: and this flag now reproduces exactly that gate (`schemes/deltaSPH.py`
    #: step 13), rather than applying `Li` unconditionally.
    pressureForceRenormalized: bool = field(default=False, metadata={'description': 'Apply gradient renormalization to the pressure-force kernel gradient, gated on kernel-sum completeness and bulk classification (DualSPHysics-style)'})

    #: Where (and whether) the mDBC no-penetration correction is applied.
    #:
    #: * ``'derivative'`` -- the historical placement: `deltaSPH_step` folds
    #:   `nopenshift / config.dt` into `dvdt` alongside pressure/gravity/
    #:   viscosity, so it is re-evaluated at every RK sub-stage and goes
    #:   through the stage weighting.
    #: * ``'finalize'`` -- DualSPHysics' placement: applied **once per step**
    #:   in `WeaklyCompressibleSystem.finalize`, after the particle shift,
    #:   as a post-integration *velocity replacement*
    #:   (`v = v^n + nopenshift`, displacement recomputed from it) rather than
    #:   a force. See `JSphGpuSimple_ker.cu`'s `MDBC2_NoPen` blocks.
    #: * ``'off'`` -- not applied at all. DualSPHysics itself gates the term on
    #:   `SlipMode >= SLIP_NoSlip` and makes it opt-in, i.e. it is **never**
    #:   applied under free slip -- which is what Marrone 2011 Sec. 3
    #:   specifies. diffSPH disables its equivalent outright (`/ dt * 0`).
    #:
    #: `DELTASPH_VALIDATION_PLAN.md` 5.9.
    mdbcNoPenShiftMode: str = field(default='finalize', metadata={'description': "Where the mDBC no-penetration correction is applied: 'finalize' (default; once per step, DualSPHysics-style velocity replacement), 'derivative' (summed into dvdt, historical), or 'off'. 'derivative' can only *oppose* an into-wall velocity, never replace it, so a particle grazing a wall keeps its normal velocity indefinitely while the correction cancels the displacement; the continuity equation then integrates that phantom velocity into a density collapse. Root cause of the sloshingTank divergence at t = 4.57 s -- measured rho 0.995 -> 0.60 on one particle pinned at the ceiling, fixed by this default (DELTASPH_VALIDATION_PLAN.md 5.13(c))."})

    shiftProperties: ShiftProperties = field(default_factory=buildDefaultShiftProperties, metadata={'description': 'Properties for the delta-SPH shift'})

    regions: List[ParticleRegion] = field(default_factory=list, metadata={'description': 'List of particle regions in the simulation'})
    rigidBodies: List[RigidBody] = field(default_factory=list, metadata={'description': 'List of rigid bodies in the simulation'})

    surfaceDetectionConfig: SurfaceDetectionConfig = field(default_factory=buildDefaultSurfaceDetectionConfig, metadata={'description': 'Configuration for surface detection module'})

    gravityConfig: gravityConfiguration = field(default_factory=buildDefaultGravityConfiguration, metadata={'description': 'Configuration for gravity module'})

    bandwith: float = field(default=10.0, metadata={'description': 'Bandwith for the divergence-free noise sampling module'})

    #: Which mDBC wall-density extrapolation `schemes/deltaSPH.py` calls.
    #: `'ramped'` (default): `density2025.py`'s `computeMdbcDensity` --
    #: English et al. 2022 Eq. (12) ghost-node extrapolation, smoothly
    #: ramped down to a 0th-order Shepard fallback on `numNeighbors`/
    #: `|det(A_g)|`, clamped to `>= rho0` on the fallback share only.
    #: `'band'`: `densityBand.py`'s `computeMdbcDensityBand` -- Band et al.
    #: 2018-style fit directly at the boundary particle with a centroid-
    #: decoupled, Tikhonov-damped gradient block, UNCLAMPED. Prototyped in
    #: `scripts/probe_bandMlsPressureBoundary.py` against the "few particle
    #: large sheet" open item (DELTASPH_VALIDATION_PLAN.md); validate against
    #: the Marrone dam break before trusting it beyond that.
    mdbcDensityScheme: str = field(default='english2025', metadata={'description': "mDBC wall-density extrapolation: 'english2025' (default since WCSPH_DEFAULT_CLOSEOUT_PLAN.md item E, english2025.py, Band's value fit + English 2025's analytic-hydrostatic extrapolation, BOUNDARY_DENSITY_PLAN.md §5 -- beats 'ramped'/'band' on every Marrone config tested, at multiple resolutions, with no known regression), 'ramped' (density2025.py's English et al. 2022 ghost-node extrapolation + smooth det/neighbour-count ramp, the previous default), or 'band' (densityBand.py, unclamped Band et al. 2018-style fit -- a research tool, not a default candidate: still the worst of the three on every Marrone metric even after its own Shepard-precision-hole fix)"})

def _buildSun2017ShiftProperties() -> ShiftProperties:
    """`buildDefaultShiftProperties()` with Sun et al. 2017 Eq. (7)'s own
    constants -- the `+` of delta+-SPH at the intensity the paper specifies.

    The shared default is 1/8 of Eq. (7) (measured:
    `scripts/probe_deltaPlusShiftMagnitude.py`), which on Sun et al. 2019
    Sec. 3.1's Taylor-Green benchmark leaves delta+-SPH sitting on top of
    plain delta-SPH instead of improving on it -- `eps_V = 0.227 %` against
    the delta-SPH leg's 0.206 %, where the paper's delta+ reaches 0.125 %
    from the same 0.22 %. See `ShiftProperties.sun2017Eq7Shift`.
    """
    props = buildDefaultShiftProperties()
    props.sun2017Eq7Shift = True
    return props


@dataclass
class Sun2017DeltaSPHConfig(WeaklyCompressibleSPHConfig):
    """Sun et al. 2017's own delta+-SPH prescription -- identical to
    `WeaklyCompressibleSPHConfig` (same step function, `schemes/deltaSPH.py`;
    no new physics) except for two field defaults:

    * `freezeDiffusionAcrossStages` -> `True` (Sec. 2's
      RK4-with-frozen-diffusion pairing, Antuono/Jameson technique);
    * `shiftProperties.sun2017Eq7Shift` -> `True` (Eq. (7)'s own shift
      constants -- `(2h)^2`, `2 m_j/(rho_i+rho_j)`, `R = 0.2`).

    Selected via `WeaklyCompressibleSPHScheme.sun2017DeltaSPH` /
    `--scheme sun2017DeltaSPH` (`schemes/builder.py`) -- a *named* scheme
    rather than a changed default on the generic `deltaSPH` scheme, so opting
    into this paper's exact prescription is explicit, not a silent behaviour
    change for every existing `deltaSPH` case. `DELTASPH_VALIDATION_PLAN.md`
    Part 5.1 confirmed the un-frozen RK4 combination produces a spurious,
    extended (~1.4 t*) violent-impact pressure transient a single-stage
    integrator (e.g. DualSPHysics' symplectic Euler) never exhibits, since it
    has no cross-stage inconsistency to freeze against in the first place --
    frozen diffusion is specifically an RK-multi-stage companion technique.

    Kernel and integrator (Wendland C2, RK4) are case-level settings either
    way, not part of this config -- `cases/dambreak.py` already defaults to
    both regardless of which `WeaklyCompressibleSPHScheme` is selected.
    """
    freezeDiffusionAcrossStages: bool = field(
        default=True,
        metadata={'description': "Sun et al. 2017 Sec. 2: freeze delta-SPH's "
                  "diffusive terms across RK sub-stages -- this preset's "
                  "default (the base WeaklyCompressibleSPHConfig defaults False)"})
    shiftProperties: ShiftProperties = field(
        default_factory=_buildSun2017ShiftProperties,
        metadata={'description': "Sun et al. 2017 Eq. (7)'s own shift constants "
                  "(sun2017Eq7Shift=True) -- this preset's default; the shared "
                  "buildDefaultShiftProperties() is 1/8 of Eq. (7)"})


from typing import Dict, Any


def weaklyCompressibleConfigToDict(config: WeaklyCompressibleSPHConfig) -> Dict[str, Any]:
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
        'pressureForceRenormalized': config.pressureForceRenormalized,
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
        'regions': [region.toDict() for region in config.regions],
        'rigidBodies': [body.toDict() for body in config.rigidBodies],
        'freezeDiffusionAcrossStages': config.freezeDiffusionAcrossStages,
    }

def dictToWeaklyCompressibleConfig(configDict: Dict[str, Any]) -> WeaklyCompressibleSPHConfig:
    config = WeaklyCompressibleSPHConfig()
    config.fluid.eosType = EquationOfState[configDict['eosType']] if isinstance(configDict['eosType'], str) else configDict['eosType']
    config.fluid.restDensity = float(configDict['restDensity'])
    config.fluid.polytropicExponent = float(configDict['polytropicExponent'])
    config.fluid.kappa = float(configDict['kappa'])
    config.fluid.gas_constant = float(configDict['gas_constant'])
    config.fluid.molarMass = float(configDict['molarMass'])
    config.fluid.fixedSoundSpeed = float(configDict['fixedSoundSpeed'])
    config.adaptiveSupportScheme = AdaptiveSupportScheme[configDict['adaptiveSupportScheme']] if isinstance(configDict['adaptiveSupportScheme'], str) else configDict['adaptiveSupportScheme']
    config.adaptiveSupportIterations = int(configDict['adaptiveSupportIterations'])
    config.adaptiveSupportThreshold = float(configDict['adaptiveSupportThreshold'])
    config.adaptiveSupportCorrections = bool(configDict['adaptiveSupportCorrections'])
    config.diffusionParams = dictToWCDiffusionParams(configDict['diffusionParams'])
    config.viscositySwitchParams = dictToViscositySwitchConfig(configDict['viscositySwitchParams'])
    config.schemeName = configDict['schemeName']
    config.boundaryConditions = [dictToBoundaryCondition(bcDict) for bcDict in configDict['boundaryConditions']]
    config.dt_viscosityConstraint = bool(configDict['dt_viscosityConstraint'])
    config.dt_accelerationConstraint = bool(configDict['dt_accelerationConstraint'])
    config.dt_acousticConstraint = bool(configDict['dt_acousticConstraint'])
    # config.densityDiffusionTerm = DensityDiffusionScheme[configDict['densityDiffusionTerm']] if isinstance(configDict['densityDiffusionTerm'], str) else configDict['densityDiffusionTerm']
    config.pressureForceTerm = PressureForceScheme[configDict['pressureForceTerm']] if isinstance(configDict['pressureForceTerm'], str) else configDict['pressureForceTerm']
    config.pressureForceRenormalized = bool(configDict.get('pressureForceRenormalized', False))
    config.bandwith = float(configDict.get('bandwith', 10.0))
    shiftPropsDict = configDict.get('shiftProperties', {})
    config.shiftProperties = ShiftProperties(
        iterations=int(shiftPropsDict.get('iterations', 1)),
        CFL=float(shiftPropsDict.get('CFL', 0.3)),
        computeMach=bool(shiftPropsDict.get('computeMach', False)),
        maxC=float(shiftPropsDict.get('maxC', 0.3)),
        active=bool(shiftPropsDict.get('active', True)),
        scheme=ShiftingScheme[shiftPropsDict.get('scheme', ShiftingScheme.deltaSPH.name)] if isinstance(shiftPropsDict.get('scheme', ShiftingScheme.deltaSPH.name), str) else shiftPropsDict.get('scheme', ShiftingScheme.deltaSPH),
        projectionScheme=ShiftingProjectionScheme[shiftPropsDict.get('projectionScheme', ShiftingProjectionScheme.dot.name)] if isinstance(shiftPropsDict.get('projectionScheme', ShiftingProjectionScheme.dot.name), str) else shiftPropsDict.get('projectionScheme', ShiftingProjectionScheme.dot),
        summationDensity=bool(shiftPropsDict.get('summationDensity', False)),
        surfaceScaling=float(shiftPropsDict.get('surfaceScaling', 0.1)),
        threshold=float(shiftPropsDict.get('threshold', 0.5)),
        projectQuantities=bool(shiftPropsDict.get('projectQuantities', False)),
    )
    surfaceConfigDict = configDict.get('surfaceDetectionConfig')
    if surfaceConfigDict is not None:
        config.surfaceDetectionConfig = SurfaceDetectionConfig(
            active=surfaceConfigDict.get('active', buildDefaultSurfaceDetectionConfig().active),
            colorFieldThreshold=float(surfaceConfigDict.get('colorFieldThreshold', buildDefaultSurfaceDetectionConfig().colorFieldThreshold)),
            colorFieldGradThreshold=float(surfaceConfigDict.get('colorFieldGradThreshold', buildDefaultSurfaceDetectionConfig().colorFieldGradThreshold)),
            barecascoThreshold=float(surfaceConfigDict.get('barecascoThreshold', buildDefaultSurfaceDetectionConfig().barecascoThreshold)),
            expansionIterations=int(surfaceConfigDict.get('expansionIterations', buildDefaultSurfaceDetectionConfig().expansionIterations)),
            scheme=SurfaceDetectionScheme[surfaceConfigDict.get('scheme', buildDefaultSurfaceDetectionConfig().scheme.name)] if isinstance(surfaceConfigDict.get('scheme', buildDefaultSurfaceDetectionConfig().scheme.name), str) else surfaceConfigDict.get('scheme', buildDefaultSurfaceDetectionConfig().scheme),
            normalSource=NormalSource[surfaceConfigDict.get('normalSource', buildDefaultSurfaceDetectionConfig().normalSource.name)] if isinstance(surfaceConfigDict.get('normalSource', buildDefaultSurfaceDetectionConfig().normalSource.name), str) else surfaceConfigDict.get('normalSource', buildDefaultSurfaceDetectionConfig().normalSource),
        )
    config.gravityConfig = dictToGravityConfiguration(configDict['gravityConfig']) if configDict.get('gravityConfig') is not None else buildDefaultGravityConfiguration()
    config.regions = [ParticleRegion.fromDict(regionDict) for regionDict in configDict.get('regions', [])]
    config.rigidBodies = [RigidBody.fromDict(bodyDict) for bodyDict in configDict.get('rigidBodies', [])]
    config.freezeDiffusionAcrossStages = bool(configDict.get('freezeDiffusionAcrossStages', False))

    return config