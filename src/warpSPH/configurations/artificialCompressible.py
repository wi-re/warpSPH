"""`ArtificialCompressibleSPHConfig`, the scheme config for
artificial-compressibility SPH (`schemes/artificialCompressible.py`, De Courcy
et al. 2024; `ACSPH_PLAN.md`), registered in `schemes/builder.py`.

Modelled on `WeaklyCompressibleSPHConfig` -- same fluid / adaptive-support /
viscosity / boundary-condition / shifting / regions / rigid-body / surface-
detection / gravity blocks -- with the equation-of-state and density-diffusion
half replaced by `acParams` (`ArtificialCompressibilityParams`), which carries
every constant Part 2 of the plan lists.

Two fields deserve their own note, because they look like they mean something
they do not:

- **`fluid.fixedSoundSpeed` is not used by this scheme.** ACSPH has no acoustic
  wave speed; `beta = CFL_tau h / dtau` plays that role and is derived, not
  configured. It is kept in the config only because `fluidProperties` is
  shared, and reading it here would be a bug.
- **`acParams.referenceSoundSpeedForViscosity` exists for exactly one purpose**:
  the paper defines the physical viscosity as `nu = alpha_nu h c0 / K` with
  `alpha_nu = 0.01`, i.e. it still needs a *reference* `c0` to fix `nu` in the
  comparison against delta-SPH, even though it says the scheme "does not
  require the definition of c0". So a `c0` is an input to the viscosity and to
  nothing else. It is a separate field so that is explicit rather than a silent
  reuse of `fluid.fixedSoundSpeed` (ACSPH_PLAN.md Part 2).
"""

__all__ = ['ArtificialCompressibilityParams', 'ArtificialCompressibleSPHConfig',
           'buildDefaultArtificialCompressibilityParams',
           'artificialCompressibleConfigToDict',
           'dictToArtificialCompressibleConfig']

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from warpSPHCore import *

from ..enumTypes import (AdaptiveSupportScheme, PressureForceScheme,
                         PressureSmoothingScheme, ViscositySwitch)
from .moduleConfigurations import *
from .moduleConfigurations.boundaryConditions import (BoundaryCondition,
                                                      boundaryConditionToDict,
                                                      dictToBoundaryCondition)
from .moduleConfigurations.surfaceDetection import (SurfaceDetectionConfig,
                                                    buildDefaultSurfaceDetectionConfig)
from .moduleConfigurations.viscositySwitchParameters import (
    ViscositySwitchConfig, dictToViscositySwitchConfig, viscositySwitchConfigToDict)
from .region import ParticleRegion
from .rigidBody import RigidBody


@dataclass
class ArtificialCompressibilityParams:
    """Every constant the paper fixes for the dual-time solve. Defaults are the
    paper's own recommended operating point: AC-2L, RK2, `CFL_t = 0.2`,
    `dt/dtau = 5`, `eps_v = -6`."""

    #: The `D^p` operator, Eqs. (32)-(37). `renormalizedBiLaplacian` (AC-2L) is
    #: the paper's working default; `laplacian` (AC-2) is its negative control.
    pressureSmoothing: PressureSmoothingScheme = field(
        default=PressureSmoothingScheme.renormalizedBiLaplacian,
        metadata={'description': "Pressure-smoothing operator D^p (AC-2/2L/4/JST)"})

    #: Pseudo-time CFL, set by the RK order: 0.5 / 1.0 / 1.5 for RK2 / RK3 / RK4
    #: (paper Sec. 3.1.3). `beta = CFL_tau h / dtau`, `k1 = beta^2`.
    cflTau: float = field(default=0.5, metadata={'description': 'Pseudo-time CFL (Eq. 24)'})
    #: Real-time CFL, Eq. (46). ~0.2 in the paper; Tables 1-2 measure a sharp
    #: accuracy cliff above 0.4, which is a hard ceiling, not a guideline.
    cflT: float = field(default=0.2, metadata={'description': 'Real-time CFL (Eq. 46); <= 0.4'})
    #: `R` in `dtau = dt / R`. 2 is the best cost/accuracy point in Table 1;
    #: 5-10 buys accuracy at near-linear cost.
    dtOverDtau: float = field(default=5.0, metadata={'description': 'dt / dtau'})
    #: Pseudo-time RK stage count. Sec. 4.3: higher order buys no accuracy here
    #: (BDF2 sets it) and costs near-linearly in stages. 2 is the default.
    rkStages: int = field(default=2, metadata={'description': 'Pseudo-time RK stages (2/3/4)'})

    #: `log10` convergence target for `eps_v` (Eqs. 47-48). -6 general use,
    #: -8 for violent impact.
    epsilonV: float = field(default=-6.0, metadata={'description': 'log10 convergence target'})
    #: The floor in `U_eps = max(min(|v|_max, U_char), eps_s)`, Eq. (48).
    epsilonS: float = field(default=1e-5, metadata={'description': 'U_eps floor (Eq. 48)'})
    #: The case's characteristic velocity in Eq. (48). The paper never defines
    #: it per case; ACSPH_PLAN.md Sec. 5.5 makes it a required per-case value.
    #: `None` means "use `|v|_max` alone", which is *not* what the paper does.
    uChar: Optional[float] = field(default=None, metadata={'description': 'U_char in Eq. (48)'})
    maxPseudoIterations: int = field(default=200, metadata={'description': 'pseudo-time iteration cap'})
    minPseudoIterations: int = field(default=1, metadata={'description': 'pseudo-time iteration floor'})

    #: `k2 = k2Factor * h * beta`, Eq. (24). 0.1 for consistency with delta-SPH
    #: practice; the measured stability ceiling is 0.2.
    k2Factor: float = field(default=0.1, metadata={'description': 'k2 = k2Factor h beta (Eq. 24)'})
    #: JST blend constants, Eq. (37).
    kappa2: float = field(default=0.5, metadata={'description': 'JST kappa_2 (Eq. 37)'})
    kappa4: float = field(default=1.0 / 32.0, metadata={'description': 'JST kappa_4 (Eq. 37)'})
    #: `eps_4 = max(0, kappa_4 - eps_2)`. The paper prints `min`, which makes
    #: the JST operator vanish in smooth flow -- the opposite of its stated
    #: design, and not what standard JST does. Set True to reproduce the
    #: printed form; see ACSPH_PLAN.md Sec. 5.1, and **ask the authors**.
    jstUsePrintedMin: bool = field(default=False, metadata={'description': 'reproduce Eq. (37) as printed'})

    #: The `k3 D(div v)/Dtau` term of Eqs. (9)/(22). The paper zeroes it
    #: ("little influence"); the field exists so the choice is visible.
    k3: float = field(default=0.0, metadata={'description': 'k3 term (dropped by the paper)'})
    #: Point-implicit source treatment, `alpha_PI = 1 + alpha_s dtau alpha_t`
    #: (Eqs. 43-45). The paper reports `alpha_PI = 1` works fine here.
    usePointImplicit: bool = field(default=True, metadata={'description': 'point-implicit alpha_PI (Eq. 43)'})
    #: The `tilde v` material-derivative corrections, Eqs. (27)-(31). The paper
    #: concludes: leave them off (Sec. 4.2).
    useTildeVAdvection: bool = field(default=False, metadata={'description': 'tilde-v advective terms (Eqs. 27-31)'})
    #: Shift inside the pseudo-time loop (Eq. 60) rather than outside it
    #: (Eq. 58). Sec. 4.2 tested both and chose *outside*.
    shiftInsidePseudoLoop: bool = field(default=False, metadata={'description': 'internal shifting (Eq. 60)'})
    #: The BDF correction of Eq. (59) that keeps the real-time derivative
    #: Lagrangian across the shift. Cheap; Sec. 4.2 found it makes little
    #: difference either way.
    bdfShiftCorrection: bool = field(default=True, metadata={'description': 'shift BDF correction (Eq. 59)'})

    #: See the module docstring: an input to `nu = alpha_nu h c0 / K` and to
    #: nothing else. `None` means the viscosity is configured directly.
    referenceSoundSpeedForViscosity: Optional[float] = field(
        default=None, metadata={'description': 'c0 used ONLY to fix nu (Part 2)'})
    #: `alpha_nu` in that same expression.
    alphaNu: float = field(default=0.01, metadata={'description': 'alpha_nu (Sec. 4)'})
    #: Kinematic viscosity, used directly when `referenceSoundSpeedForViscosity`
    #: is `None`. Exactly one of the two fixes `nu`.
    nu: float = field(default=1e-6, metadata={'description': 'kinematic viscosity (used when referenceSoundSpeedForViscosity is None)'})


def buildDefaultArtificialCompressibilityParams() -> ArtificialCompressibilityParams:
    return ArtificialCompressibilityParams()


def buildDefaultACSPHShiftProperties() -> ShiftProperties:
    """`buildDefaultShiftProperties()` (shared with WCSPH/incompressible)
    defaults to `active=True`, `scheme=deltaSPH` -- wrong for ACSPH on both
    counts: `deltaSPH` is Mach-scaled and ACSPH has no sound speed
    (`PST_ALE_PLAN.md` Part 1.2), and until `ArtificialCompressibleSystem.
    finalize` actually called `solveShifting`, `active=True` here was a
    silent no-op every existing case/test relied on. Now that it is wired in
    (`PST_ALE_PLAN.md` Stage A), this keeps shifting **off** by default (no
    behavior change for anything that does not opt in) but points `scheme`/
    `projectionScheme` at the one PST pair ACSPH can actually use, so opting
    in (`hydrostaticColumn.py`'s `ACSPH_PLAN.md` step 7) does not also
    require rediscovering which scheme is valid here."""
    props = buildDefaultShiftProperties()
    props.active = False
    props.scheme = ShiftingScheme.michel2022
    props.projectionScheme = ShiftingProjectionScheme.michel2022
    return props


@dataclass
class ArtificialCompressibleSPHConfig:
    fluid: fluidProperties = field(default_factory=buildDefaultFluidProperties, metadata={'description': 'Fluid properties (note: fixedSoundSpeed is NOT used by this scheme)'})

    acParams: ArtificialCompressibilityParams = field(
        default_factory=buildDefaultArtificialCompressibilityParams,
        metadata={'description': 'Artificial-compressibility / dual-time parameters'})

    adaptiveSupportScheme: AdaptiveSupportScheme = field(default=AdaptiveSupportScheme.NoScheme, metadata={'description': 'Adaptive support scheme to use'})
    adaptiveSupportIterations: int = field(default=1, metadata={'description': 'Number of iterations for adaptive support scheme'})
    adaptiveSupportThreshold: float = field(default=1e-3, metadata={'description': 'Threshold for adaptive support scheme'})
    adaptiveSupportCorrections: bool = field(default=True, metadata={'description': 'Whether to apply corrections in the adaptive support scheme (grad-H terms)'})

    viscositySwitchParams: ViscositySwitchConfig = field(default_factory=ViscositySwitchConfig)

    schemeName: str = field(default='Artificial Compressible SPH', metadata={'description': 'Name of the scheme'})

    boundaryConditions: List[BoundaryCondition] = field(default_factory=list, metadata={'description': 'List of boundary conditions to apply in the simulation'})

    #: Eq. (25) is the plain symmetric `(p_i + p_j)` gradient, which is this
    #: repo's (confusingly named) `nonConservative`. `Antuono` swaps in the
    #: antisymmetric form for negative-pressure non-surface pairs as a
    #: tensile-instability guard; the paper does not, so it is not the default.
    pressureForceTerm: PressureForceScheme = field(default=PressureForceScheme.nonConservative, metadata={'description': 'Pressure force term (Eq. 25 is nonConservative == (p_i+p_j))'})

    #: The repo's mDBC no-penetration correction, applied as an acceleration the
    #: way `deltaSPH_step` applies it. **A safeguard, not particle shifting** --
    #: the actual shift is a `finalize`-step displacement (Eq. 58, applied
    #: outside the pseudo-time loop; `ShiftProperties`/`PST_ALE_PLAN.md` Stage
    #: A). Off by default: it is not in the paper (Eq. 62 relies on the
    #: velocity mirror alone). Measured (`PST_ALE_PLAN.md` sec. 7.1) that the
    #: shift does *not* make this redundant -- it fixes interior particle
    #: pairing (completely: pairedFraction 0.065 -> 0.000), not the near-wall
    #: corner velocity blow-up this guards against; `hydrostaticColumn` with
    #: the Michel shift alone still peaks `|v|~2.3` vs. `~0.75` with this
    #: safeguard alone. Turn it on to see what a walled case
    #: does without either -- measured on `hydrostaticColumn` in
    #: `ACSPH_PLAN.md` step 5b.
    noPenetrationShift: bool = field(default=False, metadata={'description': 'mDBC no-penetration safeguard (off; not in the paper, and not the shift -- see ACSPH_PLAN.md step 7)'})

    dt_viscosityConstraint: bool = field(default=True, metadata={'description': 'Whether to apply the viscous constraint 0.125 h^2/nu in Eq. (46)'})
    dt_accelerationConstraint: bool = field(default=True, metadata={'description': 'Whether to apply the acceleration constraint in the timestep'})

    shiftProperties: ShiftProperties = field(default_factory=buildDefaultACSPHShiftProperties, metadata={'description': 'Particle-shifting properties (Michel et al. 2022 for this scheme; off by default, see buildDefaultACSPHShiftProperties)'})

    regions: List[ParticleRegion] = field(default_factory=list, metadata={'description': 'List of particle regions in the simulation'})
    rigidBodies: List[RigidBody] = field(default_factory=list, metadata={'description': 'List of rigid bodies in the simulation'})

    surfaceDetectionConfig: SurfaceDetectionConfig = field(default_factory=buildDefaultSurfaceDetectionConfig, metadata={'description': 'Configuration for surface detection module'})

    gravityConfig: gravityConfiguration = field(default_factory=buildDefaultGravityConfiguration, metadata={'description': 'Configuration for gravity module'})

    bandwith: float = field(default=10.0, metadata={'description': 'Bandwith for the divergence-free noise sampling module'})


def _acParamsToDict(p: ArtificialCompressibilityParams) -> Dict[str, Any]:
    return {
        'pressureSmoothing': p.pressureSmoothing.name,
        'cflTau': p.cflTau, 'cflT': p.cflT, 'dtOverDtau': p.dtOverDtau,
        'rkStages': p.rkStages,
        'epsilonV': p.epsilonV, 'epsilonS': p.epsilonS, 'uChar': p.uChar,
        'maxPseudoIterations': p.maxPseudoIterations,
        'minPseudoIterations': p.minPseudoIterations,
        'k2Factor': p.k2Factor, 'kappa2': p.kappa2, 'kappa4': p.kappa4,
        'jstUsePrintedMin': p.jstUsePrintedMin, 'k3': p.k3,
        'usePointImplicit': p.usePointImplicit,
        'useTildeVAdvection': p.useTildeVAdvection,
        'shiftInsidePseudoLoop': p.shiftInsidePseudoLoop,
        'bdfShiftCorrection': p.bdfShiftCorrection,
        'referenceSoundSpeedForViscosity': p.referenceSoundSpeedForViscosity,
        'alphaNu': p.alphaNu, 'nu': p.nu,
    }


def _dictToAcParams(d: Optional[Dict[str, Any]]) -> ArtificialCompressibilityParams:
    defaults = ArtificialCompressibilityParams()
    if not d:
        return defaults
    smoothing = d.get('pressureSmoothing', defaults.pressureSmoothing.name)
    return ArtificialCompressibilityParams(
        pressureSmoothing=PressureSmoothingScheme[smoothing] if isinstance(smoothing, str) else smoothing,
        cflTau=float(d.get('cflTau', defaults.cflTau)),
        cflT=float(d.get('cflT', defaults.cflT)),
        dtOverDtau=float(d.get('dtOverDtau', defaults.dtOverDtau)),
        rkStages=int(d.get('rkStages', defaults.rkStages)),
        epsilonV=float(d.get('epsilonV', defaults.epsilonV)),
        epsilonS=float(d.get('epsilonS', defaults.epsilonS)),
        uChar=None if d.get('uChar', defaults.uChar) is None else float(d['uChar']),
        maxPseudoIterations=int(d.get('maxPseudoIterations', defaults.maxPseudoIterations)),
        minPseudoIterations=int(d.get('minPseudoIterations', defaults.minPseudoIterations)),
        k2Factor=float(d.get('k2Factor', defaults.k2Factor)),
        kappa2=float(d.get('kappa2', defaults.kappa2)),
        kappa4=float(d.get('kappa4', defaults.kappa4)),
        jstUsePrintedMin=bool(d.get('jstUsePrintedMin', defaults.jstUsePrintedMin)),
        k3=float(d.get('k3', defaults.k3)),
        usePointImplicit=bool(d.get('usePointImplicit', defaults.usePointImplicit)),
        useTildeVAdvection=bool(d.get('useTildeVAdvection', defaults.useTildeVAdvection)),
        shiftInsidePseudoLoop=bool(d.get('shiftInsidePseudoLoop', defaults.shiftInsidePseudoLoop)),
        bdfShiftCorrection=bool(d.get('bdfShiftCorrection', defaults.bdfShiftCorrection)),
        referenceSoundSpeedForViscosity=(
            None if d.get('referenceSoundSpeedForViscosity', defaults.referenceSoundSpeedForViscosity) is None
            else float(d['referenceSoundSpeedForViscosity'])),
        alphaNu=float(d.get('alphaNu', defaults.alphaNu)),
        nu=float(d.get('nu', defaults.nu)),
    )


def artificialCompressibleConfigToDict(config: ArtificialCompressibleSPHConfig) -> Dict[str, Any]:
    return {
        'eosType': config.fluid.eosType.name,
        'restDensity': config.fluid.restDensity,
        'polytropicExponent': config.fluid.polytropicExponent,
        'kappa': config.fluid.kappa,
        'gas_constant': config.fluid.gas_constant,
        'molarMass': config.fluid.molarMass,
        'fixedSoundSpeed': config.fluid.fixedSoundSpeed if not isinstance(config.fluid.fixedSoundSpeed, torch.Tensor) else config.fluid.fixedSoundSpeed.detach().cpu().item(),

        'acParams': _acParamsToDict(config.acParams),

        'adaptiveSupportScheme': config.adaptiveSupportScheme.name,
        'adaptiveSupportIterations': config.adaptiveSupportIterations,
        'adaptiveSupportThreshold': config.adaptiveSupportThreshold,
        'adaptiveSupportCorrections': config.adaptiveSupportCorrections,
        'viscositySwitchParams': viscositySwitchConfigToDict(config.viscositySwitchParams),
        'schemeName': config.schemeName,
        'boundaryConditions': [boundaryConditionToDict(bc) for bc in config.boundaryConditions],
        'pressureForceTerm': config.pressureForceTerm.name,
        'noPenetrationShift': config.noPenetrationShift,
        'dt_viscosityConstraint': config.dt_viscosityConstraint,
        'dt_accelerationConstraint': config.dt_accelerationConstraint,
        'bandwith': config.bandwith,
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
    }


def dictToArtificialCompressibleConfig(configDict: Dict[str, Any]) -> ArtificialCompressibleSPHConfig:
    from ..enumTypes import EquationOfState
    from .moduleConfigurations.surfaceDetection import NormalSource, SurfaceDetectionScheme

    config = ArtificialCompressibleSPHConfig()
    config.fluid.eosType = EquationOfState[configDict['eosType']] if isinstance(configDict['eosType'], str) else configDict['eosType']
    config.fluid.restDensity = float(configDict['restDensity'])
    config.fluid.polytropicExponent = float(configDict['polytropicExponent'])
    config.fluid.kappa = float(configDict['kappa'])
    config.fluid.gas_constant = float(configDict['gas_constant'])
    config.fluid.molarMass = float(configDict['molarMass'])
    config.fluid.fixedSoundSpeed = float(configDict['fixedSoundSpeed'])

    config.acParams = _dictToAcParams(configDict.get('acParams'))

    config.adaptiveSupportScheme = AdaptiveSupportScheme[configDict['adaptiveSupportScheme']] if isinstance(configDict['adaptiveSupportScheme'], str) else configDict['adaptiveSupportScheme']
    config.adaptiveSupportIterations = int(configDict['adaptiveSupportIterations'])
    config.adaptiveSupportThreshold = float(configDict['adaptiveSupportThreshold'])
    config.adaptiveSupportCorrections = bool(configDict['adaptiveSupportCorrections'])
    config.viscositySwitchParams = dictToViscositySwitchConfig(configDict['viscositySwitchParams'])
    config.schemeName = configDict['schemeName']
    config.boundaryConditions = [dictToBoundaryCondition(bcDict) for bcDict in configDict['boundaryConditions']]
    config.pressureForceTerm = PressureForceScheme[configDict['pressureForceTerm']] if isinstance(configDict['pressureForceTerm'], str) else configDict['pressureForceTerm']
    config.noPenetrationShift = bool(configDict.get('noPenetrationShift', False))
    config.dt_viscosityConstraint = bool(configDict['dt_viscosityConstraint'])
    config.dt_accelerationConstraint = bool(configDict['dt_accelerationConstraint'])
    config.bandwith = float(configDict.get('bandwith', 10.0))

    shiftPropsDict = configDict.get('shiftProperties', {})
    defaultShift = buildDefaultACSPHShiftProperties()
    config.shiftProperties = ShiftProperties(
        iterations=int(shiftPropsDict.get('iterations', defaultShift.iterations)),
        CFL=float(shiftPropsDict.get('CFL', defaultShift.CFL)),
        computeMach=bool(shiftPropsDict.get('computeMach', defaultShift.computeMach)),
        maxC=float(shiftPropsDict.get('maxC', defaultShift.maxC)),
        active=bool(shiftPropsDict.get('active', defaultShift.active)),
        scheme=ShiftingScheme[shiftPropsDict['scheme']] if isinstance(shiftPropsDict.get('scheme'), str) else shiftPropsDict.get('scheme', defaultShift.scheme),
        projectionScheme=ShiftingProjectionScheme[shiftPropsDict['projectionScheme']] if isinstance(shiftPropsDict.get('projectionScheme'), str) else shiftPropsDict.get('projectionScheme', defaultShift.projectionScheme),
        summationDensity=bool(shiftPropsDict.get('summationDensity', defaultShift.summationDensity)),
        surfaceScaling=float(shiftPropsDict.get('surfaceScaling', defaultShift.surfaceScaling)),
        threshold=float(shiftPropsDict.get('threshold', defaultShift.threshold)),
        projectQuantities=bool(shiftPropsDict.get('projectQuantities', defaultShift.projectQuantities)),
    )

    surfaceConfigDict = configDict.get('surfaceDetectionConfig')
    if surfaceConfigDict is not None:
        defaults = buildDefaultSurfaceDetectionConfig()
        config.surfaceDetectionConfig = SurfaceDetectionConfig(
            active=surfaceConfigDict.get('active', defaults.active),
            colorFieldThreshold=float(surfaceConfigDict.get('colorFieldThreshold', defaults.colorFieldThreshold)),
            colorFieldGradThreshold=float(surfaceConfigDict.get('colorFieldGradThreshold', defaults.colorFieldGradThreshold)),
            barecascoThreshold=float(surfaceConfigDict.get('barecascoThreshold', defaults.barecascoThreshold)),
            expansionIterations=int(surfaceConfigDict.get('expansionIterations', defaults.expansionIterations)),
            scheme=SurfaceDetectionScheme[surfaceConfigDict['scheme']] if isinstance(surfaceConfigDict.get('scheme'), str) else surfaceConfigDict.get('scheme', defaults.scheme),
            normalSource=NormalSource[surfaceConfigDict['normalSource']] if isinstance(surfaceConfigDict.get('normalSource'), str) else surfaceConfigDict.get('normalSource', defaults.normalSource),
        )
    config.gravityConfig = dictToGravityConfiguration(configDict['gravityConfig']) if configDict.get('gravityConfig') is not None else buildDefaultGravityConfiguration()
    config.regions = [ParticleRegion.fromDict(regionDict) for regionDict in configDict.get('regions', [])]
    config.rigidBodies = [RigidBody.fromDict(bodyDict) for bodyDict in configDict.get('rigidBodies', [])]
    return config
