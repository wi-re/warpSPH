"""`DiffusionParameters` (a `wp.struct` of artificial-viscosity / thermal-
conductivity coefficients) and `ViscosityTerms` (which formulation each
coefficient feeds, e.g. Monaghan1992, Price2012_98), embedded as
`diffusionParams` on `CompressibleSPHConfig`/`CompSPHConfig`/`CRKSPHConfig` and
read by the compressible dissipation modules. Each of the three scheme configs
supplies its own `buildDefaultDiffusionParamsCompressibleSPH`/`CompSPH`/`CRKSPH`
default-factory with slightly different tuned values (e.g. `C_q`/`Cu_q` are 0
for the base compressible scheme's defaults here, but 1 or 2 for CompSPH/CRKSPH)
-- only this module's `buildDefaultDiffusionParamsCompressibleSPH` lives here;
the CompSPH/CRKSPH variants live in their own config files.
"""

__all__ = ['ViscosityTerms', 'BetaMode', 'VelocityPairPolicy', 'LimiterType', 'RiemannSolver', 'DiffusionParameters', 'resolveReconstructionLimiter', 'buildDefaultDiffusionParamsCompressibleSPH', 'diffusionParamsToDict', 'dictToDiffusionParams']

from typing import Dict, Any
from warpSPHCore import *
from dataclasses import dataclass, field
import warp as wp
from enum import Enum


class ViscosityTerms(Enum):
    Default = 0
    MonaghanGingold1983 = 1
    Cleary1998 = 2
    Monaghan1992 = 3
    Monaghan1997a = 4
    Monaghan1997b = 5
    Dukowicz = 6
    Price2012_98 = 7
    Price2012 = 8
    Price2008 = 9
    Wadsley2008 = 10
    DeltaSPH = 11
    RiemannDissipation = 13   # Pi = (p*(u_ij) - p*(0)) (1/rho_i^2 + 1/rho_j^2): the velocity-jump part of the star pressure of the pair's Riemann problem (AV_PLAN Phase 7b; solver = `DiffusionParameters.riemannSolver`)
    Frontiere2017 = 12   # CRKSPH's one-sided Q_i: Monaghan (1992) mu, own h, rho, c, alpha, eps^2 = 1e-2 (Frontiere et al. 2017)

class BetaMode(Enum):
    """How the quadratic coefficient `beta` (`C_q`) responds to the switched alpha.

    The two halves of AV_PLAN's `CoefficientPolicy`: `SwitchedAlphaCoupledBeta` and
    `SwitchedAlphaFixedBeta` (an un-switched alpha is `NoneSwitch`, alpha = 1, in
    either mode).

    * `Coupled` (default, the historical behaviour): `beta = alpha_bar * C_q`, i.e.
      beta follows the switch -- Chen & Nixon (2025)'s recommendation.
    * `Fixed`: `beta = C_q` whatever alpha is -- what Garcia-Senz & Cabezon (2026)
      (`beta = 2`), Sphenix (`beta_V = 3`) and Wadsley (`beta = 2`) specify, and what
      PHANTOM does. Before this existed that setting was inexpressible.

    `DiffusionParameters.scaleBeta` is a third, legacy behaviour (`beta = alpha_bar^2 *
    C_l * C_q`, a second alpha factor that matches no paper); it only applies in
    `Coupled` mode and is ignored in `Fixed`.
    """
    Coupled = 0
    Fixed = 1


class VelocityPairPolicy(Enum):
    """Which pair velocity difference `u_ij` the artificial viscosity sees (AV_PLAN Phase 3, `modules/reconstruction`).

    * `Raw` (default, the historical behaviour): `u_ij = v_i - v_j`.
    * `Linear`: both velocities extrapolated to the pair midpoint with the velocity Jacobian,
      `v'_i = v_i - 1/2 J_i x_ij`, `v'_j = v_j + 1/2 J_j x_ij` (Garcia-Senz & Cabezon 2026 Eqs. 11-12 with phi = 1).
      Annihilates any linear velocity field; unlimited, so not for shocks.
    * `Limited`: the same with the van Leer-like limiter `phi_ij` and the close-pair taper `kappa_ij`
      (Frontiere et al. 2017 Eqs. 51-53, Garcia-Senz & Cabezon 2026 Eqs. 13-17) -- CRKSPH's own reconstruction.
    * `BalsaraLimited`: `Limited` with `phi_ij` further scaled by `1 - Bbar_ij^p` (Garcia-Senz & Cabezon 2026
      Eqs. 18-19, `p = reconstructionBalsaraPower`): less reconstruction -- more dissipation -- where the flow is
      compressive (B -> 1), full reconstruction in shear (B -> 0). Their recommended AVSLRB2 is p = 2.

    Only the viscous force and its heating use it; the conductivity keeps the raw velocity.
    """
    Raw = 0
    Linear = 1
    Limited = 2
    BalsaraLimited = 3


class RiemannSolver(Enum):
    """Estimate of the star state `(p*, u*)` of an ideal-gas Riemann problem (`modules/riemann/solvers.py`, Toro 3rd ed.).

    * `Acoustic`: linearised with the impedances `rho_K a_K` (Godunov-SPH, Inutsuka 2002): exact for weak waves, linear in
      the velocity jump -- the Monaghan-1997a linear viscosity.
    * `PVRS`: the same with the mean impedance (Toro 9.20).
    * `TRRS`, `TSRS`: two-rarefaction / two-shock estimates (Toro 9.31 / 9.41); TSRS has the quadratic term of a shock.
    * `Adaptive`: PVRS for weak jumps, else TRRS / TSRS (Toro 9.5.2). Default.
    * `HLLC`: the star state of the HLLC wave-speed model (Toro 10.37).
    """
    Acoustic = 0
    PVRS = 1
    TRRS = 2
    TSRS = 3
    Adaptive = 4
    HLLC = 5


class LimiterType(Enum):
    """The slope limiter of the pair-velocity reconstruction (AV_PLAN Phase 7b.2, `modules/reconstruction/limiters.py`).

    The limiter is a function `psi(r)` of `r = min(r_i, r_j)`, with `r_i = g_i / g_j` and `r_j = g_j / g_i` the ratio of
    the two particles' extrapolated gradients along `x_ij` -- reciprocals, so `r` is at most 1 and a limiter is only ever
    evaluated on [0, 1] (`r <= 0`, a sign change of the gradient or no flow at all, gives `phi = 0` for every one of them).
    `psi(x) = psi(1 / x)` extends them to x > 1, and `psi(1) = 1` (a linear field is reconstructed in full) for all.

    * `VanLeerFrontiere` (default): `4 r / (1 + r)^2`, Frontiere et al. (2017) Eq. (52) / Garcia-Senz & Cabezon (2026)
      Eq. (13). Smooth at r = 1. What CRKSPH has always used; NOT the textbook van Leer limiter.
    * `Minmod`: `r`. The most diffusive TVD limiter.
    * `VanLeer`: `2 r / (1 + r)` (van Leer 1974, the textbook form), smooth.
    * `VanAlbada`: `r (1 + r) / (1 + r^2)` (van Albada et al. 1982), smooth.
    * `MC`: `min(2 r, (1 + r) / 2)` (van Leer 1977's monotonised central).
    * `Superbee`: `min(2 r, 1)` (Roe 1986), the least diffusive TVD limiter (the `max(.., min(r, 2))` branch is above 1).
    * `Ospre`: `3 r (1 + r) / (2 (1 + r + r^2))` (Waterson & Deconinck 1995), smooth.
    """
    VanLeerFrontiere = 0
    Minmod = 1
    VanLeer = 2
    VanAlbada = 3
    MC = 4
    Superbee = 5
    Ospre = 6


@wp.struct
class DiffusionParameters:
    c_s: scalar_t = field(default=scalar_t(1.0)) # Speed of sound, used in some formulations to compute the signal velocity
    C_l: scalar_t = field(default=scalar_t(1.0)) # Linear viscosity coefficient, also referred to as alpha in some formulations
    C_q: scalar_t = field(default=scalar_t(2.0)) # Quadratic viscosity coefficient, also referred to as beta in some formulations
    Cu_l: scalar_t = field(default=scalar_t(1.0)) # Linear thermal conductivity coefficient, also referred to as alpha_u in some formulations
    Cu_q: scalar_t = field(default=scalar_t(2.0)) # Quadratic thermal conductivity coefficient, also referred to as beta_u in some formulations
    
    K: scalar_t = field(default=scalar_t(1.0)) # Overall viscosity scaling factor
    thermalConductivity: scalar_t = field(default=scalar_t(0.5)) # Overall thermal conductivity scaling factor
    viscosityTerm: wp.int32 = field(default=ViscosityTerms.Price2012_98.value) # Viscosity formulation to use, e.g. Monaghan1992, Monaghan1997, Cleary1998 etc.
    thermalConductivityTerm: wp.int32 = field(default=ViscosityTerms.Price2012_98.value) # Thermal conductivity formulation to use, e.g. Monaghan1997 thermal conductivity term, Cleary1998 thermal conductivity term etc.
    betaMode: wp.int32 = field(default=BetaMode.Coupled.value) # `BetaMode`: Coupled (beta = alpha_bar * C_q, the default) or Fixed (beta = C_q, independent of the switched alpha).
    scaleBeta: wp.bool = field(default=False) # LEGACY, `Coupled` mode only: historically `beta = alpha_bar^2 * C_l * C_q` (a second alpha factor -- the comment here used to describe only the first half). Matches no paper; kept so stored configs reproduce.  # If true then the quadratic viscosity term is scaled by the linear viscosity term, as suggested in some papers to reduce excessive viscosity in certain scenarios. This is only relevant for formulations that use a quadratic term, such as Monaghan1992 and Monaghan1997.
    monaghanSwitch: wp.bool = field(default=True) # Whether to apply the Monaghan switch that turns off viscosity for diverging particles, i.e. particles that are moving away from each other. This is a common technique to reduce excessive viscosity in expanding flows and is used in many formulations such as Monaghan1992 and Monaghan1997.
    correctXi: wp.bool = field(default=True) # Divide the viscosity by the kernel-dependent length factor `sphKernel_xi` (packing ratio x kernel scale) so the coefficients are comparable across kernels. Unrelated to Cullen & Dehnen's limiter Xi (`ViscositySwitchConfig.limitXi`). The name is kept because it is serialised in stored configs.
    riemannSolver: wp.int32 = field(default=RiemannSolver.Adaptive.value) # `RiemannSolver` used by `ViscosityTerms.RiemannDissipation`
    riemannReconstruction: wp.bool = field(default=False) # `RiemannDissipation`: left / right states of rho and P are the limited midpoint extrapolations (GODUNOV_SPH_PLAN layer 1; needs the (rho, P) gradients, `reconstruction.computeStateGradients`), not the particle values
    limiterType: wp.int32 = field(default=LimiterType.VanLeerFrontiere.value) # `LimiterType`: the slope limiter of a `Limited` / `BalsaraLimited` pair velocity (VanLeerFrontiere = CRKSPH's own)
    velocityPairPolicy: wp.int32 = field(default=VelocityPairPolicy.Raw.value) # `VelocityPairPolicy`: the pair velocity the viscosity sees (Raw, Linear, Limited, BalsaraLimited)
    reconstructionEtaCrit: scalar_t = field(default=scalar_t(-1.0)) # `Limited` / `BalsaraLimited` close-pair taper eta_crit in units of r/H (H = support radius); <= 0: 1/n_h, resolved per step like CRKSPH's (`resolveReconstructionLimiter`)
    reconstructionEtaFold: scalar_t = field(default=scalar_t(-1.0)) # `Limited` taper width eta_fold, r/H units; <= 0: 0.2/n_h
    correctReconstructionGradient: wp.bool = field(default=True) # host-side: reconstruct with the M^-1-corrected velocity Jacobian (exact for linear fields) instead of the plain SPH gradient
    reconstructionBalsaraPower: scalar_t = field(default=scalar_t(2.0)) # `BalsaraLimited`: p in phi_ij (1 - Bbar_ij^p) (Garcia-Senz & Cabezon 2026 Eqs. 18-19)
    balsaraPairLimiter: wp.bool = field(default=False) # multiply the viscous Pi by Bbar_ij = (B_i + B_j)/2 (Balsara 1995 as a pair limiter, Garcia-Senz & Cabezon 2026 Eq. 7; with `C_q` coupled it is Sphenix's alpha_ij = alpha_bar Bbar, Borrow et al. 2022 Eq. 19)

    
def buildDefaultDiffusionParamsCompressibleSPH():
    """The Monaghan host's default (`CompressibleSPHConfig`; CompSPH / CRKSPH / WC have their own builders).

    Since 2026-10-08 (user, after the AV_PLAN Phases 3-6 sweep): `C_q = 2` with the coupled `BetaMode` (beta =
    2 alpha, Chen & Nixon 2025: the quadratic term is there in shocks -- with `C_q = 0` cold-gas shocks get no
    viscosity, Noh's post-shock density was 50 % off -- and switches off with the linear one in smooth flow) and
    the limited reconstruction of the pair velocity (`VelocityPairPolicy.Limited`), with the Rosswog (2020) switch
    as the scheme's default (`buildDefaultViscositySwitchConfigCompressibleSPH`). Before: `C_q = 0`, raw pair
    velocity, no switch -- `scripts/av_report.py` pins that for its M0-era columns."""
    diffusionParams = DiffusionParameters()
    diffusionParams.c_s = 1
    diffusionParams.C_l = 1
    diffusionParams.C_q = 2
    diffusionParams.Cu_l = 1
    diffusionParams.Cu_q = 0
    diffusionParams.K = 1.0
    diffusionParams.thermalConductivity = 0.5
    diffusionParams.viscosityTerm = ViscosityTerms.Price2012_98.value
    diffusionParams.thermalConductivityTerm = ViscosityTerms.Price2008.value
    diffusionParams.scaleBeta = False
    diffusionParams.monaghanSwitch = True
    diffusionParams.correctXi = True
    diffusionParams.betaMode = BetaMode.Coupled.value
    diffusionParams.velocityPairPolicy = VelocityPairPolicy.Limited.value

    return diffusionParams



def diffusionParamsToDict(diffusionParams: DiffusionParameters) -> Dict[str, Any]:
    return {
        'c_s': diffusionParams.c_s,
        'C_l': diffusionParams.C_l,
        'C_q': diffusionParams.C_q,
        'Cu_l': diffusionParams.Cu_l,
        'Cu_q': diffusionParams.Cu_q,
        'K': diffusionParams.K,
        'thermalConductivity': diffusionParams.thermalConductivity,
        'viscosityTerm': diffusionParams.viscosityTerm.name if isinstance(diffusionParams.viscosityTerm, ViscosityTerms) else diffusionParams.viscosityTerm,
        'thermalConductivityTerm': diffusionParams.thermalConductivityTerm.name if isinstance(diffusionParams.thermalConductivityTerm, ViscosityTerms) else diffusionParams.thermalConductivityTerm,
        'scaleBeta': diffusionParams.scaleBeta,
        'betaMode': BetaMode(diffusionParams.betaMode).name,
        'monaghanSwitch': diffusionParams.monaghanSwitch,
        'correctXi': diffusionParams.correctXi,
        'velocityPairPolicy': VelocityPairPolicy(diffusionParams.velocityPairPolicy).name,
        'limiterType': LimiterType(diffusionParams.limiterType).name,
        'riemannSolver': RiemannSolver(diffusionParams.riemannSolver).name,
        'riemannReconstruction': bool(diffusionParams.riemannReconstruction),
        # plain Python scalars: a scalar_t default reads back as a numpy float32, which json cannot write (the
        # export path dumps this dict)
        'reconstructionEtaCrit': float(diffusionParams.reconstructionEtaCrit),
        'reconstructionEtaFold': float(diffusionParams.reconstructionEtaFold),
        'correctReconstructionGradient': bool(diffusionParams.correctReconstructionGradient),
        'reconstructionBalsaraPower': float(diffusionParams.reconstructionBalsaraPower),
        'balsaraPairLimiter': bool(diffusionParams.balsaraPairLimiter),
    }

def dictToDiffusionParams(diffusionParamsDict: Dict[str, Any]) -> DiffusionParameters:
    diffusionParams = DiffusionParameters()
    diffusionParams.c_s = diffusionParamsDict['c_s']
    diffusionParams.C_l = diffusionParamsDict['C_l']
    diffusionParams.C_q = diffusionParamsDict['C_q']
    diffusionParams.Cu_l = diffusionParamsDict['Cu_l']
    diffusionParams.Cu_q = diffusionParamsDict['Cu_q']
    diffusionParams.K = diffusionParamsDict['K']
    diffusionParams.thermalConductivity = diffusionParamsDict['thermalConductivity']
    diffusionParams.viscosityTerm = ViscosityTerms[diffusionParamsDict['viscosityTerm']] if isinstance(diffusionParamsDict['viscosityTerm'], str) else diffusionParamsDict['viscosityTerm']
    diffusionParams.thermalConductivityTerm = ViscosityTerms[diffusionParamsDict['thermalConductivityTerm']] if isinstance(diffusionParamsDict['thermalConductivityTerm'], str) else diffusionParamsDict['thermalConductivityTerm']
    diffusionParams.scaleBeta = diffusionParamsDict['scaleBeta']
    # absent in configs stored before the mode existed -> the historical Coupled behaviour
    mode = diffusionParamsDict.get('betaMode', BetaMode.Coupled.name)
    diffusionParams.betaMode = (BetaMode[mode] if isinstance(mode, str) else BetaMode(mode)).value
    diffusionParams.monaghanSwitch = diffusionParamsDict['monaghanSwitch']
    diffusionParams.correctXi = diffusionParamsDict['correctXi']
    # absent in configs stored before Phase 3 -> the raw pair velocity
    policy = diffusionParamsDict.get('velocityPairPolicy', VelocityPairPolicy.Raw.name)
    diffusionParams.velocityPairPolicy = (VelocityPairPolicy[policy] if isinstance(policy, str) else VelocityPairPolicy(policy)).value
    # absent in configs stored before Phase 7b -> CRKSPH's own limiter
    limiter = diffusionParamsDict.get('limiterType', LimiterType.VanLeerFrontiere.name)
    diffusionParams.limiterType = (LimiterType[limiter] if isinstance(limiter, str) else LimiterType(limiter)).value
    solver = diffusionParamsDict.get('riemannSolver', RiemannSolver.Adaptive.name)
    diffusionParams.riemannSolver = (RiemannSolver[solver] if isinstance(solver, str) else RiemannSolver(solver)).value
    diffusionParams.riemannReconstruction = diffusionParamsDict.get('riemannReconstruction', False)
    diffusionParams.reconstructionEtaCrit = diffusionParamsDict.get('reconstructionEtaCrit', -1.0)
    diffusionParams.reconstructionEtaFold = diffusionParamsDict.get('reconstructionEtaFold', -1.0)
    diffusionParams.correctReconstructionGradient = diffusionParamsDict.get('correctReconstructionGradient', True)
    diffusionParams.reconstructionBalsaraPower = diffusionParamsDict.get('reconstructionBalsaraPower', 2.0)
    diffusionParams.balsaraPairLimiter = diffusionParamsDict.get('balsaraPairLimiter', False)

    return diffusionParams


def resolveReconstructionLimiter(diffusionParams: DiffusionParameters, n_h) -> DiffusionParameters:
    """`diffusionParams` with a non-positive `reconstructionEtaCrit` / `reconstructionEtaFold` replaced by the
    n_h-derived value (1/n_h, 0.2/n_h -- CRKSPH's `resolveCRKLimiter`). Returns `diffusionParams` itself when
    the policy is not a limited one or both values are explicit."""
    if (diffusionParams.velocityPairPolicy not in (VelocityPairPolicy.Limited.value, VelocityPairPolicy.BalsaraLimited.value)
            and not diffusionParams.riemannReconstruction):
        return diffusionParams
    if diffusionParams.reconstructionEtaCrit > 0 and diffusionParams.reconstructionEtaFold > 0:
        return diffusionParams
    out = dictToDiffusionParams(diffusionParamsToDict(diffusionParams))
    if out.reconstructionEtaCrit <= 0:
        out.reconstructionEtaCrit = 1.0 / float(n_h)
    if out.reconstructionEtaFold <= 0:
        out.reconstructionEtaFold = 0.2 / float(n_h)
    return out