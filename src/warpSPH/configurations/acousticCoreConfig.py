"""Configuration for the rudimentary WCSPH acoustic core (`JFNK_PLAN.md`
Phase B): which kernel/operator modes `f_acoustic_core`
(`schemes/experimental/acousticCore.py`) uses for its continuity divergence and pressure
gradient, plus the (isothermal) equation-of-state constants.

Deliberately as small as `WaveEquationConfig` -- no diffusion, no surface
detection, no shifting, no boundary conditions -- because the scheme this
feeds has none of those code paths either; see the scheme module's docstring
for why.

`forcingAmplitude`/`forcingWavenumber` (`JFNK_PLAN.md` Phase E1) add the one
piece of `cases/periodic/kolmogorov.py`'s physics this core can express with zero new
machinery: a steady sinusoidal body acceleration, `dv_x/dt += forcingAmplitude
* sin(forcingWavenumber * pi * y)`. `forcingAmplitude=0.0` (the default)
recovers Phase B's unforced core exactly -- opt-in, like `JFNKSolver` itself.
Unlike the real case, this is an acceleration directly, not a per-particle
force divided by mass elsewhere (`schemes/weaklyCompressible/deltaSPH.py`'s
`dvdt_forcing = forcing / currentState.masses`) -- same net physics, one
fewer indirection since there's no `BoundaryCondition`/`forcingFunctions`
machinery here to route through. Also omits that case's symmetry-breaking
y-noise term -- not needed to test JFNK's Newton iteration against a
*driven*, non-decaying flow, which is the point of this rung.

`densityDiffusionCoefficient`/`velocityDiffusionCoefficient`/`laplacianMode`/
`laplacianGradientMode` (`JFNK_PLAN.md` Phase E1's own finding: the
Kolmogorov base flow is linearly unstable, so zero-dissipation JFNK
eventually diverges too, just later than explicit/Picard -- dissipation is
physically load-bearing here, unlike Phase B's vortex) add Laplacian-based
dissipation with, again, zero new `warpSPHCore` derivation: `Laplacian` is
already one of the six value+geometry JVP-wrapped operators. **One real
finding building this, fixed at the root rather than routed around**: the
exact-JVP path first crashed with a kernel-argument dtype mismatch --
`wp_laplacianJVP.py`'s `Brookshaw`/`Naive` geometry-JVP kernels were
scalar-field-only (fixed `scalar_t` arguments), so neither could
differentiate `Laplacian(velocities)`, a vector field. Fixed 2026-08-25 in
`warpSPHCore` by generalizing both to generic `Any`-typed fields, the same
typing `Dot`/`Default` already used -- `q_ij = (fj-fi)*B_ij` and every
downstream op are elementwise scalar-times-field regardless of whether the
field is scalar or vector, so the fix was confined to kernel signatures and
the public wrapper's zero-tangent-default/`outputDtype` plumbing, no formula
change (gradchecked against a reverse-mode-Jacobian reference on a vector
field, `warpSPHCore/tests/operations/test_forward_mode_geometry_jvp_laplacian_
brookshaw.py`'s/`..._naive.py`'s new `*_vectorField_2d` tests). `Brookshaw` is
therefore usable again as the default here, matching `OperationProperties`'
own library-wide default and the classic Brookshaw/Morris viscosity
estimator's name. `Dot` remains unusable for the shared `densities` term
(it rejects a scalar field in >1D, `wp_laplacian.py`'s own `ValueError`);
`Default` is the other scheme both terms could share, kept only as an
explicit alternative now, not a workaround. Both coefficients default to
`0.0` -- opt-in, recovering Phase B/E1 exactly, same pattern as
`forcingAmplitude`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from warpSPHCore import GradientScheme, KernelFunctions, LaplacianScheme, SupportScheme

__all__ = ['AcousticCoreConfig', 'acousticCoreConfigToDict', 'dictToAcousticCoreConfig']


@dataclass
class AcousticCoreConfig:
    kernel: KernelFunctions = field(default=KernelFunctions.Wendland2)
    supportMode: SupportScheme = field(default=SupportScheme.SuperSymmetric)
    #: Coefficient scheme for the continuity divergence, matching
    #: `modules/momentum/inconsistent.py`'s `computeMomentum` exactly.
    continuityGradientMode: GradientScheme = field(default=GradientScheme.Difference)
    #: Coefficient scheme for the pressure-gradient force, matching
    #: `modules/pressure/symmetricForce.py`'s `computePressureForceSymmetric`.
    pressureGradientMode: GradientScheme = field(default=GradientScheme.Symmetric)

    #: Isothermal EOS: `p = soundSpeed**2 * (rho - restDensity)`
    #: (`modules/eos/weaklyCompressible.py`'s `isoThermalEOS`).
    restDensity: float = field(default=1.0)
    soundSpeed: float = field(default=10.0)

    #: Kolmogorov-flow body acceleration (`JFNK_PLAN.md` Phase E1), zero
    #: (off) by default. `forcingWavenumber=4` matches `cases/periodic/kolmogorov.py`'s
    #: own default `k`.
    forcingAmplitude: float = field(default=0.0)
    forcingWavenumber: float = field(default=4.0)

    #: Density-diffusion coefficient (`JFNK_PLAN.md` Phase E1.5), the same
    #: dimensionless `delta` `modules/deltaSPH/densityDiffusion.py` uses,
    #: applied here as a plain Fickian `delta * h * soundSpeed *
    #: Laplacian(densities)` term on `drho/dt` -- the real scheme's own
    #: renormalized-gradient/multi-scheme flux collapsed to the one Laplacian
    #: call this core needs no new derivation for. `0.0` (off) by default.
    densityDiffusionCoefficient: float = field(default=0.0)
    #: Velocity-diffusion (viscosity) coefficient, a kinematic viscosity `nu`
    #: applied as `nu * Laplacian(velocities)` on `dv/dt` -- the classic
    #: Brookshaw/Morris SPH-viscosity Laplacian. `0.0` (off) by default.
    velocityDiffusionCoefficient: float = field(default=0.0)
    #: Laplacian scheme shared by both dissipation terms above.
    #: `OperationProperties`' own library-wide default, usable for both the
    #: scalar `densities` term and the vector `velocities` term since
    #: `warpSPHCore`'s Brookshaw geometry-JVP fix (see the module docstring's
    #: finding) -- `Dot` remains unusable here regardless (it rejects a
    #: scalar field in >1D); `Default` is an available alternative, not a
    #: requirement.
    laplacianMode: LaplacianScheme = field(default=LaplacianScheme.Brookshaw)
    #: `q_ij` weighting for the Laplacian estimator above. `Difference`
    #: matches `continuityGradientMode` and is the scheme
    #: `coreOperations/wp_laplacian.py`'s own docstring calls out as
    #: satisfying the constant-field-cancellation constraint the Laplacian
    #: estimator needs by construction (`Naive`/`Summation` reduce to it;
    #: `Symmetric` is a distinct, also-consistent density-weighted variant).
    laplacianGradientMode: GradientScheme = field(default=GradientScheme.Difference)


def acousticCoreConfigToDict(config: AcousticCoreConfig) -> Dict[str, Any]:
    return {
        'kernel': config.kernel.name,
        'supportMode': config.supportMode.name,
        'continuityGradientMode': config.continuityGradientMode.name,
        'pressureGradientMode': config.pressureGradientMode.name,
        'restDensity': config.restDensity,
        'soundSpeed': config.soundSpeed,
        'forcingAmplitude': config.forcingAmplitude,
        'forcingWavenumber': config.forcingWavenumber,
        'densityDiffusionCoefficient': config.densityDiffusionCoefficient,
        'velocityDiffusionCoefficient': config.velocityDiffusionCoefficient,
        'laplacianMode': config.laplacianMode.name,
        'laplacianGradientMode': config.laplacianGradientMode.name,
    }


def dictToAcousticCoreConfig(configDict: Dict[str, Any]) -> AcousticCoreConfig:
    config = AcousticCoreConfig()
    config.kernel = KernelFunctions[configDict['kernel']] if isinstance(configDict['kernel'], str) else configDict['kernel']
    config.supportMode = SupportScheme[configDict['supportMode']] if isinstance(configDict['supportMode'], str) else configDict['supportMode']
    config.continuityGradientMode = GradientScheme[configDict['continuityGradientMode']] if isinstance(configDict['continuityGradientMode'], str) else configDict['continuityGradientMode']
    config.pressureGradientMode = GradientScheme[configDict['pressureGradientMode']] if isinstance(configDict['pressureGradientMode'], str) else configDict['pressureGradientMode']
    config.restDensity = float(configDict['restDensity'])
    config.soundSpeed = float(configDict['soundSpeed'])
    config.forcingAmplitude = float(configDict.get('forcingAmplitude', 0.0))
    config.forcingWavenumber = float(configDict.get('forcingWavenumber', 4.0))
    config.densityDiffusionCoefficient = float(configDict.get('densityDiffusionCoefficient', 0.0))
    config.velocityDiffusionCoefficient = float(configDict.get('velocityDiffusionCoefficient', 0.0))
    laplacianMode = configDict.get('laplacianMode', LaplacianScheme.Brookshaw)
    config.laplacianMode = LaplacianScheme[laplacianMode] if isinstance(laplacianMode, str) else laplacianMode
    laplacianGradientMode = configDict.get('laplacianGradientMode', GradientScheme.Difference)
    config.laplacianGradientMode = GradientScheme[laplacianGradientMode] if isinstance(laplacianGradientMode, str) else laplacianGradientMode
    return config
