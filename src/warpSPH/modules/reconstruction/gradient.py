"""The per-particle velocity Jacobian the reconstruction extrapolates with.

`computeVelocityJacobian` returns `J` with `J[i] @ dx` the velocity change along `dx` (`J_ab = dv_a/dx_b`), the
convention `linearPairVelocity` uses -- which is the layout `warpOperation`'s vector gradient already has
(`Vs[c, g] = dv_c/dx_g`, checked on `v = (y, 0)`). The difference gradient is optionally corrected by `M^-1`
(`shockCapturing.common.computeM`), which makes it exact for a linear field on any particle distribution -- the
premise of the linear-field annihilation test (AV_PLAN Phase 3). For `v = A x`, `Vs = A M`, so the correction acts
on the gradient index, `J = Vs M^-1` (`M^-1 Vs` = `M^-1 A M` keeps only the trace; `computeShearTensor` did that until
OPEN_PROBLEMS §22 was fixed). CRKSPH has its own CRK-corrected gradient and does not use this.
"""

from warpSPHCore import *
import torch
from typing import Optional, Union
from ...configurations.moduleConfigurations.diffusionParameters import VelocityPairPolicy, ViscosityTerms

__all__ = ['stateGradientInputs', 'computeStateGradients', 'needsStateGradients', 'stateGradientArguments', 'computeVelocityJacobian', 'velocityTensorArguments', 'needsJacobian', 'needsBalsara', 'balsaraFromJacobian',
           'reconstructionInputs', 'requireRawPairVelocity']


def computeVelocityJacobian(
        particleState,
        simulationConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None,
        corrected: bool = True):
    from ..shockCapturing.common import computeM   # the switch package imports the schemes' state types
    supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode
    Vs = warpOperation(
        particleState,
        OperationProperties(
            kernel = simulationConfig.kernel,
            operation = WarpOperation.Gradient,
            supportMode = supportMode,
            gradientMode = GradientScheme.Difference
        ),
        domain = simulationConfig.domain,
        adjacency = adjacency,
        queryValues = particleState.velocities
    )
    if corrected:
        # M = sum_j V_j x_ji (x) grad W, the sum the difference gradient is built from, so Vs M^-1 is exact for a
        # linear field
        M = computeM(particleState, simulationConfig, None, supportMode, adjacency)
        Vs = torch.einsum('ijk, ikl -> ijl', Vs, torch.linalg.pinv(M))
    return Vs.contiguous()


def computeStateGradients(
        particleState,
        simulationConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None,
        corrected: bool = True):
    """The `(n, 2, dim)` tensor of the density gradient (`[:, 0]`) and the pressure gradient (`[:, 1]`), `g . dx` the change of the
    field along `dx`: `reconstruction.reconstructPairRiemannStates` extrapolates the Riemann problem's left / right states with
    it (GODUNOV_SPH_PLAN layer 1). Same estimator as the velocity Jacobian: the difference gradient, optionally times `M^-1`
    (exact for a linear field on any particle distribution)."""
    from ..shockCapturing.common import computeM   # the switch package imports the schemes' state types
    supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode
    props = OperationProperties(kernel=simulationConfig.kernel, operation=WarpOperation.Gradient,
                                supportMode=supportMode, gradientMode=GradientScheme.Difference)
    grads = [warpOperation(particleState, props, domain=simulationConfig.domain, adjacency=adjacency, queryValues=values)
             for values in (particleState.densities, particleState.pressures)]
    G = torch.stack(grads, dim=1)
    if corrected:
        M = computeM(particleState, simulationConfig, None, supportMode, adjacency)
        G = torch.einsum('ipj, ijk -> ipk', G, torch.linalg.pinv(M))
    return G.contiguous()


def needsStateGradients(diffusionParams) -> bool:
    """Whether the viscosity kernels read the (rho, P) gradients: the Riemann dissipation with reconstructed states."""
    return bool(diffusionParams.riemannReconstruction) and diffusionParams.viscosityTerm == ViscosityTerms.RiemannDissipation.value


def stateGradientArguments(diffusionParams, queryStateGradients, referenceStateGradients, positions):
    """The `(rho, P)` gradient arrays a viscosity kernel is launched with: the given ones when it reads them (an error if
    missing), a cached placeholder otherwise."""
    dim = positions.shape[1]
    if needsStateGradients(diffusionParams):
        if queryStateGradients is None:
            raise ValueError('riemannReconstruction needs the (rho, P) gradients (queryStateGradients, see computeStateGradients)')
        return queryStateGradients, (referenceStateGradients if referenceStateGradients is not None else queryStateGradients)
    dummy = getCachedDummyTensor((1, 2, dim), dtype=positions.dtype, device=positions.device)
    return dummy, dummy


def needsJacobian(diffusionParams) -> bool:
    """Whether the viscosity kernels read the velocity Jacobian (a reconstructing pair-velocity policy)."""
    return diffusionParams.velocityPairPolicy != VelocityPairPolicy.Raw.value


def needsBalsara(diffusionParams) -> bool:
    """Whether the viscosity kernels read the per-particle Balsara factor: `BalsaraLimited` reconstruction or the
    Balsara pair limiter."""
    return (diffusionParams.velocityPairPolicy == VelocityPairPolicy.BalsaraLimited.value
            or bool(diffusionParams.balsaraPairLimiter))


def balsaraFromJacobian(J: torch.Tensor, soundspeeds: torch.Tensor, supports: torch.Tensor,
                        const: float = 1e-4) -> torch.Tensor:
    """The Balsara factor `|div v| / (|div v| + |curl v| + const c / h)` (Garcia-Senz & Cabezon 2026 Eq. 8, Sphenix
    Eq. 20; `shockCapturing.Balsara1995.balsaraFactor`, same `h` convention) from the Jacobian the reconstruction
    uses, so the corrected gradient feeds both."""
    from ..shockCapturing.Balsara1995 import balsaraFactor   # the switch package imports the schemes' state types
    div = torch.einsum('...ii->...', J)
    dim = J.shape[-1]
    if dim == 1:
        curl = torch.zeros_like(div)
    elif dim == 2:
        curl = (J[:, 1, 0] - J[:, 0, 1]).abs()
    else:
        curl = torch.stack([J[:, 2, 1] - J[:, 1, 2], J[:, 0, 2] - J[:, 2, 0], J[:, 1, 0] - J[:, 0, 1]], dim=-1).norm(dim=-1)
    return balsaraFactor(div, curl, soundspeeds, supports, const)


def velocityTensorArguments(diffusionParams, queryVelocityTensor, referenceVelocityTensor, positions,
                            queryBalsara=None, referenceBalsara=None):
    """The Jacobian and Balsara arrays a viscosity kernel is launched with: the given ones when the kernel reads them
    (an error if missing), a cached one-element placeholder otherwise (the kernels do not index it then)."""
    dim = positions.shape[1]
    if needsJacobian(diffusionParams):
        if queryVelocityTensor is None:
            raise ValueError(f'velocityPairPolicy {VelocityPairPolicy(diffusionParams.velocityPairPolicy).name} needs '
                             'the velocity Jacobian (queryVelocityTensor, see computeVelocityJacobian)')
        J_q = queryVelocityTensor
        J_r = referenceVelocityTensor if referenceVelocityTensor is not None else queryVelocityTensor
    else:
        J_q = J_r = getCachedDummyTensor((1, dim, dim), dtype=positions.dtype, device=positions.device)
    if needsBalsara(diffusionParams):
        if queryBalsara is None:
            raise ValueError('BalsaraLimited / balsaraPairLimiter need the per-particle Balsara factor '
                             '(queryBalsara, see balsaraFromJacobian)')
        B_q = queryBalsara
        B_r = referenceBalsara if referenceBalsara is not None else queryBalsara
    else:
        B_q = B_r = getCachedDummyTensor((1,), dtype=positions.dtype, device=positions.device)
    return J_q, J_r, B_q, B_r


def reconstructionInputs(particleState, simulationConfig, diffusionParams, adjacency,
                         supportScheme: SupportScheme = SupportScheme.SuperSymmetric):
    """What the Monaghan viscosity kernels need besides the state, as `schemes/monaghan.py` evaluates it:
    `(params, J, B)` -- the params with the n_h-derived limiter constants resolved, the velocity Jacobian (None if not
    read) and the Balsara factor (None if not read). One place, so the scheme and the diagnostics that re-evaluate
    the operator (`dissipation.avPower`, `scripts/probe_monaghanEnergy.py`) see the same thing."""
    from ...configurations.moduleConfigurations.diffusionParameters import resolveReconstructionLimiter
    J = B = None
    if not (needsJacobian(diffusionParams) or needsBalsara(diffusionParams)):
        return diffusionParams, J, B
    params = resolveReconstructionLimiter(diffusionParams, simulationConfig.n_h)
    J = computeVelocityJacobian(particleState, simulationConfig, supportScheme, adjacency,
                                corrected=params.correctReconstructionGradient)
    if needsBalsara(params):
        B = balsaraFromJacobian(J, particleState.soundspeeds, particleState.supports)
    return params, (J if needsJacobian(params) else None), B


def stateGradientInputs(particleState, simulationConfig, diffusionParams, adjacency,
                        supportScheme: SupportScheme = SupportScheme.SuperSymmetric):
    """`(params, G)` for the Riemann dissipation's reconstructed states: the params with the n_h-derived taper constants resolved
    and the `(n, 2, dim)` (rho, P) gradients, or `(diffusionParams, None)` when not read. Called after `reconstructionInputs`
    (resolving twice is a no-op)."""
    from ...configurations.moduleConfigurations.diffusionParameters import resolveReconstructionLimiter
    if not needsStateGradients(diffusionParams):
        return diffusionParams, None
    params = resolveReconstructionLimiter(diffusionParams, simulationConfig.n_h)
    G = computeStateGradients(particleState, simulationConfig, supportScheme, adjacency,
                              corrected=params.correctReconstructionGradient)
    return params, G


def requireRawPairVelocity(diffusionParams, scheme: str) -> None:
    """For hosts whose viscosity kernels do not implement the pair-velocity policies or the Balsara pair limiter
    (CompSPH; CRKSPH, which has its own reconstruction): fail loudly instead of silently ignoring the setting."""
    if needsJacobian(diffusionParams) or needsBalsara(diffusionParams):
        raise NotImplementedError(
            f'{scheme}: velocityPairPolicy={VelocityPairPolicy(diffusionParams.velocityPairPolicy).name}, '
            f'balsaraPairLimiter={bool(diffusionParams.balsaraPairLimiter)} are only implemented on the Monaghan host '
            '(modules/dissipation/wp_diffusion.py, wp_dissipation.py; AV_PLAN Phases 3-5B)')
