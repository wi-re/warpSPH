"""Phase 4 step 4 of `warpSPHCore/warpier_forward_mode_plan.md`: "swap the
matvec, keep the solver". Same Krylov-solver call as
`implicitShifting.computeImplicitShift` (selected by
`ShiftingImplicitSolver`, BiCGStab by default), same relaxation/
boundary/initializer handling,
but the Newton matvec `Hess(C) @ x` comes from `warpSPHCore.warpOperationHVP`
(Phase 4 step 3's composed-JVP-of-a-JVP) instead of `H`/
`_multiplyLaplacianBlock`'s hand-assembled per-pair kernel Hessian +
`torch.einsum`. `grad C` (the RHS `B`) is likewise `warpOperationJVP`
(Phase 4 step 2), not `wp_implicitShifting.computeShiftingPairTerms`'s `J`.

No `sphKernelHessian`/`sphKernelGradient` call, no per-pair `scatter_sum`
assembly, no hand-derived block-symmetry sign -- every SPH-specific building
block below is a call into `warpSPHCore`'s Tier-2 bridge (Density's position
JVP/HVP, `omega` standing in for mass). The one piece of Newton-solver
machinery this file still writes itself is what `warpOperationJVP`/HVP
*don't* provide: which particles are boundary (zeroed rows/RHS) and what the
Jacobi preconditioner's diagonal is -- built the same way `_buildSystem`
does, just sourced from `warpOperationHVP` called with each coordinate basis
vector (`tangentReferencePositions=0`, isolating `diagBlock_i = sum_{j != i}
omega_j H_ij` from the general `HVP_i = sum_{j != i} omega_j H_ij @ (v_i -
v_j)` formula) instead of assembling it from a dense per-pair `H` tensor. The
self-pair exclusion in both is an exact translation-invariance identity, not
a numerical-safety measure -- see `wp_densityHVP.py`'s docstring
(`warpSPHCore`).
"""

from typing import Any, Optional, Tuple
import torch
from warpSPHCore.profiling import record_function
from warpSPHCore import ParticleState, ParticleTangentState, SupportScheme, OperationProperties, warpOperationJVP, warpOperationHVP
from warpSPHCore.enumTypes import WarpOperation, OperationDirection

from warpSPH.configurations.simulationConfig import SimulationConfig
from ...configurations.moduleConfigurations.shifting import ShiftingImplicitInitializer, ShiftingImplicitSolver, ShiftingImplicitFallback, ShiftingImplicitPreconditioner

from .solverDriver import solveImplicitSystem
from .delta import computeDeltaShift
from .preconditioner import buildScalarJacobiPrecond, buildBlockJacobiPrecond

__all__ = ['computeImplicitShiftAutomatic']


def computeImplicitShiftAutomatic(
    currentState: Any,
    config: SimulationConfig,
    schemeConfig: Any,
    domain: Any,
    adjacency: Any,
    iters: int = -1,
    fallback_override: Optional[ShiftingImplicitFallback] = None,
):
    """Drop-in replacement for `implicitShifting.computeImplicitShift`, same
    signature and return value (`(delta, adjacency)`), Newton matvec/RHS
    sourced entirely from `warpOperationJVP`/`warpOperationHVP` instead of a
    hand-written per-pair kernel Hessian.
    """
    with record_function("[warpSPH] - (shift) - automatic"):
        numParticles = currentState.positions.shape[0]
        dim = currentState.positions.shape[1]
        device, dtype = currentState.positions.device, currentState.positions.dtype
        rho0 = schemeConfig.fluid.restDensity

        if schemeConfig.shiftProperties.summationDensity:
            omega = currentState.masses / currentState.densities
        else:
            omega = currentState.masses / rho0

        omegaState = ParticleState(positions=currentState.positions, supports=currentState.supports,
                                   masses=omega, kinds=currentState.kinds, densities=None)
        props = OperationProperties(kernel=config.kernel, operation=WarpOperation.Density,
                                    supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)

        zero = torch.zeros(numParticles, dim, device=device, dtype=dtype)
        zeroSupports = torch.zeros(numParticles, device=device, dtype=dtype)
        basisVectors = torch.eye(dim, device=device, dtype=dtype)

        # grad C -- Phase 4 step 2: one warpOperationJVP call per coordinate.
        # warpOperationJVP's geometry-tangent path takes a bundled
        # ParticleTangentState (positions/supports/masses) rather than the
        # loose tangentQueryPositions/tangentReferencePositions kwargs this
        # call used before the Tier-2 correction-JVP work replaced them
        # (warpier_tier2_correction_jvp_plan.md phase a1); a zero supports
        # tangent and masses=None isolate the pure position-JVP this needs,
        # matching the rest of this codebase's own gradcheck scripts.
        Jw = torch.stack([
            warpOperationJVP(omegaState, props, domain, adjacency=adjacency,
                             queryTangentState=ParticleTangentState(
                                 positions=basisVectors[d].expand(numParticles, dim),
                                 supports=zeroSupports, masses=None))
            for d in range(dim)
        ], dim=1)
        B = Jw.flatten().clone()

        # diag(Hess C) -- Phase 4 step 3, isolated the same way _buildSystem's
        # diagBlock is: query-side tangent = basis vector, reference-side
        # tangent = 0, so only the "own position" second-derivative term
        # survives (HVP_i = sum_j omega_j H_ij @ (e_d - 0) = diagBlock_i @ e_d).
        diagBlock = torch.stack([
            warpOperationHVP(omegaState, props, domain, adjacency=adjacency,
                             tangentQueryPositions=basisVectors[d].expand(numParticles, dim),
                             tangentReferencePositions=zero)
            for d in range(dim)
        ], dim=2)

        # opt-in Tikhonov lift of the operator's near-null-space eigenvalues
        # (A -> A + lift*I); see computeImplicitShift for the rationale.
        # 0.0 (default) = off.
        nullLift = schemeConfig.shiftProperties.implicitNullSpaceLift
        if nullLift > 0:
            diagBlock = diagBlock + nullLift * torch.eye(dim, device=device, dtype=dtype).unsqueeze(0)

        initializer = schemeConfig.shiftProperties.implicitInitializer
        if initializer in (ShiftingImplicitInitializer.deltaPlus, ShiftingImplicitInitializer.deltaMinus):
            delta, _ = computeDeltaShift(currentState, config, schemeConfig, domain, adjacency, iters=1)
            sign = -0.5 if initializer == ShiftingImplicitInitializer.deltaPlus else 0.5
            x0 = (delta.flatten() * sign).clone()
        else:
            x0 = torch.zeros(numParticles * dim, device=device, dtype=dtype)

        boundary = currentState.kinds != 0
        hasBoundary = torch.any(boundary)
        if hasBoundary:
            B.view(numParticles, dim)[boundary] = 0
            x0.view(numParticles, dim)[boundary] = 0

        precondMode = schemeConfig.shiftProperties.implicitPreconditioner
        if precondMode == ShiftingImplicitPreconditioner.off:
            precond = None
        elif precondMode == ShiftingImplicitPreconditioner.block:
            precond = buildBlockJacobiPrecond(diagBlock)
        else:
            precond = buildScalarJacobiPrecond(diagBlock)

        def matvec(x: torch.Tensor) -> torch.Tensor:
            v = x.view(numParticles, dim)
            Hv = warpOperationHVP(omegaState, props, domain, adjacency=adjacency,
                                  tangentQueryPositions=v, tangentReferencePositions=v)
            if hasBoundary:
                # Matches computeImplicitShift's activeMask (dropping pairs
                # whose query particle is boundary before scattering): zeroing
                # those output rows post-hoc has the same effect, since that
                # mask only ever depended on the query index.
                Hv = Hv.clone()
                Hv[boundary] = 0
            return Hv.flatten()

        dx = config.dx.cpu().item() if isinstance(config.dx, torch.Tensor) else config.dx
        threshold = schemeConfig.shiftProperties.implicitSolverThreshold
        if threshold is None:
            threshold = dx / 2

        solverArgs = dict(
            tol=schemeConfig.shiftProperties.implicitTolerance,
            rtol=schemeConfig.shiftProperties.implicitRelativeTolerance,
            maxiter=schemeConfig.shiftProperties.implicitMaxSolverIter,
            precond=precond,
            threshold=threshold,
            dim=dim,
        )
        fallback = (fallback_override if fallback_override is not None
                    else schemeConfig.shiftProperties.implicitFallback)
        xk, solverIters, convergence = solveImplicitSystem(
            matvec, B, x0, solverArgs,
            primary_solver=schemeConfig.shiftProperties.implicitSolver,
            restart=schemeConfig.shiftProperties.implicitRestart,
            fallback=fallback)

        update = -xk.view(numParticles, dim) * schemeConfig.shiftProperties.implicitRelaxation
        return update, adjacency
