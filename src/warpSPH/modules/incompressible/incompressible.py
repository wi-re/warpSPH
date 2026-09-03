"""IISPH constant-density pressure solver: relaxed-Jacobi iteration that
solves for a pressure field driving the predicted density back to
`schemeConfig.fluid.restDensity` (source term `rho0 - rhoStar`, the
density-error formulation, as opposed to `divergenceFree.py`'s divergence
formulation).

Each iteration computes the pressure acceleration (`computePressureAccelIISPH`,
scatter mode), its position-drift residual (`dt**2 * computePressureShiftIISPH`),
and updates `pressure += omega * residual / alpha` (`dt**2 * computeAlpha`'s
IISPH diagonal term), keeping pressures non-negative each step -- either by
clamping at zero (`ShiftPressureGauge.nonNegativeClamp`, the historical
default) or by subtracting the fluid minimum (`ShiftPressureGauge.minShift`,
which additionally pins the operator's constant null-space mode; see that
enum's docstring, since the clamp alone lets that mode drift to a run-ending
magnitude).
`rhoStar` is clamped to a minimum of 0.9 and `alpha` to a maximum of -1e-6 to
avoid division blow-up. Iterates between `solverConfig.pressureSolver.
{minIterations,maxIterations,tolerance,relaxationFactor}`, stopping early once
past `minIterations` and below `tolerance`.
"""

from warpSPHCore import *
import torch
from warpSPH.systems import *
from warpSPH.modules import *
from warpSPH.configurations import SimulationConfig
from typing import Optional, Union, Any



from warpSPH.configurations import SimulationConfig
from typing import Optional


from typing import Any, Optional, Union
from .wp_alpha import computeAlpha
from ..momentum.incompressible import computeMomentumIncompressible
from ..pressure.iisph import computePressureAccelIISPH
from .drift import computePressureShiftIISPH
from ...configurations import PressureSolverType, JacobiRelaxationMode, ShiftPressureGauge, BoundaryOperatorTerms, resolveBoundaryOperatorTerms
from .convergence import evaluateResidual, sourceNorm
from .krylov import solvePressureKrylov
from .consistent import applyConsistentCoupling
from .wallPressure import wallPressureExtrapolation
from ...configurations import BoundaryPressureMode

__all__ = ['solveIncompressible']

#: EXPERIMENT (DFSPH_IMPROVEMENT_PLAN.md "Immediate A1", Part 48) -- the VD+PS
#: shift's constant-density solve on a fully wall-closed box (`randomFlow
#: Incompressible --bounded`) is a pure-Neumann system: `A.1 ~ 0`, so the
#: constant (n_h=4 lattice-bias) component of the `1 - rhoStar/rho0` source has
#: no solution and `p` ramps until the run detonates. `'auto'` subtracts the
#: fluid-mean source only when it is mean-dominated (the closed-box signature,
#: Part 42's `omniIncompressible.CD_SOURCE_PROJECT`); `True`/`False` force it.
#: Default `False` == byte-identical to the historical solve.
_CLOSED_BOX_SOURCE_PROJECT = False
_CLOSED_BOX_PROJECT_THRESHOLD = 0.7

#: EXPERIMENT (same) -- recompute the `kind==1` wall pressure from the current
#: fluid iterate every Jacobi sweep (Part 41's `wallPressureExtrapolation`,
#: no relaxation / no carried state), instead of freezing `boundaryPressure`
#: at its pre-solve snapshot, so the near-wall iteration matrix is consistent.
#: `None` (default) == the historical frozen-boundary solve; `'shepard'` /
#: `'mls'` pick the extrapolation order.
_SHIFT_WALL_PRESSURE = None


def _solveIncompressibleImpl(

        particles: Any,
        config: SimulationConfig,
        schemeConfig: Any,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]],
        dvdt: torch.Tensor,
        dt : float,
        verbose: bool = False,
        warmStartPressure: Optional[torch.Tensor] = None,
):
        minIters = schemeConfig.solverConfig.pressureSolver.minIterations
        maxIters = schemeConfig.solverConfig.pressureSolver.maxIterations
        threshold = schemeConfig.solverConfig.pressureSolver.tolerance
        omega = schemeConfig.solverConfig.pressureSolver.relaxationFactor

        predictedVelocities = particles.velocities + dt * dvdt
        # dt = config.dt

        print(f'Predicted velocities: mean: {predictedVelocities.mean().cpu().item():.6g}, min: {predictedVelocities.min().cpu().item():.6g}, max: {predictedVelocities.max().cpu().item():.6g}')

        print(f'Particles masses: mean: {particles.masses.mean().cpu().item():.6g}, min: {particles.masses.min().cpu().item():.6g}, max: {particles.masses.max().cpu().item():.6g}')
        print(f'Particles density: mean: {particles.densities.mean().cpu().item():.6g}, min: {particles.densities.min().cpu().item():.6g}, max: {particles.densities.max().cpu().item():.6g}')
        apparentArea = particles.masses / particles.densities    

        print(f'Apparent area: {apparentArea.mean().cpu().item():.6g}, min: {apparentArea.min().cpu().item():.6g}, max: {apparentArea.max().cpu().item():.6g}')

        divergence = computeMomentumIncompressible(
                currentState = particles, 
                config = config, 
                schemeConfig = schemeConfig, 
                adjacency = adjacency, 
                advectionVelocities = predictedVelocities
        )

        rho0 = schemeConfig.fluid.restDensity
        rhoStar = particles.densities + dt * divergence

        # rhoStar = torch.clamp(rhoStar, min = 0.9)  # Clamp to avoid extreme density values

        # sourceTerm = config.dt * divergence
        sourceTerm = (1 - rhoStar/rho0)# / config.dt
        print(f'[IS] Source term: {sourceTerm.mean().cpu().item():.6g}, min: {sourceTerm.min().cpu().item():.6g}, max: {sourceTerm.max().cpu().item():.6g} abs mean: {sourceTerm.abs().mean().cpu().item():.6g}')

        # sourceTerm = sourceTerm - sourceTerm.mean()  # Remove mean to ensure zero-mean source term
        _fluidRows = particles.kinds == 0
        if _CLOSED_BOX_SOURCE_PROJECT and bool(_fluidRows.any()):
                _sf = sourceTerm[_fluidRows]
                _mean = _sf.mean()
                _frac = 1.0 - (_sf - _mean).norm() / (_sf.norm() + 1e-30)
                if (_CLOSED_BOX_SOURCE_PROJECT is True
                                or _frac > _CLOSED_BOX_PROJECT_THRESHOLD):
                        sourceTerm = sourceTerm - _mean
        if verbose:
            print(f'Incompressible Solver')
            print(f'[IS] Source term: {sourceTerm.mean().cpu().item():.6g}, min: {sourceTerm.min().cpu().item():.6g}, max: {sourceTerm.max().cpu().item():.6g} abs mean: {sourceTerm.abs().mean().cpu().item():.6g}')
            print(f'[IS] Mean density error: {(particles.densities - schemeConfig.fluid.restDensity).abs().mean().cpu().item():.6g}')

        # Opt-in Krylov pressure solvers (BiCGStab/GMRES/CG/BiCG/MINRES) share the same
        # matrix-free operator and IISPH-diagonal preconditioner as the relaxed
        # Jacobi path below, which stays the byte-identical default
        # (solverType == relaxedJacobi). The constant-density variant scales the
        # operator by dt**2 and clamps the pressure non-negative (gauge='nonnegative').
        # psSolver = schemeConfig.solverConfig.pressureSolver
        # if psSolver.solverType != PressureSolverType.relaxedJacobi:
        #     return solvePressureKrylov(
        #         particles, config, schemeConfig, adjacency, sourceTerm, dt**2,
        #         psSolver, gauge='nonnegative', verbose=verbose)

        # if psSolver.relaxationMode is JacobiRelaxationMode.optimal:
        #     raise ValueError(
        #         "relaxationMode 'optimal' is only supported by the divergenceFree "
        #         "(IISPH) solver: the constant-density solver clamps pressures to "
        #         "non-negative each iteration, which breaks the exact residual "
        #         "recurrence the optimal step relies on")

        # Which terms a static (`kind != 0`) neighbour contributes to. `full`
        # is the historical AllToAll behaviour; `staticBoundary` drops its
        # reaction on both sides at once -- out of `alpha`'s second sum here,
        # and out of the operator's neighbour-acceleration term in the loop
        # below -- so the diagonal keeps matching the operator it
        # preconditions. Read from *this* solver's config, since the two
        # solves are configured separately (Part 14); the bundle-level field
        # still overrides both. See `BoundaryOperatorTerms`.
        # boundaryTerms = resolveBoundaryOperatorTerms(schemeConfig.solverConfig, psSolver)

        # `BoundaryPressureMode.consistent` is Bender/Westhofen/Jeske 2023 in
        # full: their Eqs. 32 and 34 *are* `staticBoundary`, so the mode forces
        # it on rather than letting the two settings disagree.
        # boundaryPressureMode = getattr(schemeConfig.solverConfig, 'boundaryPressureMode',
                                #        BoundaryPressureMode.mdbcDensity)
        # if boundaryPressureMode is BoundaryPressureMode.consistent:
                # boundaryTerms = BoundaryOperatorTerms.staticBoundary

        alphas = dt**2 * computeAlpha(
                currentState = particles,
                config = config,
                schemeConfig = schemeConfig,
                adjacency = adjacency,
                apparentVolumes = apparentArea,
                # includeBoundaryReaction = boundaryTerms.alphaIncludesBoundaryReaction,
        )

        print(f'[IS] Alpha: {alphas.mean().cpu().item():.6g}, min: {alphas.min().cpu().item():.6g}, max: {alphas.max().cpu().item():.6g}')
        alphas = torch.clamp(alphas, max=-1e-6)  # Avoid division by zero

        # How the constant (null-space) component of the pressure field is
        # pinned each iteration -- see `ShiftPressureGauge`'s docstring for why
        # this solver needs a gauge at all and why mean-centering (what
        # `solveDivergenceFree` does) is not the answer here.
        # gauge = getattr(schemeConfig.solverConfig, 'shiftPressureGauge',
        #                 ShiftPressureGauge.nonNegativeClamp)
        # print(f'Alpha: {alphas.mean().cpu().item():.6g}, min: {alphas.min().cpu().item():.6g}, max: {alphas.max().cpu().item():.6g}')

        # kind==1 (boundary) and kind==2 (ghost) particles are not pressure unknowns:
        # their pressure is held fixed at its incoming `particles.pressures` value
        # (0 under `plain`, the mDBC-extrapolated/-projected value otherwise -- see
        # `BoundaryPressureMode`'s docstring and `divergenceFree.py`'s copy of this
        # comment for why freezing at the incoming value, not literal 0, matters),
        # excluded from the gauge statistic (under the default gauge there is
        # none to exclude them from -- it is a non-negativity clamp, not a
        # mean-center; under `minShift` the minimum is taken over fluid rows
        # only), and their `a_p` is zeroed post-solve. A no-op when there are no boundary
        # particles (`fluidMask` all-True).
        fluidMask = particles.kinds !=2
        boundaryPressure = particles.pressures.clone()
        # if boundaryPressureMode is BoundaryPressureMode.consistent:
                # Eq. 33 has no boundary pressure term at all, so there is no
                # value to carry: pin it at exactly 0 rather than at whatever
                # the state happens to hold.
                # boundaryPressure = torch.zeros_like(boundaryPressure)

        # `minShift` is a *gauge* fix, so it is only valid where the constant
        # mode is genuinely forceless: where the kernel support is truncated
        # the gradients no longer sum to zero, a *constant* pressure exerts a
        # large real force, and the offset stops being a gauge choice and
        # becomes a background pressure blowing the truncated particles
        # outward. **Free surfaces are truncated; this codebase's walls are
        # not** (`probe_wallSupportCompleteness.py`: Shepard ~1.00 and
        # `|A.1|/|A.rand|` 0.19 in the wall-adjacent bin against 0.17 in the
        # bulk, because `BOUNDED_BAND = 5` samples a solid band wider than the
        # kernel). So this guard tests for a free surface, and nothing else.
        #
        # It used to also downgrade whenever *any* pressure row was pinned
        # (`kind != 0`), on the argument that Dirichlet data already fixes the
        # constant so there is no null space left to gauge. That argument was
        # measured wrong, twice over. Part 4's evidence for it -- `minShift`
        # diverging at t=0.69 on the bounded case -- was taken at 3x [BK]'s
        # CFL, and at the published CFL the same configuration does not
        # diverge at all. And Part 13's factorial found the gauge and the
        # static-boundary operator are one fix applied at two points: together
        # they hold the bounded case's density band at 4.48e-3 against the old
        # default's 1.78e-1, 40x, and 5.4x better than the two composing
        # independently. Boundary rows are not really Dirichlet data here
        # either -- they are pinned at an mDBC extrapolation that moves with
        # the fluid field, not at a fixed level that anchors the constant.
        #
        # `forceShiftPressureGauge` now bypasses only the free-surface half.
        # if gauge is ShiftPressureGauge.minShift and not getattr(
        #                 schemeConfig.solverConfig, 'forceShiftPressureGauge', False):
        #         surface = getattr(particles, 'surfaceIndicators', None)
        #         if surface is not None and bool((surface > 0.5).any()):
        #                 gauge = ShiftPressureGauge.nonNegativeClamp

        # Warm start. Historically this solve starts cold every step
        # (`* 0.`), which is fine for the position-shift application -- the
        # shift is a one-shot displacement -- but wrong for a velocity-coupled
        # application (`ShiftApplication.inStepVelocity`): a standing pressure
        # field (a hydrostatic column) then has to be rebuilt from zero every
        # step and the column falls before it is supported. When the caller
        # passes the previous step's constant-density pressure, start from it.
        if warmStartPressure is not None:
                pressureA = warmStartPressure.clone()
        else:
                pressureA = particles.pressures.clone() * 0.
        pressureA = torch.where(fluidMask, pressureA, boundaryPressure)
        pressureB = pressureA.clone()

        errors = []
        pressures = []
        i = 0
        error = 0.
        gaugeOffset = torch.zeros((), device=pressureB.device, dtype=pressureB.dtype)

        # The stopping test: an absolute test on the configured statistic, and
        # (when `rtol > 0`) a relative disjunct on `mean|r|`. See
        # `convergence.py` and `JacobiConvergenceCriterion`; this solver's
        # historical statistic is `flooredOneSided`, which is the one §1.7
        # calls broken.
        # criterion = psSolver.convergenceCriterion
        # bNorm = sourceNorm(sourceTerm, fluidMask, psSolver.rtol)
        # relTarget = None if bNorm is None else psSolver.atol + psSolver.rtol * bNorm

        # print(f"Solving for divergence-free velocities with maxIters={maxIters}, threshold={threshold:.6g}, omega={omega:.6g}")

        for i in range(maxIters):
                pressureA = pressureB.clone()
                if _SHIFT_WALL_PRESSURE is not None:
                        # Recompute the wall pressure from the current fluid
                        # iterate (Part 41's Robin closure) so the near-wall
                        # iteration matrix is consistent -- see
                        # `_SHIFT_WALL_PRESSURE`.
                        pressureA = wallPressureExtrapolation(
                                particles, config, adjacency, pressureA,
                                _fluidRows, mode=_SHIFT_WALL_PRESSURE)
                a_p = computePressureAccelIISPH(
                        state = particles,
                        pressureValues = pressureA,
                        config = config,
                        supportScheme = SupportScheme.Scatter,
                        adjacency = adjacency,
                )
                # if not boundaryTerms.operatorMovesBoundary:
                        # `dx_p_i = sum_j V_j (a_i - a_j).gradW_ij` counts the
                        # neighbour's pressure displacement; a static particle
                        # has none, so only `i`'s own term survives for those
                        # `j` (this is SPlisHSPlasH's boundary loop in
                        # `TimeStepIISPH::pressureSolveIteration`). `i`'s own
                        # acceleration still feels their frozen pressure --
                        # that is the wall force, computed above.
                        # a_p = torch.where(fluidMask.unsqueeze(-1), a_p, torch.zeros_like(a_p))
                a_p = torch.where(particles.kinds.unsqueeze(-1)==0, a_p, torch.zeros_like(a_p))
                dx_p = dt**2 * computePressureShiftIISPH(
                        state = particles,
                        config = config,
                        pressureAccels = a_p,
                        supportScheme = SupportScheme.Scatter,
                        adjacency = adjacency,
                )
                print(f'\t[IS] Pressure acceleration: mean: {a_p.mean().cpu().item():.6g}, min: {a_p.min().cpu().item():.6g}, max: {a_p.max().cpu().item():.6g}')
                print(f'\t[IS] Pressure shift: mean: {dx_p.mean().cpu().item():.6g}, min: {dx_p.min().cpu().item():.6g}, max: {dx_p.max().cpu().item():.6g}')
                omega = 0.3
                residual = sourceTerm - dx_p
                pressureB = pressureA + omega * residual / alphas
                print(f'\t[IS] Pressure before gauge: mean: {pressureB.mean().cpu().item():.6g}, min: {pressureB.min().cpu().item():.6g}, max: {pressureB.max().cpu().item():.6g}')
                # if gauge is ShiftPressureGauge.minShift:
                #         # Non-negative *and* gauge-fixed: pinning the fluid
                #         # minimum at zero constrains the constant null-space mode
                #         # (which the clamp below does not -- it is a floor, so
                #         # that mode is free to drift upward without bound) while
                #         # translating rather than discarding the field's negative
                #         # part. The offset is a *gauge*, so it has to move the
                #         # frozen boundary rows with it (`gaugeOffset` accumulates
                #         # it, since `boundaryPressure` is a fixed pre-solve
                #         # snapshot): shifting the fluid rows alone would open a
                #         # fluid-vs-wall pressure jump of the offset's size, which
                #         # is a spurious wall-normal force, not a gauge choice.
                #         shift = pressureB[fluidMask].min()
                #         gaugeOffset = gaugeOffset + shift
                #         pressureB = pressureB - shift
                #         pressureB = torch.where(fluidMask, pressureB, boundaryPressure - gaugeOffset)
                # else:
                #         pressureB = torch.clamp(pressureB, min=0.0)  # Ensure non-negative pressures
                #         pressureB = torch.where(fluidMask, pressureB, boundaryPressure)
                pressureB = torch.clamp(pressureB, min=0.0)  # Ensure non-negative pressures

                # error, rNorm = evaluateResidual(residual, fluidMask, criterion,
                                                # threshold, bNorm)
                error = torch.mean(torch.clamp(-residual, min=-threshold)).cpu().item()
                errors.append(error)

                pressures.append((pressureB.min().cpu().item(), pressureB.max().cpu().item(), pressureB.mean().cpu().item()))

                if i >= minIters and (error < threshold):
                                #       or (relTarget is not None and rNorm <= relTarget)):
                #     print(f"Converged after {i+1} iterations with error: {error:.6g}")
                    break
                
                if verbose:
                    print(f"\t[IS] Iteration {i+1}/{maxIters}, error: {error:.6g}, pressure min/max/mean: {pressures[-1]}")

                if len(errors) > 1 and error > errors[-2]:
                    if verbose:
                        print(f"!!![IS] Warning: Error increased from {errors[-2]:.6g} to {error:.6g}.!!!")
                print(f"\t[IS] Iteration {i+1}/{maxIters}, error: {error:.6g}, pressure min/max/mean: {pressures[-1]}")

        a_p = computePressureAccelIISPH(
                state = particles,
                pressureValues = pressureB,
                config = config,
                supportScheme = SupportScheme.Scatter,
                adjacency = adjacency,
        )
        a_p = torch.where(fluidMask.unsqueeze(-1), a_p, torch.zeros_like(a_p))
        # print(f"Final pressure acceleration: mean: {a_p.mean().cpu().item():.6g}, min: {a_p.min().cpu().item():.6g}, max: {a_p.max().cpu().item():.6g}")

        # print(f'final Residual: {residual.mean().cpu().item():.6g}, min: {residual.min().cpu().item():.6g}, max: {residual.max().cpu().item():.6g}')
        if verbose:
            print(f'[IS] final Residual: {residual.mean().cpu().item():.6g}, min: {residual.min().cpu().item():.6g}, max: {residual.max().cpu().item():.6g}')
            # if residual.mean() > 
        return a_p, pressureB, errors, pressures


def solveIncompressible(
        particles: Any,
        config: SimulationConfig,
        schemeConfig: Any,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]],
        dvdt: torch.Tensor,
        dt: float,
        verbose: bool = False,
        warmStartPressure: Optional[torch.Tensor] = None,
):
        """Thin wrapper that puts `kind != 0` rows into the boundary state the
        active `BoundaryPressureMode` calls for, then runs the solve.

        `warmStartPressure` (optional): a per-particle field to seed the
        relaxed-Jacobi iterate with instead of the historical cold `0` start --
        see `_solveIncompressibleImpl`. The caller owns carrying it across
        steps.

        Only `BoundaryPressureMode.consistent` changes anything: it enters the
        solve with boundary densities pinned at `rho0` (Bender/Westhofen/Jeske
        2023 treat boundary particles as "static fluid particles" at the rest
        density) and restores the mDBC-extrapolated values afterwards, so
        nothing outside the pressure solve sees the substitution. Every other
        mode passes straight through. See `consistent.py`.
        """
        mode = getattr(schemeConfig.solverConfig, 'boundaryPressureMode',
                       BoundaryPressureMode.mdbcDensity)
        with applyConsistentCoupling(particles, config, schemeConfig, adjacency, mode):
                return _solveIncompressibleImpl(
                        particles, config, schemeConfig, adjacency, dvdt, dt, verbose,
                        warmStartPressure=warmStartPressure)
