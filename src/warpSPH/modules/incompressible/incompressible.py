"""IISPH constant-density pressure solver: relaxed-Jacobi iteration that
solves for a pressure field driving the predicted density back to
`schemeConfig.fluid.restDensity` (source term `rho0 - rhoStar`, the
density-error formulation, as opposed to `divergenceFree.py`'s divergence
formulation).

Each iteration recomputes the wall pressure from the current fluid iterate
(`_SHIFT_WALL_PRESSURE`, a Robin closure -- see its docstring for why this
solve needs it), then the pressure acceleration (`computePressureAccelIISPH`,
scatter mode) and its position-drift residual (`dt**2 *
computePressureShiftIISPH`), and updates `pressure += omega * residual /
alpha` (`dt**2 * computeAlpha`'s IISPH diagonal term). Pressures are kept
non-negative each iterate, except where `_CLOSED_DOMAIN_GAUGE` fires (a
closed, wall-only domain has no free surface for the clamp to guard against,
so the unconstrained constant mode is pinned by mean-centering instead).
`rhoStar` is clamped to a minimum of 0.9 and `alpha` to a maximum of -1e-6 to
avoid division blow-up. Iterates between `solverConfig.pressureSolver.
{minIterations,maxIterations,tolerance,relaxationFactor}`, stopping early once
past `minIterations` and below `tolerance`.
"""

from warpSPHCore import *
import os
import torch
from warpSPH.systems import *
from warpSPH.modules import *
from warpSPH.configurations import SimulationConfig
from typing import Any, Optional, Union
from .wp_alpha import computeAlpha
from ..momentum.incompressible import computeMomentumIncompressible
from ..pressure.iisph import computePressureAccelIISPH
from .drift import computePressureShiftIISPH
from ...configurations import JacobiRelaxationMode, PressureSolverType
from .krylov import solvePressureKrylov
from .consistent import applyConsistentCoupling
from .wallPressure import wallPressureExtrapolation
from ...configurations import BoundaryPressureMode

__all__ = ['solveIncompressible']

#: EXPERIMENT (DFSPH_IMPROVEMENT_PLAN.md Part 48; superseded as the fix by
#: Part 56's `_SHIFT_WALL_PRESSURE`, kept as a still-available lever) -- the
#: VD+PS shift's constant-density solve on a fully wall-closed box (`randomFlow
#: Incompressible --bounded`) is a pure-Neumann system: `A.1 ~ 0`, so the
#: constant (n_h=4 lattice-bias) component of the `1 - rhoStar/rho0` source has
#: no solution and `p` ramps until the run detonates. `'auto'` subtracts the
#: fluid-mean source only when it is mean-dominated (the closed-box signature,
#: Part 42's `omniIncompressible.CD_SOURCE_PROJECT`); `True`/`False` force it.
#: Default `False` == byte-identical to the historical solve. Part 56 fixed
#: the actual divergence via the wall-pressure closure instead (see
#: `_SHIFT_WALL_PRESSURE` below), so this projection is no longer load-bearing
#: for that case -- it remains here as an independent, still-untested-in-
#: combination lever, not as the current explanation for why the case holds.
_CLOSED_BOX_SOURCE_PROJECT = False
_CLOSED_BOX_PROJECT_THRESHOLD = 0.7

#: LANDED DEFAULT (DFSPH_IMPROVEMENT_PLAN.md "Active track" item 2a, Part 56)
#: -- recompute the `kind==1` wall pressure from the current fluid iterate
#: every Jacobi sweep (Part 41's `wallPressureExtrapolation`, no relaxation /
#: no carried state), instead of freezing `boundaryPressure` at its pre-solve
#: snapshot, so the near-wall iteration matrix is consistent (the same fix
#: Part 41 landed for `omniIncompressible`, just never previously pointed at
#: this solve). This is the actual fix for `randomFlowIncompressible
#: --bounded`, which the `_CLOSED_DOMAIN_GAUGE` mean-centering trick (Part 55)
#: only delayed: `'shepard'` holds it cleanly at every resolution in
#: {32,48,56,64,96,128}, 300-400 steps each, KE decaying throughout, no
#: `_CLOSED_DOMAIN_GAUGE` needed at all (left default `'off'` -- superseded
#: for this case, not proven harmful elsewhere, so not removed). `'mls'`
#: measured comparably or slightly better here but is not the default,
#: matching `omniIncompressible.WALL_PRESSURE_MODE`'s own Part 42 choice of
#: `'shepard'` over `'mls'` on the same case family (cheaper, no ghost Liu-Liu
#: fit, no risk from `'mls'`'s linear term on a sheared flow). `'mirror'`
#: (Part 55/56, `wallPressureExtrapolation`'s new Adami-reflection mode) also
#: holds alone but is the weakest of the three (least KE decay at nx=128) and
#: is measurably unstable combined with `_CLOSED_DOMAIN_GAUGE='always'` (an
#: unexplained interaction, not chased further since the combination is not
#: needed). Blast radius: the only case in the codebase with both `kind==1`
#: rows and `_psShift` active (this function only fires on non-gravity cases
#: -- `finalize`'s `_RESTORE_PS_SHIFT='auto'` gate) is
#: `randomFlowIncompressible --bounded`/`--obstacle`; every other caller
#: (`tgv`/`shearWave`/`staticBlob`'s `relaxLattice`, `kolmogorovIncompressible`,
#: the periodic random flow) has no boundary particles, so `wallPressureExtrapolation`
#: no-ops there regardless of this setting -- verified bit-close on/off.
#: `None` reverts to the historical frozen-boundary solve; `'mls'` / `'mirror'`
#: are the other extrapolation orders.
#:
#: `WARPSPH_WALL_PRESSURE_MODE` overrides this at import time (`'none'` maps to
#: `None`) -- for `scripts/validate_scheme.py` comparison sweeps
#: (`--wallPressure`) across schemes/solves without editing this file. Shared
#: with `schemes/omniIncompressible.py`'s `WALL_PRESSURE_MODE`, which reads the
#: same variable: the two module globals govern different call sites
#: (`_psShift`'s VD+PS shift here vs `INSTEP_CD`'s in-step fold there), so a
#: `--scheme divergenceFree` sweep needs both toggled together to cover every
#: boundary case regardless of whether it runs under gravity.
_WALL_PRESSURE_ENV = os.environ.get('WARPSPH_WALL_PRESSURE_MODE')
_SHIFT_WALL_PRESSURE = (
    (None if _WALL_PRESSURE_ENV.lower() == 'none' else _WALL_PRESSURE_ENV)
    if _WALL_PRESSURE_ENV is not None else 'shepard')

#: EXPERIMENT (DFSPH_IMPROVEMENT_PLAN.md "Active track" item 2a) -- the
#: `band2018pb.CLOSED_DOMAIN_GAUGE` fix (Part 50), ported to this operator.
#: `_CLOSED_BOX_SOURCE_PROJECT` above patches an incompatible *source*
#: (`omniIncompressible.CD_SOURCE_PROJECT`'s failure mode); this patches the
#: other closed-domain failure Part 50 found -- an unpinned *solution*. This
#: solve's pressure accel (`computePressureAccelIISPH`) is a summation
#: gradient (BWJ23 Eq. 33's `-(V_i/m_i) sum_j m_j/rho_j (p_i+p_j) gradW_ij`
#: form), so a uniform fluid `p = c` (boundary held at its frozen value) gives
#: an accel that vanishes wherever the kernel support is complete -- true
#: everywhere in a fully enclosed box (`randomFlowIncompressible --bounded`).
#: The `p >= 0` clamp then only ever pushes the unpinned constant up, and it
#: ratchets away every iterate. `'auto'` keys on the same
#: `rms|A.1| / rms|a_ii|` test `band2018pb.bandConstantModeRatio` uses (cheap:
#: one extra operator application per step, not per iteration); when it fires
#: each iterate is mean-centered over fluid rows instead of clamped, since a
#: closed domain has no free surface for the clamp to be guarding against.
#: `'off'` (default) == byte-identical to the historical solve.
_CLOSED_DOMAIN_GAUGE = 'off'
_NULL_MODE_THRESHOLD = 0.25


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
        # This solve clamps pressures non-negative each iterate (except where
        # `_CLOSED_DOMAIN_GAUGE` fires), which breaks the exact residual
        # recurrence `optimal` relies on -- that mode is divergenceFree-only.
        if schemeConfig.solverConfig.pressureSolver.relaxationMode is JacobiRelaxationMode.optimal:
                raise ValueError(
                        "relaxationMode 'optimal' is only supported by the divergenceFree "
                        "(IISPH) solver: the constant-density solver clamps pressures to "
                        "non-negative each iteration, which breaks the exact residual "
                        "recurrence the optimal step relies on")

        predictedVelocities = particles.velocities + dt * dvdt
        apparentArea = particles.masses / particles.densities

        divergence = computeMomentumIncompressible(
                currentState = particles,
                config = config,
                schemeConfig = schemeConfig,
                adjacency = adjacency,
                advectionVelocities = predictedVelocities
        )

        rho0 = schemeConfig.fluid.restDensity
        rhoStar = particles.densities + dt * divergence
        sourceTerm = (1 - rhoStar/rho0)

        _fluidRows = particles.kinds == 0
        if _CLOSED_BOX_SOURCE_PROJECT and bool(_fluidRows.any()):
                _sf = sourceTerm[_fluidRows]
                _mean = _sf.mean()
                _frac = 1.0 - (_sf - _mean).norm() / (_sf.norm() + 1e-30)
                if (_CLOSED_BOX_SOURCE_PROJECT is True
                                or _frac > _CLOSED_BOX_PROJECT_THRESHOLD):
                        sourceTerm = sourceTerm - _mean
        if verbose:
            print(f'[IS] Predicted velocities: mean: {predictedVelocities.mean().cpu().item():.6g}, min: {predictedVelocities.min().cpu().item():.6g}, max: {predictedVelocities.max().cpu().item():.6g}')
            print(f'[IS] Source term: {sourceTerm.mean().cpu().item():.6g}, min: {sourceTerm.min().cpu().item():.6g}, max: {sourceTerm.max().cpu().item():.6g} abs mean: {sourceTerm.abs().mean().cpu().item():.6g}')
            print(f'[IS] Mean density error: {(particles.densities - schemeConfig.fluid.restDensity).abs().mean().cpu().item():.6g}')

        # Opt-in Krylov pressure solvers (BiCGStab/GMRES/CG/BiCG/MINRES) share
        # the same matrix-free operator and IISPH-diagonal preconditioner as
        # the relaxed-Jacobi path below, which stays the byte-identical
        # default (solverType == relaxedJacobi). `gauge='nonnegative'`
        # matches this solve's own clamp (`divergenceFree.py`'s analogous
        # dispatch uses `gauge='center'` instead, since that solve has no
        # tensile/free-surface concern to clamp against).
        # KNOWN LIMITATION (DFSPH_IMPROVEMENT_PLAN.md Parts 43-44,
        # `DFSPH_FINDINGS.md` §9 rows 43-44): non-symmetric Krylov breaks down
        # on every wall-bounded case tried (free surface *and* closed box) --
        # the composed operator is rank-deficient at the wall, which a
        # symmetric method has nothing to converge to either. Safe on
        # well-conditioned wall-free cases (this is what
        # `tests/test_incompressibleKrylov.py` exercises, on periodic `tgv`);
        # not validated as a general substitute for the Jacobi default.
        psSolver = schemeConfig.solverConfig.pressureSolver
        if psSolver.solverType != PressureSolverType.relaxedJacobi:
            return solvePressureKrylov(
                particles, config, schemeConfig, adjacency, sourceTerm, dt**2,
                psSolver, gauge='nonnegative', verbose=verbose)

        alphas = dt**2 * computeAlpha(
                currentState = particles,
                config = config,
                schemeConfig = schemeConfig,
                adjacency = adjacency,
                apparentVolumes = apparentArea,
        )
        alphas = torch.clamp(alphas, max=-1e-6)  # Avoid division by zero
        if verbose:
            print(f'[IS] Alpha: {alphas.mean().cpu().item():.6g}, min: {alphas.min().cpu().item():.6g}, max: {alphas.max().cpu().item():.6g}')

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

        # Closed-domain gauge test (see `_CLOSED_DOMAIN_GAUGE`): is the
        # constant fluid mode a null mode of this operator? `rms|A.1| /
        # rms|a_ii|` with boundary rows held at 0 -- the homogeneous system a
        # fluid-only constant actually sees, since boundary rows are frozen
        # during the real solve too. One extra operator application, not one
        # per iteration.
        closedDomainGauge = False
        if _CLOSED_DOMAIN_GAUGE != 'off' and bool(_fluidRows.any()):
                if _CLOSED_DOMAIN_GAUGE == 'always':
                        closedDomainGauge = True
                elif _CLOSED_DOMAIN_GAUGE == 'auto':
                        onesTest = torch.where(_fluidRows, torch.ones_like(alphas),
                                               torch.zeros_like(alphas))
                        a1 = computePressureAccelIISPH(
                                state = particles, pressureValues = onesTest, config = config,
                                supportScheme = SupportScheme.Scatter, adjacency = adjacency)
                        a1 = torch.where(particles.kinds.unsqueeze(-1)==0, a1, torch.zeros_like(a1))
                        A1 = dt**2 * computePressureShiftIISPH(
                                state = particles, config = config, pressureAccels = a1,
                                supportScheme = SupportScheme.Scatter, adjacency = adjacency)
                        rmsA1 = float(A1[_fluidRows].pow(2).mean().sqrt())
                        rmsD = float(alphas[_fluidRows].abs().pow(2).mean().sqrt())
                        ratio = rmsA1 / rmsD if rmsD > 0.0 else float('inf')
                        closedDomainGauge = ratio < _NULL_MODE_THRESHOLD
                        if verbose:
                                print(f'[IS] closed-domain null-mode ratio: {ratio:.4g}'
                                     f' (gauge {"ON" if closedDomainGauge else "off"})')
                else:
                        raise ValueError(
                                f'Unknown _CLOSED_DOMAIN_GAUGE: {_CLOSED_DOMAIN_GAUGE!r}')

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

        # The stopping test is an absolute test on the mean floored residual
        # (`flooredOneSided`, the statistic §1.7 in DFSPH_FINDINGS.md calls
        # broken -- kept for continuity with the historical behaviour rather
        # than fixed here). See `convergence.py` /
        # `JacobiConvergenceCriterion` for the general form this solve does
        # not currently use.

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
                a_p = torch.where(particles.kinds.unsqueeze(-1)==0, a_p, torch.zeros_like(a_p))
                dx_p = dt**2 * computePressureShiftIISPH(
                        state = particles,
                        config = config,
                        pressureAccels = a_p,
                        supportScheme = SupportScheme.Scatter,
                        adjacency = adjacency,
                )
                if verbose:
                    print(f'\t[IS] Pressure acceleration: mean: {a_p.mean().cpu().item():.6g}, min: {a_p.min().cpu().item():.6g}, max: {a_p.max().cpu().item():.6g}')
                    print(f'\t[IS] Pressure shift: mean: {dx_p.mean().cpu().item():.6g}, min: {dx_p.min().cpu().item():.6g}, max: {dx_p.max().cpu().item():.6g}')
                # NOTE: pins the relaxation at 0.3 regardless of
                # `schemeConfig.solverConfig.pressureSolver.relaxationFactor`
                # (read into the outer `omega` above, then never used) --
                # long-standing, not touched here; changing it is a numerical
                # behaviour change that needs its own validation, not a
                # cleanup. See DFSPH_IMPROVEMENT_PLAN.md.
                omega = 0.3
                residual = sourceTerm - dx_p
                pressureB = pressureA + omega * residual / alphas
                if verbose:
                    print(f'\t[IS] Pressure before gauge: mean: {pressureB.mean().cpu().item():.6g}, min: {pressureB.min().cpu().item():.6g}, max: {pressureB.max().cpu().item():.6g}')
                if closedDomainGauge:
                        # Pin the unconstrained constant instead of clamping --
                        # a closed domain has no free surface, so there is no
                        # tensile instability for the clamp to be guarding
                        # against (`band2018pb.CLOSED_DOMAIN_GAUGE`'s pairing).
                        offset = pressureB[_fluidRows].mean()
                        pressureB = torch.where(_fluidRows, pressureB - offset,
                                                boundaryPressure)
                else:
                        pressureB = torch.clamp(pressureB, min=0.0)  # Ensure non-negative pressures
                        pressureB = torch.where(fluidMask, pressureB, boundaryPressure)

                error = torch.mean(torch.clamp(-residual, min=-threshold)).cpu().item()
                errors.append(error)

                pressures.append((pressureB.min().cpu().item(), pressureB.max().cpu().item(), pressureB.mean().cpu().item()))

                if i >= minIters and (error < threshold):
                    break

                if verbose:
                    print(f"\t[IS] Iteration {i+1}/{maxIters}, error: {error:.6g}, pressure min/max/mean: {pressures[-1]}")
                    if len(errors) > 1 and error > errors[-2]:
                        print(f"!!![IS] Warning: Error increased from {errors[-2]:.6g} to {error:.6g}.!!!")

        a_p = computePressureAccelIISPH(
                state = particles,
                pressureValues = pressureB,
                config = config,
                supportScheme = SupportScheme.Scatter,
                adjacency = adjacency,
        )
        a_p = torch.where(fluidMask.unsqueeze(-1), a_p, torch.zeros_like(a_p))

        if verbose:
            print(f'[IS] final Residual: {residual.mean().cpu().item():.6g}, min: {residual.min().cpu().item():.6g}, max: {residual.max().cpu().item():.6g}')
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
