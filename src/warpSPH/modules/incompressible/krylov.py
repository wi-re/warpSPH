"""Matrix-free Krylov dispatch for the DFSPH incompressible pressure solve.

The live incompressible pressure solve (``divergenceFree.solveDivergenceFree`` /
``incompressible.solveIncompressible``) is already matrix-free: it drives the
scalar pressure ``p`` so that

    A p = b,   A = dt_scale * (IISPH pressure shift o IISPH pressure accel),

with the IISPH diagonal ``dt_scale * computeAlpha`` as a Jacobi-style
preconditioner and ``b`` the variant's source term. This module builds exactly
those operators as closures and hands them to the Krylov solvers in
``modules/shifting`` (BiCGStab, GMRES, CG, BiCG, MINRES). The
relaxed-Jacobi path in ``divergenceFree.py`` is left untouched and remains the
default (``PressureSolverType.relaxedJacobi``).

The solvers precondition by *multiplication* (``Mx = precond * x``), so the
preconditioner handed to them is ``1/D`` -- the reciprocal of the IISPH
diagonal -- not ``D`` itself.

See ``docs/historic_plans/INCOMPRESSIBLE_SOLVER_PLAN.md`` for the operator mapping, the
SPD/symmetry probe, and the per-method phase plan.
"""

from typing import Any, Callable, List, Optional, Tuple, Union

import torch

from warpSPHCore import SupportScheme

from .wp_alpha import computeAlpha
from .drift import computePressureShiftIISPH
from ..pressure.iisph import computePressureAccelIISPH
from ..shifting.bicgstab import bicgstabSolve
from ..shifting.gmres import gmresSolve
from ..shifting.cg import cgSolve
from ..shifting.bicg import bicgSolve
from ..shifting.minres import minresSolve
from ...configurations import PressureSolverType, BoundaryOperatorTerms, resolveBoundaryOperatorTerms

__all__ = [
    'buildIISPHMatvec',
    'buildIISPHMatvecT',
    'buildIISPHPrecond',
    'solvePressureKrylov',
]


def buildIISPHMatvec(state, config, schemeConfig, adjacency, dt_scale,
                     supportScheme=SupportScheme.Scatter,
                     boundaryTerms=None) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the matrix-free pressure operator ``A(p) = dt_scale * shift(accel(p))``.

    This is exactly the inner iteration of the relaxed-Jacobi pressure solve
    (``computePressureAccelIISPH`` -> ``computePressureShiftIISPH``), scaled by
    ``dt_scale`` (``dt`` for the divergence-free variant, ``dt**2`` for the
    constant-density variant). It is a closure over the current state, so it
    stays matrix-free: each call runs two SPH gather/scatter passes.
    """
    # Static (`kind != 0`) neighbours contribute no pressure displacement of
    # their own under `BoundaryOperatorTerms.staticBoundary`; the Krylov path
    # builds the same operator as the Jacobi one, so it reads the same setting.
    # The setting is per-solver (Part 14) and this builder does not know which
    # solver it is serving, so `solvePressureKrylov` passes the resolved value
    # in; the fallback is the bundle-level override, then `full`.
    # See `BoundaryOperatorTerms`.
    if boundaryTerms is None:
        boundaryTerms = getattr(schemeConfig.solverConfig, 'boundaryOperatorTerms',
                                None) or BoundaryOperatorTerms.full
    fluidMask = None if boundaryTerms.operatorMovesBoundary else (state.kinds == 0).unsqueeze(-1)

    def matvec(p):
        a_p = computePressureAccelIISPH(
            state=state, pressureValues=p, config=config,
            supportScheme=supportScheme, adjacency=adjacency)
        if fluidMask is not None:
            a_p = torch.where(fluidMask, a_p, torch.zeros_like(a_p))
        return dt_scale * computePressureShiftIISPH(
            state=state, config=config, pressureAccels=a_p,
            supportScheme=supportScheme, adjacency=adjacency)
    return matvec


def buildIISPHMatvecT(state, config, schemeConfig, adjacency, dt_scale,
                      supportScheme=SupportScheme.Scatter,
                      boundaryTerms=None) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the adjoint (transpose) pressure operator ``A^T``.

    Needed only by BiCG. ``warpSPHCore`` does not expose a ready adjoint of the
    pressure ``accel``/``shift`` pair, so this is a *self-adjoint placeholder*:
    it aliases the forward matvec. That is correct only if the discrete operator
    is symmetric, which the Phase-0 operator probe is meant to quantify. A
    rigorously derived ``A^T`` (rearranged kernel weights) is the Phase-4 work
    item; until then BiCG results should be treated as provisional.
    """
    # Phase-0 stub: assume A is (approximately) symmetric -> A^T ~= A.
    return buildIISPHMatvec(state, config, schemeConfig, adjacency, dt_scale,
                            supportScheme=supportScheme, boundaryTerms=boundaryTerms)


def buildIISPHPrecond(state, config, schemeConfig, adjacency, dt_scale,
                      supportScheme=SupportScheme.Scatter,
                      boundaryTerms=None) -> torch.Tensor:
    """Return the IISPH Jacobi preconditioner ``1/D`` as a flat vector.

    ``D = dt_scale * computeAlpha(...)`` is the IISPH diagonal (negated, hence
    clamped away from zero exactly as the relaxed-Jacobi path does). The Krylov
    solvers apply the preconditioner by elementwise multiplication
    (``Mx = precond * x``), so we return the reciprocal ``1/D``, not ``D``.
    """
    apparentArea = state.masses / state.densities
    if boundaryTerms is None:
        boundaryTerms = getattr(schemeConfig.solverConfig, 'boundaryOperatorTerms',
                                None) or BoundaryOperatorTerms.full
    alphas = dt_scale * computeAlpha(
        currentState=state, config=config, schemeConfig=schemeConfig,
        adjacency=adjacency, apparentVolumes=apparentArea,
        includeBoundaryReaction=boundaryTerms.alphaIncludesBoundaryReaction)
    alphas = torch.clamp(alphas, max=-1e-6)
    return 1.0 / alphas


def solvePressureKrylov(
        particles: Any,
        config,
        schemeConfig: Any,
        adjacency: Optional[Any],
        sourceTerm: torch.Tensor,
        dt_scale: float,
        solverCfg: Any,
        gauge: Optional[str] = 'center',
        x0: Optional[torch.Tensor] = None,
        verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[Tuple[float, float, float]]]:
    """Solve the IISPH pressure problem ``A p = b`` with a Krylov method.

    Dispatches on ``solverCfg.solverType`` to the appropriate solver in
    ``modules/shifting``, applies the variant's gauge fix, and returns the same
    ``(a_p, pressure, errors, pressures)`` tuple the relaxed-Jacobi solvers
    return, so the call sites in ``divergenceFree.py`` / ``incompressible.py``
    are unchanged.

    Parameters
    ----------
    sourceTerm : the RHS ``b`` (per-particle scalar), computed by the caller
        variant (``-divergence`` for divergence-free, ``rho0 - rhoStar`` for
        constant-density).
    dt_scale : the operator scale (``dt`` for divergence-free, ``dt**2`` for
        constant-density), matching the variant's matvec/precond.
    solverCfg : a ``RelaxedJacobiSolverConfig``; only ``solverType``, ``rtol``,
        ``atol``, ``maxIterations`` and ``restart`` are read.
    gauge : ``'center'`` (subtract the mean -- pure-Neumann divergence-free),
        ``'nonnegative'`` (clamp to >= 0 -- constant-density) or ``None``.
    x0 : warm-start guess; defaults to the previous step's pressure
        (``particles.pressures``) when available.
    """
    solverType = solverCfg.solverType
    rtol = solverCfg.rtol
    atol = solverCfg.atol
    maxiter = solverCfg.maxIterations
    restart = solverCfg.restart

    # kind==1 (boundary) and kind==2 (ghost) particles are not pressure unknowns (see
    # `BoundaryPressureMode`'s docstring): the operator is wrapped so every matvec
    # sees (and returns) 0 at boundary rows, restricting the Krylov iterate `x` to
    # the fluid subspace -- the standard Dirichlet-lifting trick for an
    # inhomogeneous boundary value in an iterative solve. Boundary pressure is
    # frozen at its incoming `particles.pressures` value (0 under `plain`, the
    # mDBC-extrapolated/-projected value otherwise), folded into the RHS as the
    # `A_fb p_b` correction (`boundaryCorrection` below) rather than left inside the
    # matvec/iterate, since a generic Krylov method needs a fixed linear operator on
    # a fixed unknown subspace -- unlike the relaxed-Jacobi solvers in
    # `divergenceFree.py`/`incompressible.py`, which can just bake the frozen value
    # into the pressure field their SPH neighbor sums read directly. A no-op when
    # there are no boundary particles (`fluidMask` all-True, `boundaryCorrection`
    # all-0).
    fluidMask = particles.kinds == 0
    boundaryPressure = particles.pressures.clone()
    # The static-neighbour operator terms are configured per solver (Part 14),
    # and `solverCfg` is the one this call is solving for.
    boundaryTerms = resolveBoundaryOperatorTerms(schemeConfig.solverConfig, solverCfg)
    _rawMatvec = buildIISPHMatvec(particles, config, schemeConfig, adjacency, dt_scale,
                                  boundaryTerms=boundaryTerms)

    def matvec(p):
        p = torch.where(fluidMask, p, torch.zeros_like(p))
        out = _rawMatvec(p)
        return torch.where(fluidMask, out, torch.zeros_like(out))

    boundaryOnly = torch.where(fluidMask, torch.zeros_like(boundaryPressure), boundaryPressure)
    boundaryCorrection = _rawMatvec(boundaryOnly)
    b = torch.where(fluidMask, sourceTerm - boundaryCorrection, torch.zeros_like(sourceTerm))
    precond = buildIISPHPrecond(particles, config, schemeConfig, adjacency, dt_scale,
                                boundaryTerms=boundaryTerms)
    if x0 is None:
        x0 = particles.pressures.clone() if particles.pressures is not None else None
    if x0 is not None:
        x0 = torch.where(fluidMask, x0, torch.zeros_like(x0))

    if getattr(solverCfg, 'krylovFp64', False):
        # Run the Krylov recurrence in fp64 while the SPH matvec stays at
        # production fp32. This operator is ill-conditioned (kappa(M^-1 A) is
        # O(1e8) on a uniform lattice, i.e. the raw operator condition number),
        # so the fp32 recurrence loses the orthogonality/conjugacy it relies on
        # (BiCGStab's shadow system, CG's conjugacy) and BiCGStab in particular
        # stagnates ~10x higher than it could. fp64 bookkeeping costs only a few
        # extra vectors; the matvec (the expensive part) is unchanged. Measured
        # on the TGV probe: BiCGStab 1.9e-3 -> 1.1e-4, CG 3.9e-5 -> 3.6e-6.
        # See docs/regression/incompressible_pressure_solver_choice.md.
        _m = matvec
        _dt = particles.densities.dtype
        matvec = (lambda p: _m(p.to(_dt)).double())
        b = b.double()
        precond = precond.double()
        if x0 is not None:
            x0 = x0.double()

    if solverType == PressureSolverType.bicgStab:
        x, status, conv = bicgstabSolve(matvec, b, x0, tol=atol, rtol=rtol,
                                        maxiter=maxiter, precond=precond, dim=1,
                                        verbose=verbose)
    elif solverType == PressureSolverType.gmres:
        x, status, conv = gmresSolve(matvec, b, x0, tol=atol, rtol=rtol,
                                     maxiter=maxiter, precond=precond, dim=1,
                                     restart=restart, verbose=verbose)
    elif solverType == PressureSolverType.cg:
        # CG requires a symmetric positive-definite operator. The IISPH operator
        # is expected to have a negative diagonal (hence negative-definite), so
        # flip the sign of the operator, RHS and preconditioner to hand PCG a
        # positive-definite system. If the operator is genuinely indefinite (not
        # merely negative-definite) -- the question the Phase-0 operator probe
        # settles -- cgSolve bails with status -16.
        if float(precond.mean()) < 0:
            _m = matvec
            matvec = (lambda p: -_m(p))
            precond = -precond
            b = -b
        x, status, conv = cgSolve(matvec, b, x0, tol=atol, rtol=rtol,
                                  maxiter=maxiter, precond=precond, dim=1,
                                  verbose=verbose)
    elif solverType == PressureSolverType.bicg:
        _rawMatvecT = buildIISPHMatvecT(particles, config, schemeConfig, adjacency, dt_scale,
                                        boundaryTerms=boundaryTerms)

        def matvecT(p):
            p = torch.where(fluidMask, p, torch.zeros_like(p))
            out = _rawMatvecT(p)
            return torch.where(fluidMask, out, torch.zeros_like(out))
        # The operator is (to fp32 precision) symmetric, so buildIISPHMatvecT's
        # self-adjoint placeholder is exact; the sign-flip below (same rationale
        # as the CG branch) hands BiCG a positive-definite system. Note BiCG is
        # still the least robust of the four on the indefinite/gauge-mode part of
        # the spectrum -- see the Phase-0 operator probe in
        # docs/historic_plans/INCOMPRESSIBLE_SOLVER_PLAN.md.
        if float(precond.mean()) < 0:
            _m = matvec
            _mT = matvecT
            matvec = (lambda p: -_m(p))
            matvecT = (lambda q: -_mT(q))
            precond = -precond
            b = -b
        x, status, conv = bicgSolve(matvec, matvecT, b, x0, tol=atol, rtol=rtol,
                                    maxiter=maxiter, precond=precond, dim=1,
                                    verbose=verbose)
    elif solverType == PressureSolverType.minres:
        # MINRES needs symmetry, not positive definiteness: no sign flip, and
        # the flat 1/D diagonal is passed as-is -- minresSolve turns it into
        # its symmetrizing congruence (d = 1/sqrt(|precond|)) internally.
        x, status, conv = minresSolve(matvec, b, x0, tol=atol, rtol=rtol,
                                      maxiter=maxiter, precond=precond, dim=1,
                                      verbose=verbose)
    else:
        raise ValueError(f'Unrecognized pressure solver type: {solverType}')

    # If the recurrence ran in fp64 (krylovFp64), the iterate is fp64; cast back
    # to the production dtype before the gauge fix / final accel / return.
    x = x.to(particles.densities.dtype)

    # Gauge fix (the pressure is defined up to an additive constant),
    # excluding boundary rows from the centering mean, then re-pinning them
    # to their frozen `boundaryPressure` (the sign-flip branches above, and the
    # nonnegative clamp, may have left tiny fp noise or an incorrect clamp there
    # since `x` at boundary rows should already be ~0 by construction of
    # `matvec`/`b` above).
    if gauge == 'center':
        x = x - x[fluidMask].mean()
    elif gauge == 'nonnegative':
        x = torch.clamp(x, min=0.0)
    elif gauge is not None:
        raise ValueError(f'Unknown gauge fix: {gauge!r}')
    x = torch.where(fluidMask, x, boundaryPressure)

    a_p = computePressureAccelIISPH(
        state=particles, pressureValues=x, config=config,
        supportScheme=SupportScheme.Scatter, adjacency=adjacency)
    a_p = torch.where(fluidMask.unsqueeze(-1), a_p, torch.zeros_like(a_p))

    errors = [float(e) for e in conv]
    pressures = [(float(x.min()), float(x.max()), float(x.mean()))]

    if verbose:
        print(f'[Krylov:{solverType.name}] status={status} '
              f'history={len(conv)} final={conv[-1].item():.6g} '
              f'pressure min/max/mean={pressures[0]}')

    return a_p, x, errors, pressures