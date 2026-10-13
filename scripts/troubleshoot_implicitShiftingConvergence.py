#!/usr/bin/env python3
"""Troubleshooting harness for implicit particle shifting's inner Krylov
solve (`modules/shifting/implicitShifting.computeImplicitShift`, and its
composed-JVP twin `modules/shifting/implicitShiftingAutomatic.computeImplicitShiftAutomatic`;
solver selectable via `--solver`, BiCGStab by default).

**Why this exists.** `warpSPHCore/warpier_forward_mode_plan.md`'s Status
section documents that `tests/test_implicitShifting.py`'s convergence test is
flaky, and that Phase 4's own comparison tests only pass because
`implicitRelaxation`'s heavy outer damping smooths over individual bad inner
solves -- not because the inner Newton solve actually converges. A jitter
sweep (`--jitters 0,0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2`, the
default) narrowed this to an ill-posedness effect, not noise: the Newton step
linearizes `grad C = 0` at a *frozen* adjacency, but finding the equilibrium
distribution is a genuinely global problem where a particle's true
connectivity changes as it moves -- so the further the current state is from
equilibrium, the worse a local model the frozen matvec/RHS is.

**This finding is specific to `ShiftingImplicitOperator.exactHessian`, not
the shipped default.** This script was added (commit `664cee9`) *before* the
very next commit (`61f783e`, same day) introduced `ShiftingImplicitOperator`
and flipped the default to `legacyPairwise` -- precisely *because*
`exactHessian`'s diagonal is configuration-dependent and unbounded, which is
what produces the divergence described below (see
`implicitShifting.py`'s module docstring and
`docs/regression/implicit_shifting_operator_choice.md`). This script builds
its system from `schemeConfig.shiftProperties.implicitOperator`
(`--operator`, default `legacyPairwise` to match production), so **running
it with the default settings does *not* reproduce the divergence this
docstring describes** -- pass `--operator exactHessian` for that. Verified
2026-08-24: on the default `legacyPairwise` operator, the same jitter sweep
(nx=16, 2D, seeds 1234/1/42) converges cleanly at every jitter through 0.2
(`rel_resid` ~5e-5 to 9e-5, under the `1e-4` target) with only `jitter=0`
itself hitting `rho-breakdown` (`|b|` is ~1e-6 there, an exact-lattice
degeneracy, not the ill-posedness this docstring is about). With
`--operator exactHessian`, the same sweep reproduces the description below
almost exactly: `rho-breakdown` by `jitter=0.01` (`rel_resid` ~8e-3, two
orders of magnitude over target) and `threshold-bailout` with `max|xk|/dx`
up to ~9.6 (a genuine blow-up, not merely slow convergence) from
`jitter=0.02` on.

At `jitter=0` the solve is trivial; by `jitter=0.01` it already hits the
divergence threshold almost immediately, with the *unconverged* residual
growing monotonically with jitter from there. Every existing test uses
`jitter=0.1`, squarely in that regime -- **for `exactHessian`**; the
production-default `legacyPairwise` tests
(`test_implicitShiftingConvergesFromFullyRandomPositions`) pass because the
default operator itself converges there, not because outer damping is
covering for a non-converged inner solve (see that test's own docstring,
which states this directly: `exactHessian`'s instability on far-from-
equilibrium starts is "the known, documented, accepted tradeoff, not a
regression to guard"). (This was originally also a silent gap on top of the
above: a bailed-out `xk` got used exactly as if it had converged. It is now
closed by the opt-in fallback chain in `solverDriver.solveImplicitSystem` --
enabled by `shiftProperties.implicitFallback` or `ShiftingScheme.dynamic`,
default `none` keeps the historical behavior. This harness still reports the
raw solver status directly, *bypassing* that fallback, for diagnosis.)

**What this script does, concretely.** For each jitter level: builds a
jittered Cartesian lattice, assembles the exact same linear system
`computeImplicitShift` solves internally (via the same `_buildSystem`/
`_multiplyLaplacianBlock`/`computeShiftingPairTerms` production code, not a
reimplementation), runs the selected Krylov solver (`--solver`,
BiCGStab by default), and reports whether it actually converged
(`iters >= 0`) or bailed out (`-10`..`-14`, see
`bicgstab.py`/`gmres.py`), plus the raw (possibly unconverged) final residual and the
largest per-particle shift relative to the divergence threshold. Optionally
compares the composed-JVP matvec (`warpOperationHVP`) against the hand-built
one on the same system, sweeps Jacobi-preconditioner choices (the production
`D_excl`-diagonal one vs. a diffSPH-style raw self-Hessian diagonal), and can
print a full per-iteration residual trace for one selected case.

Usage:
    python scripts/troubleshoot_implicitShiftingConvergence.py
    python scripts/troubleshoot_implicitShiftingConvergence.py --nx 8 24 32
    python scripts/troubleshoot_implicitShiftingConvergence.py --jitters 0,0.05,0.1 --trace-jitter 0.1
    python scripts/troubleshoot_implicitShiftingConvergence.py --compare-preconditioners --jitters 0.1
    python scripts/troubleshoot_implicitShiftingConvergence.py --compare-automatic --jitters 0.1
    python scripts/troubleshoot_implicitShiftingConvergence.py --no-threshold --maxiter 200 --jitters 0.1
    python scripts/troubleshoot_implicitShiftingConvergence.py --solver both --jitters 0,0.05,0.1
"""

from __future__ import annotations

import argparse
from typing import Any

import warp as wp
wp.init()

import torch

from warpSPHCore import (
    ParticleState, SupportScheme, buildVerletList, warpOperationHVP,
    OperationProperties, castTorchToWarpAsBuiltins, castTorchToWarp, allocateTorchWarp,
)
from warpSPHCore.enumTypes import WarpOperation, OperationDirection, supportSchemeToUint
from warpSPHCore.kernels.hessian import sphKernelHessian
from warpSPHCore.dataTypes.domain_t import domainData
from warpSPHCore.dataTypes.kernelState_t import kernelState
from warpSPHCore.type_config import scalar_t
from warp.types import matrix

from warpSPH.utils import buildDomainDescription
from warpSPH.configurations.simulationConfig import buildConfig
from warpSPH.configurations.weaklyCompressible import WeaklyCompressibleSPHConfig
from warpSPH.configurations.moduleConfigurations.shifting import ShiftingImplicitOperator
from warpSPH.sample.regular import sampleRegularParticles
from warpSPH.modules.density import computeDensities
from warpSPH.modules.shifting.implicitShifting import _multiplyLaplacianBlock, _buildSystem, _buildDiagBlock
from warpSPH.modules.shifting.wp_implicitShifting import computeShiftingPairTerms
from warpSPH.modules.shifting.bicgstab import bicgstabSolve
from warpSPH.modules.shifting.gmres import gmresSolve

DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Self-Hessian probe (diffSPH's own Jacobi-preconditioner ingredient: `H_ii`
# alone, not `sum_{j!=i} omega_j H_ij` like the production `_buildSystem`
# uses -- see the plan doc's Phase 4 step 3 correction for why the self term
# is legitimately excluded from the *matvec*, but diffSPH still reads it
# directly for a cheap preconditioner diagonal).
# ---------------------------------------------------------------------------

@wp.kernel
def _selfHessianKernel(
    positions: wp.array(dtype=Any),
    supports: wp.array(dtype=scalar_t),
    domainState: domainData,
    kernelProperties: kernelState,
    outH: wp.array(dtype=Any),
):
    i = wp.tid()
    x = positions[i]
    h = supports[i]
    outH[i] = sphKernelHessian(x, x, h, h, kernelProperties, domainState)


def _selfHessian(positions: torch.Tensor, supports: torch.Tensor, domain, kernel) -> torch.Tensor:
    dim = domain.dim
    n = positions.shape[0]
    posW = castTorchToWarpAsBuiltins(positions.contiguous())
    supW = castTorchToWarp(supports.contiguous())
    d = domainData()
    d.domainMin = castTorchToWarp(domain.min)
    d.domainMax = castTorchToWarp(domain.max)
    d.periodicity = castTorchToWarp(domain.periodic)
    d.dim = dim
    k = kernelState()
    k.kernelFunction = kernel.value
    k.supportMode = supportSchemeToUint(SupportScheme.Gather)
    H_t, H_w = allocateTorchWarp(n, matrix(shape=(dim, dim), dtype=scalar_t), posW.device)
    wp.launch(_selfHessianKernel, dim=n, inputs=[posW, supW, d, k, H_w], device=posW.device)
    return H_t


# ---------------------------------------------------------------------------
# Test-case construction
# ---------------------------------------------------------------------------

def jitteredLatticeState(nx: int, dim: int, L: float, jitter: float, seed: int, device: torch.device):
    domain = buildDomainDescription(l=L, dim=dim, periodic=True, device=device, dtype=DTYPE)
    config, _integrator = buildConfig(dim=dim, nx=nx, domain=domain, device=device,
                                      dtype=DTYPE, dx=L / nx, cflFactor=0.3)
    template = sampleRegularParticles(nx=nx, domain=domain, targetNeighbors=config.targetNeighbors)
    n = template.positions.shape[0]
    particleDx = template.masses.pow(1 / dim).mean().item()
    gen = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(template.positions.shape, generator=gen).to(device=device, dtype=DTYPE)
    positions = template.positions + jitter * particleDx * noise
    state = ParticleState(positions=positions, supports=template.supports, masses=template.masses,
                          kinds=torch.zeros(n, device=device, dtype=torch.int32),
                          densities=torch.ones(n, device=device, dtype=DTYPE))
    return state, config, domain


def buildSystem(state, config, domain, schemeConfig):
    """Reuses the exact production code `computeImplicitShift` calls
    internally -- same matvec, same RHS, same preconditioner-ingredient
    diagonal -- so this script's numbers are the real solver's numbers, not
    a reimplementation that could silently diverge from it. (The harness's
    states always have `kinds == 0`, so `computeImplicitShift`'s
    boundary-`activeMask` on the matvec's `i`/`j`/`Hw` is the identity here.)"""
    dim = state.positions.shape[1]
    numParticles = state.positions.shape[0]
    rho0 = schemeConfig.fluid.restDensity

    adjacency = buildVerletList(state, domain, config.verletScale, SupportScheme.SuperSymmetric, None)
    state.densities = computeDensities(state, config, schemeConfig, adjacency)

    _K, J, H = computeShiftingPairTerms(state, domain, config.kernel, adjacency)
    i_all, j_all = adjacency.livePairs()[0], adjacency.livePairs()[1]
    Hw_all, B, x0 = _buildSystem(state, config, schemeConfig, domain, adjacency, i_all, j_all, J, H,
                                 rho0, dim, numParticles)
    diagBlock, i, j, Hw = _buildDiagBlock(schemeConfig.shiftProperties.implicitOperator,
                                          i_all, j_all, Hw_all, numParticles, dim,
                                          state.positions.device, state.positions.dtype)

    def matvec(x, diagBlock=diagBlock, i=i, j=j, Hw=Hw, numParticles=numParticles, dim=dim):
        return _multiplyLaplacianBlock(diagBlock, Hw, x, i, j, numParticles, dim)

    dx = config.dx.cpu().item() if isinstance(config.dx, torch.Tensor) else config.dx
    threshold = schemeConfig.shiftProperties.implicitSolverThreshold or dx / 2
    return dict(matvec=matvec, B=B, x0=x0, diagBlock=diagBlock, adjacency=adjacency,
               dx=dx, threshold=threshold, numParticles=numParticles, dim=dim, state=state)


def preconditionerDiagonal(sys: dict, kind: str, state, config, domain, schemeConfig) -> torch.Tensor:
    if kind == 'current':
        diag = torch.diagonal(sys['diagBlock'], dim1=-2, dim2=-1).flatten()
    elif kind == 'selfHessian':
        rho0 = schemeConfig.fluid.restDensity
        omega = state.masses / state.densities if schemeConfig.shiftProperties.summationDensity else state.masses / rho0
        Hself = _selfHessian(state.positions, state.supports, domain, config.kernel)
        diag = torch.diagonal(Hself * omega[:, None, None], dim1=-2, dim2=-1).flatten()
    elif kind == 'none':
        return None
    else:
        raise ValueError(f"unknown preconditioner kind {kind!r}")
    return torch.where(diag.abs() > 1e-8, 1.0 / diag, torch.zeros_like(diag))


def runOnce(sys: dict, schemeConfig, precond: torch.Tensor, maxiter: int, useThreshold: bool,
            verbose: bool, solver: str = 'bicgstab', restart: int = 30):
    common = dict(
        tol=schemeConfig.shiftProperties.implicitTolerance,
        rtol=schemeConfig.shiftProperties.implicitRelativeTolerance,
        maxiter=maxiter,
        precond=precond,
        threshold=sys['threshold'] if useThreshold else None,
        dim=sys['dim'], verbose=verbose,
    )
    if solver == 'gmres':
        xk, iters, convergence = gmresSolve(sys['matvec'], sys['B'], sys['x0'],
                                            restart=restart, **common)
    else:
        xk, iters, convergence = bicgstabSolve(sys['matvec'], sys['B'], sys['x0'], **common)
    resid = torch.linalg.norm(sys['matvec'](xk) - sys['B']).item()
    bnorm = torch.linalg.norm(sys['B']).item()
    maxShift = xk.view(-1, sys['dim']).norm(dim=-1).max().item()
    return dict(iters=iters, resid=resid, bnorm=bnorm, relResid=resid / max(bnorm, 1e-300),
               maxShift=maxShift, xk=xk)


ITER_LABEL = {-10: 'rho-breakdown', -11: 'omega-breakdown', -12: 'threshold-bailout',
              -13: 'stagnation', -14: 'maxiter-budget'}


def fmtIters(iters: int, maxiter: int) -> str:
    if iters < 0:
        return ITER_LABEL.get(iters, f'bailout({iters})')
    if iters >= maxiter - 1:
        return f'{iters}(maxiter)'
    return str(iters)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nx', type=int, nargs='+', default=[16], help='particles per side (repeat for multiple resolutions)')
    ap.add_argument('--dim', type=int, default=2)
    ap.add_argument('--L', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--jitters', type=str, default='0,0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2',
                    help='comma-separated jitter amounts, in particle-spacing units')
    ap.add_argument('--maxiter', type=int, default=None, help='override implicitMaxSolverIter (config default: 64)')
    ap.add_argument('--no-threshold', action='store_true', help='disable the divergence-threshold bailout to see the true convergence trend')
    ap.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--trace-jitter', type=float, default=None, help='print the solvers\' (verbose=True) full per-iteration trace for this one jitter value')
    ap.add_argument('--solver', type=str, default='bicgstab', choices=['bicgstab', 'gmres', 'both'],
                    help='which Krylov solver to run (default bicgstab; "both" prints one row per solver)')
    ap.add_argument('--restart', type=int, default=30, help='GMRES restart length m (only used with --solver gmres/both)')
    ap.add_argument('--compare-preconditioners', action='store_true', help='also solve with a diffSPH-style self-Hessian-only Jacobi diagonal and with no preconditioner')
    ap.add_argument('--compare-automatic', action='store_true', help='also check warpOperationHVP\'s matvec against the hand-built one on the same system')
    ap.add_argument('--operator', type=str, default='legacyPairwise', choices=['legacyPairwise', 'exactHessian'],
                    help='which ShiftingImplicitOperator to assemble (default legacyPairwise, matching production; '
                         'the jitter>=0.01 divergence this script documents is an exactHessian-specific finding -- '
                         'pass --operator exactHessian to reproduce it)')
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    jitters = [float(j) for j in args.jitters.split(',') if j.strip() != '']
    operator = ShiftingImplicitOperator[args.operator]

    for nx in args.nx:
        print(f"\n{'=' * 90}\nnx={nx}  dim={args.dim}  L={args.L}  device={device}  operator={args.operator}\n{'=' * 90}")
        header = f"{'solver':>10s} {'jitter':>8s} {'status':>18s} {'rel_resid':>11s} {'raw_resid':>11s} {'|b|':>10s} {'max|xk|/dx':>10s} {'thresh/dx':>10s}"
        print(header)

        solvers = {'bicgstab': ['bicgstab'], 'gmres': ['gmres'],
                   'both': ['bicgstab', 'gmres']}[args.solver]

        for jitter in jitters:
            state, config, domain = jitteredLatticeState(nx, args.dim, args.L, jitter, args.seed, device)
            schemeConfig = WeaklyCompressibleSPHConfig()
            schemeConfig.shiftProperties.active = True
            schemeConfig.surfaceDetectionConfig.active = False
            schemeConfig.shiftProperties.implicitOperator = operator

            sys = buildSystem(state, config, domain, schemeConfig)
            maxiter = args.maxiter or schemeConfig.shiftProperties.implicitMaxSolverIter

            verbose = args.trace_jitter is not None and abs(jitter - args.trace_jitter) < 1e-12
            if verbose:
                print(f"\n--- verbose trace: jitter={jitter} ---")

            precond = preconditionerDiagonal(sys, 'current', state, config, domain, schemeConfig)
            for solver in solvers:
                r = runOnce(sys, schemeConfig, precond, maxiter, not args.no_threshold,
                            verbose, solver=solver, restart=args.restart)
                print(f"{solver:>10s} {jitter:8.3f} {fmtIters(r['iters'], maxiter):>18s} {r['relResid']:11.3e} "
                     f"{r['resid']:11.3e} {r['bnorm']:10.3e} {r['maxShift']/sys['dx']:10.3f} "
                     f"{sys['threshold']/sys['dx']:10.3f}")

            if args.compare_preconditioners:
                for kind, label in (('selfHessian', 'diffSPH-style (H_ii*omega)'), ('none', 'no preconditioner')):
                    p = preconditionerDiagonal(sys, kind, state, config, domain, schemeConfig)
                    rr = runOnce(sys, schemeConfig, p, maxiter, not args.no_threshold, False)
                    print(f"    [{label:28s}] {fmtIters(rr['iters'], maxiter):>18s} rel_resid={rr['relResid']:.3e}")

            if args.compare_automatic:
                rho0 = schemeConfig.fluid.restDensity
                omega = state.masses / rho0
                omegaState = ParticleState(positions=state.positions, supports=state.supports, masses=omega,
                                           kinds=state.kinds, densities=None)
                props = OperationProperties(kernel=config.kernel, operation=WarpOperation.Density,
                                            supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)
                v = r['xk'].view(-1, sys['dim'])
                automaticHv = warpOperationHVP(omegaState, props, domain, adjacency=sys['adjacency'],
                                               tangentQueryPositions=v, tangentReferencePositions=v)
                handBuiltHv = sys['matvec'](r['xk']).view(-1, sys['dim'])
                diff = (automaticHv - handBuiltHv).abs().max().item()
                print(f"    [automatic-vs-hand-built matvec @ xk]  max_abs_diff={diff:.3e}")

    print()
    print("status legend: negative codes are the solvers' early-exit statuses (see")
    print("  bicgstab.py/gmres.py): -10 rho-breakdown, -11 omega-breakdown (BiCGStab),")
    print("  -12 threshold-bailout, -13 stagnation (GMRES), -14 max-iteration budget")
    print("  exhausted, -15 stagnation (Richardson). These statuses are acted on by")
    print("  solverDriver.solveImplicitSystem once shiftProperties.implicitFallback is set:")
    print("  the default `none` keeps the historical use-the-bailed-iterate behavior, while")
    print("  `krylov` retries the other Krylov solver and `krylov_richardson` adds a bounded")
    print("  Richardson polish (see docs/regression/implicit_shifting_operator_choice.md).")
    print("  raw_resid is this harness's own ||b - A xk|| recompute")
    print("  (independent of the solvers' convergence lists; both solvers now append")
    print("  that same value as their final history entry on every return). Watch")
    print("  rel_resid: production tolerances target ~1e-4 relative. With the default")
    print("  --operator legacyPairwise (production default) rel_resid stays near or")
    print("  under that target through jitter=0.2. With --operator exactHessian,")
    print("  rel_resid at jitter=0.1 is typically 2-3 orders of magnitude above it.")


if __name__ == "__main__":
    main()
