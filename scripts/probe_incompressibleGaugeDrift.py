"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 4, 2026-08-26): `solveIncompressible`
(the DI/PS constant-density pressure solver, called every step from
`IncompressibleSystem.finalize`) has **no working gauge fix** for the pure-
Neumann null space its PPE operator shares with `solveDivergenceFree` (see
`probe_pressureGaugeDrift.py`, which found that solver's per-iteration mean-
recentering works correctly). `solveIncompressible`'s only "gauge fix" is a
physically-motivated `torch.clamp(pressureB, min=0.0)` (pressure isn't
tensile) -- a floor, not a mean-center, so nothing stops the field's mean
from drifting *upward* without bound.

This is a self-contained reimplementation of `solveIncompressible`'s
relaxed-Jacobi loop (matched byte-for-byte against the real solver on the
baseline case before trusting any variant below), run against
`kolmogorovIncompressible` (periodic, no solid boundary, no free surface --
the case this investigation deliberately isolated to, since it's the
building block every other DFSPH case sits on top of). It exists as one
script with toggles, not several, because the three questions it answers
share almost all of their plumbing:

  --no-clamp        remove the non-negative floor entirely
  --maxIters N       override `pressureSolver.maxIterations`
  --jitter X         monkeypatch `sampleRegularParticles`'s own `jitter`
                      argument (kolmogorovIncompressible hardcodes
                      `jitter=0.0` and does not read `CaseSpec.samplingScheme`
                      at all -- confirmed by testing `--samplingScheme
                      jittered`/`optimal` through the real CLI path, which
                      changed nothing; this flag is the only way to actually
                      perturb the initial lattice for this case)

Findings so far (all at nx=64, 300 steps unless noted):
  - Baseline (clamp on, default maxIterations=64): `pMean` climbs from ~1e-3
    to ~9.8 (comparable to `pStd` itself -- i.e. much of the "pressure
    signal" at that point is an unphysical uniform offset, not real spatial
    variation), `nIter` pegs at 64 (never converges) from step ~30-40 onward.
    A longer nx=128/1000-step run under the same config actually reaches
    NaN (`pMean` -> 2.38e6 -> NaN by step 574) -- a live, reproducible
    blowup, not a hypothetical.
  - `--no-clamp`: nearly identical trajectory to the baseline (peaks ~8.5 vs
    ~9.8). The clamp isn't *causing* the drift, it's just not defending
    against it.
  - `--jitter 0.1`: a large one-time initial spike (`pMean=36.8` at step 0)
    that resolves in 6 iterations, then the trajectory tracks the unjittered
    baseline closely from step 1 on. Breaking the initial lattice symmetry
    trades a one-step transient for the same subsequent drift, it doesn't
    prevent it -- the mechanism re-emerges dynamically regardless of the
    starting condition.
  - `--maxIters 1024` (16x default): makes it **categorically worse**, not
    better -- `pMean` peaks at 61.4 (vs 9.8), `pMax` at 194 (vs 34), still
    pegged at the *new* cap throughout. This is the decisive result: the
    null-space/mean component isn't attributable to any degree of freedom
    the residual can actually correct, so every additional iteration adds
    another unopposed increment. More iterations is strictly more of the
    problem -- the fix has to be architectural, not a convergence-tuning
    knob.
  - The persistent negative bias driving all of this (`sourceTerm.mean()`,
    logged every step) is *not* from the `rhoStar` floor-clamp at 0.9
    (confirmed: 0 particles clamped throughout a 200-step run) -- it's a
    genuine, systematic tendency for predicted density to sit above `rho0`.
    Separately confirmed (`probe_densitySign.py` in this directory, or see
    the plan doc) that DFSPH's bulk density genuinely runs dense on average
    (`mean(rho-1) = +2.7e-3`, ~76% of the unsigned error, 82% of particles
    above `rho0`) -- the same sign/magnitude, suggesting this drift and
    Part 2's still-open "DFSPH bulk density-band gap vs deltaSPH" share one
    upstream cause.

Resolved (2026-08-27), using this script's `--null-test`, `--gauge` and
`--project-source` flags:
  - `--null-test` measures how near-null the constant mode is: on the t=0
    lattice `|A*1| = 6.8e-11` against `|A*rand| = 2.8e-5`, and by step 59 of
    the developed flow `|A*1|` has risen to only ~5% of a same-magnitude
    non-constant field's response. So the mean error is weakly correctable,
    but only at ~20x the amplitude a normal mode would need -- which is the
    pressure mean reaching 2.4e6 before float32 gives out.
  - The source term's mean is *structurally* unreachable, not buggy: the SPH
    summation density's particle average is minimised by the lattice and
    rises quadratically with disorder (`probe_densityBiasVsDisorder.py`), so
    the solve is an integral controller winding up against a setpoint it
    cannot reach.
  - `--gauge` A/B (nx=64, 300 steps, second-half means): `clamp` (historical)
    `mean|rho-1|`=3.83e-3, drifts; `center` (what `solveDivergenceFree` does)
    **NaN at step 155**; `center-clamp` **NaN at step 136**; `minshift`
    (subtract the fluid min) 2.86e-3 with no drift -- the winner, now landed
    as `ShiftPressureGauge.minShift`.
  - `--project-source` (zero-mean the source) makes the solver *converge*
    (nIter 64 -> 13.4) but the density gets worse: part of the source's mean
    is the genuine de-clump signal, not just the unreachable part. Rejected.
See `DFSPH_IMPROVEMENT_PLAN.md` Part 4, and `probe_shiftPressureGauge.py`
for the end-to-end A/B through the real solver.

Still not done: the Krylov path's intra-solve behavior (it returns through
`solvePressureKrylov` with `gauge='nonnegative'` before ever reaching the
relaxed-Jacobi loop, so it never sees `ShiftPressureGauge` and still has the
"floor, not a gauge" problem this script diagnosed).

Usage: `python scripts/probe_incompressibleGaugeDrift.py [--nx 64]
[--nsteps 300] [--gauge minshift] [--null-test] [--project-source]
[--no-clamp] [--maxIters 1024] [--jitter 0.1]`
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=64)
parser.add_argument('--nsteps', type=int, default=300)
parser.add_argument('--no-clamp', action='store_true')
parser.add_argument('--maxIters', type=int, default=None)
parser.add_argument('--jitter', type=float, default=None)
parser.add_argument('--project-source', action='store_true',
                    help="subtract the fluid mean from `sourceTerm` before the "
                         "solve (the PPE compatibility projection: see --null-test)")
parser.add_argument('--gauge', default='clamp',
                    choices=['clamp', 'center', 'center-clamp', 'minshift',
                             'quantile', 'none'],
                    help="how the iterate's null-space (constant) component is "
                         "pinned each iteration: 'clamp' = the real solver's "
                         "non-negative floor (baseline); 'center' = subtract the "
                         "fluid mean, as `solveDivergenceFree` does; "
                         "'center-clamp' = subtract the mean, then re-floor at 0; "
                         "'minshift' = subtract the fluid min (non-negative *and* "
                         "gauge-fixed, shape preserved); 'quantile' = subtract "
                         "the `--gaugeQuantile` quantile, then re-floor at 0 (a "
                         "minshift that is not hostage to a single outlier "
                         "particle); 'none' = nothing")
parser.add_argument('--gaugeQuantile', type=float, default=0.01)
parser.add_argument('--cflFactor', type=float, default=None,
                    help="override the case's cflFactor. Part 8 item 3: the "
                         "default 0.3 is applied to the support radius h, and "
                         "n_h=4 makes that 1.2 particle spacings per step, 3x "
                         "Bender & Koschier's published `dt <= 0.4 d/|v_max|`. "
                         "0.1 matches the paper.")
parser.add_argument('--setpointEps', type=float, default=0.0,
                    help="DFSPH_IMPROVEMENT_PLAN.md Part 8 item 1: move the "
                         "solve's setpoint off the lattice floor. The case "
                         "normalises mass so the *lattice* summation density "
                         "equals rho0 (kolmogorovIncompressible.py:71), and "
                         "Part 4 proved the lattice MINIMISES mean(rho) -- so "
                         "rho0 sits exactly at a value no disordered "
                         "configuration can reach, and `sourceTerm = rho0 - "
                         "rhoStar` keeps a permanent negative mean. This "
                         "rescales mass by 1/(1+eps) after the case builds, "
                         "putting the lattice at rho0/(1+eps) so a "
                         "configuration carrying a density bias of eps lands "
                         "*on* rho0. eps=0 reproduces the historical setpoint "
                         "byte-for-byte. Neither DFSPH nor IISPH ties rho0 to "
                         "the sampled configuration at all; this flag is the "
                         "cheapest way to ask what happens when it doesn't.")
parser.add_argument('--null-test', action='store_true',
                    help="each step, apply the IISPH operator to a constant "
                         "field and to a random field and log the means, to "
                         "test whether constants are in the operator's null "
                         "space (=> a nonzero-mean source term is infeasible)")
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch

from warpSPH.cases import kolmogorovIncompressible as mod
from warpSPH.runner.cli import caseMain
from warpSPHCore import SupportScheme
from warpSPH.modules.incompressible.wp_alpha import computeAlpha
from warpSPH.modules.pressure.iisph import computePressureAccelIISPH
from warpSPH.modules.incompressible.drift import computePressureShiftIISPH
from warpSPH.modules.incompressible.incompressible import computeMomentumIncompressible

log = []

import warpSPH.systems.incompressible as sysmod


def custom_solveIncompressible(particles, config, schemeConfig, adjacency, dvdt, dt, verbose=False):
    """Mirrors `modules/incompressible/incompressible.py::solveIncompressible`
    exactly, except the clamp and `maxIterations` are overridable via this
    script's CLI flags. Keep in sync with the real function if it changes --
    this script's baseline (no flags) is checked to reproduce the real
    solver's trajectory byte-for-byte before trusting any variant."""
    minIters = schemeConfig.solverConfig.pressureSolver.minIterations
    maxIters = args.maxIters if args.maxIters is not None else schemeConfig.solverConfig.pressureSolver.maxIterations
    threshold = schemeConfig.solverConfig.pressureSolver.tolerance
    omega = schemeConfig.solverConfig.pressureSolver.relaxationFactor

    predictedVelocities = particles.velocities + dt * dvdt
    apparentArea = particles.masses / particles.densities
    divergence = computeMomentumIncompressible(
        currentState=particles, config=config, schemeConfig=schemeConfig,
        adjacency=adjacency, advectionVelocities=predictedVelocities)
    rho0 = schemeConfig.fluid.restDensity
    rhoStarRaw = particles.densities + dt * divergence
    rhoStar = torch.clamp(rhoStarRaw, min=0.9)
    sourceTerm = rho0 - rhoStar
    nClamped = int((rhoStarRaw < 0.9).sum().item())
    stMeanRaw = sourceTerm.mean().item()
    if args.project_source:
        sourceTerm = sourceTerm - sourceTerm[particles.kinds == 0].mean()

    alphas = dt ** 2 * computeAlpha(
        currentState=particles, config=config, schemeConfig=schemeConfig,
        adjacency=adjacency, apparentVolumes=apparentArea)
    alphas = torch.clamp(alphas, max=-1e-6)

    fluidMask = particles.kinds == 0

    def applyOperator(p):
        """The solver's own PPE operator, `A p = dt^2 * shift(accel(p))` --
        exactly the `dx_p` of the iteration below, factored out so it can be
        probed on fields the iteration never visits (a constant, a random
        field) without changing what the iteration does."""
        ap = computePressureAccelIISPH(state=particles, pressureValues=p, config=config,
                                       supportScheme=SupportScheme.Scatter, adjacency=adjacency)
        return dt ** 2 * computePressureShiftIISPH(state=particles, config=config, pressureAccels=ap,
                                                   supportScheme=SupportScheme.Scatter, adjacency=adjacency)

    nullStats = {}
    if args.null_test:
        ones = torch.ones_like(sourceTerm)
        dxOnes = applyOperator(ones)
        g = torch.Generator(device=sourceTerm.device); g.manual_seed(0)
        rand = torch.rand(sourceTerm.shape, generator=g, device=sourceTerm.device, dtype=sourceTerm.dtype)
        dxRand = applyOperator(rand)
        nullStats = dict(
            # |A*1| relative to |A| applied to a same-scale non-constant field:
            # ~0 => constants are in the null space.
            dxOnesAbs=dxOnes.abs().mean().item(),
            dxRandAbs=dxRand.abs().mean().item(),
            # mean(A*x) relative to mean|A*x|: ~0 => the operator's *range* is
            # mean-zero, i.e. no pressure field can produce a nonzero-mean
            # density correction, i.e. a nonzero-mean source term is infeasible.
            dxRandMean=dxRand.mean().item(),
            dxOnesMean=dxOnes.mean().item(),
        )

    boundaryPressure = particles.pressures.clone()
    pressureA = particles.pressures.clone() * 0.
    pressureA = torch.where(fluidMask, pressureA, boundaryPressure)
    pressureB = pressureA.clone()

    errors = []
    for i in range(maxIters):
        pressureA = pressureB.clone()
        a_p = computePressureAccelIISPH(state=particles, pressureValues=pressureA, config=config,
                                         supportScheme=SupportScheme.Scatter, adjacency=adjacency)
        dx_p = dt ** 2 * computePressureShiftIISPH(state=particles, config=config, pressureAccels=a_p,
                                                     supportScheme=SupportScheme.Scatter, adjacency=adjacency)
        residual = sourceTerm - dx_p
        pressureB = pressureA + omega * residual / alphas
        if args.gauge == 'clamp':
            if not args.no_clamp:
                pressureB = torch.clamp(pressureB, min=0.0)
        elif args.gauge == 'center':
            pressureB = pressureB - pressureB[fluidMask].mean()
        elif args.gauge == 'center-clamp':
            pressureB = torch.clamp(pressureB - pressureB[fluidMask].mean(), min=0.0)
        elif args.gauge == 'minshift':
            pressureB = pressureB - pressureB[fluidMask].min()
        elif args.gauge == 'quantile':
            q = torch.quantile(pressureB[fluidMask], args.gaugeQuantile)
            pressureB = torch.clamp(pressureB - q, min=0.0)
        pressureB = torch.where(fluidMask, pressureB, boundaryPressure)

        residual_clamped = torch.clamp(-residual, min=-threshold)
        error = torch.mean(residual_clamped[fluidMask]).cpu().item()
        errors.append(error)
        if not torch.isfinite(pressureB).all():
            break
        if i >= minIters and error < threshold:
            break

    a_p = computePressureAccelIISPH(state=particles, pressureValues=pressureB, config=config,
                                     supportScheme=SupportScheme.Scatter, adjacency=adjacency)
    a_p = torch.where(fluidMask.unsqueeze(-1), a_p, torch.zeros_like(a_p))

    log.append(dict(
        **nullStats,
        stMeanRaw=stMeanRaw,
        dxMean=dx_p.mean().item(), dxAbsMean=dx_p.abs().mean().item(),
        resMean=residual.mean().item(), resAbsMean=residual.abs().mean().item(),
        stMean=sourceTerm.mean().item(), stAbsMean=sourceTerm.abs().mean().item(),
        pMean=pressureB.mean().item(), pStd=pressureB.std().item(), pMax=pressureB.max().item(),
        nIter=len(errors), finalErr=errors[-1] if errors else float('nan'),
        alphaStd=alphas.std().item(), rhoStd=particles.densities.std().item(),
        rhoBias=(particles.densities - rho0).mean().item(),
        rhoErr=(particles.densities - rho0).abs().mean().item(),
        rhoMax=(particles.densities - rho0).abs().max().item(),
        nClamped=nClamped,
        finite=bool(torch.isfinite(pressureB).all()),
    ))
    return a_p, pressureB, errors, [(pressureB.min().item(), pressureB.max().item(), pressureB.mean().item())]


sysmod.solveIncompressible = custom_solveIncompressible

if args.jitter is not None:
    import warpSPH.sample.weaklyCompressible as wcsample
    import warpSPH.sample.regular as regmod
    _orig_sample = regmod.sampleRegularParticles

    def _jittered(nx, domain, targetNeighbors, jitter=0.0, band=0, shortEdge=True):
        return _orig_sample(nx, domain, targetNeighbors, jitter=args.jitter, band=band, shortEdge=shortEdge)

    wcsample.sampleRegularParticles = _jittered

if args.setpointEps != 0.0:
    # Post-scale the case's own mass normalisation rather than editing it, so
    # the eps=0 path stays byte-identical to the shipped case.
    _origBuild = mod.kolmogorovIncompressibleCase.buildSystem

    def _offsetBuild(ctx, _origBuild=_origBuild):
        system = _origBuild(ctx)
        system.state.masses = system.state.masses / (1.0 + args.setpointEps)
        # Recompute rather than clear: the runner takes a t=0 diagnostics
        # sample before the first step, which reads `densities` directly.
        # (Every step recomputes it anyway -- `integrateRho` is False by
        # default -- so this only matters for that one pre-step sample.)
        from warpSPH.modules.density import computeDensities
        system.state.densities = computeDensities(
            system.state, ctx.config, ctx.schemeConfig, None)
        # The forcing is `accel * mass` and `divergenceFree_step` divides it back out,
        # so the applied acceleration is unchanged by the rescale.
        return system

    mod.kolmogorovIncompressibleCase.buildSystem = _offsetBuild

_argv = ['--nx', str(args.nx), '--nSteps', str(args.nsteps), '--tLimit', '1000.0',
         '--quiet', '--no-store', '--no-plot']
if args.cflFactor is not None:
    _argv += ['--cflFactor', str(args.cflFactor)]
result = caseMain(mod.kolmogorovIncompressibleCase, argv=_argv)

n = len(log)
label = (f"nx={args.nx} no_clamp={args.no_clamp} maxIters={args.maxIters} jitter={args.jitter} "
         f"project_source={args.project_source} setpointEps={args.setpointEps} gauge={args.gauge} "
         f"cflFactor={args.cflFactor}")
print(f"\n=== {label}: {n} steps recorded ===")
print(f"{'step':>5} {'stMean':>11} {'pMean':>11} {'pStd':>11} {'pMax':>11} {'nIter':>6} "
      f"{'rhoBias':>11} {'rhoErr':>11} {'rhoMax':>11} {'nClamped':>9} {'fin':>4}")
show = sorted(set([0, 1, 2, 5, 10, 20] + list(range(0, n, max(1, n // 30))) + [n - 1]))
for i in show:
    if i >= n:
        continue
    e = log[i]
    print(f"{i:5d} {e['stMean']:11.3e} {e['pMean']:11.3e} {e['pStd']:11.3e} {e['pMax']:11.3e} {e['nIter']:6d} "
          f"{e['rhoBias']:11.3e} {e['rhoErr']:11.3e} {e['rhoMax']:11.3e} {e['nClamped']:9d} {str(e['finite']):>4}")

if n:
    print(f"\npMean: first={log[0]['pMean']:.4e} last={log[-1]['pMean']:.4e} max={max(e['pMean'] for e in log):.4e}")
    print(f"finite throughout: {all(e['finite'] for e in log)}")
    fin = [e for e in log if e['finite']]
    tail = fin[len(fin) // 2:]
    if tail:
        print(f"2nd-half means: rhoBias={sum(e['rhoBias'] for e in tail) / len(tail):.4e} "
              f"rhoErr={sum(e['rhoErr'] for e in tail) / len(tail):.4e} "
              f"rhoMax={sum(e['rhoMax'] for e in tail) / len(tail):.4e} "
              f"|pMean|={sum(abs(e['pMean']) for e in tail) / len(tail):.4e} "
              f"pStd={sum(e['pStd'] for e in tail) / len(tail):.4e} "
              f"nIter={sum(e['nIter'] for e in tail) / len(tail):.1f}")
    if args.null_test:
        print(f"\n--- null-space test (is the constant field annihilated by A?) ---")
        print(f"{'step':>5} {'|A*1|':>11} {'|A*rand|':>11} {'mean(A*1)':>11} {'mean(A*rand)':>13} "
              f"{'mean(A*p_k)':>12} {'|A*p_k|':>11} {'mean(res)':>11} {'stMeanRaw':>11}")
        for i in show:
            if i >= n:
                continue
            e = log[i]
            print(f"{i:5d} {e['dxOnesAbs']:11.3e} {e['dxRandAbs']:11.3e} {e['dxOnesMean']:11.3e} "
                  f"{e['dxRandMean']:13.3e} {e['dxMean']:12.3e} {e['dxAbsMean']:11.3e} "
                  f"{e['resMean']:11.3e} {e['stMeanRaw']:11.3e}")
    nIters = [e['nIter'] for e in log]
    peakIdx = max(range(n), key=lambda i: nIters[i])
    print(f"nIter peak: step {peakIdx}, nIter={nIters[peakIdx]}, "
          f"alphaStd={log[peakIdx]['alphaStd']:.4e}, rhoStd={log[peakIdx]['rhoStd']:.4e}")
