"""Probe (`ACSPH_PLAN.md` Part 8 step 6, De Courcy et al.'s Table 1/2): the
real-time Courant number `CFL_t` (Eq. 46) and the real/pseudo time-step ratio
`Δt/Δτ` sweep on `oscillatingDroplet`, the paper's own headline acceptance
test for the dual-time machinery -- "for the same recorded error as δ-SPH the
solution cost can be matched or reduced".

Table 1's cells are `IRMSE(KE)/IRMSE(a)/w/e`, all normalised by the δ-SPH
solution:

- **`IRMSE`** (paper's Eq., §4.3): the RMS error against the *exact* inviscid
  analytic solution, "equivalent to the discrete RMSE if the data is equally
  spaced in time" -- i.e. `sqrt(mean((measured - analytic)**2))` over the
  recorded trajectory. `oscillatingDroplet.analyticSolution` is that ground
  truth: Monaghan & Rafiee (2013)'s elliptical-drop ODE (`literature/`,
  DOI 10.1002/fld.3671), re-derived from Appendix A and verified against the
  case's own encoded `DROPLET_STRETCH`/`DROPLET_PERIOD` (see that function's
  docstring for the OCR pitfall it caught).
- **`w`**: wall-clock time, already recorded per step (`stepTime_ms`) --
  summed over the run.
- **`e`** (Eq. 65): `integral_0^T (m_iter * s_RK / dt) dt`. Since diagnostics
  are recorded once per real step of size `dt`, this integral telescopes to
  `sum_steps(m_iter * s_RK)` exactly -- no numerical quadrature needed. `m_iter`
  is `pseudoIterations` (1 for delta-SPH, the real pseudo-iteration count for
  ACSPH -- `oscillatingDroplet.diagnostics` reads it off
  `ctx.scratch['lastStageUpdate']`, `runner.py`'s per-step stash of the
  scheme's own update object). `s_RK` is a *run-level* constant (the
  pseudo-time RK stage count for ACSPH, the real-time integrator's stage count
  for delta-SPH), not a per-step column, per Eq. (65)'s own reading of `s_RK`,
  the number of Runge-Kutta stages.

**Scale**: this reproduces the *mechanism* on a scaled-down grid and step
count, not the paper's own hardware-scale reproduction (their Table 1 runs to
enough oscillations that the "10th oscillation" pressure field in their Fig. 17
is meaningful; a single real step already costs ~1s at nx=32 on one CPU core
here, so a faithful multi-period, `L/Δx=200`-scale sweep is a much larger,
separately-budgeted run). `--nx`/`--periods`/`--cflT`/`--dtOverDtau` are all
exposed so the grid can be widened once the mechanism is confirmed correct.

**`maxDt` must be raised for this sweep to mean anything.** `config.maxDt`
defaults to `1e-2` (`runner/caseSpec.py`, a generic global default with no
relation to ACSPH's own natural timescale) and is *always* one of Eq. (46)'s
`min()` candidates (`modules/timestep/artificialCompressible.py`) -- so if it
is smaller than every `cflT`-scaled advective candidate being tested, `dt`
silently pins to `maxDt` regardless of `cflT`, and the whole sweep measures
one flat row. Caught exactly this way on the first run of this script: all
four `cflT` values in `[0.1, 0.6]` produced bit-identical trajectories.
`--maxDt` is set generously above what any tested `cflT` needs by default.

Usage:
  python scripts/probe_acsphOscillatingDropletTable1.py [--nx 32]
      [--periods 1.0] [--cflT 0.1 0.2 0.4 0.6] [--dtOverDtau 2 5]
      [--rkStages 3]
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=32)
parser.add_argument('--periods', type=float, default=1.0)
parser.add_argument('--cflT', type=float, nargs='+', default=[0.1, 0.2, 0.4, 0.6])
parser.add_argument('--dtOverDtau', type=float, nargs='+', default=[2.0, 5.0])
parser.add_argument('--rkStages', type=int, default=3)
parser.add_argument('--maxDt', type=float, default=1.0,
                     help='config.maxDt -- must exceed every cflT candidate tested, or dt pins here regardless of cflT (see module docstring)')
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import numpy as np

from warpSPH.cases.oscillatingDroplet import (DROPLET_PERIOD, analyticSolution,
                                               oscillatingDropletCase as case)
from warpSPH.runner import run

TLIMIT = args.periods * DROPLET_PERIOD
#: Real-time integrator stage count for the delta-SPH reference (Eq. 65's
#: `s_RK`) -- `oscillatingDropletCase`'s own default, `rungeKutta2`.
DELTA_SPH_RK_STAGES = 2


def _irmse(measured, analytic):
    return float(np.sqrt(np.mean((measured - analytic) ** 2)))


def _measure(r, rkStages: int):
    """`(IRMSE(KE), IRMSE(a), w, e)` for one completed run `r`."""
    t = r.series('t')
    ke = r.series('kineticEnergy')
    a = r.series('semiAxisA')
    stepTimeMs = r.series('stepTime_ms')
    pseudoIters = r.series('pseudoIterations')

    aAnalytic, _, keAnalytic = analyticSolution(t)
    irmseKE = _irmse(ke, keAnalytic)
    irmseA = _irmse(a, aAnalytic)
    w = float(np.nansum(stepTimeMs)) / 1000.0  # seconds
    e = float(np.nansum(pseudoIters)) * rkStages
    return irmseKE, irmseA, w, e


def _configureAcsph(cflT, dtOverDtau, rkStages):
    base = case.configureScheme

    def wrapped(ctx):
        base(ctx)
        acParams = ctx.schemeConfig.acParams
        acParams.cflT = cflT
        acParams.dtOverDtau = dtOverDtau
        acParams.rkStages = rkStages

    return wrapped


def main():
    print(f"nx={args.nx}, periods={args.periods} (tLimit={TLIMIT:.3f}), "
          f"RK stages={args.rkStages}")

    print("Running delta-SPH reference...", flush=True)
    ref = run(case, nx=args.nx, tLimit=TLIMIT, nSteps=None, maxDt=args.maxDt,
              store=False, plot=False, quiet=True, progress=False)
    refIrmseKE, refIrmseA, refW, refE = _measure(ref, DELTA_SPH_RK_STAGES)
    print(f"  delta-SPH: diverged={ref.diverged} steps={ref.nSteps} "
          f"IRMSE(KE)={refIrmseKE:.4g} IRMSE(a)={refIrmseA:.4g} "
          f"w={refW:.4g}s e={refE:.4g}")
    print()

    hdr = f"{'CFL_t':>6} {'dt/dtau':>8} {'steps':>6} {'div':>4}  {'IRMSE(KE)':>10} {'IRMSE(a)':>9}  {'w':>9} {'e':>9}"
    print(hdr)
    print('-' * len(hdr))

    for cflT in args.cflT:
        for dtOverDtau in args.dtOverDtau:
            _orig = case.configureScheme
            case.configureScheme = _configureAcsph(cflT, dtOverDtau, args.rkStages)
            try:
                r = run(case, scheme='artificialCompressible', nx=args.nx,
                        tLimit=TLIMIT, nSteps=None, maxDt=args.maxDt,
                        store=False, plot=False, quiet=True, progress=False)
            finally:
                case.configureScheme = _orig

            if r.trajectory and 'kineticEnergy' in r.trajectory[-1]:
                irmseKE, irmseA, w, e = _measure(r, args.rkStages)
                print(f"{cflT:6.2f} {dtOverDtau:8.1f} {r.nSteps:6d} "
                      f"{('yes' if r.diverged else 'no'):>4}  "
                      f"{irmseKE / refIrmseKE:10.3f} {irmseA / refIrmseA:9.3f}  "
                      f"{w / refW:9.3f} {e / refE:9.3f}")
            else:
                print(f"{cflT:6.2f} {dtOverDtau:8.1f}   (no finite step)")


if __name__ == '__main__':
    main()
