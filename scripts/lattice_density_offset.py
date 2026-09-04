#!/usr/bin/env python3
"""The SPH summation density of a *perfect* lattice is not exactly `rho0`.

Why. A particle's density is `rho_i = sum_j m_j W(|x_ij|, h)`, which is a
midpoint quadrature of `int W dV = 1` sampled on the lattice. The quadrature is
not exact, so a defect-free lattice of spacing `s` carrying the nominal mass
`m = rho0 s^d` integrates to `rho0 * L` with `L != 1`. `L` depends only on the
**ratio `h/s`**: sliding `h` moves every neighbour's `q = r/h` and so changes
its weight continuously, and at particular ratios a whole lattice shell crosses
the support boundary and enters or leaves the sum. Nothing about this is a bug
or a free-surface effect -- it is uniform through the bulk.

Consequence, and the reason this script exists: a case whose particle mass is
the nominal `rho0 * s^d` hands the pressure solver a fluid whose measured
density is `rho0 * L`, i.e. a uniform apparent compression it will immediately
try to relieve. On `sloshingTank` that shows up as a startup velocity impulse.

TWO legitimate fixes, and one that is NOT:

  * **Correct the mass** (`--mass-correction`). Scale `m` by `1/L` at fixed
    spacing and fixed `h`. Free, exact for the ideal lattice, and the
    perturbation is small (0.12 % at `h/s = 4`) -- but it *is* a change to the
    physical fluid mass.
  * **Widen the support** (`--solve-h`). Keep the mass exact and choose `h` so
    the quadrature error itself is below a tolerance. More numerically
    sensible, and it is the only one of the two that improves the *operators*
    rather than just the density; the cost is neighbour count, which grows as
    `(h/s)^d`.
  * **NOT: change the particle spacing / sampling frequency.** `dx` is chosen
    so it evenly divides the domain, which is what puts the domain bounds a
    clean `dx/2` from the real surface -- the boundary and ghost-particle
    representation depends on that. Re-sampling to chase a nicer `L` perturbs
    the initial geometry discontinuously and unpredictably. Picking a "good"
    `nx` is dodging the problem, not fixing it.

    python scripts/lattice_density_offset.py                     # h/s sweep
    python scripts/lattice_density_offset.py --hs 4.0            # one ratio
    python scripts/lattice_density_offset.py --solve-h --tol 1e-3
    python scripts/lattice_density_offset.py --validate sloshingTank

Method. `warpSPHCore.util.latticeDensity` -- see its module docstring. It
evaluates the repo's OWN kernel (`eval_k`/`eval_C_d`), so there is no
transcription here to drift out of sync, and it offers three routes: the exact
shell sum (default), the Poisson/reciprocal-space identity, and a sum-free
closed form for the Wendland family. `--method` picks between them; they agree
to 1e-6 or better and `shells` is exact to machine precision.

How omniSPH handles this (checked in `~/dev/omniSPH`, same Wendland2 kernel:
`7/pi/h^2 (1-q)^4 (1+4q)`, `simulation/2DMath.h:68`). It does address it, and by
a third route -- it fixes the RATIO and snaps the geometry to the lattice,
which is the mirror image of this codebase's convention:

  * `simulation/SPH.h`: `packing_2D = 0.399200743165053487`, and the spacing is
    `s = packing_2D * h`, i.e. `h/s = 2.505005` is a hard constant. Particle
    area is `pi r^2` with `h = r sqrt(targetNeighbors)`, `targetNeighbors = 20`.
  * `simulation/config.cpp:428`: emitter bounds are snapped OUTWARD to an
    integer multiple of `packing` ("Changed emitterMax from ... to ..."), and
    then `domain.min/max = emitterExtent -/+ packing/2`. The domain is derived
    from the lattice, half a spacing outside the outermost particles.

So omniSPH gets the clean half-spacing boundary offset by fixing `s` and moving
the domain; warpSPH gets it by fixing the domain so `dx` divides it and letting
the sampler pick `s`. That is exactly why the offset is a single constant there
and varies with `nx` here.

Their calibration is this script's `areaRatio * L` product: assigned area over
cell area times the quadrature factor. At their shipped constant that is
`0.985683 * 1.013588 = 0.999076` -- so a perfect omniSPH lattice reads 0.09 %
BELOW rest density, not exactly at it. The packing that would make it exactly 1
is `0.399007431645928`. (Curiosity, not a claim: the shipped and exact values
share the digit run `0074316` offset by one place -- `0.3992|0074316|5053` vs
`0.399|0074316|45928` -- which is what a single inserted digit would look like.
Their commented-out earlier value `0.39960767069134208` gives 0.997137, so the
live one is the better of the two either way.)

Validation. `--validate <case>` runs the case for one step, reads back the
*achieved* lattice spacing, `h`, the particle mass and the measured summation
density, and reports predicted vs measured. On `sloshingTank` this separates
the two independent error sources:

  * `L` itself -- constant ~1.0012 at `h/s ~ 4`, the same at every `nx`;
  * `m / (rho0 s^d)` -- the *sampler* fitting the fluid block to the region and
    leaving the achieved spacing `s` different from the nominal `dx` the mass
    was computed from. This is the nx-dependent part, and the larger one
    (1.0004 at nx=200, 1.0142 at nx=100).

Both are removed by computing the mass from the achieved spacing and dividing
by `L`.
"""
from __future__ import annotations

import argparse
import itertools
import math

import numpy as np

from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.util import latticeDensity as _latticeDensity

#: Every kernel the core evaluator knows about, by name.
KERNELS = tuple(k.name for k in KernelFunctions)


def latticeDensity(hOverS: float, dim: int = 2, kernel: str = 'Wendland2',
                   method: str = 'shells') -> float:
    """`L` -- the summation density of a unit-rest-density perfect lattice.

    Thin alias for `warpSPHCore.util.latticeDensity`, kept so this script reads
    the same as it did when it carried its own transcribed kernel table.
    """
    return _latticeDensity(KernelFunctions[kernel], hOverS, dim, method)


def solveH(args) -> None:
    """Smallest `h/s` whose ideal-lattice density error is within tolerance.

    On a square lattice `L(h/s) > 1` and decreases strictly monotonically (no
    interior optimum, verified to 0.005 resolution over h/s in [3, 6]), so
    "the h that minimises the error" is unbounded -- the well-posed question is
    the *cheapest* `h` meeting a target. Neighbour count grows as `(h/s)^d`, so
    the table below is the accuracy/cost curve.
    """
    print(f'kernel={args.kernel} dim={args.dim}   (L>1 and monotone decreasing:'
          f' no exact L=1 on a square lattice)')
    print(f"{'target |L-1|':>14} {'h/s needed':>12} {'L':>12} "
          f"{'neighbours':>11} {'cost vs h/s=4':>14}")
    base = None
    grid = [i / 1000 for i in range(1500, 12001)]
    for tol in args.tol:
        hit = None
        for hs in grid:
            if abs(latticeDensity(hs, args.dim, args.kernel, args.method) - 1.0) <= tol:
                hit = hs
                break
        if hit is None:
            print(f'{tol:14.0e} {"unreachable":>12} {"-":>12} {"-":>11} {"-":>14}')
            continue
        L = latticeDensity(hit, args.dim, args.kernel, args.method)
        n = _neighbourCount(hit, args.dim)
        if base is None:
            base = _neighbourCount(4.0, args.dim)
        print(f'{tol:14.0e} {hit:12.3f} {L:12.8f} {n:11d} {n / base:13.2f}x')


def _neighbourCount(hs: float, dim: int) -> int:
    reach = int(math.ceil(hs))
    return sum(1 for k in itertools.product(range(-reach, reach + 1), repeat=dim)
               if 0 < math.sqrt(sum(i * i for i in k)) / hs < 1.0)


def massCorrection(args) -> None:
    hs = args.hs or 4.0
    L = latticeDensity(hs, args.dim, args.kernel, args.method)
    print(f'kernel={args.kernel} dim={args.dim} h/s={hs}')
    print(f'  ideal-lattice density   rho/rho0 = L = {L:.8f}')
    print(f'  mass correction factor      1/L  = {1.0 / L:.8f}')
    print(f'  i.e. set m = rho0 * s^{args.dim} / L  (spacing and h unchanged)')
    print(f'  residual density error after     = {abs(1.0 - 1.0):.1e} '
          f'(exact for a defect-free lattice)')


def sweep(args) -> None:
    print(f'kernel={args.kernel} dim={args.dim}')
    print(f"{'h/s':>8} {'L (lattice rho/rho0)':>22} {'mass correction 1/L':>22} "
          f"{'neighbours':>11}")
    ratios = ([args.hs] if args.hs else
              [r / 100 for r in range(int(args.lo * 100), int(args.hi * 100) + 1,
                                      int(args.step * 100))])
    for hs in ratios:
        L = latticeDensity(hs, args.dim, args.kernel, args.method)
        reach = int(math.ceil(hs))
        n = sum(1 for k in itertools.product(range(-reach, reach + 1), repeat=args.dim)
                if 0 < math.sqrt(sum(i * i for i in k)) / hs < 1.0)
        print(f'{hs:8.3f} {L:22.6f} {1.0 / L:22.6f} {n:11d}')


def validate(args) -> None:
    from warpSPH.cases import importAll
    importAll()
    from warpSPH.runner import getCase, run
    from warpSPH.schemes import builder

    bundle = builder.buildScheme(args.scheme)
    step = bundle.stepFunction
    mod = __import__(step.__module__, fromlist=['x'])
    cap: dict = {}

    def spy(system, dt, config, schemeConfig, verbose=False):
        if 'h' not in cap:
            st = system.state
            f = (st.kinds == 0).cpu().numpy()
            pos = st.positions.detach().cpu().numpy()[f]
            cap['h'] = float(np.median(st.supports.detach().cpu().numpy()[f]))
            cap['m'] = float(np.median(st.masses.detach().cpu().numpy()[f]))
            cap['dx'] = float(config.dx)
            cap['dim'] = int(config.dim)
            cap['rho0'] = float(schemeConfig.fluid.restDensity)
            # Achieved spacing, PER AXIS. The sampler fits the fluid block to
            # its region independently in each direction, so the lattice comes
            # out slightly anisotropic (`sloshingTank` nx=60: sx 0.009375 vs
            # sy 0.009400). The mass ratio needs the true cell volume
            # `prod(s_i)`, not `s^d` from an isotropic estimate -- using the
            # latter mispredicts by exactly the anisotropy (2.7e-3 at nx=60).
            spacings = []
            for ax in range(cap['dim']):
                u = np.unique(np.round(pos[:, ax], 9))
                spacings.append(float(np.median(np.diff(u))) if len(u) > 1
                                else float('nan'))
            cap['spacings'] = spacings
            cap['cell'] = float(np.prod(spacings))
            # Geometric mean is the isotropic-equivalent spacing for `h/s`.
            cap['s'] = float(np.prod(spacings) ** (1.0 / cap['dim']))
        return step(system, dt, config, schemeConfig, verbose)

    print(f'case={args.validate} scheme={args.scheme} kernel={args.kernel}')
    print(f"{'nx':>5} {'dx':>10} {'s (achieved)':>13} {'h':>10} {'h/s':>7} "
          f"{'m/(rho0 s^d)':>13} {'L (this tool)':>14} {'predicted':>10} "
          f"{'measured':>10} {'err':>9}")
    for nx in args.nx:
        cap.clear()
        setattr(mod, step.__name__, spy)
        try:
            kw = dict(nx=nx, nSteps=1, scheme=args.scheme, kernel=args.kernel,
                      supportMode='SuperSymmetric',
                      integrationScheme='semiImplicitEuler', quiet=True,
                      plot=False, store=False, progress=False)
            if args.validate == 'sloshingTank':
                kw.update(cflFactor=0.2, dt=1e-3, maxDt=2e-3)
            r = run(getCase(args.validate), **kw)
        finally:
            setattr(mod, step.__name__, step)
        rows = [x for x in r.trajectory if x.get('step', -1) >= 0]
        measured = rows[0].get('densityMedian')
        d, s, h = cap['dim'], cap['s'], cap['h']
        L = latticeDensity(h / s, d, args.kernel, args.method)
        massRatio = cap['m'] / (cap['rho0'] * cap['cell'])
        predicted = massRatio * L
        err = (predicted - measured) / measured if measured else float('nan')
        print(f'{nx:5d} {cap["dx"]:10.6f} {s:13.6f} {h:10.6f} {h / s:7.4f} '
              f'{massRatio:13.5f} {L:14.6f} {predicted:10.5f} '
              f'{measured:10.5f} {err:+9.2e}')


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kernel', default='Wendland2', choices=sorted(KERNELS))
    ap.add_argument('--dim', type=int, default=2)
    ap.add_argument('--method', default='shells',
                    choices=['shells', 'fourier', 'closed'],
                    help='exact shell sum / Poisson identity / sum-free closed form')
    ap.add_argument('--hs', type=float, default=None, help='a single h/s ratio')
    ap.add_argument('--lo', type=float, default=1.5)
    ap.add_argument('--hi', type=float, default=5.0)
    ap.add_argument('--step', type=float, default=0.25)
    ap.add_argument('--solve-h', action='store_true',
                    help='cheapest h/s meeting a density-error tolerance')
    ap.add_argument('--mass-correction', action='store_true',
                    help='the 1/L mass factor at a given --hs')
    ap.add_argument('--tol', type=float, nargs='*',
                    default=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5])
    ap.add_argument('--validate', default=None, help='case name to check against')
    ap.add_argument('--scheme', default='divergenceFree')
    ap.add_argument('--nx', type=int, nargs='*', default=[60, 100, 200])
    args = ap.parse_args()
    if args.validate:
        validate(args)
    elif args.solve_h:
        solveH(args)
    elif args.mass_correction:
        massCorrection(args)
    else:
        sweep(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
