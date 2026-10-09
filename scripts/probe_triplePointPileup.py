#!/usr/bin/env python
"""Replicate / inspect the CRKSPH triple-point lattice pile-up (OPEN_PROBLEMS.md §15).

The equal-spacing triple point (nx 256, CRKSPH, B7, n_h 4, cfl 0.3) is clean for
~86 steps, then one step later |v| ~ 1e5 and the run dies. The mechanism is a
pile-up of light-gas columns at the contact: nearest-neighbour spacing collapses
from step ~16 until pairs coincide (r = 0) around x = 1.3125 and 12.6875.

This runs the case with a per-step diagnostic and prints, every step:
dt, max|v|, and (value @ location) extrema of rho, h, P, u; every `--nnEvery`
steps the smallest nearest-neighbour spacing in the contact/shock window (x in [1, 2.5] triple point, [-0.05, 0.6] sod2d) (in units of the
lattice dx = 6/256) and how many particles sit closer than 0.5 dx / 0.1 dx to a
neighbour. The first step with max|v| > `--blowVelocity` dumps the last clean
snapshot and the blown one to `<dump>` (npz) and stops.

Replication (about 10 s to the blow-up, once the kernels are cached)::

    python scripts/probe_triplePointPileup.py --nnEvery 8               # cfl 0.3: blows at step ~88
    python scripts/probe_triplePointPileup.py --nnEvery 8 -- --cflFactor 0.05 --nSteps 600
                                       # survives, but the same pile-up (min spacing ~0.009 dx, not 0)
    python scripts/probe_triplePointPileup.py -- --rho_II 1.0           # equal masses: still blows (~step 208)
    python scripts/probe_triplePointPileup.py -- --equalMass --nx 181   # the equal-mass sampling: fine
    python scripts/probe_triplePointPileup.py --case sod2d --nnEvery 25   # Sod 2D slab under CRKSPH, same lattice both sides

Everything after ``--`` goes to the case CLI (``--nx``, ``--kernel``, ``--n_h``,
``--cflFactor`` ...). Video is off: this is a short crash probe; the shipped
example (`examples/compressible/14-triplePoint/triplePoint_equalSpacing.py`)
is the one to watch.
"""
import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from warpSPHBootstrap import bootstrap  # noqa: E402

bootstrap(precision='float32')

import numpy as np  # noqa: E402
import torch  # noqa: E402

from warpSPH.cases.compressible import compressibleDiagnostics  # noqa: E402
from warpSPH.cases.sod import sodCase  # noqa: E402
from warpSPH.cases.sodND import sod2dCase  # noqa: E402
from warpSPH.cases.triplePoint import triplePointCase  # noqa: E402
from warpSPH.runner import caseMain  # noqa: E402

# case -> (Case, lattice dx, x window of the contact/shock zone, default CLI prefix)
CASES = {
    'triplePoint': (triplePointCase, 6.0 / 256.0, (1.0, 2.5),
                    ['--no-equalMass', '--nx', '256', '--nSteps', '300']),
    # 1D Sod (regular sampling, samplingRatio 4: equal mass) for the control
    'sod1d': (sodCase, 2.0 / 800.0, (-0.05, 0.6),
              ['--scheme', 'CRKSPH', '--supportMode', 'KernelMeanSymmetric', '--tLimit', '0.6']),
    # Sod 2D slab on the same lattice for both states (--no-equalMass): dense side 4x heavier, dx 0.01
    'sod2d': (sod2dCase, 0.01, (-0.05, 0.6),
              ['--no-equalMass', '--scheme', 'CRKSPH', '--supportMode', 'KernelMeanSymmetric', '--tLimit', '0.6']),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', choices=sorted(CASES), default='triplePoint')
    ap.add_argument('--nnEvery', type=int, default=0,
                    help='every N steps print the smallest nearest-neighbour spacing in the contact/shock window (x in [1, 2.5] triple point, [-0.05, 0.6] sod2d) (0 = never; CPU KD-tree, ~0.3 s each)')
    ap.add_argument('--blowVelocity', type=float, default=5.0, help='max|v| that counts as the blow-up')
    ap.add_argument('--dump', default='triplePointBlow.npz', help='npz for the last clean + blown snapshot')
    ap.add_argument('--exportRoot', default='scripts/out_triplePointPileup')
    ap.add_argument('rest', nargs=argparse.REMAINDER, help='-- then case CLI flags')
    args = ap.parse_args()
    rest = [a for a in args.rest if a != '--']

    case0, DX, (xlo, xhi), prefix = CASES[args.case]
    state = dict(step=0, prev=None, dumped=False)
    fields = ('positions', 'velocities', 'densities', 'supports', 'pressures', 'internalEnergies',
              'masses', 'soundspeeds', 'alphas', 'divergence')

    def diagnostics(ctx, st):
        d = compressibleDiagnostics(ctx, st)
        s = st.state
        state['step'] += 1
        speed = torch.linalg.norm(s.velocities, dim=-1)
        dt = ctx.config.dt
        dt = dt.item() if torch.is_tensor(dt) else dt

        def ext(name, x):
            i, j = int(torch.argmin(x)), int(torch.argmax(x))
            p = s.positions
            def at(k):
                return ','.join(f'{c:.2f}' for c in p[k].tolist())
            return f"{name} [{x[i].item():.4g}@({at(i)}), {x[j].item():.4g}@({at(j)})]" 

        print(f"[pileup] step {state['step']} dt={dt:.3e} vmax={speed.max().item():.4g} "
              + ext('rho', s.densities) + ' ' + ext('h', s.supports) + ' ' + ext('P', s.pressures)
              + ' ' + ext('u', s.internalEnergies), flush=True)
        if args.nnEvery and state['step'] % args.nnEvery == 0:
            from scipy.spatial import cKDTree
            x = s.positions.detach().cpu().numpy().astype(np.float64)
            xs = x[(x[:, 0] > xlo) & (x[:, 0] < xhi)]
            nn = cKDTree(xs).query(xs, k=2)[0][:, 1]
            print(f"[pileup]   nn spacing min {nn.min() / DX:.4f} dx at x={xs[nn.argmin(), 0]:.4f}; "
                  f"n(<0.5dx)={int((nn < 0.5 * DX).sum())} n(<0.1dx)={int((nn < 0.1 * DX).sum())}", flush=True)
        snap = {k: getattr(s, k).detach().cpu().numpy().copy() for k in fields if getattr(s, k, None) is not None}
        if speed.max().item() > args.blowVelocity and state['prev'] is not None and not state['dumped']:
            np.savez(args.dump, **{'prev_' + k: v for k, v in state['prev'].items()},
                     **{'cur_' + k: v for k, v in snap.items()})
            state['dumped'] = True
            print(f"[pileup] BLOW-UP at step {state['step']} (vmax {speed.max().item():.3g}); "
                  f"snapshots -> {args.dump}", flush=True)
            os._exit(0)
        state['prev'] = snap
        return dict(d, maxVelocity=speed.max().item())

    case = dataclasses.replace(case0, diagnostics=diagnostics)
    argv = prefix + ['--caseName', 'pileupProbe', '--no-plot', '--no-video', '--no-show',
                     '--exportRoot', args.exportRoot, '--velocityAlarmFactor', '1e9', '--stallProgress', '0'] + rest
    caseMain(case, argv)


if __name__ == '__main__':
    main()
