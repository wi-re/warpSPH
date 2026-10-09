"""Compare a solid-wall case against its mirror-symmetric periodic twin.

COMPRESSIBLE_WALLS_PLAN.md step 5. A periodic run whose initial state is
mirror-symmetric about the wall planes *is* a run with exactly reflecting
walls there (every wall sees its own mirror image), so it is the reference a
wall treatment should reproduce:

* `woodwardColella` (periodic, [-1, 1], mirrored about 0) vs
  `woodwardColellaWalls` (walls at the tube's ends, [-0.5, 0.5] here);
* `sedov` (periodic, [-1, 1]^d, blast at the origin) vs `sedovWalls` (walls at
  +-1): equivalent because the initial state is even in every coordinate.

Both runs use the same scheme, resolution and end time; each is a normal run
with video and progress. The fields are binned onto a common grid (cell means)
and compared as L1 errors relative to the reference's mean magnitude.

    python scripts/compare_wallMirror.py --case woodwardColella --scheme CRKSPH
    python scripts/compare_wallMirror.py --case sedov --dim 2 --nx 101 --goalRadius 1.4
"""
import argparse
import json

import torch

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.runner import run  # noqa: E402


def binned(x, values, lo, hi, bins):
    """Cell means of `values` over a regular grid on [lo, hi]^d (x: (N, d)); NaN where empty."""
    d = x.shape[-1]
    idx = ((x - lo) / (hi - lo) * bins).floor().long().clamp(0, bins - 1)
    flat = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    for k in range(d):
        flat = flat * bins + idx[:, k]
    total = torch.zeros(bins ** d, dtype=values.dtype, device=x.device).index_add_(0, flat, values)
    count = torch.zeros(bins ** d, dtype=values.dtype, device=x.device).index_add_(0, flat, torch.ones_like(values))
    return torch.where(count > 0, total / count.clamp_min(1), torch.full_like(total, float('nan')))


def fields(state, mask, shift):
    s = state.state
    x = s.positions[mask] + shift
    v = s.velocities[mask]
    return x, dict(density=s.densities[mask], pressure=s.pressures[mask],
                   speed=torch.linalg.norm(v, dim=-1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', choices=('woodwardColella', 'sedov'), default='woodwardColella')
    ap.add_argument('--scheme', default='CRKSPH')
    ap.add_argument('--supportMode', default=None, help='unset: KernelMeanSymmetric for CRKSPH, else Gather')
    ap.add_argument('--dim', type=int, default=2, help='sedov only')
    ap.add_argument('--nx', type=int, default=None, help='fluid particles per unit-2 axis (sedov) / per tube (WC)')
    ap.add_argument('--goalRadius', type=float, default=1.4, help='sedov only: run until the free shock would reach this')
    ap.add_argument('--tLimit', type=float, default=None)
    ap.add_argument('--bins', type=int, default=None)
    ap.add_argument('--exportRoot', default='results/compressibleWalls/mirror')
    a = ap.parse_args()
    supportMode = a.supportMode or ('KernelMeanSymmetric' if a.scheme == 'CRKSPH' else 'Gather')
    common = dict(scheme=a.scheme, supportMode=supportMode, plot=True, video=True, progress=True,
                  exportRoot=a.exportRoot, velocityAlarmPlotInterval=1, stallProgress=1e-3)
    if a.tLimit is not None:
        common['tLimit'] = a.tLimit

    if a.case == 'woodwardColella':
        from warpSPH.cases.woodwardColella import woodwardColellaCase
        from warpSPH.cases.woodwardColellaWalls import woodwardColellaWallsCase
        n = a.nx or 1000
        nw = woodwardColellaWallsCase.params['wallLayers']
        ref = run(woodwardColellaCase, nx=2 * n, L=2.0, **common)
        wal = run(woodwardColellaWallsCase, nx=n + 2 * nw, **common)
        xr = ref.state.state.positions[:, 0]
        rx, rf = fields(ref.state, (xr >= 0) & (xr <= 1), 0.0)
        wx, wf = fields(wal.state, wal.state.state.kinds == 0, 0.5)
        lo, hi, bins = 0.0, 1.0, a.bins or n // 4
    else:
        from warpSPH.cases.sedov import sedovCase
        from warpSPH.cases.sedovWalls import sedovWallsCase
        n = a.nx or 101
        params = dict(goalRadius=a.goalRadius)
        ref = run(sedovCase, dim=a.dim, nx=n, L=2.0, periodic=True, params=params, **common)
        wal = run(sedovWallsCase, dim=a.dim, nx=n, params=params, **common)
        rx, rf = fields(ref.state, torch.ones_like(ref.state.state.kinds, dtype=torch.bool), 0.0)
        wx, wf = fields(wal.state, wal.state.state.kinds == 0, 0.0)
        lo, hi, bins = -1.0, 1.0, a.bins or n // 3

    print(f'reference: t={float(ref.state.t):.5f} diverged={ref.diverged}; '
          f'walls: t={float(wal.state.t):.5f} diverged={wal.diverged}', flush=True)
    out = {}
    for k in rf:
        r, w = binned(rx, rf[k], lo, hi, bins), binned(wx, wf[k], lo, hi, bins)
        both = torch.isfinite(r) & torch.isfinite(w)
        scale = r[both].abs().mean()
        out[k] = dict(L1=((w - r)[both].abs().mean() / scale).item(),
                      Linf=((w - r)[both].abs().max() / r[both].abs().max()).item(),
                      refMax=r[both].max().item(), wallMax=w[both].max().item())
        print(f'  {k:9s} L1 {out[k]["L1"]:.4f}  Linf {out[k]["Linf"]:.4f}  '
              f'max ref {out[k]["refMax"]:.4g} walls {out[k]["wallMax"]:.4g}', flush=True)
    print(json.dumps(dict(case=a.case, scheme=a.scheme, nx=n, bins=bins, **out)), flush=True)


if __name__ == '__main__':
    main()
