#!/usr/bin/env python3
"""Probe: the relaxed-Jacobi omega stability window of the IISPH pressure
operator, as a function of kernel, support radius (neighbor count), and
resolution.

**Why this exists.** The default pressure solve (`solveDivergenceFree` with
`solverType=relaxedJacobi`) updates `p <- p + omega * r / diag(A)`. Because
the IISPH operator A is symmetric negative-semi-definite (with a gauge null
mode) and D = diag(A) < 0, D^{-1}A is similar to a symmetric PSD matrix, and
the fixed-omega iteration is stable **iff** `0 < omega < 2/rho(D^{-1}A)`.
The window is set by the particle geometry (A does not depend on velocity)
and is narrow enough that the config default omega=0.5 diverges while the
scheme default omega=0.3 has limited margin. See the
"omega stability window" section of
`docs/regression/incompressible_pressure_solver_choice.md` and the Session-4
update in `docs/historic_plans/INCOMPRESSIBLE_SOLVER_PLAN.md` for the full analysis.

**What this script does, concretely.** For each combination of
  * kernel function (`--kernels`: Wendland2/4/6, B7, splines, ...),
  * support radius (`--n-h`, in cell units; converted to a target neighbor
    count via `n_h_to_nH(n_h, dim)`, so the *same* n_h means more neighbors
    in 3D, and reported as the actual per-particle neighbor counts),
  * particle resolution (`--nx` in 2D, `--nx3` in 3D; N = nx^dim),
  * dimension (`--dims 2 3`: note the density of the operator changes with
    dim -- more neighbors and a stiffer 3D lattice),
plus an optional grid deformation (`--deform`, in domain units, along the
TGV field -- the state change a running sim produces, and what actually
moves A), it:
  1. builds the same TGV start state as `tests/test_incompressibleKrylov.py`
     (same `_buildCase`),
  2. assembles the *exact production* matrix-free operator densely (column j
     = matvec(e_j), fp32 -> fp64), so it measures what the matvec closure
     actually computes,
  3. reports the spectrum of D^{-1}A: `mu = rho`, the stability window
     `2/mu`, the gauge mode, the smallest non-gauge eigenvalue, the spectral
     spread, the top-5 (degenerate) cluster, and power-iteration estimates
     (5/10/20 iters, to show how unreliable a spectral seed is),
  4. runs `--steps` (default 64) of fixed-omega Jacobi at omega = 0.3, 0.5,
     1/mu, 0.8*2/mu and the per-step optimal residual minimizer (the
     `relaxationMode: optimal` update), reporting relative-residual
     checkpoints, iterations to 1e-2/1e-3, and monotonicity.

The velocity field only scales the source term b (initial residual); mu and
the window are velocity-independent, so `--velocity-scale` (default 0.05,
matching the unit-test state) is rarely interesting.

Memory note: the dense operator is fp64, so 8*N^2 bytes with N = nx^dim --
2D: nx=48 -> 42 MB, nx=64 -> 134 MB; 3D: nx3=12 -> 24 MB, nx3=16 -> 134 MB,
nx3=24 -> 1.5 GB. Keep N at/below ~4096 on a laptop (the eigendecomposition
is O(N^3) on top); each case also costs N matvec launches to assemble A.

Usage:
    python scripts/probe_relaxedJacobiOmega.py
    python scripts/probe_relaxedJacobiOmega.py --kernels Wendland2 Wendland4 QuarticSpline B7
    python scripts/probe_relaxedJacobiOmega.py --n-h 3 4 6
    python scripts/probe_relaxedJacobiOmega.py --nx 24 32 48 64
    python scripts/probe_relaxedJacobiOmega.py --dims 2 3 --nx 32 --nx3 8 12
    python scripts/probe_relaxedJacobiOmega.py --dims 3 --kernels Wendland2 B7 --nx3 8
    python scripts/probe_relaxedJacobiOmega.py --deform 0 0.25 1.0 --nx 32
    python scripts/probe_relaxedJacobiOmega.py --csv sweep.csv --device cpu --threads 8
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / 'src', _REPO / 'tests'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

# Case construction and operator assembly shared with the unit tests, so the
# probe measures exactly the operator tests/test_incompressibleKrylov.py pins.
from test_incompressibleKrylov import _assembleA, _buildCase, _sourceTerm  # noqa: E402

from warpSPHCore import buildVerletList, SupportScheme  # noqa: E402
from warpSPHCore.enumTypes import KernelFunctions  # noqa: E402
from warpSPH.modules.density import computeDensities  # noqa: E402

DENSITY_KERNELS = ['Wendland2', 'Wendland4', 'Wendland6', 'B7',
                   'QuarticSpline', 'QuinticSpline', 'CubicSpline',
                   'Poly6', 'Spiky']
# Guard against drift from the warpSPHCore kernel enum (the viscosity /
# cohesion kernels are not density kernels and are deliberately excluded).
_unknown = [k for k in DENSITY_KERNELS if not hasattr(KernelFunctions, k)]
if _unknown:
    raise ValueError(f'unknown kernel(s) { _unknown }; valid: '
                     f'{[m.name for m in KernelFunctions]}')

def deformCase(case, deform: float):
    """Displace the (uniform) grid along a divergence-free TGV field by
    `deform` (in domain units), wrap, and rebuild adjacency + densities --
    the non-uniform particle distribution a running sim produces. In 3D the
    field is the 2D TGV with zero z-component, which is still
    divergence-free."""
    x = case.state.positions
    dom = case.config.domain
    L = torch.tensor([float(dom.max[i] - dom.min[i]) for i in range(x.shape[1])],
                     device=x.device, dtype=x.dtype)
    u = torch.zeros_like(x)
    u[:, 0] = torch.sin(math.pi * x[:, 0] / L[0]) * torch.cos(math.pi * x[:, 1] / L[1])
    u[:, 1] = -torch.cos(math.pi * x[:, 0] / L[0]) * torch.sin(math.pi * x[:, 1] / L[1])
    newpos = x + deform * u
    newpos = dom.min + torch.remainder(newpos - dom.min, L)
    case.state.positions = newpos
    adjacency = buildVerletList(case.state, case.config.domain,
                                verletScale=case.config.verletScale,
                                supportMode=SupportScheme.SuperSymmetric,
                                priorNeighborhood=None, verbose=False)
    case.adjacency = adjacency
    case.state.densities = computeDensities(case.state, case.config,
                                            case.schemeConfig, adjacency)
    case.state.masses = case.state.masses / case.state.densities.mean() * 1.0
    case.state.densities = computeDensities(case.state, case.config,
                                            case.schemeConfig, adjacency)
    return case


def jacobiFixed(A, b, d, omega, steps, recompute_every=25):
    """Relative-residual trajectory of the fixed-omega update p += omega*r/d
    (gauge mean-subtracted each step, as production does)."""
    N = b.shape[0]
    p = torch.zeros(N, dtype=torch.float64, device=b.device)
    r = b.clone()
    r0 = r.norm()
    out = []
    for k in range(steps):
        p = p + omega * r / d
        p = p - p.mean()
        r = r - omega * A @ (r / d)
        if (k + 1) % recompute_every == 0:
            r = b - A @ p
        out.append(r.norm() / r0)
        if not torch.isfinite(torch.tensor(out[-1])):
            break
    return torch.tensor(out, dtype=torch.float64)


def jacobiOptimal(A, b, d, steps, recompute_every=25):
    """Per-step exact 1-D residual minimizer omega_k = (r . A D^{-1} r) /
    ||A D^{-1} r||^2 -- the `relaxationMode: optimal` update, standalone."""
    N = b.shape[0]
    p = torch.zeros(N, dtype=torch.float64, device=b.device)
    r = b.clone()
    r0 = r.norm()
    out, omegas = [], []
    for k in range(steps):
        u = r / d
        q = A @ u
        num = torch.dot(r, q)
        den = torch.dot(q, q)
        w = float(num / den) if den > 0 else 0.0
        p = p + w * u
        p = p - p.mean()
        r = r - w * q
        if (k + 1) % recompute_every == 0:
            r = b - A @ p
        out.append(r.norm() / r0)
        omegas.append(w)
        if not torch.isfinite(torch.tensor(out[-1])):
            break
    return torch.tensor(out, dtype=torch.float64), omegas


def powerEstimate(A, d, iters):
    """mu_hat = rho(D^{-1}A) via power iteration on the similar symmetric
    PSD matrix S = |d|^{-1/2} (-A) |d|^{-1/2}."""
    N = A.shape[0]
    s = 1.0 / torch.sqrt(d.abs())
    i = torch.arange(N, dtype=torch.float64, device=A.device)
    v = torch.sin(0.35 * i + 1.7) * (1.0 - 2.0 * (i % 2))
    v = v / v.norm()
    rhos = []
    for _ in range(iters):
        w = s * (-(A @ (s * v)))
        v = w / w.norm()
        sv = s * (-(A @ (s * v)))
        rhos.append(float(torch.dot(v, sv) / torch.dot(v, v)))
    return max(rhos), rhos


def _it_to(tr, target):
    idx = (tr <= target).nonzero().flatten()
    return str(int(idx[0]) + 1) if idx.numel() else '-'


def _mono(tr):
    return bool((tr[1:] <= tr[:-1] * 1.001).all()) if tr.numel() > 1 else True


def _at(tr, k):
    """1-based checkpoint value, or 'div' if the trajectory broke early."""
    return f'{tr[k - 1]:.2e}' if k <= tr.numel() else '  div  '

def analyzeCase(label, case, steps, do_power=True):
    """Assemble the operator, measure the spectrum, run the convergence
    candidates, print the per-case block, return a flat row for the CSV."""
    A = _assembleA(case).double()
    N = A.shape[0]
    d = torch.diagonal(A).clone()
    b = _sourceTerm(case, torch.zeros_like(case.state.velocities)).double()
    nbrs = case.adjacency.numNeighbors.float()
    supports = getattr(case.state, 'supports', None)

    print(f'--- {label} ---')
    hstr = (f'  h={float(supports.mean()):.4g} mean' if supports is not None else '')
    print(f'N={N}  neighbors {float(nbrs.mean()):.1f} mean / {int(nbrs.max())} max{hstr}')
    row = {'case': label, 'N': N,
           'nbrs_mean': round(float(nbrs.mean()), 2), 'nbrs_max': int(nbrs.max()),
           **({'h_mean': round(float(supports.mean()), 6)} if supports is not None else {})}

    if (d.abs() < 1e-30).any():
        print('  WARNING: |diag(A)| has near-zero entries; skipping spectrum.')
        return row

    dmin, dmax = float(d.min()), float(d.max())
    row['d_min'], row['d_max'] = dmin, dmax
    row['d_rel_spread'] = abs(dmax - dmin) / abs(dmin)
    s = 1.0 / torch.sqrt(d.abs())
    S = 0.5 * (s.unsqueeze(0) * (-A) * s.unsqueeze(1)
               + s.unsqueeze(1) * (-A.T) * s.unsqueeze(0))  # symmetrize
    eig = torch.linalg.eigvalsh(S)
    mu = float(eig.max())
    window = 2.0 / mu
    nz = eig[eig > 1e-9 * mu]
    As = 0.5 * (A + A.T)
    eA = torch.linalg.eigvalsh(As)
    row.update(mu=mu, window=window, spec_A_min=float(eA.min()),
               spec_A_max=float(eA.max()), nongauge_min=float(nz.min()),
               nongauge_spread=float(nz.max() / nz.min()))
    print(f'|d| in [{dmin:.4e}, {dmax:.4e}]  rel-spread {row["d_rel_spread"]:.3e}')
    print(f'spec(A) in [{float(eA.min()):.4e}, {float(eA.max()):.4e}]')
    print(f'mu = rho(D^-1 A) = {mu:.4f}   window: omega < {window:.4f}')
    top5 = [round(float(v), 3) for v in eig[-5:].tolist()[::-1]]
    row['top5_cluster'] = ';'.join(str(v) for v in top5)
    print(f'non-gauge: {nz.numel()}, top-5 (desc): {top5}')
    print(f'smallest non-gauge {float(nz.min()):.4e}, spread {float(nz.max() / nz.min()):.3e}')
    if do_power:
        ests = {n: powerEstimate(A, d, n)[0] for n in (5, 10, 20)}
        for n in (5, 10, 20):
            row[f'power{n}_err_pct'] = (float(ests[n]) / mu - 1.0) * 100.0
            print(f'power {n:2d} iters -> mu_hat {ests[n]:.4f} '
                  f'({(float(ests[n]) / mu - 1.0) * 100.0:+.1f}%)')

    checkpoints = sorted({10, steps // 2, steps})
    print(f'{"scheme":24s}' + ''.join(f'{f"rel@{k}":>9s}' for k in checkpoints)
          + f'{"it->1e-2":>9s}{"it->1e-3":>9s}{"mono":>6s}')
    runs = [
        ('fixed w=0.30', 'w0.30', jacobiFixed(A, b, d, 0.30, steps)),
        ('fixed w=0.50', 'w0.50', jacobiFixed(A, b, d, 0.50, steps)),
        (f'fixed w=1/mu={1.0 / mu:.3f}', 'w1mu', jacobiFixed(A, b, d, 1.0 / mu, steps)),
        (f'fixed w=0.8*2/mu={0.8 * window:.3f}', 'w08win',
         jacobiFixed(A, b, d, 0.8 * window, steps)),
    ]
    tr, omegas = jacobiOptimal(A, b, d, steps)
    runs.append(('optimal (per-step)', 'optimal', tr))
    for name, key, trajectory in runs:
        for k in checkpoints:
            row[f'{key}_rel{k}'] = _at(trajectory, k)
        row[f'{key}_it1e-2'] = _it_to(trajectory, 1e-2)
        row[f'{key}_it1e-3'] = _it_to(trajectory, 1e-3)
        row[f'{key}_mono'] = _mono(trajectory)
        print(f'{name:24s}' + ''.join(f'{_at(trajectory, k):>9s}' for k in checkpoints)
              + f'{row[f"{key}_it1e-2"]:>9s}{row[f"{key}_it1e-3"]:>9s}'
              f'{str(row[f"{key}_mono"]):>6s}')
    row['opt_omega_first'] = omegas[0]
    row['opt_omega_max'] = max(omegas)
    row['opt_omega_last'] = omegas[-1]
    print(f'optimal omega_k: first={omegas[0]:.4f} max={max(omegas):.4f} '
          f'last={omegas[-1]:.4f}  (fixed bound 2/mu={window:.4f})')
    print()
    return row

def main():
    parser = argparse.ArgumentParser(
        description='Sweep the relaxed-Jacobi omega stability window of the '
                    'IISPH pressure operator over kernel, support radius, and '
                    'resolution (see the module docstring).')
    parser.add_argument('--dims', type=int, nargs='+', default=[2], choices=[2, 3],
                        help='dimensions to sweep (N = nx^dim)')
    parser.add_argument('--nx', type=int, nargs='+', default=[24, 32, 48],
                        help='2D particles per side (N = nx^2); default 24 32 48')
    parser.add_argument('--nx3', type=int, nargs='+', default=[8, 12],
                        help='3D particles per side (N = nx3^3); default 8 12 '
                             '-- keep N = nx3^3 at/below ~4096')
    parser.add_argument('--kernels', nargs='+', default=['Wendland2'],
                        choices=DENSITY_KERNELS,
                        help='density kernel(s) to sweep')
    parser.add_argument('--n-h', dest='n_h', type=float, nargs='+', default=[4.0],
                        help='support radius in cell units (target neighbor '
                             'count); default 4.0 (the tgv case default)')
    parser.add_argument('--deform', type=float, nargs='+', default=[0.0],
                        help='TGV-field grid deformation, domain units '
                             '(0 = uniform start state)')
    parser.add_argument('--velocity-scale', type=float, default=0.05,
                        help='velocity noise scale for the source term only '
                             '(mu/window are velocity-independent)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=64,
                        help='Jacobi steps per run (default 64, the '
                             'divergenceFree maxIterations)')
    parser.add_argument('--device', default=None,
                        help='device for case + operator (default: cuda if '
                             'available, else cpu)')
    parser.add_argument('--threads', type=int, default=None,
                        help='torch CPU threads')
    parser.add_argument('--csv', type=str, default=None,
                        help='also write all measured values to this CSV')
    parser.add_argument('--no-power', action='store_true',
                        help='skip the power-iteration mu estimates')
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    for dim in args.dims:
        for nx in (args.nx if dim == 2 else args.nx3):
            if nx ** dim > 12000:
                print(f'WARNING: dim={dim} nx={nx} -> N={nx ** dim} -> dense '
                      f'fp64 operator > 1 GB; consider a smaller sweep.',
                      file=sys.stderr)

    rows = []
    for dim in args.dims:
        nxs = args.nx if dim == 2 else args.nx3
        for kernel in args.kernels:
            for n_h in args.n_h:
                for nx in nxs:
                    for deform in args.deform:
                        label = (f'{kernel} dim={dim} n_h={n_h:g} nx={nx}'
                                 + (f' deform={deform:g}' if deform > 0 else ''))
                        case = _buildCase(nx=nx, velocityScale=args.velocity_scale,
                                          seed=args.seed, kernel=kernel, n_h=n_h,
                                          device=args.device, dim=dim)
                        if deform > 0.0:
                            deformCase(case, deform)
                        rows.append(analyzeCase(label, case, args.steps,
                                                do_power=not args.no_power))
                        del case

    # Summary table: the numbers that answer "does kernel / n_h / nx move the
    # window, and does any candidate lose to optimal?"
    end = sorted({10, args.steps // 2, args.steps})[-1]
    keys = [('w0.30', 'w0.3'), ('w0.50', 'w0.5'), ('optimal', 'opt')]
    hdr = (f'{"case":42s}{"N":>6s}{"nbrs":>7s}{"mu":>9s}{"window":>9s}'
           + ''.join(f'{n + "@" + str(end):>12s}' for n, _ in keys)
           + f'{"opt it1e-2":>11s}{"opt mono":>10s}')
    print('=' * len(hdr))
    print(hdr)
    for row in rows:
        if 'mu' not in row:
            continue
        print(f'{row["case"]:42s}{row["N"]:>6d}{row["nbrs_mean"]:>7.1f}'
              f'{row["mu"]:>9.4f}{row["window"]:>9.4f}'
              + ''.join(f'{row.get(k + f"_rel{end}", "  div "):>12s}'
                        for k, _ in keys)
              + f'{row.get("optimal_it1e-2", "-"):>11s}'
              f'{str(row.get("optimal_mono", "-")):>10s}')

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f'wrote {args.csv}')


if __name__ == '__main__':
    main()

