"""Sod shock-tube validation in 2D / 3D (Monaghan scheme) for the three
viscosity switches (NoneSwitch / CullenDehnen2010 / ReadHayfield2012).

The N-D tube is a periodic slab: the shock travels along x and the solution is
uniform along y (and z), so the SPH profile is binned along x on the window
[0, L/2] = [0,1] (L=2) with the Riemann interface at x=+0.5 -- the same
convention as the 1D case and `sodUtil.plotSod_` (geometry=(0., 1., 0.5)). The
exact 1D Riemann solution (D&A IC, gamma=5/3) is overlaid.

Acceptance (Phase 6): R&H 2012 suppresses the contact-discontinuity
thermal-energy/pressure overshoot vs NoneSwitch, and all three switches
reproduce the exact discontinuity positions (head/foot/contact/shock) within
SPH smearing. R&H is run with its designed alpha range (0.2-1.0).

Usage: scripts/probe_sodND.py <dim 2|3> [nx] [tLimit]
"""
import contextlib
import os
import io
import sys
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from warpSPH.runner import run
from warpSPH.cases.sodND import sod2dCase, sod3dCase
from warpSPH.caseUtils.compressible.sod.sodSolution import solve

L = 2.0
XL, XR, XI = 0.0, 1.0, 0.5   # window + interface (x = +L/4)
GAMMA = 5 / 3
# D&A IC (consistent with the 1D validation): left (P, rho, u) / right.
IC = ((1.0, 1.0, 0.0), (0.1, 0.125, 0.0))
PARAMS_DA = dict(right_rho=0.125, right_pressure=0.1)
ALPHA_OVERRIDES = {'ReadHayfield2012': dict(alpha_min=0.2, alpha_max=1.0)}
# transverseSpacings keeps the periodic slab wider than 2x the support radius
# at the (coarser) resolutions used here -- the sampler raises otherwise. 3D
# squares the transverse count, so it uses a smaller value.
TRANSVERSE = {2: 30, 3: 17}


def _quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def exact(t, npts=1000):
    return solve(IC[0], IC[1], (XL, XR, XI), t, gamma=GAMMA, npts=npts)


def profile(state):
    x = state.positions[:, 0].detach().cpu().numpy()
    rho = state.densities.detach().cpu().numpy()
    P = state.pressures.detach().cpu().numpy()
    u = state.velocities[:, 0].detach().cpu().numpy()
    e = state.internalEnergies.detach().cpu().numpy()
    a = state.alphas.detach().cpu().numpy() if state.alphas is not None else None
    return dict(x=x, rho=rho, P=P, u=u, e=e, alpha=a, A=P / (rho ** GAMMA))


def binwindow(p, n=200):
    m = (p['x'] >= XL) & (p['x'] <= XR)
    edges = np.linspace(XL, XR, n + 1)
    xc = 0.5 * (edges[:-1] + edges[1:])
    out = dict(x=xc)
    for k in ('rho', 'P', 'u', 'e', 'A'):
        vals = np.full(n, np.nan)
        for b in range(n):
            bm = m & (p['x'] >= edges[b]) & (p['x'] <= edges[b + 1])
            if bm.any():
                vals[b] = p[k][bm].mean()
        out[k] = vals
    return out


def run_case(case, label, switch, nx, tLimit, dim):
    print(f'\n===== {label} (nx={nx}, tLimit={tLimit}) =====')
    params = dict(PARAMS_DA, transverseSpacings=TRANSVERSE[dim],
                  viscositySwitch=switch, **ALPHA_OVERRIDES.get(switch, {}))
    res = _quiet(lambda: run(case, progress=False, quiet=True,
                             scheme='Monaghan', nx=nx, tLimit=tLimit,
                             params=params))
    st = res.state.state
    t = float(res.state.t)
    p = profile(st)
    print(f't = {t:.5f}  nSteps = {res.nSteps}  N = {st.positions.shape[0]}  '
          f'diverged = {res.diverged}')
    if p['alpha'] is not None and p['alpha'].min() < 1.0:
        print(f'  alpha: min={p["alpha"].min():.4f} max={p["alpha"].max():.4f} '
              f'mean={p["alpha"].mean():.4f}')
    return res, t, p, binwindow(p)


def contact_spike(p, exact_vals, cx, xshock):
    """Contact overshoot in a window around the exact contact, vs the EXACT
    post-contact (region-4) value (consistent with the 1D probe). The exact
    reference is taken from the middle of the constant region 4 (between the
    contact and the shock), away from both smeared fronts."""
    lo = cx + 0.5 * (xshock - cx) * 0.4
    hi = cx + 0.5 * (xshock - cx) * 0.9
    sel = (exact_vals['x'] > lo) & (exact_vals['x'] < hi)
    p4 = float(np.nanmedian(exact_vals['p'][sel]))
    rho4 = float(np.nanmedian(exact_vals['rho'][sel]))
    e4 = p4 / (rho4 * (GAMMA - 1))
    A4 = p4 / (rho4 ** GAMMA)
    m = (p['x'] > cx - 0.07) & (p['x'] < cx + 0.07)
    return dict(
        P=float(np.nanmax(p['P'][m]) / p4 - 1.0),
        e=float(np.nanmax(p['e'][m]) / e4 - 1.0),
        A=float(np.nanmax(p['A'][m]) / A4 - 1.0),
        p4=p4, A4=A4)


if __name__ == '__main__':
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    case = sod2dCase if dim == 2 else sod3dCase
    nx = int(sys.argv[2]) if len(sys.argv) > 2 else (40 if dim == 2 else 20)
    tLimit = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2

    switches = ['NoneSwitch', 'CullenDehnen2010', 'ReadHayfield2012']
    results = []
    for sw in switches:
        res, t, p, b = run_case(case, f'Monaghan + {sw} ({dim}D)', sw, nx, tLimit, dim)
        results.append((sw, t, p, b, res))

    t = results[0][1]
    positions, regions, vals = exact(t)
    print(f'\n===== exact Riemann ({dim}D window [0,1]) at t = {t:.5f} =====')
    for k in ('Head of Rarefaction', 'Foot of Rarefaction',
              'Contact Discontinuity', 'Shock'):
        print(f'  {k:22s}: {positions[k]:+.6f}')

    # overlay PNG
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for label, ax in (('rho', axes[0, 0]), ('P', axes[0, 1]),
                      ('u', axes[1, 0]), ('e', axes[1, 1])):
        ek = {'rho': 'rho', 'P': 'p', 'u': 'u', 'e': 'energy'}[label]
        ax.plot(vals['x'], vals[ek], 'k:', lw=1.8, label='exact')
        for sw, _, _, bb, _ in results:
            ax.plot(bb['x'], bb[label], '.', ms=2, label=sw)
        ax.set_title(f'{label} ({dim}D)')
        ax.legend()
    for k in ('Head of Rarefaction', 'Foot of Rarefaction',
              'Contact Discontinuity', 'Shock'):
        for ax in axes.ravel():
            ax.axvline(positions[k], color='gray', ls='--', alpha=0.4)
    fig.suptitle(f'Sod {dim}D (Monaghan), t={t:.4f}, interface x=0.5, '
                 f'nx={nx}, switches: {", ".join(s for s, *_ in results)}')
    fig.tight_layout()
    os.makedirs('results/probes', exist_ok=True)
    out_png = f'results/probes/sod_{dim}d_overlay.png'
    fig.savefig(out_png, dpi=120)
    print(f'\nsaved overlay -> {out_png}')

    # contact-spike comparison (the acceptance)
    cx = positions['Contact Discontinuity']
    xshock = positions['Shock']
    print(f'\n-- contact spike @ {cx:.4f} ({dim}D) --')
    spikes = {}
    for sw, _, p, b, res in results:
        cs = contact_spike(p, vals, cx, xshock)
        spikes[sw] = cs
        print(f'  {sw:20s}: P {cs["P"]:+7.3%}  e {cs["e"]:+7.3%}  A {cs["A"]:+7.3%}  '
              f'(diverged={res.diverged})')
    ns, rh = spikes['NoneSwitch'], spikes['ReadHayfield2012']
    ok = (rh['P'] < ns['P']) and (rh['e'] < ns['e']) and (rh['A'] < ns['A'])
    print(f'\n  R&H suppresses contact overshoot vs NoneSwitch: '
          f'{"PASS" if ok else "CHECK"} (P {rh["P"]:+.2%} vs {ns["P"]:+.2%}, '
          f'e {rh["e"]:+.2%} vs {ns["e"]:+.2%}, A {rh["A"]:+.2%} vs {ns["A"]:+.2%})')
