"""Probe: Cullen & Dehnen (2010) switch on the basic Monaghan scheme, 1D Sod.

Runs the 1D Sod tube with the Monaghan scheme under the switches passed on the
command line (default: NoneSwitch + CullenDehnen2010), overlays the exact
Riemann solution, and reports a region-by-region comparison plus contact
spike metrics.

GEOMETRY: `buildSod1D` lays out a mirror-symmetric two-interface tube -- the
dense (left) state fills |x| <= L/4 and the light (right) state the two outer
quarters, interfaces at x = +/-L/4. The mirror symmetry makes x = 0 and
x = +/-L/2 act as reflecting walls, so the analytic Riemann solution describes
ONLY the window x in [0, L/2] = [0, 1] (L=2) with the interface at x = +0.5 --
exactly what `plotSod_`/`sodSolution.solve` use (geometry=(0., 1., 0.5)). The
SPH profile is therefore binned on x in [0, 1] and the exact solution is
evaluated there.
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
from warpSPH.cases.sod import sodCase
from warpSPH.caseUtils.compressible.sod.sodSolution import solve

L = 2.0
XI = 0.5          # Riemann interface location (x = +L/4)
XL, XR = 0.0, 1.0  # the window the analytic solution describes
GAMMA = 5 / 3


def exact(ic, t, gamma, npts=1000):
    (pl, rhol, ul), (pr, rhor, ur) = ic
    return solve((pl, rhol, ul), (pr, rhor, ur), (XL, XR, XI), t, gamma=gamma, npts=npts)


def profile(state):
    x = state.positions[:, 0].detach().cpu().numpy()
    rho = state.densities.detach().cpu().numpy()
    P = state.pressures.detach().cpu().numpy()
    u = state.velocities[:, 0].detach().cpu().numpy()
    e = state.internalEnergies.detach().cpu().numpy()
    a = state.alphas.detach().cpu().numpy() if state.alphas is not None else None
    A = P / (rho ** GAMMA)
    return dict(x=x, rho=rho, P=P, u=u, e=e, alpha=a, A=A)


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


def region_mean(p, lo, hi):
    """Mean SPH field in [lo, hi] (an interior slice of one exact region)."""
    m = (p['x'] >= lo) & (p['x'] <= hi)
    if not m.any():
        return dict(rho=np.nan, P=np.nan, u=np.nan, e=np.nan, A=np.nan)
    return dict(rho=p['rho'][m].mean(), P=p['P'][m].mean(), u=p['u'][m].mean(),
                e=p['e'][m].mean(), A=p['A'][m].mean())


def contact_spike(p, vals, contact_x, gamma, window=0.07):
    """Spike of P, e and specific entropy A above the exact region-4 value in a
    window around the contact discontinuity."""
    sel = (vals['x'] > contact_x) & (vals['x'] < contact_x + 0.04)
    p4 = float(np.median(vals['p'][sel]))
    rho4 = float(np.median(vals['rho'][sel]))
    e4 = p4 / (rho4 * (gamma - 1))
    A4 = p4 / (rho4 ** gamma)
    m = (p['x'] > contact_x - window) & (p['x'] < contact_x + window)
    out = dict(p4=p4, e4=e4, A4=A4)
    for key, exact in (('P', p4), ('e', e4), ('A', A4)):
        out[f'{key}_peak'] = float(np.nanmax(p[key][m]))
        out[f'{key}_overs'] = out[f'{key}_peak'] / exact - 1.0
    return out


def run_case(label, **overrides):
    print(f'\n===== {label} =====')
    with contextlib.redirect_stdout(io.StringIO()):
        res = run(sodCase, progress=False, quiet=True, **overrides)
    st = res.state.state
    t = float(res.state.t)
    p = profile(st)
    print(f't = {t:.5f}  nSteps = {res.nSteps}  diverged = {res.diverged}')
    if p['alpha'] is not None and p['alpha'].min() < 1.0:
        print(f'  alpha: min={p["alpha"].min():.4f} max={p["alpha"].max():.4f} '
              f'mean={p["alpha"].mean():.4f}')
    return res, t, p, binwindow(p)


def report(name, p, b, positions, regions, vals):
    xhd = positions['Head of Rarefaction']
    xft = positions['Foot of Rarefaction']
    xcd = positions['Contact Discontinuity']
    xsh = positions['Shock']
    # interior slices of the constant regions (away from the smeared boundaries)
    slices = {
        'R1 (x<hd)': (0.0, xhd - 0.03),
        'R3 (ft<xcd)': (xft + 0.03, xcd - 0.03),
        'R4 (xcd<xsh)': (xcd + 0.03, xsh - 0.03),
        'R5 (x>xsh)': (xsh + 0.03, 1.0),
    }
    exact = {'R1 (x<hd)': regions['Region 1'], 'R3 (ft<xcd)': regions['Region 3'],
             'R4 (xcd<xsh)': regions['Region 4'], 'R5 (x>xsh)': regions['Region 5']}
    print(f'\n-- {name}: region-by-region (SPH mean vs exact) --')
    print(f'   {"region":14s} {"rho":>22s} {"P":>22s} {"u":>22s}')
    for rname, (lo, hi) in slices.items():
        pe = exact[rname]  # (p, rho, u)
        sp = region_mean(p, lo, hi)
        def cell(sp_v, ex_v):
            if abs(ex_v) < 0.1:  # ~zero reference: show absolute deviation, not %
                return f'{sp_v:7.4f} / {ex_v:7.4f} ({sp_v - ex_v:+7.4f})'
            return f'{sp_v:7.4f} / {ex_v:7.4f} ({sp_v / ex_v - 1.0:+5.1%})'
        print(f'   {rname:14s} {cell(sp["rho"], pe[1]):22s} {cell(sp["P"], pe[0]):22s} '
              f'{cell(sp["u"], pe[2]):22s}')
    cs = contact_spike(p, vals, xcd, GAMMA)
    print(f'   contact spike @ {xcd:.4f}: P {cs["P_overs"]:+.3%}  e {cs["e_overs"]:+.3%}  '
          f'A {cs["A_overs"]:+.3%}  (A4={cs["A4"]:.4f})')
    return cs


if __name__ == '__main__':
    switches = sys.argv[1:] or ['NoneSwitch', 'CullenDehnen2010']
    ic = ((1.0, 1.0, 0.0), (0.1, 0.125, 0.0))  # (p, rho, u) left / right (D&A IC)
    params_da = dict(right_rho=0.125, right_pressure=0.1)
    common = dict(scheme='Monaghan', nx=400, nSteps=400)
    # R&H 2012 is designed with a narrower, higher-baseline alpha range.
    alpha_overrides = {'ReadHayfield2012': dict(alpha_min=0.2, alpha_max=1.0)}

    results = []
    for sw in switches:
        params = dict(params_da, viscositySwitch=sw, **alpha_overrides.get(sw, {}))
        res, t, p, b = run_case(f'Monaghan + {sw}',
                                **common, params=params)
        results.append((sw, t, p, b))

    t = results[0][1]
    positions, regions, vals = exact(ic, t, GAMMA)
    print('\n===== exact Riemann (window [0,1], interface x=0.5) at t = %.5f =====' % t)
    for k in ('Head of Rarefaction', 'Foot of Rarefaction', 'Contact Discontinuity', 'Shock'):
        print(f'  {k:22s}: {positions[k]:+.6f}')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for label, ax in (('rho', axes[0, 0]), ('P', axes[0, 1]),
                      ('u', axes[1, 0]), ('e', axes[1, 1])):
        ek = {'rho': 'rho', 'P': 'p', 'u': 'u', 'e': 'energy'}[label]
        ax.plot(vals['x'], vals[ek], 'k:', lw=1.8, label='exact')
        for sw, _, _, bb in results:
            ax.plot(bb['x'], bb[label], '.', ms=2, label=sw)
        ax.set_title(label)
        ax.legend()
    for k in ('Head of Rarefaction', 'Foot of Rarefaction', 'Contact Discontinuity', 'Shock'):
        for ax in axes.ravel():
            ax.axvline(positions[k], color='gray', ls='--', alpha=0.4)
    fig.suptitle(f'Sod 1D (Monaghan), t={t:.4f}, interface x=0.5, '
                 f'switches: {", ".join(s for s, _, _, _ in results)}')
    fig.tight_layout()
    os.makedirs('results/probes', exist_ok=True)
    out_png = 'results/probes/sod_1d_cd_overlay.png'
    fig.savefig(out_png, dpi=120)
    print(f'\nsaved overlay -> {out_png}')

    for sw, _, p, b in results:
        report(sw, p, b, positions, regions, vals)

    cd = next((p for sw, _, p, _ in results if sw == 'CullenDehnen2010'), None)
    if cd is not None and cd['alpha'] is not None:
        a = cd['alpha']
        hi = a > 0.5
        if hi.any():
            print(f'\n  C&D alpha>0.5 over x in [{cd["x"][hi].min():+.4f}, '
                  f'{cd["x"][hi].max():+.4f}] ({hi.sum()} p); peak={a.max():.4f} at '
                  f'x={cd["x"][int(np.argmax(a))]:+.4f}; exact shock {positions["Shock"]:+.4f}')
