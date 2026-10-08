#!/usr/bin/env python3
"""GODUNOV_SPH_PLAN: Inutsuka's identity by quadrature, and how far the pair-axis interpolation of `1/rho` is from it.

For constant pressure the exact convolution form of the SPH force (Inutsuka 2002 Eq. 22-23) is zero whatever the density,

    a_i = 2 P sum_j m_j I_ij x_ij / h^2 = 0,     I_ij = int rho(x)^-2 W(x - x_i, h) W(x - x_j, h) dx,     rho(x) = sum_k m_k W(x - x_k, h)

(1D, Gaussian W of width h; the factor 2 is the one of Eq. 58, the standard-SPH limit). This evaluates `I_ij` by quadrature on a density jump, checks that
`a_i` vanishes, and prints the kernel force (`computeInutsukaWarp`: `I_ij` replaced by `V_ij^2(h) W(x_ij, sqrt2 h)`, with the cubic or the linear specific-volume
interpolant) next to it: the difference is the interpolation error across a sharp jump.

    python scripts/probe_densityJumpQuadrature.py [--ratio 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import probe_densityJumpForce as P   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--ratio', type=float, default=4.0)
    args = ap.parse_args()
    system, config = P.contactState(args.ratio, uniformSupport=True)
    st = system.state
    x = st.positions[:, 0].double().cpu().numpy()
    m = st.masses.double().cpu().numpy()
    h = float(st.supports.mean()) / 3.0
    W = lambda d: (np.pi * h * h) ** -0.5 * np.exp(-d * d / (h * h))
    L = x.max() - x.min() + (np.sort(x)[1] - np.sort(x)[0])
    wrap = lambda d: d - L * np.round(d / L)
    rho_p = np.array([np.sum(m * W(wrap(xi - x))) for xi in x])
    order = np.argsort(x)
    xs, rs = x[order], rho_p[order]
    mid = np.sqrt(rho_p.max() * rho_p.min())
    xj = 0.5 * (xs[:-1] + xs[1:])[((rs[:-1] - mid) * (rs[1:] - mid)) < 0]
    x0 = xj[0]
    gx = x0 + np.linspace(-14 * h, 14 * h, 6001)
    dgx = gx[1] - gx[0]
    nb = np.where(np.abs(wrap(x - x0)) < 20 * h)[0]
    rho_g = np.array([np.sum(m[nb] * W(wrap(g - x[nb]))) for g in gx])
    sel = np.where(np.abs(wrap(x - x0)) < 5 * h)[0]
    sel = sel[np.argsort(x[sel])]
    aCubic, rho = P.inutsukaForce(system, config, cubic=True, widthFactor=1.0)
    aLinear, _ = P.inutsukaForce(system, config, cubic=False, widthFactor=1.0)
    print(f'density ratio {args.ratio:g}, constant h = {h:.4f}; 1D quadrature on the jump at x = {x0:.3f}; P0 = 1')
    print(f'{"(x - x0) / h":>14s} {"rho":>7s} {"exact (quad.)":>14s} {"cubic V":>10s} {"linear V":>10s}')
    for i in sel[::max(1, len(sel) // 14)]:
        a = 0.0
        for j in nb:
            if j == i:
                continue
            d = wrap(x[i] - x[j])
            if abs(d) > 6 * h:
                continue
            a += 2.0 * m[j] * np.sum(W(wrap(gx - x[i])) * W(wrap(gx - x[j])) / rho_g ** 2) * dgx * d / h ** 2
        print(f'{(x[i] - x0) / h:14.2f} {float(rho[i]):7.3f} {a:14.4f} {float(aCubic[i, 0]):10.3f} {float(aLinear[i, 0]):10.3f}')


if __name__ == '__main__':
    main()
