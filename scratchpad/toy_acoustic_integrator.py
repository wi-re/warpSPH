"""CEILING_STICKING_PLAN.md §3.4 toy: is symplecticEuler's (rho, v) update
explicit midpoint (energy growth ~ (omega dt)^4 / 4 per step) and does
`timeCentredContinuity` make it Stormer-Verlet (bounded)?

Periodic [0,1]^2, regular lattice, no walls / gravity / PST, alpha = delta = 0,
pressure force P_i + P_j (the exact adjoint of the Difference divergence the
continuity equation uses), so the semi-discrete system conserves
E = 1/2 sum m|v|^2 + sum m c^2 (rho - rho0)^2 / (2 rho0^2). Initial state: at
rest, rho = rho0 (1 + eps cos(k x)), a standing acoustic wave of `lam` dx.
Measures omega (zero crossings of the mode amplitude), the per-step energy
growth, and prints h^4/8 for comparison (growth of |amplitude| = half the
energy growth rate: E ~ |lambda|^(2n), |lambda|^2 = 1 + h^4/4).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

ap = argparse.ArgumentParser()
ap.add_argument('--nx', type=int, default=64)
ap.add_argument('--c0', type=float, default=10.0)
ap.add_argument('--courant', type=float, nargs='+', default=[0.3, 0.6])
ap.add_argument('--lam', type=float, nargs='+', default=[4.0, 6.0], help='wavelength in dx')
ap.add_argument('--nSteps', type=int, default=400)
ap.add_argument('--eps', type=float, default=1e-4)
ap.add_argument('--integrators', nargs='+', default=None, help='sweep integrators instead of the tcc/pade matrix (no tcc)')
ap.add_argument('--pade', action='store_true', help='also run the Parshikov/DualSPHysics Pade(1,1) density map')
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import numpy as np
import torch
from warpSPH.cases import importAll
importAll()
from warpSPH.cases.tgvWeaklyCompressible import tgvWeaklyCompressibleCase as case
from warpSPH.runner import run
from warpSPH.enumTypes import PressureForceScheme

_cfg0, _ic0, _diag0 = case.configureScheme, case.initialConditions, case.diagnostics


def runOne(courant, lam, tcc, pade=False, integrator='symplecticEuler'):
    dx = 1.0 / args.nx
    dt = courant * dx / args.c0
    k = 2 * np.pi / (lam * dx)
    # an integer number of wavelengths in the box
    k = 2 * np.pi * max(1, round(1.0 / (lam * dx)))
    series = []

    def cfg(ctx):
        _cfg0(ctx)
        sc = ctx.schemeConfig
        sc.diffusionParams.inviscidAlpha = 0.0
        sc.diffusionParams.densityDelta = 0.0
        sc.shiftProperties.active = False
        sc.pressureForceTerm = PressureForceScheme.nonConservative
        sc.timeCentredContinuity = tcc

    def ic(ctx, system):
        _ic0(ctx, system)
        st = system.state
        st.velocities.zero_()
        st.densities = 1.0 + args.eps * torch.cos(k * st.positions[:, 0])

    def diag(ctx, system):
        st = system.state
        fl = st.kinds == 0
        m = st.masses[fl]; v = st.velocities[fl]; r = st.densities[fl]
        c = float(ctx.schemeConfig.fluid.fixedSoundSpeed)
        KE = 0.5 * float((m * (v * v).sum(-1)).sum())
        PE = float((m * c * c * (r - 1.0) ** 2 / 2.0).sum())
        amp = float(((r - 1.0) * torch.cos(k * st.positions[fl, 0])).sum()) * 2 / int(fl.sum())
        series.append((float(system.t), KE + PE, amp))
        return dict(E=KE + PE, amp=amp)

    case.configureScheme, case.initialConditions, case.diagnostics = cfg, ic, diag
    from warpSPH.systems.weaklyCompressible import WeaklyCompressibleSystem as WCS
    _fin0 = WCS.finalize

    def finPade(self, initialState, dt, returnValues, updateValues, *a, **kw):
        # DualSPHysics JSphCpu::ComputeSymplecticCorr / Parshikov (B.12):
        # rho^{n+1} = rho^n (2 - eps)/(2 + eps), eps = -(R/rho_h) dt, R = the rate
        # the step actually used (the linear update's (rho_lin - rho^n)/dt)
        out = _fin0(self, initialState, dt, returnValues, updateValues, *a, **kw)
        rho0 = initialState.state.densities
        rhoH = returnValues[-1][1].densities
        R = (self.state.densities - rho0) / dt
        e = -(R / rhoH) * dt
        fl = self.state.kinds == 0
        self.state.densities = torch.where(fl, rho0 * (2 - e) / (2 + e), self.state.densities)
        return out
    if pade:
        WCS.finalize = finPade
    try:
        run(case, scheme='deltaSPH', integrationScheme=integrator, nx=args.nx, L=1.0,
            nSteps=args.nSteps, quiet=True, progress=True, plot=False, video=False,
            cudaGraph=False, pipelineOutputs=False, adaptiveDt=False,
            params=dict(soundSpeed=args.c0, targetDt=dt, shuffleIters=0, jitter=0.0,
                        inviscid=True, alpha=0.0, nu=0.0, k=1, uMag=0.0, shifting=False))
    finally:
        case.configureScheme, case.initialConditions, case.diagnostics = _cfg0, _ic0, _diag0
        WCS.finalize = _fin0
    s = np.array(series[1:])
    E, amp = s[:, 1], s[:, 2]
    zc = np.nonzero(np.diff(np.sign(amp)) != 0)[0]
    period = 2 * np.mean(np.diff(zc)) if zc.size > 2 else np.nan
    omega_dt = 2 * np.pi / period
    n = np.arange(E.size)
    g = np.polyfit(n, np.log(E), 1)[0]   # per-step log energy growth
    return omega_dt, g, E[-1] / E[0]


if args.integrators:
    print(f'{"integrator":>18} {"drift":>5} {"courant":>8} {"omega*dt":>9} {"dlnE/step":>11} {"E_end/E0":>10}')
    for integ in args.integrators:
        for courant in args.courant:
            for lam in args.lam:
              for tcc in (False, True):
                try:
                    h, g, ratio = runOne(courant, lam, tcc, False, integ)
                    print(f'{integ:>18} {str(tcc):>5} {courant:8.2f} {h:9.4f} {g:11.3e} {ratio:10.4g}', flush=True)
                except Exception as e:
                    print(f'{integ:>18} {courant:8.2f} FAILED {type(e).__name__}: {str(e)[:80]}', flush=True)
    raise SystemExit
print(f'{"courant":>8} {"lam/dx":>7} {"tcc":>5} {"pade":>5} {"omega*dt":>9} {"dlnE/step":>11} {"h^4/4 pred":>11} {"E_end/E0":>9}')
for courant in args.courant:
    for lam in args.lam:
        for tcc, pade in ((False, False), (False, True), (True, False), (True, True)) if args.pade else ((False, False), (True, False)):
            h, g, ratio = runOne(courant, lam, tcc, pade)
            print(f'{courant:8.2f} {lam:7.1f} {str(tcc):>5} {str(pade):>5} {h:9.4f} {g:11.3e} {h**4/4:11.3e} {ratio:9.4f}', flush=True)
