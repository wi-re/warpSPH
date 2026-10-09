"""Control test: Gresho-Chan vortex (2D, CRKSPH) with the C&D 2010 switch on
vs off (NoneSwitch).

The Gresho vortex is a steady shear-dominated flow with NO shocks. A good
shock capturer must stay OFF here (alpha ~ alpha_min) so it does not add
shear-driven dissipation. The exact solution is time-independent, so any
drift in the vortex peak speed is dissipation. The control test checks that
C&D-on does not dissipate the vortex more than the NoneSwitch baseline.

Metric: the vortex peak speed (max |v|; exact = 1.0 at r=0.2) and its
persistence ratio (final/initial). If C&D-on preserves the vortex at least as
well as NoneSwitch, the switch is not over-dissipating in shear.
"""
import contextlib
import io
import sys
import numpy as np
import torch

from warpSPH.runner import run
from warpSPH.cases.greshoVortex import greshoVortexCase


def _quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def run_case(label, switch, nx, tLimit):
    print(f'\n===== {label} (nx={nx}, tLimit={tLimit}) =====')
    res = _quiet(lambda: run(
        greshoVortexCase, progress=False, quiet=True,
        nx=nx, tLimit=tLimit,
        params=dict(viscositySwitch=switch)))
    st = res.state.state
    t = float(res.state.t)
    v = st.velocities.detach().cpu().numpy()
    speed = np.linalg.norm(v, axis=1)
    rho = st.densities.detach().cpu().numpy()
    m = st.masses.detach().cpu().numpy()
    ke = float(0.5 * np.sum(m * speed ** 2))
    te = float(res.trajectory[-1]['totalEnergy'])
    alpha = st.alphas.detach().cpu().numpy() if st.alphas is not None else None
    print(f't = {t:.4f}  nSteps = {res.nSteps}  diverged = {res.diverged}')
    print(f'  peak speed = {speed.max():.5f}   (exact 1.0)')
    print(f'  kinetic E  = {ke:.6f}   total E = {te:.6f}')
    if alpha is not None:
        print(f'  alpha: min={alpha.min():.4f} max={alpha.max():.4f} mean={alpha.mean():.4f}')
    return dict(t=t, peak=speed.max(), ke=ke, te=te, alpha=alpha, res=res)


if __name__ == '__main__':
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    tLimit = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0

    # initial (t=0) peak speed for the persistence ratio
    res0 = _quiet(lambda: run(
        greshoVortexCase, progress=False, quiet=True, nx=nx, tLimit=0.0,
        params=dict(viscositySwitch='NoneSwitch')))
    v0 = res0.state.state.velocities.detach().cpu().numpy()
    peak0 = float(np.linalg.norm(v0, axis=1).max())
    print(f'initial (t=0) peak speed = {peak0:.5f}')

    base = run_case('NoneSwitch (baseline)', 'NoneSwitch', nx, tLimit)
    cd = run_case('CullenDehnen2010 (switch on)', 'CullenDehnen2010', nx, tLimit)

    print('\n===== control-test summary =====')
    print(f'peak-speed persistence (final/initial):')
    print(f'  NoneSwitch: {base["peak"] / peak0:.5f}')
    print(f'  C&D 2010:   {cd["peak"] / peak0:.5f}')
    d_base = 1.0 - base['peak'] / peak0
    d_cd = 1.0 - cd['peak'] / peak0
    print(f'vortex dissipation (1 - persistence):')
    print(f'  NoneSwitch: {d_base:.5f}')
    print(f'  C&D 2010:   {d_cd:.5f}')
    if d_cd <= d_base + 0.02:
        print('PASS: C&D-on does not dissipate the vortex more than the baseline '
              '(within 2 pp tolerance).')
    else:
        print('CHECK: C&D-on dissipates the vortex more than the baseline -- '
              'inspect the alpha field.')
