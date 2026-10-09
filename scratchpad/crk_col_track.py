"""Follow the columns next to the Sod-2D right interface (x0 = 0.5) by initial position: mean displacement / velocity / u / rho / P of
dense columns H1-H3 and light columns L1-L5, every 4 steps until t=0.25. Same probe args as crk_pile_watch."""
import os, runpy, sys
import numpy as np, torch
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import warpSPH.schemes.crkSPH as crk, warpSPH.schemes.builder as builder, warpSPH.schemes as schemes
_orig = crk.crkSPH_step
N = [0]; INIT = {}; ROWS = []
def wrapped(system, dt, config, schemeConfig, verbose=False):
    out = _orig(system, dt, config, schemeConfig, verbose)
    N[0] += 1
    st = system.state
    if 'x0' not in INIT:
        x0 = st.positions[:, 0].detach().cpu().numpy().copy(); uid = st.UIDs.detach().cpu().numpy().copy()
        INIT['x0'] = dict(zip(uid.tolist(), x0.tolist()))
    if N[0] % 8 == 0 and system.t < 0.25:
        x = st.positions.detach().cpu().numpy(); uid = st.UIDs.detach().cpu().numpy(); v = st.velocities.detach().cpu().numpy()[:, 0]
        u = st.internalEnergies.detach().cpu().numpy(); rho = st.densities.detach().cpu().numpy(); P = st.pressures.detach().cpu().numpy()
        x0 = np.array([INIT['x0'][k] for k in uid.tolist()])
        row = [float(system.t)]
        for k in (-3, -2, -1, 1, 2, 3, 4, 5):   # columns relative to the interface at 0.5: -1 = last dense, +1 = first light
            c = 0.5 + (k - 0.5 * np.sign(k)) * 0.01
            s = np.abs(x0 - c) < 0.004
            row += [float(x[s, 0].mean() - c), float(v[s].mean()), float(rho[s].mean()), float(P[s].mean())]
        ROWS.append(row)
    return out
crk.crkSPH_step = wrapped; builder.crkSPH_step = wrapped; schemes.crkSPH_step = wrapped
sys.argv = ['probe_triplePointPileup.py'] + sys.argv[1:]
try:
    runpy.run_path('scripts/probe_triplePointPileup.py', run_name='__main__')
finally:
    names = ['H3', 'H2', 'H1', 'L1', 'L2', 'L3', 'L4', 'L5']
    print('[track] columns: displacement [units dx=0.01] / vx', flush=True)
    print('[track]   t    ' + ' '.join(f'{n:>13s}' for n in names), flush=True)
    for r in ROWS:
        print('[track] %.3f ' % r[0] + ' '.join('%6.2f/%5.2f  ' % (r[1 + 4 * k] / 0.01, r[2 + 4 * k]) for k in range(8)), flush=True)
    print('[track] density / pressure of the same columns', flush=True)
    for r in ROWS[::2]:
        print('[track] %.3f ' % r[0] + ' '.join('%5.2f/%5.2f  ' % (r[3 + 4 * k], r[4 + 4 * k]) for k in range(8)), flush=True)
