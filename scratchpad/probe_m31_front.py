#!/usr/bin/env python
"""Marrone 3.1: front position vs time, sheet thickness, and the wall pressure
under the front.

Is something obstructing the surge before it reaches the far wall? Ritter's
analytic dry-bed dam break gives front speed 2*sqrt(gH) from the dam; a front
that tracks that and only *then* decelerates at the wall is fine, and the tip
spray is a thin-sheet artifact. A front that stalls or rebounds early is being
obstructed by something that should not be there.

Also reports, at each sample: the tongue's thickness near the tip (in dx), and
the mDBC boundary pressure on the bed directly under the tip -- a bed that
reports p_b = 0 under a sub-dx sheet gives it no support and lets it sink.
"""
import sys
import numpy as np, torch
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases import importAll; importAll()
from warpSPH.cases.dambreak import dambreakCase
from warpSPH.runner import run

WALL_BC = sys.argv[1] if len(sys.argv) > 1 else 'constant'
H, TANK_L, G = 0.6, 1.0, 9.81
U_MAX = 1.95 * (G * H) ** 0.5
DAM_X = -1.602 + 1.2          # dam face, from the IC dump
WALL_X = 1.602
params = dict(W=3.2196, fillRatio=H / TANK_L, fluidWidth=1.2 / 3.2196,
              gravityMagnitude=G, pressureProbeHeights=[0.160, 0.584, 1.000],
              pressureProbeInset=0.0, pressureProbeDiscRadius=0.045,
              referenceVelocity=U_MAX, machTarget=U_MAX / (40.0 * (G * H) ** 0.5),
              shifting=False, wallBC=WALL_BC)

samples = []
def hook(ctx, state, step):
    t = float(state.t)
    tS = t * (G / H) ** 0.5
    if samples and tS - samples[-1][0] < 0.2:
        return
    st = state.state
    pos = st.positions.detach().cpu().numpy()
    k = st.kinds.detach().cpu().numpy()
    rho = st.densities.detach().cpu().numpy()
    dx = float(ctx.config.dx)
    f = pos[k == 0]
    xf = f[:, 0]
    front = np.percentile(xf, 99.5)                 # robust front, ignores fliers
    tipMax = xf.max()
    # tongue thickness in the 10dx behind the robust front
    band = f[(xf > front - 10 * dx) & (xf <= front)]
    thick = ((band[:, 1].max() - band[:, 1].min()) / dx) if len(band) > 3 else 0.0
    # bed boundary pressure under the tip
    b = pos[k == 1]
    rb = rho[k == 1]
    # The BED is the FLOOR band's top row. Selecting on `y < 0` is not enough:
    # the side walls span the full height, so the topmost boundary particle
    # below mid-height is a side-wall particle, not the floor. Anchor on the
    # global minimum and take the rows within the band thickness above it.
    yMin = b[:, 1].min()
    floorBand = b[b[:, 1] < yMin + 6.0 * dx]
    bed = floorBand[:, 1].max()
    onBed = np.abs(b[:, 1] - bed) < 0.6 * dx
    under = onBed & (np.abs(b[:, 0] - front) < 5 * dx)
    pb = float(np.mean(rb[under]) - 1.0) if under.sum() else float('nan')
    deep = onBed & (b[:, 0] < -1.2)
    if not samples:
        print(f'  [bed check] bed y={bed:.4f} (expect -0.4925); '
              f'{onBed.sum()} particles on the bed row; '
              f'{deep.sum()} under the reservoir', flush=True)
    pbDeep = float(np.mean(rb[deep]) - 1.0) if deep.sum() else float('nan')
    samples.append((tS, front, tipMax, thick, pb, pbDeep))

dambreakCase.postStep = hook
print(f'wallBC={WALL_BC}  dam face x={DAM_X:.3f}  far wall x={WALL_X:.3f}  Ritter front speed '
      f'2*sqrt(gH)={2*(G*H)**0.5:.2f} m/s  U_max={U_MAX:.2f}')
r = run(dambreakCase, scheme='deltaSPH', L=TANK_L, nx=67, tLimit=0.80,
        integrationScheme='rungeKutta4', quiet=True, store=False,
        progress=False, plot=False, params=params)

print(f'\n{"t*":>6} {"front x":>9} {"tip x":>9} {"frac to wall":>13} '
      f'{"v_front":>9} {"thick/dx":>9} {"pb@front":>9} {"pb@res":>8}')
prev = None
for tS, fr, tip, th, pb, pbDeep in samples:
    frac = (fr - DAM_X) / (WALL_X - DAM_X)
    v = (fr - prev[1]) / ((tS - prev[0]) * (H / G) ** 0.5) if prev else float('nan')
    print(f'{tS:6.2f} {fr:9.4f} {tip:9.4f} {frac:13.3f} {v:9.3f} {th:9.1f} {pb:9.5f} {pbDeep:8.5f}')
    prev = (tS, fr)
