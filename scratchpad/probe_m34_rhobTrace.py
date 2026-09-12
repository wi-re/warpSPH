"""Diagnostic: trace min/max boundary-particle rho_b (kind==1) each step on the
failing Marrone 3.4 config, to see whether the cavitation floor in
modules/mdbc/density2025.py is actually being engaged before/during the
corner blowup, or whether the mechanism lies elsewhere.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch
import numpy as np
from warpSPH.cases.dambreak import dambreakCase
from warpSPH.modules.mdbc import computeMdbcDensity
from warpSPH.enumTypes import isIncompressibleScheme
from warpSPH.runner import run

H = 1.0
G = 9.81
TANK_L = 8.0 * H
TANK_W = 10.0 * H
COL_W = 3.0 * H
COL_H = 2.4 * H
SQRT_H_G = (H / G) ** 0.5

trace = []

def _hook(ctx, state, step):
    particles = getattr(state, 'state', state)
    kinds = getattr(particles, 'kinds', None)
    if kinds is None or not bool((kinds == 1).any()):
        return
    try:
        if isIncompressibleScheme(ctx.scheme):
            return
    except Exception:
        pass
    adjacency = getattr(state, 'adjacency', None)
    rho_b_full = computeMdbcDensity(particles, ctx.config, ctx.schemeConfig, adjacency)
    particles.densities = rho_b_full
    boundaryMask = kinds == 1
    rho_b = rho_b_full[boundaryMask]
    tRaw = state.t
    t = float(tRaw.item()) if hasattr(tRaw, 'item') else float(tRaw)
    trace.append((step, t, float(rho_b.min().item()), float(rho_b.max().item())))

dambreakCase.postStep = _hook

# reuse probe_deltaSPHMarrone34's _params for the exact same case config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import importlib
m34 = importlib.import_module('probe_deltaSPHMarrone34')

nx = 256
c0Ratio = 28.3
tStar = 3.0
machTarget = 1.95 / c0Ratio
tLimit = tStar * SQRT_H_G

params = m34._params('deltaSPH', cornerOnly=False, shifting='on', Re=None)
params['machTarget'] = machTarget
params['wallBC'] = 'freeSlip'

r = run(dambreakCase, scheme='deltaSPH', L=TANK_L, nx=nx, tLimit=tLimit,
        quiet=True, store=False, progress=True, params=params,
        integrationScheme='symplecticEuler')

print(f'diverged={r.diverged} steps={r.nSteps}')
rho0 = float(r.ctx.schemeConfig.fluid.restDensity)
c_s = float(r.ctx.schemeConfig.fluid.fixedSoundSpeed)
floor = rho0 + (-101.325) / c_s**2
print(f'rho0={rho0} c_s={c_s} expected cavitation floor={floor:.5f}')

for i in range(0, len(trace), max(1, len(trace)//80)):
    step, t, rmin, rmax = trace[i]
    print(f'step={step:5d} t={t:.4f} rho_b[min,max]=[{rmin:.4f},{rmax:.4f}]')

# also print the last 40 entries densely, to see the onset in detail
print('--- dense tail ---')
for step, t, rmin, rmax in trace[-60:]:
    print(f'step={step:5d} t={t:.4f} rho_b[min,max]=[{rmin:.4f},{rmax:.4f}]')
