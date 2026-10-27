"""Before / after of the Wadsley detector: the torch-op reference (`wadsleyDetectorReference`, four host syncs per call) against the
warp-kernel detector (`wadsleyDetector`), and the full switch terms, per call at several resolutions (`microBench`, skill `module-microbench`).

    python scripts/microbench_wadsleyDetector.py --nx 100 200 400
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument('--nx', type=int, nargs='+', default=[100, 200, 400])
ap.add_argument('--steps', type=int, default=20)
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch
from warpSPHCore import SupportScheme
from warpSPH.cases import importAll
from warpSPH.modules.shockCapturing import computeViscositySwitchTerms
from warpSPH.modules.shockCapturing.Wadsley2017 import wadsleyDetector, wadsleyDetectorReference
from warpSPH.runner import run
from warpSPH.runner.case import getCase
from warpSPH.utils.microBench import microBench, formatTable, formatSyncSites

importAll()
for nx in args.nx:
    r = run(getCase('sod2d'), quiet=True, progress=False, nSteps=args.steps, scheme='CompSPH', plot=False, video=False,
            params={'viscositySwitch': 'Wadsley2017'}, nx=nx)
    system, ctx = r.state, r.ctx
    st, config, sc, adj = system.state, ctx.config, ctx.schemeConfig, system.adjacency
    dt = config.dt
    rows = [
        microBench(lambda: wadsleyDetectorReference(st, config, adj, sc.gamma), 'detector (torch ops, before)'),
        microBench(lambda: wadsleyDetector(st, config, adj, sc.gamma), 'detector (warp kernels, after)'),
        microBench(lambda: computeViscositySwitchTerms(dt, st, config, sc, SupportScheme.SuperSymmetric, adj), 'Wadsley2017 terms (after)'),
    ]
    print(f'=== sod2d CompSPH, nx={nx}, N={st.positions.shape[0]}, per call ===\n' + formatTable(rows))
    sites = formatSyncSites(rows)
    if sites:
        print(sites)
