import sys, dataclasses
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import warpSPH.systems.compSPH as sysm, warpSPH.modules.compSPH.pairWork as pw
slack = float(sys.argv[1]); integ = sys.argv[2]; steps = int(sys.argv[3])
import warpSPH.modules.compSPH.supportGrid as sg; sg.USAGE_GUARD = float(sys.argv[4]) if len(sys.argv) > 4 else sg.USAGE_GUARD
n = dict(rebuilds=0, reruns=0, finalize=0)
ob = sysm.buildVerletList
def bvl(*a, **k):
    n['rebuilds'] += 1; return ob(*a, **k)
sysm.buildVerletList = bvl
of = sysm.CompSPHSystem.finalize
def fin(self, *a, **k):
    out = of(self, *a, **k); n['finalize'] += 1
    if self.rerunRequest is not None: n['reruns'] += 1
    return out
sysm.CompSPHSystem.finalize = fin
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase
importAll()
case = getCase('sod2d'); inner = case.configureScheme
def conf(ctx, *a, **k):
    o = inner(ctx, *a, **k) if inner else None; ctx.schemeConfig.supportSolveOnGrid = True; return o
r = run(dataclasses.replace(case, configureScheme=conf), quiet=True, progress=False, nSteps=steps, scheme='CompSPH', integrationScheme=integ)
print(f'slack {slack:4.2f} {integ:12s} steps {steps}: end-of-step rebuilds {n["rebuilds"]:3d}  second passes {n["reruns"]:3d}  wall {r.wallTime:.2f}s')
