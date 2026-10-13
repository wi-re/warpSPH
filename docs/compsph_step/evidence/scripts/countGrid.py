import sys, dataclasses
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import warpSPH.systems.compSPH as sysm
n=dict(finalize=0, rerunCons=0, rerunEnergy=0)
orig=sysm.CompSPHSystem.finalize
def fin(self,*a,**k):
    out=orig(self,*a,**k); n['finalize']+=1
    rr=self.rerunRequest
    if rr is not None: n['rerunCons' if rr.halfVelocity is None else 'rerunEnergy']+=1
    return out
sysm.CompSPHSystem.finalize=fin
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase
importAll()
case=getCase('sod2d'); inner=case.configureScheme
def conf(ctx,*a,**k):
    o=inner(ctx,*a,**k) if inner else None; ctx.schemeConfig.supportSolveOnGrid=True; return o
case=dataclasses.replace(case, configureScheme=conf)
r=run(case,quiet=True,progress=False,nSteps=int(sys.argv[1]),scheme='CompSPH',integrationScheme=sys.argv[2])
print(sys.argv[1:], n)
