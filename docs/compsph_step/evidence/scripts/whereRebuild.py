import sys, collections
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import warpSPH.modules.adaptiveSupport.optimalSupportOwen as owen
import warpSPH.schemes.compSPH as sch
stage=[0]; cnt=collections.Counter()
origSolve=sch.evaluateOptimalSupport
def solve(*a,**k):
    stage[0]+=1
    return origSolve(*a,**k)
sch.evaluateOptimalSupport=solve
ob=owen.buildVerletList
def bvl(*a,**k):
    prior=k.get('priorNeighborhood')
    r=ob(*a,**k)
    cnt[('solver iteration', (stage[0]-1)%2, 'rebuilt' if (prior is None or r is not prior) else 'reused')]+=1
    return r
owen.buildVerletList=bvl
sb=sch.buildVerletList
def sbl(*a,**k):
    prior=k.get('priorNeighborhood'); r=sb(*a,**k)
    cnt[('scheme call 02', (stage[0]-1)%2, 'rebuilt' if (prior is None or r is not prior) else 'reused')]+=1
    return r
sch.buildVerletList=sbl
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase
importAll()
run(getCase(sys.argv[1]),quiet=True,progress=False,nSteps=int(sys.argv[2]),scheme='CompSPH')
for k,v in sorted(cnt.items()): print(k,v)
