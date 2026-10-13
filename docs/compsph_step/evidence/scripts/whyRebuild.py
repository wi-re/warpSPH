import sys, collections
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import warpSPHCore.radiusSearch.verlet.build as vb
orig=vb.verletValidityMetrics
ev=[]
def wrap(*a, **k):
    out=orig(*a, **k)
    flag,md,sf,ms,buf=out
    ev.append((bool(flag), float(md), float(sf), float(buf)))
    return out
vb.verletValidityMetrics=wrap
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase
importAll()
run(getCase(sys.argv[1]),quiet=True,progress=False,nSteps=int(sys.argv[3]),scheme=sys.argv[2])
rb=[e for e in ev if e[0]]
c=collections.Counter()
for _,md,sf,buf in rb:
    c['distance alone exceeds' if md>buf else 'support change alone exceeds' if sf>buf else 'only the sum exceeds']+=1
print('checks',len(ev),'rebuilds',len(rb)); print(dict(c))
import statistics
print('at rebuild: median distance/buffer %.2f, support/buffer %.2f'%(statistics.median(md/buf for _,md,sf,buf in rb), statistics.median(sf/buf for _,md,sf,buf in rb)))
nr=[e for e in ev if not e[0]]
print('median over non-rebuild checks: distance/buffer %.2f support/buffer %.2f'%(statistics.median(md/buf for _,md,sf,buf in nr), statistics.median(sf/buf for _,md,sf,buf in nr)))
