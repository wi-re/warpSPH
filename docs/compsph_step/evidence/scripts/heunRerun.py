import sys
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import warpSPH.systems.compSPH as sysm
n=[0,0]
orig=sysm.energyIncrement
def wrap(pw,*a,**k):
    r=orig(pw,*a,**k); n[0]+=1; n[1]+= r[1] is not None and pw.halfVelocity is None; return r
sysm.energyIncrement=wrap
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase
importAll()
r=run(getCase('sod2d'),quiet=True,progress=False,nSteps=int(sys.argv[2]),scheme=sys.argv[1],integrationScheme=sys.argv[3])
st=r.state.state
print(sys.argv[1:],'finalize',n[0],'reruns',n[1],'wall',round(r.wallTime,2),'u mean',float(st.internalEnergies.double().mean()),'E',r.trajectory[-1]['totalEnergy'],'E0',r.trajectory[0]['totalEnergy'])
