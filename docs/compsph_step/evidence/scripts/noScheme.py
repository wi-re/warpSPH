import sys
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase
importAll()
r=run(getCase('sod2d'),quiet=True,progress=False,nSteps=60,scheme=sys.argv[1],params=dict(adaptiveSupportScheme=sys.argv[2]))
print(sys.argv[1:], 'steps', r.nSteps, 'E', r.trajectory[-1]['totalEnergy'])
