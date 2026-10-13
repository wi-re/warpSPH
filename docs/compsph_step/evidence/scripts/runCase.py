import sys
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase
importAll()
r = run(getCase(sys.argv[1]), quiet=True, progress=False, nSteps=int(sys.argv[2]))
print(sys.argv[1], 'ok steps', r.nSteps)
