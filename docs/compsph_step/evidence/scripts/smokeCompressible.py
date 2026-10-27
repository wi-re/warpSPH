import sys, traceback
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase, listCases
importAll()
names = [c for c in listCases() if getCase(c).scheme in ('CRKSPH', 'CompSPH')]
bad = []
for c in names:
    try:
        r = run(getCase(c), quiet=True, progress=False, nSteps=6)
        import torch; torch.cuda.synchronize()
        print('ok  ', c, getCase(c).scheme, r.nSteps, flush=True)
    except Exception as e:
        bad.append(c); print('FAIL', c, type(e).__name__, str(e).splitlines()[0][:100], flush=True)
        break   # a CUDA error poisons the process
print('FAILED:', bad)
