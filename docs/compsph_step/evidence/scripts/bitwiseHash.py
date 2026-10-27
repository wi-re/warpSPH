import hashlib, sys, os
cg = sys.argv[1] == 'graph'
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import warpSPH, warpSPHCore
print('using', warpSPH.__file__, warpSPHCore.__file__, flush=True)
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase
importAll()
r = run(getCase('dambreak'), quiet=True, progress=False, nSteps=55, scheme='artificialCompressible', nx=70, cudaGraph=cg)
st = r.state.state
for k in ('positions', 'velocities', 'pressures', 'densities'):
    t = getattr(st, k).detach().cpu().contiguous()
    print(k, hashlib.sha1(t.numpy().tobytes()).hexdigest()[:16], flush=True)
print('steps', r.nSteps, 'pseudo', r.trajectory[-1].get('pseudoIterations'))
