import hashlib, sys
case, scheme, n, cg = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == 'graph'
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import warpSPH
print('using', warpSPH.__file__, flush=True)
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase
importAll()
kw = dict(nx=70) if case == 'dambreak' else {}
r = run(getCase(case), quiet=True, progress=False, nSteps=n, scheme=scheme, cudaGraph=cg, **kw)
st = r.state.state
for k in ('positions', 'velocities', 'densities', 'internalEnergies', 'pressures'):
    if hasattr(st, k) and getattr(st, k) is not None:
        t = getattr(st, k).detach().cpu().contiguous()
        print(k, hashlib.sha1(t.numpy().tobytes()).hexdigest()[:16], flush=True)
print('steps', r.nSteps, 't', r.trajectory[-1]['t'])
