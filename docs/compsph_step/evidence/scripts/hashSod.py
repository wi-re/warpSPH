import hashlib, sys
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import warpSPH
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase
importAll()
case, scheme, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = run(getCase(case), quiet=True, progress=False, nSteps=n, scheme=scheme)
st = r.state.state
print('using', warpSPH.__file__)
for k in ('positions','velocities','internalEnergies','densities'):
    t = getattr(st,k).detach().cpu().contiguous()
    print(k, hashlib.sha1(t.numpy().tobytes()).hexdigest()[:16], float(t.double().abs().mean()))
print('E_total', r.trajectory[-1].get('totalEnergy'), 'E0', r.trajectory[0].get('totalEnergy'), 'wall', round(r.wallTime,2))
