import sys, hashlib
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import warpSPH.schemes.crkSPH as crk, warpSPH.schemes.compSPH as comp
mode = sys.argv[3]
if mode == 'summation':
    for m in (crk, comp):
        o = m.evaluateOptimalSupport
        m.evaluateOptimalSupport = (lambda o: (lambda *a, **k: o(*a, **{**k, 'computeDensity': True})))(o)
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase
importAll()
r = run(getCase(sys.argv[1]), quiet=True, progress=False, nSteps=int(sys.argv[2]))
st = r.state.state
h = st.supports.double(); u = st.internalEnergies.double(); rho = st.densities.double()
print(sys.argv[1], mode, 'steps', r.nSteps, 'clampMode', getattr(r.ctx.schemeConfig, 'supportVolumeClamp', None), '| h mean %.8f max %.8f | u mean %.8f | rho mean %.8f | E %.8f' % (h.mean(), h.max(), u.mean(), rho.mean(), r.trajectory[-1].get('totalEnergy', float('nan'))))
