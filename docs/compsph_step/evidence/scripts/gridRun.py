import sys, dataclasses, hashlib
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase
importAll()
case = getCase(sys.argv[1])
onGrid = sys.argv[3] == 'grid'
inner = case.configureScheme
def configure(ctx, *a, **k):
    out = inner(ctx, *a, **k) if inner else None
    ctx.schemeConfig.supportSolveOnGrid = onGrid
    return out
case = dataclasses.replace(case, configureScheme=configure)
scheme = sys.argv[4] if len(sys.argv) > 4 else None
kw = dict(scheme=scheme) if scheme else {}
r = run(case, quiet=True, progress=False, nSteps=int(sys.argv[2]), **kw)
st = r.state.state
print(sys.argv[1:], 'steps', r.nSteps, 'E', r.trajectory[-1].get('totalEnergy'), 'wall', round(r.wallTime, 2), '| h mean %.8f u mean %.8f rho mean %.8f' % (st.supports.double().mean(), st.internalEnergies.double().mean(), st.densities.double().mean()))
