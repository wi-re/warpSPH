import time, torch, warp as wp
wp.init()
from warpSPHCore import SupportScheme
from warpSPHCore.radiusSearch.compactHash import GraphedBuild, buildCompactHashMap, buildFixed, makeHashMapSlots
from warpSPHCore.util import generateNeighborTestData
from warpSPHCore import callAudit
dev=torch.device('cuda')
for nx in (50, 316):
    pos, sup, n, dom, dx = generateNeighborTestData(nx, 20, 2, True, dev)
    H=n+1; mode=SupportScheme.SuperSymmetric
    slots=makeHashMapSlots(n, dom, float(sup.max()), device=dev, mode=mode, hashMapLength=H)
    slots.positions.copy_(pos); slots.supports.copy_(sup)
    g=GraphedBuild(slots)
    def t(fn, reps=30):
        for _ in range(3): fn()
        torch.cuda.synchronize(); a=time.perf_counter()
        for _ in range(reps): fn()
        torch.cuda.synchronize(); return (time.perf_counter()-a)/reps*1e3
    print(f'N={n}: eager build {t(lambda: buildCompactHashMap(pos,pos,sup,sup,dom.periodic,dom,mode,H)):.2f} ms | fixed eager {t(lambda: buildFixed(slots)):.2f} ms | fixed graphed {t(g.replay):.3f} ms')
callAudit.enable(torchOps=True, warmupSteps=0)
from warpSPHCore.profiling import record_function
pos, sup, n, dom, dx = generateNeighborTestData(50, 20, 2, True, dev)
slots=makeHashMapSlots(n, dom, float(sup.max()), device=dev, hashMapLength=n+1); slots.positions.copy_(pos); slots.supports.copy_(sup)
for name, fn in (('eager', lambda: buildCompactHashMap(pos,pos,sup,sup,dom.periodic,dom,SupportScheme.SuperSymmetric,n+1)), ('fixed', lambda: buildFixed(slots))):
    fn(); callAudit.reset(); callAudit.markStep()
    with record_function(f'AUDIT {name}'): fn()
    callAudit.markStep()
    print(name); callAudit.printReport() if False else None
    r=callAudit.report(); print([l for l in r.splitlines() if f'AUDIT {name}' in l][:1])
