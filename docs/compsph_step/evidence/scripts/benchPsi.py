import time, torch
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPHCore import SupportScheme, OperationProperties, ParticleState, buildVerletList
from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.radiusSearch.compactHash import buildFixed, makeHashMapSlots
from warpSPHCore.util import generateNeighborTestData
from warpSPH.modules.adaptiveSupport.wp_psi0 import computePsi0Warp
dev=torch.device('cuda')
for nx in (50, 316):
    pos, sup, n, dom, dx = generateNeighborTestData(nx, 50, 2, True, dev)
    st=ParticleState(positions=pos, supports=sup, masses=torch.ones(n,device=dev)*dx*dx, densities=torch.ones(n,device=dev), kinds=torch.zeros(n,dtype=torch.int32,device=dev))
    props=OperationProperties(kernel=KernelFunctions.B7, supportMode=SupportScheme.Gather)
    adj=buildVerletList(st, dom, verletScale=1.41, supportMode=SupportScheme.SuperSymmetric)
    slots=makeHashMapSlots(n, dom, float(sup.max())*1.2, device=dev, hashMapLength=n+1); slots.positions.copy_(pos); slots.supports.copy_(sup)
    grid=buildFixed(slots)
    def t(fn, reps=50):
        for _ in range(5): fn()
        torch.cuda.synchronize(); a=time.perf_counter()
        for _ in range(reps): fn()
        torch.cuda.synchronize(); return (time.perf_counter()-a)/reps*1e3
    a=computePsi0Warp(st, props, dom, adjacency=adj); b=computePsi0Warp(st, props, dom, adjacency=grid)
    print(f'N={n}: psi0 on Verlet list {t(lambda: computePsi0Warp(st,props,dom,adjacency=adj)):.3f} ms | on grid {t(lambda: computePsi0Warp(st,props,dom,adjacency=grid)):.3f} ms | max|diff| {float((a[0]-b[0]).abs().max()):.2e}')
from warpSPHCore.radiusSearch.compactHash import buildCompactHashMap
pos, sup, n, dom, dx = generateNeighborTestData(50, 50, 2, True, dev)
st=ParticleState(positions=pos, supports=sup, masses=torch.ones(n,device=dev)*dx*dx, densities=torch.ones(n,device=dev), kinds=torch.zeros(n,dtype=torch.int32,device=dev))
props=OperationProperties(kernel=KernelFunctions.B7, supportMode=SupportScheme.Gather)
adj=buildVerletList(st, dom, verletScale=1.41, supportMode=SupportScheme.SuperSymmetric)
eg=buildCompactHashMap(pos,pos,sup,sup,dom.periodic,dom,SupportScheme.Gather,n+1)
egS=buildCompactHashMap(pos,pos,sup,sup,dom.periodic,dom,SupportScheme.SuperSymmetric,n+1)
a=computePsi0Warp(st, props, dom, adjacency=adj)[0]
for name,g in (('eager grid (Gather)',eg),('eager grid (SuperSymmetric)',egS)):
    b=computePsi0Warp(st, props, dom, adjacency=g)[0]; print(name, 'max|diff| vs list', float((a-b).abs().max()), 'a range', float(a.min()), float(a.max()))
slots=makeHashMapSlots(n, dom, float(sup.max()), device=dev, mode=SupportScheme.Gather, hashMapLength=n+1); slots.positions.copy_(pos); slots.supports.copy_(sup)
b=computePsi0Warp(st, props, dom, adjacency=buildFixed(slots))[0]; print('fixed grid (Gather) diff vs list', float((a-b).abs().max()))
b=computePsi0Warp(st, props, dom, adjacency=eg)[0]
print('list', a[:3].tolist(), 'grid', b[:3].tolist(), 'ratio', (b/a)[:3].tolist())
from warpSPHCore import warpOperation
from warpSPHCore.enumTypes import WarpOperation
ra=warpOperation(st, OperationProperties(kernel=KernelFunctions.B7, operation=WarpOperation.Density, supportMode=SupportScheme.Gather), domain=dom, adjacency=adj)
rb=warpOperation(st, OperationProperties(kernel=KernelFunctions.B7, operation=WarpOperation.Density, supportMode=SupportScheme.Gather), domain=dom, adjacency=eg)
print('density list', ra[:2].tolist(), 'grid', rb[:2].tolist())
ia=adj.numNeighbors[:3].tolist(); print('list numNeighbors', ia)
