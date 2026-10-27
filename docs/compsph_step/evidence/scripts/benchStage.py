import time, torch
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPHCore import SupportScheme, OperationProperties, ParticleState, buildVerletList, warpOperation
from warpSPHCore.enumTypes import KernelFunctions, WarpOperation
from warpSPHCore.radiusSearch.compactHash import GraphedBuild, makeHashMapSlots
from warpSPHCore.util import generateNeighborTestData
from warpSPH.modules.adaptiveSupport.wp_psi0 import computePsi0Warp
dev=torch.device('cuda')
def t(fn, reps=40):
    for _ in range(5): fn()
    torch.cuda.synchronize(); a=time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-a)/reps*1e3
for nx in (100, 158, 224):
    pos, sup, n, dom, dx = generateNeighborTestData(nx, 50, 2, True, dev)
    st=ParticleState(positions=pos, supports=sup, masses=torch.ones(n,device=dev)*dx*dx, densities=torch.ones(n,device=dev), kinds=torch.zeros(n,dtype=torch.int32,device=dev))
    psiP=OperationProperties(kernel=KernelFunctions.B7, supportMode=SupportScheme.Gather)
    denP=OperationProperties(kernel=KernelFunctions.B7, operation=WarpOperation.Density, supportMode=SupportScheme.Gather)
    adj=buildVerletList(st, dom, verletScale=1.41, supportMode=SupportScheme.SuperSymmetric)
    def A():   # list: one validity check (+ its sync) then 2 iterations (psi0) and the final density on the list
        buildVerletList(st, dom, verletScale=1.41, supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=adj)
        for _ in range(2): computePsi0Warp(st, psiP, dom, adjacency=adj)
        warpOperation(st, denP, domain=dom, adjacency=adj)
    slots=makeHashMapSlots(n, dom, float(sup.max())*1.2, device=dev, hashMapLength=n+1); g=GraphedBuild(slots)
    def B():   # fresh graphed grid, then the same work on the grid, no host decision
        slots.positions.copy_(pos); slots.supports.copy_(sup); grid=g.replay()
        for _ in range(2): computePsi0Warp(st, psiP, dom, adjacency=grid)
        warpOperation(st, denP, domain=dom, adjacency=grid)
    rebuild = t(lambda: buildVerletList(st, dom, verletScale=1.41, supportMode=SupportScheme.SuperSymmetric))
    a=t(A); b=t(B)
    print(f'N={n}: A (list: check + psi0 x2 + density) {a:.2f} ms | B (fresh graphed grid + same on grid) {b:.2f} ms | one list rebuild {rebuild:.2f} ms -> A with 12% rebuilds {a+0.12*rebuild:.2f} ms')
