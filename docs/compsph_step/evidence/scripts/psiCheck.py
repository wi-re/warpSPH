import torch
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPHCore import SupportScheme, OperationProperties, ParticleState, buildVerletList
from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.radiusSearch.compactHash import buildCompactHashMap
from warpSPHCore.util import generateNeighborTestData
from warpSPH.modules.adaptiveSupport.wp_psi0 import computePsi0Warp
dev=torch.device('cuda')
for periodic in (False, True):
  for target in (20, 50):
    pos, sup, n, dom, dx = generateNeighborTestData(40, target, 2, periodic, dev)
    st=ParticleState(positions=pos, supports=sup, masses=torch.ones(n,device=dev)*dx*dx, densities=torch.ones(n,device=dev), kinds=torch.zeros(n,dtype=torch.int32,device=dev))
    props=OperationProperties(kernel=KernelFunctions.B7, supportMode=SupportScheme.Gather)
    adj=buildVerletList(st, dom, verletScale=1.0, supportMode=SupportScheme.Gather)
    eg=buildCompactHashMap(pos,pos,sup,sup,dom.periodic,dom,SupportScheme.Gather,n+1)
    a=computePsi0Warp(st, props, dom, adjacency=adj); b=computePsi0Warp(st, props, dom, adjacency=eg); c=computePsi0Warp(st, props, dom, adjacency=None)
    mid=n//2+20
    print(f'periodic={periodic} target={target}: list {float(a[0][mid]):.4f}/{float(a[1][mid]):.4f} grid {float(b[0][mid]):.4f}/{float(b[1][mid]):.4f} no-adjacency {float(c[0][mid]):.4f}/{float(c[1][mid]):.4f}')
