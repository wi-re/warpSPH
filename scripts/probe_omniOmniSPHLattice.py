"""warpSPH analytic omniIncompressible and the boundaries repo's DFSPH2D on omniSPH's exact dam-break lattice (23 x 91 centres on the block edges, V = pi r^2, h = sqrt(20) r, walls one spacing outside),
against the stored compiled-omniSPH final positions: spacing irregularity, nearest-neighbour percentiles, isolated particles, wall proximity (2026-10-09). Argument: end time (0.6).
"""
import sys, math, numpy as np, torch
sys.path.insert(0, '/home/lu26029/dev/warpSPH/scripts')
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases import importAll
importAll()
from warpSPH.runner import getCase, run
from warpSPH.runner.caseSpec import CaseSpec
from warpSPHBoundaries.sim.dfsph2d import DFSPH2D, DFSPHConfig, domain_scene
d=np.load('/home/lu26029/dev/curvatureBoundaries/.tmp/omni/cmp_dam.npz')
# omniSPH lattice: centres on the block edges
nxo,nyo=23,91; spx=0.2/22; spy=0.8/90
X,Y=np.meshgrid(0.1+spx*np.arange(nxo),0.1+spy*np.arange(nyo),indexing='ij'); xo=np.stack([X.ravel(),Y.ravel()],1)
Vo=math.pi*0.005**2; ho=math.sqrt(20)*0.005
lo_o=np.array([0.1-spx,0.1-spy]); hi_o=np.array([1.6,1.0])
T=float(sys.argv[1]) if len(sys.argv)>1 else 0.6
def metrics(x):
    D=np.linalg.norm(x[:,None]-x[None],axis=2); np.fill_diagonal(D,9); nn=D.min(1)
    cnt=(D<2.57*0.0088).sum(1); b=cnt>=10
    return nn[b].std()/nn[b].mean(), np.percentile(nn[b],[5,50,95]).round(5), int((nn>1.8*0.0088).sum())
out={}
# DFSPH2D on omniSPH's lattice (the boundaries repo's own validated configuration: hydrostatic wall, default config)
sim=DFSPH2D(xo,np.zeros_like(xo),np.full(len(xo),Vo),np.full(len(xo),ho),domain_scene('surface',lo_o,hi_o,ho,'cuda:0'),DFSPHConfig(wallPressure='hydrostatic',fluidPairs='cells',graphIterations=False),'cuda:0')
while sim.time<T-1e-9: sim.step()
out['DFSPH2D, omniSPH lattice (its default config)']=sim.x.cpu().double().numpy()
# warpSPH analytic omni on the same lattice
dx=0.9/100; off=0.5
case=getCase('dambreak'); _ic=case.initialConditions
def _ic2(ctx,system):
    if _ic is not None: _ic(ctx,system)
    st=system.state; interior=ctx.scratch['interiorDomain']
    eff_lo=np.array([float(interior.min[0]),float(interior.min[1])])-off*dx
    pos=xo-lo_o+eff_lo
    st.positions=torch.tensor(pos,dtype=st.positions.dtype,device=st.positions.device)
    st.masses=torch.full_like(st.masses,Vo); st.supports=torch.full_like(st.supports,ho)
case.initialConditions=_ic2
params={**case.params,'wallRepresentation':'analytic','analyticWallOffset':off,'W':1.5,'fluidWidth':23*dx/1.5,'fillRatio':91*dx/0.9,'wallBC':'freeSlip'}
spec=CaseSpec(caseName='o',scheme='omniIncompressible',params=params).merged(**case.defaults).merged(scheme='omniIncompressible',L=0.9,nx=100,n_h=2.57,integrationScheme='semiImplicitEuler',kernel='Wendland2',supportMode='SuperSymmetric',cflFactor=1.0,dt=1e-3,minDt=1e-4,maxDt=1e-3,tLimit=T,adaptiveDt=True,plot=False,store=False,progress=False,video=False,show=False,quiet=True)
r=run(case,spec); st=r.state.state
interior=r.ctx.scratch['interiorDomain']; eff_lo=np.array([float(interior.min[0]),float(interior.min[1])])-off*dx
xw=st.positions.cpu().double().numpy()-eff_lo+lo_o
out['warpSPH analytic omni, omniSPH lattice']=xw
vm=float(st.velocities.norm(dim=1).max())
print(f'N {len(xw)}  support {float(st.supports[0]):.5f}  mass {float(st.masses[0]):.4e}  t {float(r.state.t):.3f}')
print(f'compiled omniSPH final (t=0.6, stored)')
for n,x in [('compiled omniSPH',d['omni_x'])]+list(out.items()):
    m=metrics(x); print(f'  {n:50s} cv {m[0]:.3f}  nn 5/50/95 {m[1]}  isolated {m[2]}   front {x[:,0].max():.4f} mean y {x[:,1].mean():.4f}')

print('wall proximity at t=%.2f (distance of a particle to the nearest wall plane, in dx = 0.0088):'%T)
def wallstats(x):
    dist=np.minimum.reduce([x[:,0]-lo_o[0],hi_o[0]-x[:,0],x[:,1]-lo_o[1],hi_o[1]-x[:,1]])/0.0088
    return float(dist.min()), int((dist<0.35).sum()), int((dist<0.6).sum())
for n,x in [('compiled omniSPH',d['omni_x'])]+list(out.items()):
    m=wallstats(x); print(f'  {n:50s} min {m[0]:+.2f} dx   #<0.35dx {m[1]}   #<0.6dx {m[2]}')
