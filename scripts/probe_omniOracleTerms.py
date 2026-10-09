"""Term-by-term comparison of the analytic-wall omniIncompressible terms with the boundaries repo's DFSPH2D (the oracle) on the identical dam-break lattice: density, alpha, both source terms, the wall gradient,
the fluid and wall pressure accelerations (all equal to <= 1e-5 relative in float32 vs float64, 2026-10-09). Needs warpSPHBoundaries with its sim layer; arguments: wall offset in dx (0.5).
"""
import sys, numpy as np, torch
sys.path.insert(0, '/home/lu26029/dev/warpSPH/scripts')
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases import importAll
importAll()
from warpSPH.runner import getCase, run, buildContext
from warpSPH.runner.caseSpec import CaseSpec
from warpSPH.modules.density.density import computeDensities
import warpSPH.schemes.omniIncompressible as O
from warpSPHBoundaries.sim.dfsph2d import DFSPH2D, DFSPHConfig, domain_scene
off=float(sys.argv[1]) if len(sys.argv)>1 else 0.5
nSteps=int(sys.argv[2]) if len(sys.argv)>2 else 1
nx=100; L=0.9; W=1.5; dx=L/nx
case=getCase('dambreak')
params={**case.params,'wallRepresentation':'analytic','analyticWallOffset':off,'W':W,'fluidWidth':23*dx/W,'fillRatio':91*dx/L,'wallBC':'freeSlip'}
def spec(n):
    return CaseSpec(caseName='o',scheme='omniIncompressible',params=params).merged(**case.defaults).merged(scheme='omniIncompressible',L=L,nx=nx,n_h=2.57,integrationScheme='semiImplicitEuler',kernel='Wendland2',supportMode='SuperSymmetric',cflFactor=1.0,dt=1e-3,minDt=1e-4,maxDt=1e-3,nSteps=n,adaptiveDt=True,plot=False,store=False,progress=False,video=False,show=False,quiet=True)
# initial state (no step)
ctx=buildContext(case,spec(nSteps)); case.configureScheme(ctx); system=case.buildSystem(ctx)
if case.initialConditions is not None: case.initialConditions(ctx,system)
st=system.state; sc=ctx.schemeConfig; cfg=ctx.config
x0=st.positions.cpu().double().numpy(); m0=st.masses.cpu().double().numpy(); h0=st.supports.cpu().double().numpy()
adj=O._rebuildAdjacency(st,system,cfg)
rho_w=computeDensities(st,cfg,sc,adj).cpu().double().numpy()
print('N',len(x0),'dx',dx,'support',h0.min(),h0.max(),'mass',m0.min(),m0.max(),'calibrateNormalization',cfg.calibrateNormalization)
interior=ctx.scratch['interiorDomain']; lo=np.array([float(interior.min[0]),float(interior.min[1])])-off*dx; hi=np.array([float(interior.max[0]),float(interior.max[1])])+off*dx
print('wall box',lo,hi)
# DFSPH2D with identical inputs
dev='cuda:0'
sc2=domain_scene('surface',lo,hi,float(h0.max()),dev)
dcfg=DFSPHConfig(gravity=(0.0,-9.81),maxDt=1e-3,minDt=1e-4,boundaryInDivergence=True,wallPressure='hydrostatic',xsph=0.0,boundaryFriction=0.0,wallMass=1.0,recordForces=False,fluidPairs='cells',graphIterations=False)
sim=DFSPH2D(x0,np.zeros_like(x0),m0,h0,sc2,dcfg,dev)
sim.dt=1e-3
sim._prepare(); rho_d=(sim._sum(sim.V[sim.pj]*sim.W)+sim.lam).cpu().double().numpy()
d=rho_w-rho_d
print('density  warpSPH vs DFSPH2D: max|diff| %.2e  rms %.2e   (warpSPH rho min/max %.4f/%.4f, DFSPH2D %.4f/%.4f)'%(np.abs(d).max(),np.sqrt((d**2).mean()),rho_w.min(),rho_w.max(),rho_d.min(),rho_d.max()))

import warpSPH.modules.analyticBoundary as AB
from warpSPH.modules.incompressible.wp_alpha import computeAlpha
dt=1e-3
sim=DFSPH2D(x0,np.zeros_like(x0),m0,h0,sc2,dcfg,dev); sim.dt=dt
sim._bdiv=True; sim._clampWallDiv=True
sim._prepare(); sim.rho=sim._sum(sim.V[sim.pj]*sim.W)+sim.lam
Vt=sim.V/sim.rho
g=torch.tensor([0.0,-9.81],dtype=torch.float64,device=dev)
vp=dt*g.expand(len(x0),2).clone()
# ---- warpSPH side
cfg.dt=dt; fluid=st.kinds==0
st.densities=computeDensities(st,cfg,sc,adj)
wall=AB.resolveWall(st,cfg,sc,adj)
def cmp(name,a,b,rel=True):
    a=np.asarray(a,float).reshape(len(x0),-1); b=np.asarray(b,float).reshape(len(x0),-1)
    d=np.abs(a-b); sc_=np.sqrt((b**2).mean())
    i=np.unravel_index(np.argmax(d),d.shape)[0]
    print(f'{name:34s} max|d| {d.max():.3e}  rms|d| {np.sqrt((d**2).mean()):.3e}  rms(ref) {sc_:.3e}  rel.rms {np.sqrt((d**2).mean())/max(sc_,1e-300):.3e}  worst i={i} y={x0[i,1]:+.3f}')
cp=lambda t: t.detach().cpu().double().numpy()
alpha_w=dt*dt*computeAlpha(st,cfg,sc,adj,apparentVolumes=st.masses/st.densities,includeBoundaryReaction=False,wall=wall)
cmp('alpha (density & divergence)',cp(alpha_w),cp(sim._alpha(dt,Vt,True)))
div=O._divergence(st,cfg,adj,torch.as_tensor(vp,dtype=st.densities.dtype,device=st.densities.device),wall,bodyVelocity=True)
cmp('source, divergence solve',cp(dt*div),cp(sim._source(dt,Vt,vp,False,True)))
cmp('source, density solve',cp((1-st.densities)+dt*div),cp(sim._source(dt,Vt,vp,True,True)))
H=float(x0[:,1].max())+0.5*dx; p_t=torch.tensor(9.81*(H-x0[:,1]),dtype=torch.float64,device=dev)
a_d=sim._boundary_accel(p_t,True)+sim._fluid_accel(p_t)
a_w=O._pressureAccel(st,cfg,adj,p_t.to(st.densities.dtype),fluid,wall,sc.fluid.restDensity)
cmp('pressure accel, hydrostatic p (total)',cp(a_w),cp(a_d))
a_fl_w=O._pressureAccel(st,cfg,adj,p_t.to(st.densities.dtype),fluid,None)
cmp('  fluid part',cp(a_fl_w),cp(sim._fluid_accel(p_t)))
from warpSPH.modules.analyticBoundary import wallPressureAcceleration
pp=p_t.clamp(min=0).to(st.densities.dtype)
a_wall_w=AB.wallPressureAccelerationOmni(wall,pp,st.densities,sc.fluid.restDensity,wallMass=wall.wm,h=wall.support)
cmp('  wall part',cp(a_wall_w),cp(sim._boundary_accel(p_t,True)))
# wall part with p = 0 (the hydrostatic offset alone)
z=torch.zeros_like(pp)
cmp('  wall part at p = 0 (hydrostatic A)',cp(AB.wallPressureAccelerationOmni(wall,z,st.densities,sc.fluid.restDensity,wallMass=wall.wm,h=wall.support)),cp(sim._boundary_accel(torch.zeros_like(p_t),True)))
cmp('wall G (sum over bodies)',cp(wall.G.sum(0)),cp(sim.gk))
