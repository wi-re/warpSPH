"""MLS wall pressure (wallPressure='linear') of the omni analytic walls against DFSPH2D, term by term on the identical lattice, for a non-trivial pressure field. Derived from the probe below.

Term-by-term comparison of the analytic-wall omniIncompressible terms with the boundaries repo's DFSPH2D (the oracle) on the identical dam-break lattice: density, alpha, both source terms, the wall gradient,
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
params={**case.params,'wallRepresentation':'analytic','analyticWallOffset':off,'W':W,'fluidWidth':23*dx/W,'fillRatio':91*dx/L,'wallBC':'freeSlip','calibratedLattice':False}
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
dcfg=DFSPHConfig(gravity=(0.0,-9.81),maxDt=1e-3,minDt=1e-4,boundaryInDivergence=True,wallPressure='linear',xsph=0.0,boundaryFriction=0.0,wallMass=1.0,recordForces=False,fluidPairs='cells',graphIterations=False)
sim=DFSPH2D(x0,np.zeros_like(x0),m0,h0,sc2,dcfg,dev)
sim.dt=1e-3
sim._prepare(); rho_d=(sim._sum(sim.V[sim.pj]*sim.W)+sim.lam).cpu().double().numpy()
d=rho_w-rho_d
print('density  warpSPH vs DFSPH2D: max|diff| %.2e  rms %.2e   (warpSPH rho min/max %.4f/%.4f, DFSPH2D %.4f/%.4f)'%(np.abs(d).max(),np.sqrt((d**2).mean()),rho_w.min(),rho_w.max(),rho_d.min(),rho_d.max()))

import warpSPH.modules.analyticBoundary as AB
dt=1e-3
sim=DFSPH2D(x0,np.zeros_like(x0),m0,h0,sc2,dcfg,dev); sim.dt=dt
sim._bdiv=True; sim._clampWallDiv=True
sim._prepare(); sim.rho=sim._sum(sim.V[sim.pj]*sim.W)+sim.lam
cfg.dt=dt; fluid=st.kinds==0
st.densities=computeDensities(st,cfg,sc,adj)
wall=AB.resolveWall(st,cfg,sc,adj)
cp=lambda t: t.detach().cpu().double().numpy()
def cmp(name,a,b):
    a=np.asarray(a,float).reshape(len(x0),-1); b=np.asarray(b,float).reshape(len(x0),-1)
    d=np.abs(a-b); sc_=np.sqrt((b**2).mean())
    i=np.unravel_index(np.argmax(d),d.shape)[0]
    print(f'{name:40s} max|d| {d.max():.3e}  rms|d| {np.sqrt((d**2).mean()):.3e}  rms(ref) {sc_:.3e}  rel.rms {np.sqrt((d**2).mean())/max(sc_,1e-300):.3e}  worst i={i} x={x0[i,0]:+.3f} y={x0[i,1]:+.3f}')
H=float(x0[:,1].max())+0.5*dx
rng=np.random.default_rng(0)
for label,noise in (('hydrostatic p',0.0),('hydrostatic + 1 % noise',0.01),('hydrostatic + 10 % noise',0.1)):
    ph=9.81*(H-x0[:,1])
    p_np=ph*(1.0+noise*rng.standard_normal(len(x0)))
    p_t=torch.tensor(p_np,dtype=torch.float64,device=dev).clamp(min=0)
    pw=p_t.to(st.densities.dtype)
    fit=AB.buildMLSPressureFit(st,cfg,adj,fluid,sc.fluid.restDensity)
    a1_w=cp(fit.gradient(pw)); a1_d=cp(torch.einsum("nij,nj->ni",sim.Minv,sim._sum(( (p_t[sim.pj]-p_t[sim.pi])[:,None]*sim.wy))))
    print(label)
    cmp('  MLS gradient a1',a1_w,a1_d)
    a_wall_w=AB.wallPressureAccelerationOmni(wall,pw,st.densities,sc.fluid.restDensity,wallMass=wall.wm,h=wall.support,gradient=fit.gradient(pw))
    cmp('  wall part (MLS)',cp(a_wall_w),cp(sim._boundary_accel(p_t,True)))
