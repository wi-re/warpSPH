"""The Morris viscous acceleration with the no-slip wall closure (`modules/analyticBoundary/wallViscosity.py`) against `DFSPH2D._viscous_accel`, per particle on the identical dam-break lattice, for a smooth velocity field with wall-normal and tangential parts.

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
params={**case.params,'wallRepresentation':'analytic','analyticWallOffset':off,'W':W,'fluidWidth':23*dx/W,'fillRatio':91*dx/L,'wallBC':'noSlip','calibratedLattice':False}
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
from warpSPH.modules.deltaSPH import computeVelocityDiffusion
from warpSPH.enumTypes import ViscosityTerm
from warpSPH.modules.incompressible.compactProjection import morrisCalibration
dt=1e-3
fluid=st.kinds==0
rho0=sc.fluid.restDensity
st.densities=computeDensities(st,cfg,sc,adj)
cal=morrisCalibration(float(st.supports.double().median()),float(cfg.dx),float(st.masses.double().median())/rho0)
nu=0.01
dcfg.viscosity=nu; dcfg.morrisCalibration=cal; dcfg.boundaryFriction=0.0
cp=lambda t: t.detach().cpu().double().numpy()
def cmp(name,a,b):
    a=np.asarray(a,float).reshape(len(x0),-1); b=np.asarray(b,float).reshape(len(x0),-1)
    d=np.abs(a-b); sc_=np.sqrt((b**2).mean())
    i=np.unravel_index(np.argmax(d),d.shape)[0]
    print(f'{name:44s} max|d| {d.max():.3e}  rms|d| {np.sqrt((d**2).mean()):.3e}  rms(ref) {sc_:.3e}  rel.rms {np.sqrt((d**2).mean())/max(sc_,1e-300):.3e}  worst i={i} x={x0[i,0]:+.3f} y={x0[i,1]:+.3f}')
sim=DFSPH2D(x0,np.zeros_like(x0),m0,h0,sc2,dcfg,dev); sim.dt=dt
sim._prepare(); sim.rho=sim._sum(sim.V[sim.pj]*sim.W)+sim.lam
sim.forceViscous=torch.zeros((sim.nb,2),dtype=torch.float64,device=dev); sim.torque=torch.zeros((3,sim.nb),dtype=torch.float64,device=dev)
sc.diffusionParams.inviscid=False; sc.diffusionParams.viscidNu=nu/cal; sc.diffusionParams.viscousTerm=ViscosityTerm.morris1997; sc.morrisCalibration=cal
sc.wallViscosityClosure='noslipMoment'
cfg.dt=dt
for label,vf in (('smooth swirl',lambda p: np.stack([0.4*np.sin(2*np.pi*p[:,1]/0.8)*np.cos(2*np.pi*p[:,0]/1.5),-0.3*np.sin(2*np.pi*p[:,0]/1.5)*np.cos(np.pi*p[:,1]/0.9)],1)),
                 ('wall-normal + tangential noise',lambda p: 0.2*np.random.default_rng(3).standard_normal(p.shape))):
    vel=vf(x0)
    sim.v=torch.tensor(vel,dtype=torch.float64,device=dev); st.velocities=torch.tensor(vel,dtype=st.velocities.dtype,device=st.velocities.device)
    for comp in (True,False):
        from warpSPHBoundaries.sim.wallclosure import NoSlipClosure
        from warpSPHBoundaries.sim.pairs import dwendland2
        sim._wcObj=NoSlipClosure(sim.scene,dwendland2,float(sim.h.max()),float(cfg.dx),dev,morris=True,cal=cal,wallMass=1.0,complement=comp,sums=sim._complement_sums)      # DFSPH2D hard-wires complement=True
        sc.complementMoments=comp
        ref=sim._viscous_accel()
        mine=computeVelocityDiffusion(st,cfg,sc,adj)
        cmp(f'{label}, complement={comp}',cp(mine),cp(ref))
