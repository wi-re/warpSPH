"""XSPH and boundary friction of the omni analytic walls against DFSPH2D (the two velocity filters of its step, copied from sim/dfsph2d.py), term by term on the identical lattice, for a random velocity field, static and rotating wall. Derived from the probe below.

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
from warpSPH.modules.xsph import computeXSPH, computeBoundaryFriction
cfg.dt=1e-3; fluid=st.kinds==0
sim=DFSPH2D(x0,np.zeros_like(x0),m0,h0,sc2,dcfg,dev); sim.dt=1e-3
sim._prepare(); sim.rho=sim._sum(sim.V[sim.pj]*sim.W)+sim.lam
st.densities=computeDensities(st,cfg,sc,adj)
cp=lambda t: t.detach().cpu().double().numpy()
def cmp(name,a,b):
    a=np.asarray(a,float).reshape(len(x0),-1); b=np.asarray(b,float).reshape(len(x0),-1)
    d=np.abs(a-b); sc_=np.sqrt((b**2).mean())
    i=np.unravel_index(np.argmax(d),d.shape)[0]
    print(f'{name:44s} max|d| {d.max():.3e}  rms|d| {np.sqrt((d**2).mean()):.3e}  rms(ref) {sc_:.3e}  rel.rms {np.sqrt((d**2).mean())/max(sc_,1e-300):.3e}  worst i={i} x={x0[i,0]:+.3f} y={x0[i,1]:+.3f}')
rng=np.random.default_rng(1)
vel=0.5*rng.standard_normal((len(x0),2))
sim.v=torch.tensor(vel,dtype=torch.float64,device=dev); st.velocities=torch.tensor(vel,dtype=st.velocities.dtype,device=st.velocities.device)
i_,j_=sim.pi,sim.pj
for xs in (1e-4,0.05):
    w=2.0*sim.V[j_]/(sim.rho[i_]+sim.rho[j_])*sim.W
    dv_d=xs*sim._sum(w[:,None]*(sim.v[j_]-sim.v[i_]))
    cmp(f'XSPH c={xs}',cp(computeXSPH(st,cfg,sc,adj,fluidCoefficient=xs)),cp(dv_d))

def friction_oracle(sim,c):
    """sim/dfsph2d.py step(): the boundary-friction block, verbatim (fused provider)."""
    sim.ps=sim._particleState(torch.ones_like(sim.V))
    fw=sim._aggregate(sim.ps)
    lam,gk=sim.cfg.wallMass*fw.out["lam"],sim.cfg.wallMass*fw.out["G"]; m1=None
    dv=torch.zeros_like(sim.v)
    for bi,b in enumerate(sim.scene.bodies):
        moving=float(b.angularVelocity)!=0.0 or float(b.linearVelocity.norm())!=0.0
        vw=torch.zeros_like(sim.v)
        if moving:
            from warpSPHBoundaries.scene import WallOutput
            if m1 is None: m1=fw.evaluate((WallOutput("m1",0,"m1"),))["m1"]
            lb=lam[bi]/sim.cfg.wallMass
            num=b.velocityAt(sim.x)*lb[:,None]+float(b.angularVelocity)*torch.stack([-m1[bi][:,1],m1[bi][:,0]],1)
            ok=lam[bi]>1e-8*sim.cfg.wallMass
            vw=torch.where(ok[:,None],num*sim.cfg.wallMass/lam[bi].clamp(min=1e-300)[:,None],torch.zeros_like(num))
        nrm=gk[bi].norm(dim=1,keepdim=True); n=-gk[bi]/nrm.clamp(min=1e-300)
        vr=sim.v-vw; tang=vr-(vr*n).sum(1,keepdim=True)*n
        fac=(c*lam[bi]).clamp(max=1.0)
        dv=dv+torch.where((nrm[:,0]>0)[:,None],-fac[:,None]*tang,torch.zeros_like(tang))
    return dv
sc.boundaryFriction=5e-3
for label,om,lin in (('static wall',0.0,(0.0,0.0)),('rotating wall (omega 0.7)',0.7,(0.0,0.0)),('translating + rotating',0.7,(0.3,-0.2))):
    for b in sim.scene.bodies:
        b.angularVelocity=torch.tensor(om,dtype=torch.float64,device=dev); b.linearVelocity=torch.tensor(lin,dtype=torch.float64,device=dev)
    rb=sc.boundaryProvider.rigidBodies[0]
    rb.angularVelocity=torch.tensor(om,dtype=rb.angularVelocity.dtype,device=rb.angularVelocity.device); rb.linearVelocity=torch.tensor(lin,dtype=rb.linearVelocity.dtype,device=rb.linearVelocity.device)
    print(label, ' body centre sim', cp(sim.scene.bodies[0].center), ' warpSPH', cp(rb.centerOfMass))
    cmp('  boundary friction c=5e-3',cp(computeBoundaryFriction(st,cfg,sc,adj)),cp(friction_oracle(sim,5e-3)))
