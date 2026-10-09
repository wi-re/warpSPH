"""Match the boundaries repo's original analytic dam-break result (F_surf_r5, DFSPH2D with exact edge integrals on the calibrated lattice) with warpSPH's analytic omniIncompressible, per particle over time (2026-10-09).
The setup: omniSPH's lattice (23 x 91 centres on the block edges), `lattice_calibration` (particle mass V', wall mass mu, wall plane 0.55 dx from the first row), h = sqrt(20) r, hydrostatic wall closure, the
wall NOT in the divergence solve (omniSPH and DFSPH2D for static walls). Arguments: end time, mode (calib | omni = pi r^2, walls one spacing out). Writes a snapshot npz in the format of the boundaries repo's dfsph_video.render.
"""
import sys, math, numpy as np, torch
sys.path.insert(0, '/home/lu26029/dev/warpSPH/scripts')
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases import importAll
importAll()
from warpSPH.runner import getCase, run
from warpSPH.runner.caseSpec import CaseSpec
import warpSPH.schemes.omniIncompressible as O
from warpSPHBoundaries.sim.dfsph2d import lattice_calibration
R='/home/lu26029/dev/curvatureBoundaries/.tmp/omni/runs'
ref=np.load(R+'/F_surf_r5.npz'); omni=np.load(R+'/omni_r5.npz')
T=float(sys.argv[1]) if len(sys.argv)>1 else 0.4
mode=sys.argv[2] if len(sys.argv)>2 else 'calib'      # calib: the boundaries repo's calibrated lattice (F_surf_r5) | omni: omniSPH conventions (pi r^2, wall one spacing)
xo=ref['x'][0].astype(np.float64); h=float(ref['h']); lo_ref=ref['lo']; hi_ref=ref['hi']
dxo=0.2/22; dyo=0.8/90
if mode=='calib':
    cal=lattice_calibration(dxo,dyo,h); V=cal['V']; mu=cal['mu']; lo=xo.min(0)-np.array([cal['dwallX'],cal['dwallY']])
else:
    V=math.pi*0.005**2; mu=1.0; lo=np.array([0.1-dxo,0.1-dyo])
hi=np.array([1.6,1.0])
print(f'mode {mode}: V {V:.5e}  wall mass {mu:.4f}  lo {lo}  hi {hi}  (ref lo {lo_ref})')
dx=0.9/100; Lc=0.9; off=(hi[1]-lo[1]-Lc)/(2*dx); Wc=(hi[0]-lo[0])-2*off*dx
case=getCase('dambreak'); _ic=case.initialConditions; _ps=case.postStep; _cs=case.configureScheme
rec={'t':[],'x':[],'v':[]}
def _cs2(ctx):
    if _cs is not None: _cs(ctx)
    ctx.schemeConfig.analyticWallMass=mu
case.configureScheme=_cs2
def _ic2(ctx,system):
    if _ic is not None: _ic(ctx,system)
    st=system.state; interior=ctx.scratch['interiorDomain']
    eff_lo=np.array([float(interior.min[0]),float(interior.min[1])])-off*dx
    st.positions=torch.tensor(xo-lo+eff_lo,dtype=st.positions.dtype,device=st.positions.device)
    st.masses=torch.full_like(st.masses,V); st.supports=torch.full_like(st.supports,h)
    ctx.scratch['effLo']=eff_lo
    system.state.positions=st.positions
def _ps2(ctx,state,step):
    if _ps is not None: _ps(ctx,state,step)
    rec['t'].append(float(state.t)); rec['x'].append(state.state.positions.cpu().double().numpy()-ctx.scratch['effLo']+lo); rec['v'].append(state.state.velocities.cpu().double().numpy())
case.initialConditions=_ic2; case.postStep=_ps2
params={**case.params,'wallRepresentation':'analytic','analyticWallOffset':off,'W':Wc,'fluidWidth':23*dx/Wc,'fillRatio':91*dx/Lc,'wallBC':'freeSlip'}
spec=CaseSpec(caseName='o',scheme='omniIncompressible',params=params).merged(**case.defaults).merged(scheme='omniIncompressible',L=Lc,nx=100,n_h=2.57,integrationScheme='semiImplicitEuler',kernel='Wendland2',supportMode='SuperSymmetric',cflFactor=1.0,dt=1e-3,minDt=1e-4,maxDt=1e-3,tLimit=T,adaptiveDt=True,plot=False,store=False,progress=False,video=False,show=False,quiet=True)
r=run(case,spec)
print('warpSPH steps',len(rec['t']),'t_end',rec['t'][-1],'N',rec['x'][0].shape[0])
t=np.array(rec['t']); X=np.array(rec['x']); Vv=np.array(rec['v'])
sp=0.0088
print('rms position error [dx] and rms velocity error [m/s] of warpSPH vs the boundaries repo analytic (F_surf_r5) / vs compiled omniSPH, and reference F_surf vs omniSPH')
for tt in (0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6):
    if tt>T: break
    k=int(np.argmin(abs(t-tt))); kr=int(np.argmin(abs(ref['t']-tt))); ko=int(np.argmin(abs(omni['t']-tt)))
    e_ref=np.sqrt(((X[k]-ref['x'][kr])**2).sum(1).mean())/sp; e_om=np.sqrt(((X[k]-omni['x'][ko])**2).sum(1).mean())/sp; e_ro=np.sqrt(((ref['x'][kr]-omni['x'][ko])**2).sum(1).mean())/sp
    v_ref=np.sqrt(((Vv[k]-ref['v'][kr])**2).sum(1).mean()); v_ro=np.sqrt(((ref['v'][kr]-omni['v'][ko])**2).sum(1).mean())
    print(f'  t={tt:.2f} (warp t={t[k]:.3f}, ref t={ref["t"][kr]:.3f}):  warpSPH-vs-F_surf {e_ref:6.3f} dx  |dv| {v_ref:.3f}   warpSPH-vs-omniSPH {e_om:6.3f} dx   (F_surf-vs-omniSPH {e_ro:6.3f} dx |dv| {v_ro:.3f})')
np.savez('/tmp/claude-598314/-home-lu26029-dev-warpSPH/632d38d6-a97d-4844-9942-72f794f49a8f/scratchpad/match_%s.npz'%mode,t=t,x=X,v=Vv)

# snapshot file in the boundaries repo's video format (60 fps, nearest step), metrics at t = 0.6 and 1.0
ts=[tt for tt in ref['t'] if tt<=T+1e-9]; idx=[int(np.argmin(abs(t-tt))) for tt in ts]
snap=dict(t=np.array([t[i] for i in idx]),x=X[idx],v=Vv[idx],rho=np.zeros((len(idx),X.shape[1])),p=np.zeros((len(idx),X.shape[1])),lo=lo,hi=hi,r=0.005,h=h)
OUT='/tmp/claude-598314/-home-lu26029-dev-warpSPH/632d38d6-a97d-4844-9942-72f794f49a8f/scratchpad/warp_%s_snap.npz'%mode
np.savez(OUT,**snap); print('wrote',OUT,snap['x'].shape)
def met(x):
    D=np.linalg.norm(x[:,None]-x[None],axis=2); np.fill_diagonal(D,9); nn=D.min(1); b=(D<2.57*sp).sum(1)>=10
    dist=np.minimum.reduce([x[:,0]-lo[0],hi[0]-x[:,0],x[:,1]-lo[1],hi[1]-x[:,1]])/sp
    return nn[b].std()/nn[b].mean(), np.percentile(nn[b],5), int((nn>1.8*sp).sum()), float(dist.min()), int((dist<0.35).sum())
print('noise / wall metrics: spacing cv, nn 5th percentile [m], isolated, closest wall distance [dx], #<0.35dx')
for tt in (0.6,1.0):
    if tt>T: continue
    for n,d in (('compiled omniSPH',omni),('boundaries repo analytic (F_surf)',ref)):
        m=met(d['x'][int(np.argmin(abs(d['t']-tt)))]); print(f'  t={tt}  {n:36s} cv {m[0]:.3f}  nn5 {m[1]:.5f}  isolated {m[2]}  wall min {m[3]:+.2f}  #<0.35 {m[4]}')
    m=met(X[int(np.argmin(abs(t-tt)))]); print(f'  t={tt}  {"warpSPH analytic omni":36s} cv {m[0]:.3f}  nn5 {m[1]:.5f}  isolated {m[2]}  wall min {m[3]:+.2f}  #<0.35 {m[4]}')
