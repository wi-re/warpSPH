#!/usr/bin/env python3
"""Lobovsky ACSPH comparison using exact dynGPUSPH ICs and timestep replay.

No production solver changes. --coefficients nominal preserves the nominal
reference CFL/diffusion constants; mapped compensates the existing local h
convention to match beta and k2. Both retain current warpSPH operators.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from render import read_snapshot


def write_json(path, data):
    path.write_text(json.dumps(data,indent=2,default=lambda x:getattr(x,'name',str(x)))+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=3)
    ap.add_argument('--coefficients',choices=('nominal','mapped'),default='nominal')
    ap.add_argument('--history',choices=('bdf1','source'),default='bdf1')
    args=ap.parse_args(); args.out=args.out.resolve(); args.out.mkdir(parents=True,exist_ok=False)
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import numpy as np
    import torch
    assert torch.cuda.is_available(), 'CUDA FP64 required for this reference comparison'
    torch.set_num_threads(4)
    from warpSPH.runner import Case,run
    from warpSPH.geometry.types import ParticleSet
    from warpSPH.configurations.region import ParticleRegion,RegionType,BCType
    from warpSPH.initializers.weaklyCompressible import initializeSimulation
    from warpSPH.configurations import GravityType
    from warpSPH.configurations.moduleConfigurations.shifting import ShiftingScheme,ShiftingProjectionScheme
    from warpSPH.cases.common.plotting import Field,buildFieldPlotter,refreshFieldPlotter
    from warpSPHCore import sphKernel_xi
    from warpSPH.modules.artificialCompressible.parameters import acParameters

    ref=read_snapshot(args.reference/'snapshots/000000.bin')
    controls=[float(line.split('#')[0]) for line in (args.reference/'input_values.dat').read_text().splitlines()]
    ref_rows=json.loads((args.reference/'result.json').read_text())['steps']
    if args.steps<1 or args.steps>len(ref_rows): raise ValueError('steps must fit the recorded reference sequence')
    dt=[r['dt'] for r in ref_rows[:args.steps]]
    dx=.00075; support=4*dx; rho0=controls[6]; h_ref=controls[10]*dx
    nu=max(controls[3],controls[4]*h_ref*controls[2]/8)
    state_info={}; rows=[]; fields=[Field('velocities','Speed',mapping='L2Norm'),Field('pressures','Pressure [Pa]',colorMap='coolwarm',colorMapKind='diverging')]
    (args.out/'snapshots').mkdir()

    def configure(ctx):
        c,sc=ctx.config,ctx.schemeConfig
        c.dx=dx; c.domain.min[:]=torch.tensor(ref['x'].min(0)-support,device=ctx.device)
        c.domain.max[:]=torch.tensor(ref['x'].max(0)+support,device=ctx.device);c.domain.periodic[:]=False
        c.dt=dt[0];c.calibrateNormalization=False;c.densityCorrection=False
        sc.fluid.restDensity=rho0;sc.fluid.fixedSoundSpeed=controls[2]
        sc.gravityConfig.active=True;sc.gravityConfig.type=GravityType.Directional;sc.gravityConfig.magnitude=9.81;sc.gravityConfig.direction=[0.,-1.]
        sc.surfaceDetectionConfig.active=True
        sc.shiftProperties.active=True;sc.shiftProperties.scheme=ShiftingScheme.michel2022;sc.shiftProperties.projectionScheme=ShiftingProjectionScheme.michel2022
        sc.noPenetrationShift=False;sc.dt_accelerationConstraint=False
        p=sc.acParams;p.rkStages=3;p.cflTau=.75;p.dtOverDtau=controls[-2];p.cflT=controls[9]
        p.epsilonV=controls[11];p.maxPseudoIterations=int(controls[13]);p.convergenceCheckSchedule='every'
        p.referenceSoundSpeedForViscosity=None;p.nu=nu;p.uChar=.1*controls[2]
        p.wallPressureEdgeRelax='lambda';p.diffusionLambdaTaper=True;p.redetectSurfacePerIteration=True
        p.wallBounceBack=True;p.wallBounceTarget='position';p.wallBounceRamp=0.
        xi=float(sphKernel_xi(c.kernel.value,2))
        if args.coefficients=='mapped':p.cflTau*=xi/2;p.k2Factor*=xi/2
        ctx.velocityScale=(2*9.81*.3)**.5;ctx.velocityScaleSource='Lobovsky sqrt(2gH)'
        state_info['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path('src/warpSPH/modules/artificialCompressible/residual.py'),Path('src/warpSPH/modules/artificialCompressible/parameters.py'),Path('src/warpSPH/modules/artificialCompressible/dualTime.py'),Path('src/warpSPH/modules/pressure/wp_surfaceAware.py'))}
        state_info['parameters']={'dx':dx,'support':support,'rho0':rho0,'reference_h':h_ref,'local_xi':xi,'local_h':support/xi,'nu':nu,'acParams':asdict(p),'coefficient_mode':args.coefficients}

    def build(ctx):
        def sdf(x):
            # Nominal tank planes from the supplied boundary rectangles. Particle
            # coordinates remain exact; no phase alignment or wall movement.
            lo=x.new_tensor([0.,0.]);hi=x.new_tensor([1.61,1.4]);q=torch.maximum(lo-x,x-hi)
            return -(torch.linalg.norm(q.clamp_min(0),dim=-1)+torch.minimum(q.max(-1).values,torch.zeros(len(x),device=x.device,dtype=x.dtype)))
        regs=[]
        for code,kind in ((1,RegionType.Fluid),(3,RegionType.Boundary)):
            mask=ref['bc']==code
            tensor=lambda x:torch.as_tensor(np.array(x,copy=True),device=ctx.device,dtype=ctx.dtype)
            x=tensor(ref['x'][mask]);n=len(x)
            particles=ParticleSet(x,torch.full((n,),support,device=ctx.device,dtype=ctx.dtype),tensor(ref['volume'][mask]),tensor(ref['rho'][mask]))
            regs.append(ParticleRegion(sdf=sdf,type=kind,particles=particles,kind=BCType.freeSlip,
                initialConditions={'pressures':lambda x,p=tensor(ref['p'][mask]):p,'velocities':tensor(ref['v'][mask]),'densities':tensor(ref['rho'][mask])}))
        ctx.schemeConfig.regions=regs
        system=initializeSimulation(regs,ctx.config,ctx.schemeConfig,ctx.SimulationSystem,ctx.SimulationState,verbose=False)
        ctx.schemeConfig.rigidBodies=ctx.config.rigidBodies
        # Rigid-body COM subtraction/addition can round static wall coordinates.
        # Restore the exact physical IC after building its derived ghost geometry.
        physical=system.state.kinds!=2
        system.state.positions[physical]=torch.as_tensor(ref['x'],device=ctx.device,dtype=ctx.dtype)
        return system

    def snapshot(ctx,system,index):
        st=system.state;keep=st.kinds!=2
        vals={name:getattr(st,name)[keep].detach().cpu().numpy() for name in ('positions','velocities','pressures','densities','masses','kinds')}
        n=len(vals['positions']);bc=np.where(vals['kinds']==0,1,3).astype('<i4');vol=vals['masses']/vals['densities']
        with (args.out/'snapshots'/f'{index:06d}.bin').open('wb') as f:
            f.write(struct.pack('<id',n,float(system.t)))
            for a in (vals['positions'],vals['pressures'],vals['densities'],vals['velocities']):f.write(a.astype('<f8').tobytes())
            f.write(bc.tobytes())
            for a in (np.zeros(n),vol,np.zeros(n)):f.write(a.astype('<f8').tobytes())
        return vals

    def initialize(ctx,system):
        if args.history=='source':
            # Upstream seeds duplicate histories and dt_prev=min_dt; it uses
            # BDF2 from its first step. This is a fixture option, not a default.
            system.positionsPrev=system.state.positions.clone()
            system.positionsPrev2=system.state.positions.clone()
            system.velocitiesPrev=system.state.velocities.clone()
            system.velocitiesPrev2=system.state.velocities.clone()
            system.dtPrev=controls[7]
        state_info['startup_history']=args.history
        state_info['initial_bdf_coefficients']=system.bdfCoefficients(dt[0])
        got=snapshot(ctx,system,0)
        expected={'positions':ref['x'],'velocities':ref['v'],'pressures':ref['p'],'densities':ref['rho'],'masses':ref['rho']*ref['volume']}
        checks={name:bool(np.array_equal(got[name],value)) for name,value in expected.items()}
        physical=system.state.kinds!=2;f=system.state.kinds==0
        par=acParameters(system.state,ctx.config,ctx.schemeConfig,dt[0])
        beta_ref=.75*h_ref/(dt[0]/controls[-2]); k2_ref=.1*h_ref*beta_ref
        state_info.update(initial_exact=checks,physical_particles=int(physical.sum()),ghost_particles=int((~physical).sum()),fluid_particles=int(f.sum()),
            fluid_mass=float(system.state.masses[f].sum()),fluid_volume=float((system.state.masses[f]/system.state.densities[f]).sum()),
            local_beta=float(par[1][0]),reference_beta=beta_ref,local_k2=float(par[3][0]),reference_k2=k2_ref,
            dtype=str(system.state.positions.dtype),device=str(ctx.device),reference_snapshot_sha256=hashlib.sha256((args.reference/'snapshots/000000.bin').read_bytes()).hexdigest())
        write_json(args.out/'initial-audit.json',state_info)
        print('INITIAL AUDIT',json.dumps({k:v for k,v in state_info.items() if k!='parameters'}),flush=True)
        assert all(checks.values()), checks
        if args.coefficients=='mapped':
            assert np.isclose(state_info['local_beta'],beta_ref,rtol=1e-12)
            assert np.isclose(state_info['local_k2'],k2_ref,rtol=1e-12)

    def post(ctx,system,step):
        snapshot(ctx,system,step+1)
        ctx.scratch['completed']=step+1

    def timestep(ctx,system):return dt[min(ctx.scratch.get('completed',0),len(dt)-1)]

    def diagnostics(ctx,system):
        st=system.state;f=st.kinds==0;v=st.velocities[f];p=st.pressures[f];update=ctx.scratch.get('lastStageUpdate')
        row={'maxVelocity':float(v.norm(dim=-1).max()),'kineticEnergy':float((.5*st.masses[f]*v.square().sum(-1)).sum()),
            'meanPressure':float(p.mean()),'minPressure':float(p.min()),'maxPressure':float(p.max()),
            'pseudoIterations':getattr(update,'pseudoIterations',0),'epsilonV':getattr(update,'epsilonV',None)}
        rows.append({'t':float(system.t),**row});write_json(args.out/'progress.json',rows)
        print('LOCAL STEP',json.dumps(rows[-1]),flush=True)
        return row

    case=Case('lobovskyExactReference','artificialCompressible',configureScheme=configure,buildSystem=build,initialConditions=initialize,
        timestep=timestep,postStep=post,diagnostics=diagnostics,
        setupPlot=lambda ctx,st:buildFieldPlotter(ctx,st,fields,figsize=(12,6)),
        updatePlot=lambda ctx,st,plot,step:refreshFieldPlotter(ctx,st,plot,fields,step=step))
    start=time.monotonic()
    result=run(case,precision='float64',device='cuda:0',nx=800,L=.6,n_h=4,kernel='Wendland2',periodic=False,
        integrationScheme='forwardEuler',calibrateNormalization=False,densityCorrection=False,
        nSteps=args.steps,tLimit=sum(dt),dt=dt[0],minDt=controls[7],maxDt=controls[8],
        plot=True,video=True,plotBackend='vispy',plotInterval=1,show=False,exportRoot=str(args.out/'recording'),
        cudaGraph=True,progress=True,quiet=False,store=False,velocityAlarmPlotInterval=1,velocityAlarmStopRatio=100,stallProgress=1e-3)
    write_json(args.out/'result.json',{'reference':str(args.reference.resolve()),'mode':args.coefficients,'history':args.history,'trajectory':result.trajectory,
        'nSteps':result.nSteps,'wall_seconds':time.monotonic()-start,'reported_wall_seconds':result.wallTime,
        'diverged':result.diverged,'stop_reason':result.stopReason,'exportPath':result.exportPath,
        'graph_disabled':getattr(result.ctx.schemeConfig,'_acsphGraphDisabled',None)})
    print('LOCAL COMPLETE',args.out,'steps',result.nSteps,'wall',time.monotonic()-start,flush=True)
    if result.diverged:raise SystemExit(1)

if __name__=='__main__':main()
