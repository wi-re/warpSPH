#!/usr/bin/env python3
"""Lobovsky ACSPH comparison using exact dynGPUSPH ICs and timestep replay.

Explicit controls retain the absolute-position/unfused historical comparison
despite newer production defaults. --coefficients nominal preserves reference
CFL/diffusion constants; mapped compensates the local h convention. Optional
wall, RK and damping controls are reference-code diagnostics.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from render import read_snapshot
from fixture_geometry import lobovsky_tank_sdf, audit_flat_wall_normals, volume_ghost_initializer


def write_json(path, data):
    path.write_text(json.dumps(data,indent=2,default=lambda x:getattr(x,'name',str(x)))+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=3,help='Number of recorded reference steps; 0 replays all')
    ap.add_argument('--coefficients',choices=('nominal','mapped'),default='nominal')
    ap.add_argument('--history',choices=('bdf1','source'),default='bdf1')
    ap.add_argument('--audit-stages',action='store_true',help='Save first-step RK inputs and spatial residuals; eager observation only')
    ap.add_argument('--wall-diffusion',choices=('native','source'),default='native',help='Probe-only AC-2L wall gradient treatment')
    ap.add_argument('--acoustic-damper',choices=('off','source'),default='off',help='Probe-only reference bulk-divergence damper')
    ap.add_argument('--reference-operators',type=Path,help='Audit the damper on these exact native RK states')
    ap.add_argument('--rk-treatment',choices=('native','source'),default='native',help='Probe-only reference RK source placement, including integer-division quirk')
    ap.add_argument('--pseudo-iterations',type=int,help='Explicit bounded operator-probe budget; not a converged trajectory')
    ap.add_argument('--wall-velocity',choices=('native','source','source-consistent'),default='native',help='Static wall Shepard/mirror; source is historical RK-only, source-consistent also covers entry and pre-shift')
    ap.add_argument('--performance',choices=('profile','graph','graph-native'),help='Eager profile, historical guard-specialization control, or production graph path')
    ap.add_argument('--precision',choices=('float64','float32'),default='float64',help='FP32 explicitly rounds the supplied FP64 arrays')
    ap.add_argument('--long-run',action='store_true',help='Reference-cadence snapshots and lightweight per-step timing; explicit FP32 trajectory qualification')
    ap.add_argument('--plot-interval',type=int,default=1,help='Real steps between live VisPy frames')
    ap.add_argument('--position-form',choices=('absolute','increment'),default='absolute',help='Explicit BDF position arithmetic comparison')
    ap.add_argument('--fused-residual',action='store_true',help='Use the fused ACSPH stage pair sums')
    ap.add_argument('--wall-bounce-ramp',type=float,default=0.,help='D5 smoothing width in dx; 0 retains the native hard switch')
    ap.add_argument('--ghost-spacing',choices=('legacy-mass','volume'),default='legacy-mass',help='Explicit ghost-init unit control; volume uses (m/rho)^(1/dim)')
    ap.add_argument('--shifting',choices=('warpsph-michel','dyngpusph-reference'),default='warpsph-michel',help='External shift/field-transport model; reference includes compiled-code history and velocity-remap quirks')
    ap.add_argument('--solver-model',choices=('warpsph','dyngpusph-static-2d'),default='warpsph',help='Complete static-2D compiled-reference AC-2L solve; legacy comparison hooks stay separate')
    ap.add_argument('--eager',action='store_true',help='Explicit complete-state graph qualification control')
    args=ap.parse_args()
    if args.solver_model=='dyngpusph-static-2d':
        args.coefficients='mapped';args.history='source';args.shifting='dyngpusph-reference'
        args.wall_velocity='source-consistent';args.wall_diffusion='source'
        args.acoustic_damper='source';args.rk_treatment='source';args.fused_residual=True
        if args.audit_stages or args.performance:
            ap.error('Reference solver uses its own fixed-field qualification and solve diagnostics')
    if args.pseudo_iterations is not None and args.pseudo_iterations<1:ap.error('--pseudo-iterations must be positive')
    if args.reference_operators is not None and not args.audit_stages:ap.error('--reference-operators requires --audit-stages')
    if args.precision!='float64' and not (args.performance or args.long_run):ap.error('Rounded FP32 ICs require explicit performance or long-run qualification')
    if args.long_run and (args.performance or args.audit_stages):ap.error('Long runs use bounded observations instead of retained profiler/graph objects')
    if args.plot_interval<1:ap.error('--plot-interval must be positive')
    if args.shifting=='dyngpusph-reference' and args.wall_velocity!='source-consistent':
        ap.error('dyngpusph-reference shifting requires --wall-velocity source-consistent')
    if not 0 <= args.wall_bounce_ramp < float('inf'):ap.error('--wall-bounce-ramp must be finite and nonnegative')
    args.out=args.out.resolve(); args.out.mkdir(parents=True,exist_ok=False)
    from warpSPHBootstrap import bootstrap
    bootstrap(precision=args.precision)
    import numpy as np
    import torch
    assert torch.cuda.is_available(), 'CUDA required for this reference comparison'
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
    if args.steps==0:args.steps=len(ref_rows)
    if args.steps<1 or args.steps>len(ref_rows): raise ValueError('steps must fit the recorded reference sequence')
    dt=[r['dt'] for r in ref_rows[:args.steps]]
    snapshot_steps=None
    if args.long_run:
        from export_viewer import reference_snapshot_steps
        snapshot_steps=reference_snapshot_steps(args.reference,ref_rows)
        snapshot_steps.add(args.steps)
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
        p.referenceModel='none' if args.solver_model=='warpsph' else args.solver_model
        if p.referenceModel!='none' and controls[18:24]!=[1024.,512.,2.,-.05,-.05,-.00001]:
            raise ValueError('Qualified reference solver currently expects the supplied Lobovsky hash geometry')
        p.incrementalPositions=args.position_form=='increment'
        p.fusedResidual=args.fused_residual
        if args.shifting=='dyngpusph-reference':
            import math
            p.bdfShiftCorrection=False
            grid_z=int(controls[20])
            if grid_z<1 or grid_z&(grid_z-1):raise ValueError('Reference hash grid z size must be a power of two')
            cell_z=math.floor(-controls[23]/(2*h_ref))
            sc._dyngpusphShiftPairMultiplicity=sum(((cell_z+offset)&(grid_z-1))==(cell_z&(grid_z-1)) for offset in range(-2,3))
        p.epsilonV=controls[11];p.maxPseudoIterations=int(controls[13]);p.convergenceCheckSchedule='every'
        if args.pseudo_iterations is not None:p.maxPseudoIterations=args.pseudo_iterations
        p.referenceSoundSpeedForViscosity=None;p.nu=nu;p.uChar=.1*controls[2]
        p.wallPressureEdgeRelax='lambda';p.diffusionLambdaTaper=True;p.redetectSurfacePerIteration=True
        p.wallBounceBack=True;p.wallBounceTarget='position';p.wallBounceRamp=args.wall_bounce_ramp
        xi=float(sphKernel_xi(c.kernel.value,2))
        if args.coefficients=='mapped':p.cflTau*=xi/2;p.k2Factor*=xi/2
        ctx.velocityScale=(2*9.81*.3)**.5;ctx.velocityScaleSource='Lobovsky sqrt(2gH)'
        state_info['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path('src/warpSPH/modules/artificialCompressible/residual.py'),Path('src/warpSPH/modules/artificialCompressible/parameters.py'),Path('src/warpSPH/modules/artificialCompressible/dualTime.py'),Path('src/warpSPH/modules/pressure/wp_surfaceAware.py'),Path(__file__),Path(__file__).with_name('audit_local.py'),Path(__file__).with_name('fixture_geometry.py'))}
        state_info['parameters']={'dx':dx,'support':support,'rho0':rho0,'reference_h':h_ref,'local_xi':xi,'local_h':support/xi,'nu':nu,'acParams':asdict(p),'coefficient_mode':args.coefficients}
        state_info['parameters']['wall_velocity_closure']=args.wall_velocity
        state_info['parameters']['external_shifting']=args.shifting
        if args.shifting=='dyngpusph-reference':state_info['parameters']['reference_shift_pair_multiplicity']=sc._dyngpusphShiftPairMultiplicity

    def build(ctx):
        regs=[]
        for code,kind in ((1,RegionType.Fluid),(3,RegionType.Boundary)):
            mask=ref['bc']==code
            tensor=lambda x:torch.as_tensor(np.array(x,copy=True),device=ctx.device,dtype=ctx.dtype)
            x=tensor(ref['x'][mask]);n=len(x)
            particles=ParticleSet(x,torch.full((n,),support,device=ctx.device,dtype=ctx.dtype),tensor(ref['volume'][mask]),tensor(ref['rho'][mask]))
            regs.append(ParticleRegion(sdf=lobovsky_tank_sdf,type=kind,particles=particles,kind=BCType.freeSlip,
                initialConditions={'pressures':lambda x,p=tensor(ref['p'][mask]):p,'velocities':tensor(ref['v'][mask]),'densities':tensor(ref['rho'][mask])}))
        ctx.schemeConfig.regions=regs
        geometry_audit={'mode':args.ghost_spacing}
        context=volume_ghost_initializer(geometry_audit) if args.ghost_spacing=='volume' else nullcontext()
        with context:
            system=initializeSimulation(regs,ctx.config,ctx.schemeConfig,ctx.SimulationSystem,ctx.SimulationState,verbose=False)
        state_info['ghost_geometry']=geometry_audit
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
        ctx.scratch['last_snapshot_step']=index
        return vals

    def initialize(ctx,system):
        state_info['derived_wall_geometry']=audit_flat_wall_normals(system.state,
            margin=2*support if args.ghost_spacing=='volume' else .05)
        if args.solver_model=='warpsph' and (args.audit_stages or args.wall_diffusion != 'native' or args.acoustic_damper != 'off' or args.rk_treatment != 'native' or args.wall_velocity != 'native'):
            from audit_local import install
            install(args.out/'operators',observe=args.audit_stages,wall_diffusion=args.wall_diffusion,
                    acoustic_damper=args.acoustic_damper,reference_operators=args.reference_operators,rk_treatment=args.rk_treatment,wall_velocity=args.wall_velocity,shifting=args.shifting)
        if args.audit_stages:
            # Host-side snapshots cannot be captured. Eager is already the
            # qualified fallback for this fixture; observation is explicit.
            ctx.schemeConfig._acsphGraphDisabled='stage audit requires host snapshots'
        if args.performance:
            if args.audit_stages:raise ValueError('Performance runs must disable stage snapshots')
            from performance import install as install_performance
            ctx.scratch['finish_performance']=install_performance(args.out/'performance',system,ctx.schemeConfig,args.performance)
        if args.long_run:
            from performance import install_light
            install_light(args.out/'solve-timing.jsonl',ctx.schemeConfig)
        if args.history=='source':
            # Upstream seeds duplicate histories and dt_prev=min_dt; it uses
            # BDF2 from its first step. This is a fixture option, not a default.
            system.positionsPrev=system.state.positions.clone()
            system.positionsPrev2=system.state.positions.clone()
            system.velocitiesPrev=system.state.velocities.clone()
            system.velocitiesPrev2=system.state.velocities.clone()
            system.dtPrev=controls[7]
        state_info['startup_history']=args.history
        state_info['solver_model']=args.solver_model
        state_info['long_run']=args.long_run
        state_info['snapshot_steps']=sorted(snapshot_steps) if snapshot_steps is not None else 'every real step'
        state_info['wall_diffusion']=args.wall_diffusion
        state_info['acoustic_damper']=args.acoustic_damper
        state_info['rk_treatment']=args.rk_treatment
        state_info['wall_velocity']=args.wall_velocity
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
        if args.precision=='float64':
            assert all(checks.values()), checks
        else:
            rounded={name:bool(np.allclose(got[name],value,rtol=2e-7,atol=1e-12)) for name,value in expected.items()}
            assert all(rounded.values()),rounded
            state_info['rounded_initial_tolerance']=rounded
            state_info['initial_rounding_max_abs']={name:float(np.max(np.abs(got[name]-value))) for name,value in expected.items()}
            write_json(args.out/'initial-audit.json',state_info)
        if args.coefficients=='mapped':
            tolerance=1e-12 if args.precision=='float64' else 1e-6
            assert np.isclose(state_info['local_beta'],beta_ref,rtol=tolerance)
            assert np.isclose(state_info['local_k2'],k2_ref,rtol=tolerance)

    def post(ctx,system,step):
        if args.performance:ctx.scratch['finish_performance']()
        if snapshot_steps is None or step+1 in snapshot_steps:snapshot(ctx,system,step+1)
        ctx.scratch['completed']=step+1

    def timestep(ctx,system):return dt[min(ctx.scratch.get('completed',0),len(dt)-1)]

    def diagnostics(ctx,system):
        st=system.state;f=st.kinds==0;v=st.velocities[f];p=st.pressures[f];update=ctx.scratch.get('lastStageUpdate')
        row={'maxVelocity':float(v.norm(dim=-1).max()),'kineticEnergy':float((.5*st.masses[f]*v.square().sum(-1)).sum()),
            'meanPressure':float(p.mean()),'minPressure':float(p.min()),'maxPressure':float(p.max()),
            'pseudoIterations':getattr(update,'pseudoIterations',0),'epsilonV':getattr(update,'epsilonV',None),
            'epsilonP':getattr(update,'epsilonP',None),'exitReason':getattr(update,'exitReason',None)}
        if args.long_run:
            from export_viewer import PROBES
            probe_values=[];probe_wet=[]
            for point in PROBES:
                q=(st.positions[f]-st.positions.new_tensor(point)).norm(dim=-1)/support
                weights=(1-q).clamp_min(0).pow(4)*(1+4*q)
                total=weights.sum();wet=bool(total>0)
                probe_wet.append(wet)
                probe_values.append(float((weights*p).sum()/total) if wet else 0.)
            row.update(pressureProbes=probe_values,pressureProbesWet=probe_wet)
        rows.append({'t':float(system.t),**row})
        if args.long_run:
            with (args.out/'progress.jsonl').open('a') as log:log.write(json.dumps(rows[-1])+'\n')
        else:write_json(args.out/'progress.json',rows)
        print('LOCAL STEP',json.dumps(rows[-1]),flush=True)
        return row

    case=Case('lobovskyExactReference','artificialCompressible',configureScheme=configure,buildSystem=build,initialConditions=initialize,
        timestep=timestep,postStep=post,diagnostics=diagnostics,
        setupPlot=lambda ctx,st:buildFieldPlotter(ctx,st,fields,figsize=(12,6)),
        updatePlot=lambda ctx,st,plot,step:refreshFieldPlotter(ctx,st,plot,fields,step=step))
    start=time.monotonic()
    result=run(case,precision=args.precision,device='cuda:0',nx=800,L=.6,n_h=4,kernel='Wendland2',periodic=False,
        integrationScheme='forwardEuler',calibrateNormalization=False,densityCorrection=False,
        nSteps=args.steps,tLimit=sum(dt),dt=dt[0],minDt=controls[7],maxDt=controls[8],
        plot=True,video=True,plotBackend='vispy',plotInterval=args.plot_interval,show=False,exportRoot=str(args.out/'recording'),
        cudaGraph=not args.eager,progress=True,quiet=False,store=False,velocityAlarmPlotInterval=1,velocityAlarmStopRatio=100/(2*9.81*.3)**.5,stallProgress=1e-3)
    if result.ctx.scratch.get('last_snapshot_step') != result.nSteps:
        snapshot(result.ctx,result.state,result.nSteps)
    if args.long_run:write_json(args.out/'progress.json',rows)
    write_json(args.out/'result.json',{'reference':str(args.reference.resolve()),'solver_model':args.solver_model,'mode':args.coefficients,'history':args.history,'wall_diffusion':args.wall_diffusion,'acoustic_damper':args.acoustic_damper,'rk_treatment':args.rk_treatment,'wall_velocity':args.wall_velocity,'shifting':args.shifting,'trajectory':result.trajectory,
        'nSteps':result.nSteps,'wall_seconds':time.monotonic()-start,'reported_wall_seconds':result.wallTime,
        'diverged':result.diverged,'stop_reason':result.stopReason,'exportPath':result.exportPath,
        'graph_disabled':getattr(result.ctx.schemeConfig,'_acsphGraphDisabled',None)})
    print('LOCAL COMPLETE',args.out,'steps',result.nSteps,'wall',time.monotonic()-start,flush=True)
    if result.diverged:raise SystemExit(1)

if __name__=='__main__':main()
