"""Prepared, shared initial states for the incompressible comparison profile.

All numerical choices are declared in the manifest. Fixture creation is done
once by a canonical native DFSPH preparation, independent of the tested method.
The default cohort is the four supported native projection variants.
"""
from dataclasses import replace, fields
from pathlib import Path
import copy
import hashlib
import json
import math
import time

METHODS = ('dfsph', 'dfsphPredictor', 'dfsphShift', 'iisph')
LABELS = dict(dfsph='DFSPH', dfsphPredictor='Predictor-first DFSPH',
              dfsphShift='VD+PS (native operators)', iisph='IISPH')
ORDERS = dict(dfsph='df_np_cd', dfsphPredictor='np_df_cd', dfsphShift='np_df_shift')
CLOSED = ('tgv', 'decayingVortices', 'shearDecay', 'boundedVortices')


def morris_c4_lattice_calibration(n_h=4.):
    """Long-wave Morris viscosity on the nominal-mass C4 square lattice.

    Evaluate the runtime kernel's radial derivative and Morris eta²=0.0025H².
    Apparent volume is m/rho=dx²/S for this unnormalized summation density.
    This is a kernel moment, not a fit to the subsequent decay trajectory.
    """
    import numpy as np
    h=float(n_h);n=int(math.ceil(h))+1
    x,y=np.meshgrid(np.arange(-n,n+1),np.arange(-n,n+1),indexing='ij')
    r=np.hypot(x,y);q=np.minimum(r/h,1.)
    w=np.where(r<h,9/(math.pi*h*h)*(1-q)**6*(1+6*q+35/3*q*q),0.)
    density_sum=float(w.sum())
    dw=np.where(r<h,9/(math.pi*h**3)*(-56/3)*q*(5*q+1)*(1-q)**5,0.)
    factor=float(-np.sum(dw*r*y*y/(r*r+.0025*h*h))/density_sum)
    return dict(factor=factor,latticeDensitySum=density_sum,
                kernel='Wendland4',n_h=h,eta2=.0025,
                convention='long-wave square-lattice moment; apparent volume dx²/S')


def corrected_prepared_plan(plan):
    """Explicit revised presets; old saved rows without overrides replay as before."""
    plan=copy.deepcopy(plan)
    for row in plan['rows']:
        corrections={}
        if row['test']=='shearDecay':
            corrections.update(viscosityTerm='morris1997',targetKinematicViscosity=.0185,
                               viscosityCalibration=morris_c4_lattice_calibration(row['spec']['n_h']))
        if row['test']=='sloshing':
            row['spec'].update(dt=.0002,maxDt=.0002)
            corrections['stepCeilingSeconds']=.0002
        if row['test']=='tgv' and row['scheme']=='dfsphShift':
            corrections['divergenceTolerancePercent']=.001
        if corrections:
            row['numerical_preset']=dict(revision='stability-v1',**corrections)
            row['notes']+='; stability-v1: '+json.dumps(corrections,sort_keys=True)
    plan.update(version=7,revision='stability-v1',
        correction_policy='Declared before replay: C4-calibrated Morris shear; sloshing 0.2ms ceiling; native VD+PS TGV DF0.001%. Reuse saved physical fixtures; retain failures without tuning.')
    return plan


def consistent_prepared_plan(plan):
    """Keep each free-surface case's declared viscosity instead of blanket nu=.01."""
    plan=corrected_prepared_plan(plan)
    for row in plan['rows']:
        if row['test'] not in ('marrone31','sloshing'):
            continue
        row.setdefault('numerical_preset',{}).update(
            revision='consistency-v1',viscosityPolicy='case')
        if row['test']=='marrone31' and row['params'].get('machTarget'):
            # The projection has no acoustic CFL, but artificial viscosity
            # still contains alpha*c0. Use the case's WCSPH reference scale.
            row['numerical_preset']['artificialViscositySoundSpeed']=(
                row['params']['referenceVelocity']/row['params']['machTarget'])
        row['notes']+='; consistency-v1: case viscosity; reflected sampled-wall viscosity references'
    plan.update(version=8,revision='consistency-v1',
        correction_policy='Case viscosity for dam break/sloshing; viscosity-only sampled free-slip references. One representative dam break must be analyzed before any matrix continuation.')
    return plan


def sampling_recovery_prepared_plan(plan):
    """Use the selected recovery rule in all three native DF orderings."""
    plan=consistent_prepared_plan(plan)
    for row in plan['rows']:
        if row['test'] in CLOSED:
            # Preparation is native DFSPH even when evolution uses IISPH
            # or another comparison method. It cannot depend on row order.
            row['preparation_sampling_gate']={'minDivergenceDensityRatio':.9}
        if row.get('solver')=='dfsph':
            row['df_sampling_gate']={'minDivergenceDensityRatio':.9,
                'scope':'clamped divergence only'}
            row['notes']+='; sampling-recovery-v1: DF active at wall-completed rho>=0.9rho0'
    plan.update(version=9,revision='sampling-recovery-v1',
        correction_policy='Shared native DF sampling recovery across all orderings and walls; case viscosity and prepared physical fixtures retained. IISPH density-only and signed periodic constraints unchanged.')
    return plan


def make_prepared_plan(make_plan, methods=METHODS, scale='full', revision='sampling-recovery-v1'):
    if not set(methods) <= set(METHODS):
        raise ValueError('Unknown prepared matrix method')
    plan = make_plan(('iisph',))
    templates = plan['rows']
    rows = []
    for template in templates:
        for method in methods:
            r = copy.deepcopy(template)
            r['scheme'] = method
            r['solver'] = 'dfsph' if method in ORDERS else method
            r['id'] = f"{r['test']}__{method}__{r['boundary']}"
            r['skip'] = None
            if r['test'] == 'taylorCouette':
                if r['boundary'] == 'particles':
                    r['skip'] = 'No prescribed spinning sampled-wall adapter.'
                else:
                    r['skip'] = 'Native static-wall projections do not support rotating analytic solids.'
            spec = r['spec']
            spec.update(kernel='Wendland4', n_h=4., dt=.001, maxDt=.001,
                        adaptiveDt=True, device='cuda', pipelineOutputs=False)
            if r['test']=='taylorCouette':
                # Its existing compact rotating-wall closure is C2-only.
                spec['kernel']='Wendland2'
            if scale == 'screen':
                spec['nx'] = min(spec['nx'], 64)
            r['params'].update(shifting=False)
            if r['test'] == 'hydrostatic':
                r['case'] = 'dambreak'
                spec['L'] = 1.
                r['params'] = dict(W=1., fillRatio=.5, fluidWidth=1.,
                    hydrostaticInit=True, gravityMagnitude=9.81,
                    referenceVelocity=math.sqrt(9.81*.5), pressureProbeHeights=[],
                    wallRepresentation=r['boundary'], wallBC='freeSlip', shifting='off')
            periodic = r['boundary'] == 'periodic'
            r['initialization'] = ('periodic-jitter-damped' if periodic else
                'closed-jitter-damped' if r['test'] == 'boundedVortices' else
                'calibrated-free-surface' if r['test'] != 'taylorCouette' else 'native-couette')
            r['density_source_policy'] = ('periodicMeanFree' if periodic and
                r['solver'] in ('dfsph','iisph') else 'absolute' if r['solver'] in
                ('dfsph','iisph') else 'legacy-native-policy')
            r['ordering'] = ORDERS.get(method)
            r['scheme_settings'] = {}
            spec['calibrateNormalization'] = False
            if r['test'] in CLOSED:
                spec['L'] = 2*math.pi if r['test'] == 'tgv' else 1. if r['test']=='shearDecay' else 2.
                spec['periodic'] = periodic
                r['params'].update(rho0=1., markerSize=3)
            elif r['test'] != 'taylorCouette':
                # Case geometry is calibrated once in the common fixture path;
                # avoid fitted mass/positions being applied twice by its hooks.
                r['params'].update(calibratedLattice=False, calibrateRestDensity=False)
            r['notes'] = (LABELS[method] + '; shared prepared fixture; density source: ' +
                r['density_source_policy'] + '; fixed declared tolerances; no outcome-driven tuning.')
            rows.append(r)
    plan.update(version=6, profile='prepared-incompressible', scale=scale,
        schemes=list(methods), labels=LABELS, rows=rows,
        initialization='Shared fixtures: nominal-mass jitter/damped relaxation in closed/periodic domains; explicit mean-free periodic source; one-time mass calibration and flat-wall fit for free surfaces.',
        preparation=dict(seed=20261025,jitterDx=.01,dampingGamma=20.,
            periodicRawSeconds=0.,periodicCoolSeconds=5.,closedMaxSeconds=2.,
            restSeconds=.25,rmsQuiet=1e-5,maxQuiet=1e-4,quietWindow=.25),
        qualification='Prepared fixtures are measured and shared even when duration-limited. Absolute-density periodic controls remain distinct from the mean-free evolution variant. Accuracy and cost are separate; cap-limited trajectories cannot be called converged.')
    if revision=='sampling-recovery-v1':return sampling_recovery_prepared_plan(plan)
    # Historical revisions predate the selected gate. Keep their replay
    # policy explicit even as the native configuration defaults evolve.
    for row in plan['rows']:
        if row.get('solver')=='dfsph':
            row['df_sampling_gate']={'minDivergenceDensityRatio':0.}
    if revision=='original':return plan
    if revision=='consistency-v1':return consistent_prepared_plan(plan)
    if revision!='stability-v1':raise ValueError('Unknown prepared matrix revision')
    return corrected_prepared_plan(plan)


def wrap(x, domain):
    import torch
    return torch.where(domain.periodic, torch.remainder(x-domain.min, domain.max-domain.min)+domain.min, x)


def bounded_jitter(x, dx, seed):
    import numpy as np
    import torch
    rng=np.random.default_rng(seed)
    a=rng.uniform(0,2*np.pi,len(x)); r=.01*dx*np.sqrt(rng.uniform(0,1,len(x)))
    return x+torch.as_tensor(np.stack((r*np.cos(a),r*np.sin(a)),1),device=x.device,dtype=x.dtype)


def reset_projection(state):
    import torch
    state.velocities=torch.zeros_like(state.velocities)
    for key in ('pressures','divergencePressures','densityPressureImpulse','divergencePressureImpulse'):
        if hasattr(state,key): setattr(state,key,torch.zeros_like(state.masses))


def tensor_hash(state):
    h=hashlib.sha256()
    for key in ('positions','velocities','masses','supports','kinds'):
        h.update(getattr(state,key).detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sample_field(x, test, length):
    import torch
    if test=='tgv':
        return torch.stack((-torch.sin(x[:,0])*torch.cos(x[:,1]),torch.cos(x[:,0])*torch.sin(x[:,1])),1)
    if test=='shearDecay':
        return torch.stack((.05*torch.sin(2*math.pi*x[:,1]/length),torch.zeros_like(x[:,0])),1)
    # One deterministic analytic streamfunction, evaluated at actual positions.
    # Its closed version has a smooth envelope with zero velocity on every wall.
    z=x/length
    psi=torch.zeros_like(z[:,0]); gx=psi.clone();gy=psi.clone()
    for m,n,a,p in ((1,2,.09,.3),(2,-1,.07,1.4),(3,2,.035,2.2),(2,3,.025,.9)):
        phase=2*math.pi*(m*z[:,0]+n*z[:,1])+p
        psi+=a*torch.cos(phase);gx-=a*2*math.pi*m/length*torch.sin(phase);gy-=a*2*math.pi*n/length*torch.sin(phase)
    if test=='boundedVortices':
        ex=torch.sin(math.pi*z[:,0])**4; ey=torch.sin(math.pi*z[:,1])**4
        dex=4*math.pi/length*torch.sin(math.pi*z[:,0])**3*torch.cos(math.pi*z[:,0])
        dey=4*math.pi/length*torch.sin(math.pi*z[:,1])**3*torch.cos(math.pi*z[:,1])
        return .2*torch.stack((ex*(dey*psi+ey*gy),-ey*(dex*psi+ex*gx)),1)
    return .2*torch.stack((gy,-gx),1)


def prepared_case(case, root, row):
    """Wrap a normal runner case; every scheme consumes identical physical arrays."""
    import numpy as np
    import torch
    from warpSPH.runner import Case
    from warpSPH.caseUtils.projection import initializeProjectionDensity
    from run_fluidMatrix import write_json
    closed=row['test'] in CLOSED
    periodic=row['boundary']=='periodic'
    native=row.get('solver',row['scheme']) in ('dfsph','iisph')
    original_config,original_initial=case.configureScheme,case.initialConditions
    fixture_dir=root/'fixtures'/f"{row['test']}__{row['boundary']}"
    fixture_dir.mkdir(parents=True,exist_ok=True)
    fixture_file=fixture_dir/'state.pt'
    out=root/row['id']

    def configure(ctx):
        if not closed and original_config: original_config(ctx)
        cfg,sc=ctx.config,ctx.schemeConfig
        cfg.calibrateNormalization=False
        ctx.spec=replace(ctx.spec,calibrateNormalization=False)
        sc.iisph.boundaryRepresentation='analytic' if row['boundary']=='analytic' else 'particles'
        sc.iisph.boundaryPressure='self' if closed else 'contactShepard'
        sc.iisph.contactGravity=not closed
        sc.iisph.relaxationFactor=.25
        sc.iisph.maxIterations=100;sc.iisph.maxErrorPercent=.01
        sc.iisph.clampPressure=not periodic
        sc.iisph.densitySourcePolicy=row['density_source_policy'] if native else 'absolute'
        sc.dfsph.ordering=row.get('ordering') or 'df_np_cd'
        # Saved pre-gate manifests also lack this field; preserve their
        # original ungated policy rather than inheriting the new default.
        sc.dfsph.minDivergenceDensityRatio=row.get('df_sampling_gate',{}).get('minDivergenceDensityRatio',0.)
        sc.dfsph.validate()
        ctx.scratch['matrixPreparationDFGate']=row.get('preparation_sampling_gate',{}).get('minDivergenceDensityRatio',0.)
        for p in (sc.dfsph.density,sc.dfsph.divergence):
            p.relaxationFactor=.25;p.maxIterations=100;p.clampPressure=not periodic
        sc.dfsph.density.maxErrorPercent=.01
        sc.dfsph.divergence.maxErrorPercent=.1
        sc.shiftProperties.active=False
        preset=row.get('numerical_preset',{})
        if preset.get('viscosityPolicy')!='case':
            sc.diffusionParams.inviscid=False;sc.diffusionParams.inviscidAlpha=0.
            sc.diffusionParams.approachOnly=False
            sc.diffusionParams.viscidNu=(.0185 if row['test']=='shearDecay' else .01)
            if row['test']=='taylorCouette':sc.diffusionParams.viscidNu=.0185
        elif sc.diffusionParams.inviscid and 'artificialViscositySoundSpeed' in preset:
            sc.fluid.fixedSoundSpeed=preset['artificialViscositySoundSpeed']
        if 'viscosityTerm' in preset:
            from warpSPH.enumTypes import ViscosityTerm
            sc.diffusionParams.viscousTerm=ViscosityTerm[preset['viscosityTerm']]
            calibration=preset['viscosityCalibration']
            if cfg.kernel.name!=calibration['kernel'] or ctx.spec.n_h!=calibration['n_h']:
                raise ValueError('Viscosity calibration does not match the actual kernel/support')
            sc.diffusionParams.viscidNu=preset['targetKinematicViscosity']/calibration['factor']
            ctx.scratch['matrixViscosityCalibration']=calibration
        if 'divergenceTolerancePercent' in preset:
            sc.dfsph.divergence.maxErrorPercent=preset['divergenceTolerancePercent']
        sc.omniViscosity=True;sc.bandViscosity=True
        sc.xsphCoefficient=0.;sc.xsphBoundaryCoefficient=0.;sc.boundaryFriction=0.
        if closed:
            cfg.dx=ctx.spec.L/ctx.spec.nx
            cfg.domain.min[:]=0.;cfg.domain.max[:]=ctx.spec.L;cfg.domain.periodic[:]=periodic
            if not periodic:cfg.domain.min[:]=-6*cfg.dx;cfg.domain.max[:]=ctx.spec.L+6*cfg.dx
            sc.fluid.restDensity=1.;sc.fluid.fixedSoundSpeed=10.
            sc.gravityConfig.active=False;sc.surfaceDetectionConfig.active=False
        ctx.velocityScale=1.;ctx.velocityScaleSource='Declared matrix flow scale'

    def build(ctx):
        if not closed:
            system=case.buildSystem(ctx)
            if native and row['boundary']=='particles' and getattr(ctx.schemeConfig,'boundaryProvider',None) is None:
                from warpSPH.caseUtils.weaklyCompressible import analyticTankBody
                from warpSPH.caseUtils.attachedBoundary import obstaclePolygonVertices
                from warpSPH.configurations.region import ParticleRegion,RegionType,BCType
                from warpSPH.boundary import buildBoundaryProvider
                from warpSPH.rigidBody.build import buildAnalyticRigidBody
                vertices=obstaclePolygonVertices(ctx.scratch['obstacle']) if ctx.param('obstacleActive',False) else None
                body=analyticTankBody(ctx.scratch['interiorDomain'],obstacleVertices=vertices)
                wall_kind=(BCType[ctx.param('wallBC','freeSlip')]
                    if row.get('numerical_preset',{}).get('viscosityPolicy')=='case' else BCType.zeros)
                wall=ParticleRegion(sdf=lambda x:torch.ones(len(x),device=x.device),
                    type=RegionType.Boundary,particles=None,kind=wall_kind,representation=body)
                rigid=buildAnalyticRigidBody(wall,0,system.state)
                provider=buildBoundaryProvider([wall],ctx.device);provider.rigidBodies=[rigid]
                ctx.schemeConfig.boundaryProvider=provider;ctx.schemeConfig.rigidBodies=[rigid]
                ctx.config.regions=list(ctx.config.regions)+[wall]
                ctx.schemeConfig.regions=ctx.config.regions
            return system
        from warpSPH.geometry.types import ParticleSet
        from warpSPH.configurations.region import ParticleRegion,RegionType,BCType
        from warpSPH.initializers.weaklyCompressible import initializeState
        from warpSPH.boundary import buildBoundaryProvider
        from warpSPH.rigidBody.build import buildAnalyticRigidBody
        from warpSPHBoundaries.scene import Body
        from warpSPHBoundaries.scene.scene import BoxRep
        nx=ctx.spec.nx;dx=ctx.spec.L/nx
        axis=(torch.arange(nx,device=ctx.device,dtype=ctx.dtype)+.5)*dx
        x=torch.stack(torch.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
        def region(points,kind):
            n=len(points)
            particles=ParticleSet(points,torch.full((n,),4*dx,device=ctx.device,dtype=ctx.dtype),
                torch.full((n,),dx*dx,device=ctx.device,dtype=ctx.dtype),torch.ones(n,device=ctx.device,dtype=ctx.dtype))
            return ParticleRegion(sdf=lambda x:torch.ones(len(x),device=x.device),type=kind,particles=particles,initialConditions={},kind=BCType.zeros)
        regs=[region(x,RegionType.Fluid)]
        if row['boundary']=='particles':
            a=(torch.arange(-5,nx+5,device=ctx.device,dtype=ctx.dtype)+.5)*dx
            b=torch.stack(torch.meshgrid(a,a,indexing='ij'),-1).reshape(-1,2)
            b=b[~((b>=0)&(b<=ctx.spec.L)).all(1)]
            regs.append(region(b,RegionType.Boundary))
            from warpSPHCore import DomainDescription
            from warpSPH.regions import sampleDomainSDF
            interior=DomainDescription(torch.zeros(2,device=ctx.device,dtype=ctx.dtype),
                torch.full((2,),ctx.spec.L,device=ctx.device,dtype=ctx.dtype),
                torch.zeros(2,device=ctx.device,dtype=torch.bool),2)
            regs[-1].sdf=lambda points:sampleDomainSDF(points,interior,invert=False)
            regs[-1].kind=BCType.freeSlip
        st=initializeState(regs,ctx.config,ctx.schemeConfig,ctx.SimulationState,verbose=False)
        if row['boundary']=='particles':
            # Keep the normal sampled-wall auxiliary representation for the
            # legacy solvers too. Ghosts carry no physical fluid mass.
            from warpSPH.rigidBody.ghostParticles import addBoundaryGhostParticles
            st=addBoundaryGhostParticles(regs,st)
        if not periodic and (native or row['boundary']=='analytic'):
            length=ctx.spec.L
            body=Body(bodyId=0,reps=[BoxRep((0.,0.),(length,length),solid='outside')],center=(0.,0.))
            wall=ParticleRegion(sdf=lambda x:torch.minimum(x,length-x).min(1).values,
                type=RegionType.Boundary,particles=None,kind=BCType.freeSlip,representation=body)
            rigid=buildAnalyticRigidBody(wall,0,st)
            regs.append(wall)
            provider=buildBoundaryProvider([wall],ctx.device);provider.rigidBodies=[rigid]
            ctx.schemeConfig.boundaryProvider=provider;ctx.schemeConfig.rigidBodies=[rigid]
        else:
            ctx.schemeConfig.boundaryProvider=None;ctx.schemeConfig.rigidBodies=[]
            if row['boundary']=='particles':
                from warpSPH.rigidBody.build import buildRigidBody
                ctx.schemeConfig.rigidBodies=[buildRigidBody(st,regs,0)]
        ctx.config.rigidBodies=ctx.schemeConfig.rigidBodies
        ctx.config.regions=ctx.schemeConfig.regions=regs
        return ctx.SimulationSystem(state=st,domain=ctx.config.domain)

    def initialize(ctx,system):
        sc,cfg=ctx.schemeConfig,ctx.config
        if not closed and original_initial: original_initial(ctx,system)
        st=system.state
        created_here=not fixture_file.exists()
        if created_here:
            started=time.perf_counter()
            if closed:
                report=relax_fixture(ctx,system,fixture_dir,periodic)
            else:
                # Uniform bulk mass calibration is applied once, identically to
                # fluid and sampled solids. Analytic flat walls fit this lattice.
                from warpSPHCore.util.latticeDensity import latticeDensityFactor
                factor=latticeDensityFactor(cfg.kernel,4.,2)
                st.masses=st.masses*factor;sc.analyticWallMass=factor
                if row['boundary']=='analytic' and row['test']!='taylorCouette':
                    from warpSPH.boundary.calibration import calibrateAnalyticLattice
                    cal=calibrateAnalyticLattice(ctx,system,ctx.scratch['interiorDomain'])
                    # calibrateAnalyticLattice supplies its own fluid mass once.
                    report=dict(massFactor=factor,flatWallCalibration=cal)
                else:report=dict(massFactor=factor)
            report['wallSeconds']=time.perf_counter()-started
            reset_projection(st)
            if closed:
                f=st.kinds==0;st.velocities[f]=sample_field(st.positions[f],row['test'],ctx.spec.L)
            elif ctx.param('hydrostaticInit',False):
                from warpSPH.caseUtils.projection import initializeHydrostaticPressure
                initializeHydrostaticPressure(ctx,system)
            payload={k:v.detach().cpu().clone() for k,v in vars(st).items() if isinstance(v,torch.Tensor)}
            torch.save(payload,fixture_file)
            write_json(fixture_dir/'metadata.json',dict(report=report,
                analyticWallMass=float(getattr(sc,'analyticWallMass',1.)),
                fingerprint=tensor_hash(st),canonicalPreparation='native DFSPH',
                fluidVolume=float((st.masses[st.kinds==0]/sc.fluid.restDensity).double().sum()),
                domainVolume=ctx.spec.L**2 if closed else None))
        meta=json.loads((fixture_dir/'metadata.json').read_text())
        if not closed and row['boundary']=='analytic' and row['test']!='taylorCouette':
            # Recreate the identical deterministic wall fit in this process.
            from warpSPH.boundary.calibration import calibrateAnalyticLattice
            calibrateAnalyticLattice(ctx,system,ctx.scratch['interiorDomain'])
        data=torch.load(fixture_file,map_location=ctx.device,weights_only=True)
        for k,v in data.items():
            if hasattr(st,k):setattr(st,k,v.clone())
        sc.analyticWallMass=meta['analyticWallMass'];cfg.calibrateNormalization=False
        system.adjacency=None
        initializeProjectionDensity(ctx,system)
        if closed:
            volume=float((st.masses[st.kinds==0]/sc.fluid.restDensity).double().sum())
            if abs(volume/ctx.spec.L**2-1)>2e-6:raise AssertionError('Nominal fluid volume does not fill the declared domain')
        actual=tensor_hash(st)
        if actual!=meta['fingerprint']:raise AssertionError('Shared physical fixture changed during density initialization')
        ctx.scratch['matrixFixture']=meta
        ctx.scratch['matrixPreparationSecondsThisRun']=meta['report']['wallSeconds'] if created_here else 0.
        ctx.scratch['matrixInitialMasses']=st.masses.clone()
        write_json(out/'fixture.json',dict(path=str(fixture_file),sha256=hashlib.sha256(fixture_file.read_bytes()).hexdigest(),**meta))

    if closed:
        from warpSPH.cases.common.plotting import Field,particlePlot
        setup,update=particlePlot([Field('velocities','Speed [m/s]',mapping='L2Norm'),Field('pressures','Stored pressure')])
        return Case('prepared_'+row['test'],row.get('solver',row['scheme']),build,
            configureScheme=configure,initialConditions=initialize,
            diagnostics=lambda ctx,system:{'maxVelocity':float(system.state.velocities[system.state.kinds==0].norm(dim=1).max())},
            setupPlot=setup,updatePlot=update,
            timestep=lambda ctx,system:min(.001,max(0.,row['spec']['tLimit']-float(system.t))))
    return replace(case,buildSystem=build,configureScheme=configure,initialConditions=initialize)


def relax_fixture(ctx,system,directory,periodic):
    """Canonical preparation; keep mass/walls fixed and record a fresh rest hold."""
    import numpy as np
    import torch
    from warpSPH.systems.incompressible import DFSPHState,DFSPHSystem
    from warpSPH.modules.iisph import initializeIISPHBoundaries
    from warpSPH.modules.dfsph.composition import advanceDFSPH
    st=system.state;cfg=ctx.config;sc=copy.deepcopy(ctx.schemeConfig)
    sc.dfsph.minDivergenceDensityRatio=ctx.scratch['matrixPreparationDFGate']
    data={f.name:getattr(st,f.name) for f in fields(DFSPHState) if f.init and hasattr(st,f.name)}
    pstate=DFSPHState(**copy.deepcopy(data));f=pstate.kinds==0
    pstate.positions[f]=bounded_jitter(pstate.positions[f],cfg.dx,20261025)
    pstate.positions=wrap(pstate.positions,cfg.domain)
    mass=pstate.masses.clone();wall=pstate.positions[~f].clone()
    reset_projection(pstate)
    sc.gravityConfig.active=False;sc.diffusionParams.viscidNu=0.;sc.diffusionParams.inviscidAlpha=0.
    sc.iisph.boundaryPressure='self';sc.iisph.contactGravity=False
    sc.iisph.clampPressure=not periodic;sc.iisph.densitySourcePolicy='absolute'
    for pol in (sc.dfsph.density,sc.dfsph.divergence):pol.clampPressure=not periodic
    sim=DFSPHSystem(state=pstate,domain=cfg.domain)
    sim.boundarySupport,sim.analyticBoundaryProvider,sim.contactBoundaryProvider=initializeIISPHBoundaries(pstate,sc)
    sim._projectedStepActive=True
    stages=[('relax',5000,'periodicMeanFree')] if periodic else [('relax',2000,'absolute')]
    rows=[];snapshots=[];clock=0.;settled=False;dt=.001
    with (directory/'preparation.jsonl').open('w',buffering=1) as log:
        for name,count,source in stages+[('fresh-rest',250,'periodicMeanFree' if periodic else 'absolute')]:
            reset_projection(pstate);sc.iisph.densitySourcePolicy=source
            for step in range(count):
                before=pstate.velocities.clone()
                if name!='fresh-rest':pstate.velocities[f]*=math.exp(-20*dt)
                result=advanceDFSPH(sim,dt,cfg,sc,'df_np_cd')
                clock+=dt;sim.t=clock
                rms=float(pstate.velocities[f].square().sum(1).mean().sqrt());vmax=float(pstate.velocities[f].norm(dim=1).max())
                r=dict(t=clock,phase=name,step=step+1,rmsSpeed=rms,maxSpeed=vmax,
                    meanDensity=float(pstate.densities[f].mean()/sc.fluid.restDensity),
                    densityStd=float(pstate.densities[f].std()/sc.fluid.restDensity),
                    DFCap=result[0].divergenceResult.termination=='budget',
                    CDCap=result[0].densityResult.termination=='budget')
                rows.append(r);log.write(json.dumps(r)+'\n')
                if step%100==0:print('PREPARATION',name,step+1,count,rms,vmax,flush=True)
                if step%25==0:snapshots.append(dict(t=clock,positions=pstate.positions[f].cpu().numpy().copy()))
                if not torch.isfinite(pstate.positions).all() or vmax>20:raise RuntimeError('Canonical preparation exceeded finite/velocity gate')
                if name!='fresh-rest' and step>=499:
                    w=rows[-250:]
                    settled=max(z['rmsSpeed'] for z in w)<1e-5 and max(z['maxSpeed'] for z in w)<1e-4 and not any(z['CDCap'] or z['DFCap'] for z in w)
                    if settled:break
            if name!='fresh-rest':prepared=pstate.positions.clone()
        assert torch.equal(pstate.masses,mass) and torch.equal(pstate.positions[~f],wall)
    # The fresh rest is a validation, never another hidden preparation stage.
    st.positions=prepared
    reset_projection(st)
    np.savez_compressed(directory/'preparation_snapshots.npz',t=np.array([z['t'] for z in snapshots]),positions=np.stack([z['positions'] for z in snapshots]))
    hold=[z for z in rows if z['phase']=='fresh-rest']
    return dict(settled=settled,preparationSteps=len(rows)-250,restSteps=250,
        preparationDivergenceDensityRatio=sc.dfsph.minDivergenceDensityRatio,
        restFinal=hold[-1],restMaxSpeed=max(z['maxSpeed'] for z in hold),
        restCaps=sum(z['CDCap'] or z['DFCap'] for z in hold),
        massUnchanged=True,wallsUnchanged=True,jitterDx=.01,dampingGamma=20.,
        stages=stages,sourceVariant='periodicMeanFree' if periodic else 'absolute',
        rawDensityRetained=True)




def seed_projection_pressure(state, ordering, dt):
    """Encode a shared physical initial pressure in the selected stage's units.

    Coupled routes seed density p*dt²; shift/resample seeds physical divergence
    p*dt and leaves the auxiliary positional multiplier zero. No prior
    preparation history is reused. The caller supplies the actual first dt.
    """
    import torch
    pressure=state.pressures
    zero=torch.zeros_like(pressure)
    if ordering=='np_df_shift':
        state.densityPressureImpulse=zero
        state.divergencePressureImpulse=pressure*dt
    else:
        state.densityPressureImpulse=pressure*(dt*dt)
        state.divergencePressureImpulse=zero
