"""Execute bundled BlobTest-2d.py with explicit API/solver-selection repairs.

No Warp input is injected. The example owns sampling, startup H relaxation,
hydro, integrator, adaptive stepping and duration. Passive hooks export its
state, stream to the daemon and guard finite/runaway behavior. Upstream source
is never edited. Run 'render' in the Warp environment after 'run' in Spheral.
"""
import argparse as _blob_argparse
import ast as _blob_ast
import hashlib as _blob_hashlib
import json as _blob_json
import sys as _blob_sys
import time as _blob_time
from pathlib import Path as _blob_Path
from types import SimpleNamespace as _blob_NS
import numpy as _blob_np


def _blob_paper_counts():
    if not _blob_args.paper:return None
    import ctypes
    api=ctypes.CDLL(None);api.warpsph_paper_policy_calls.restype=ctypes.c_ulonglong
    return {k:int(api.warpsph_paper_policy_calls(i)) for i,k in enumerate(('volume','density','finalEnergy','rawPredictor','rhsEntropy'))}


def _blob_save(name,data):
    (_blob_output/name).write_text(_blob_json.dumps(data,indent=2,allow_nan=False)+'\n')


def _blob_arrays(g):
    result={k:[] for k in ('positions','velocities','masses','densities','internalEnergies','pressures','supports','materials','UIDs','kinds','H')}
    offset=0
    for material,n in enumerate(g['nodeSet']):
        N=n.numInternalNodes;dim=2
        x=n.positions().internalValues();v=n.velocity().internalValues();H=n.Hfield().internalValues()
        rho=_blob_np.array(n.massDensity().internalValues());u=_blob_np.array(n.specificThermalEnergy().internalValues())
        values=dict(positions=_blob_np.array([[p[d] for d in range(dim)] for p in x]),
            velocities=_blob_np.array([[p[d] for d in range(dim)] for p in v]),
            H=_blob_np.array([[[h.xx,h.xy],[h.yx,h.yy]] for h in H]),
            masses=_blob_np.array(n.mass().internalValues()),densities=rho,internalEnergies=u,
            pressures=(g['gamma']-1)*rho*u,supports=_blob_np.array([g['WT'].kernelExtent/h.xx for h in H]),
            materials=_blob_np.full(N,material,dtype=_blob_np.int32),UIDs=_blob_np.arange(offset,offset+N,dtype=_blob_np.int64),
            kinds=_blob_np.zeros(N,dtype=_blob_np.int32))
        offset+=N
        for k,v in values.items():result[k].append(v)
    return {k:_blob_np.concatenate(v) for k,v in result.items()}


def _blob_pre_controller(g):
    global _blob_g,_blob_publisher
    from warpsph_remote.publisher import SimulationPublisher
    assert g['crksph'] and 'CRKSPH' in type(g['hydro']).__name__,type(g['hydro']).__name__
    _blob_g=g;_blob_np.savez_compressed(_blob_output/'generatedState.npz',**_blob_arrays(g))
    if _blob_args.paper:
        import ctypes
        assert ctypes.CDLL(None).warpsph_paper_policy_version()==1
        ctypes.CDLL(None).warpsph_paper_fresh_rk2(int(_blob_args.integrator=='fresh'))
        ctypes.CDLL(None).warpsph_paper_dudt_partition(int(_blob_args.partition=='dudt'))
        if hasattr(ctypes.CDLL(None),'warpsph_paper_averaged_heating'):
            ctypes.CDLL(None).warpsph_paper_averaged_heating(int(_blob_args.heating=='averaged'))
        elif _blob_args.heating!='one-sided':
            raise RuntimeError('Selected paper overlay does not implement averaged heating')
        if hasattr(ctypes.CDLL(None),'warpsph_paper_av_gradient'):
            ctypes.CDLL(None).warpsph_paper_av_gradient(('lagged','sph','crk').index(_blob_args.av_gradient))
        elif _blob_args.av_gradient=='crk':
            raise RuntimeError('Selected paper overlay does not implement fresh CRK AV gradients')
    g['hydro'].Q.rigorousVelocityGradient=(_blob_args.av_gradient=='sph')
    if _blob_args.paper:
        assert g['WT'].kernelExtent==4 and g['nPerh']==1 and g['order']==7
        assert g['hydro'].Q.Cl==2 and g['hydro'].Q.Cq==1
        import os
        library=_blob_Path(os.environ['LD_PRELOAD']).resolve()
        _blob_save('paperOverlayLoaded.json',dict(calls=_blob_paper_counts(),library=str(library),librarySHA256=_blob_hashlib.sha256(library.read_bytes()).hexdigest(),loadedMaps=[line.strip() for line in _blob_Path('/proc/self/maps').read_text().splitlines() if 'PaperPolicies' in line]))
    _blob_publisher=SimulationPublisher(('Paper policies' if _blob_args.paper else 'Bundled')+' Spheral CRKSPH blob n'+str(g['nx1'])+(' fresh RK2 replay' if _blob_args.integrator=='fresh' else '')+(' dudt control' if _blob_args.partition=='dudt' else '')+' AV '+_blob_args.av_gradient+' heating '+_blob_args.heating,measurements=True)
    _blob_save('registration.json',dict(remoteRunID=_blob_publisher.id))
    print('Remote run registered:',_blob_publisher.id,flush=True)


def _blob_attach(control,g):
    global _blob_control,_blob_file,_blob_ctx,_blob_measure,_blob_start,_blob_last_print,_blob_progress
    import h5py
    from study_blobPartition import Measures
    _blob_control=control;initial=_blob_arrays(g)
    if _blob_args.time_grid:
        global _blob_grid
        history=_blob_json.loads(_blob_args.time_grid.read_text())
        _blob_grid=_blob_np.diff([r['time'] for r in history])
        assert (_blob_grid>0).all() and abs(_blob_grid.sum()-g['goalTime'])<1e-10
        g['dt']=float(_blob_grid[0]);g['integrator'].dtMin=g['integrator'].dtMax=g['integrator'].lastDt=g['dt']
    _blob_np.savez_compressed(_blob_output/'initializedState.npz',**initial)
    settings=dict(case='blob',gammaVariant='bundled',nx=g['nx1'],dim=2,gamma=[g['gamma']]*2,
        end=g['goalTime'],tKH=g['tKH'],dt=g['dt'],kernel='BSpline7' if _blob_args.paper else 'QuinticSpline',n_h=g['WT'].kernelExtent*g['nPerh'],
        physicalKernelOrder=g['order'],paperPolicies=_blob_args.paper,
        domainAxes=[[g['xb0'],g['xb1']],[g['yb0'],g['yb1']]],support='adaptive',
        volumePolicy='kernelSum' if _blob_args.paper else 'massOverDensity',energyPartition=('spheral' if _blob_args.partition=='dudt' else 'entropy') if _blob_args.paper else 'native',energyPredictor='raw',oneSidedWork=_blob_args.heating=='one-sided',
        frameEvery=_blob_args.frame_every,source='Paper formulas/configuration with isolated compiled overlay.' if _blob_args.paper else 'Bundled driver with documented API/dispatch repairs and passive recording hooks.')
    _blob_save('settings.json',settings)
    runtime=dict(source=str(_blob_source),sourceSHA256=_blob_manifest['sourceSHA256'],hydroClass=type(g['hydro']).__name__,
        integrator=type(g['integrator']).__name__,lastDt=g['integrator'].lastDt,dtMin=g['integrator'].dtMin,
        dtMax=g['integrator'].dtMax,dtGrowth=g['integrator'].dtGrowth,cfl=g['hydro'].cfl,
        densityUpdate=str(g['hydro'].densityUpdate),compatibleEnergy=bool(g['hydro'].compatibleEnergyEvolution),
        viscosityClass=type(g['hydro'].Q).__name__,viscosity={k:float(getattr(g['hydro'].Q,k)) for k in ('Cl','Cq','etaCritFrac','etaFoldFrac')},
        kernelTableEntries=1000,initializeH=True,cullGhostNodes=g['integrator'].cullGhostNodes,
        particles=len(initial['masses']),goalTime=g['goalTime'],tKH=g['tKH'],paperPolicies=_blob_args.paper,
        kernelOrder=g['order'],kernelExtent=g['WT'].kernelExtent,nPerh=g['nPerh'],policyCalls=_blob_paper_counts())
    runtime['timeGridReplay']=str(_blob_args.time_grid) if _blob_args.time_grid else None
    runtime['avGradient']=_blob_args.av_gradient
    runtime['heating']=_blob_args.heating
    runtime['rigorousVelocityGradient']=bool(g['hydro'].Q.rigorousVelocityGradient)
    _blob_save('nativeConfiguration.json',runtime)
    _blob_measure=Measures(settings,initial)
    _blob_file=h5py.File(_blob_output/'trajectory.h5','w')
    _blob_ctx=_blob_NS(spec=_blob_NS(store=True,storeMode='trajectory',tLimit=g['goalTime']),
        scratch={'trajectoryFile':_blob_file},exportPath=str(_blob_output.resolve()),scheme='Spheral CRKSPH bundled example',
        config=_blob_NS(dt=g['dt'],domain=_blob_NS(min=_blob_np.array([g['xb0'],g['yb0']]),max=_blob_np.array([g['xb1'],g['yb1']]),periodic=_blob_np.array([True,True])),kernel=settings['kernel'],integrationScheme=runtime['integrator']))
    _blob_start=_blob_time.perf_counter();_blob_last_print=0.;_blob_progress=(_blob_start,0.)
    _blob_record(0,float(control.time()),g['dt']);control.appendPeriodicWork(_blob_record,1)


def _blob_record(cycle,t,dt):
    global _blob_latest,_blob_last_print,_blob_progress
    # Controller.advance forces periodic work once more at the final cycle.
    if _blob_history and _blob_history[-1].get('cycle',len(_blob_history)-1)==cycle and abs(_blob_history[-1]['time']-t)<1e-12:return
    a=_blob_arrays(_blob_g);row=_blob_measure(a,t);_blob_history.append(row);_blob_latest=a
    row['cycle']=int(cycle)
    if _blob_args.paper:
        row['originalCloudMassFraction']=row['cloudMassFraction']
        mask=(a['densities']>6.4)&(a['internalEnergies']<1.35)
        row['cloudMassFraction']=float(a['masses'][mask].sum()/a['masses'][a['materials']==1].sum())
    if _blob_args.paper and cycle>0:
        assert all(v>0 for v in _blob_paper_counts().values()),_blob_paper_counts()
        import ctypes
        api=ctypes.CDLL(None)
        if not (_blob_output/'firstDensityAudit.npz').exists():
            api.warpsph_paper_audit_rows.restype=ctypes.c_ulonglong
            api.warpsph_paper_audit_data.restype=ctypes.POINTER(ctypes.c_double)
            N=api.warpsph_paper_audit_rows()
            if N:_blob_np.savez_compressed(_blob_output/'firstDensityAudit.npz',rows=_blob_np.ctypeslib.as_array(api.warpsph_paper_audit_data(),shape=(N,15)).copy())
        if not (_blob_output/'firstRHSAudit.npz').exists():
            api.warpsph_paper_rhs_audit_rows.restype=ctypes.c_ulonglong
            api.warpsph_paper_rhs_audit_data.restype=ctypes.POINTER(ctypes.c_double)
            N=api.warpsph_paper_rhs_audit_rows()
            if N:_blob_np.savez_compressed(_blob_output/'firstRHSAudit.npz',rows=_blob_np.ctypeslib.as_array(api.warpsph_paper_rhs_audit_data(),shape=(N,15)).copy())
    row['negativeEnergyParticles']=int((a['internalEnergies']<0).sum())
    if row['minEnergy']<0 and not (_blob_output/'firstNegativeEnergy.json').exists():
        index=int(a['internalEnergies'].argmin())
        _blob_save('firstNegativeEnergy.json',dict(time=t,cycle=cycle,UID=int(a['UIDs'][index]),energy=float(a['internalEnergies'][index]),
            position=a['positions'][index].tolist(),action='Diagnostic only; native evolution is unchanged.' if not _blob_args.strict_positive else 'Strict positivity stop.'))
        print('BLOB NEGATIVE ENERGY:',cycle,t,row['minEnergy'],flush=True)
    if not all(_blob_np.isfinite(v).all() for v in a.values()) or (_blob_args.strict_positive and row['minEnergy']<=0) or row['minDensity']<=0 or row['maxVelocity']>100:
        raise RuntimeError('Bundled Spheral example finite-state/velocity watchdog')
    if cycle<10 or cycle%_blob_args.frame_every==0 or t>=_blob_g['goalTime']-1e-10:
        group=_blob_file.require_group('states').create_group(f'frame_{cycle:07d}');group.attrs['time']=t
        for k,v in a.items():group.create_dataset(k,data=v)
        _blob_file.flush()
    now=_blob_time.perf_counter()
    if now-_blob_progress[0]>=60:
        if t-_blob_progress[1]<1e-3:raise RuntimeError('Spheral blob progress watchdog: simulated time advanced <1e-3 in 60 wall seconds')
        _blob_progress=(now,t)
    if cycle<10 or now-_blob_last_print>=1:
        print(f'BLOB cycle={cycle} t={t:.9g} tKH={row["tKH"]:.6g} dt={dt:.6g} minU={row["minEnergy"]:.6g} maxV={row["maxVelocity"]:.5g} dE={row["relativeEnergyDrift"]:.3g} cloud={row["cloudMassFraction"]:.5g} wall={now-_blob_start:.2f}s',flush=True)
        _blob_last_print=now
    _blob_ctx.config.dt=dt;_blob_publisher.post_step(_blob_ctx,_blob_NS(state=_blob_NS(**a),t=t),cycle)
    if _blob_args.time_grid and cycle<len(_blob_grid):
        nextDt=float(_blob_grid[cycle]);_blob_g['integrator'].dtMin=_blob_g['integrator'].dtMax=_blob_g['integrator'].lastDt=nextDt


class _blob_Compatibility(_blob_ast.NodeTransformer):
    def visit_Expr(self,node):
        if isinstance(node.value,_blob_ast.Call) and isinstance(node.value.func,_blob_ast.Attribute):
            f=node.value.func
            if isinstance(f.value,_blob_ast.Name) and f.value.id=='control' and f.attr in ('updateViz','dropRestartFile'):
                return None # output is handled by HDF5/vispy and state exports
        return self.generic_visit(node)
    def visit_Module(self,node):
        self.generic_visit(node)
        # The source has a second independent 'if fsisph', whose else replaces
        # the earlier CRK/PSPH selection with SPH. Honor the declared solver flag.
        body=[];index=0
        while index<len(node.body):
            item=node.body[index]
            if isinstance(item,_blob_ast.If) and isinstance(item.test,_blob_ast.Name) and item.test.id=='crksph':
                following=node.body[index+1]
                assert isinstance(following,_blob_ast.If) and isinstance(following.test,_blob_ast.Name) and following.test.id=='fsisph'
                last=item
                while len(last.orelse)==1 and isinstance(last.orelse[0],_blob_ast.If):last=last.orelse[0]
                assert not last.orelse
                last.orelse=[following];index+=1
            body.append(item);index+=1
        node.body=body
        return node
    def visit_ImportFrom(self,node):
        if node.module=='SpheralGnuPlotUtilities':return None # no symbols used by this driver
        return node
    def visit_Name(self,node):
        if node.id=='ASPH':node.id='asph'
        return node
    def visit_Call(self,node):
        self.generic_visit(node)
        if globals().get('_blob_args') is not None and getattr(_blob_args,'integrator','cheap')=='fresh' and isinstance(node.func,_blob_ast.Name) and node.func.id=='IntegratorConstructor':
            node.func.id='SynchronousRK2Integrator'
        if isinstance(node.func,_blob_ast.Name) and node.func.id=='CRKSPH':
            node.keywords.append(_blob_ast.keyword(arg='W',value=_blob_ast.Name(id='WT',ctx=_blob_ast.Load())))
        if isinstance(node.func,_blob_ast.Name) and node.func.id=='Qconstructor':
            node.keywords.append(_blob_ast.keyword(arg='kernel',value=_blob_ast.Name(id='WT',ctx=_blob_ast.Load())))
        if globals().get('_blob_args') is not None and _blob_args.paper and isinstance(node.func,_blob_ast.Name):
            if node.func.id=='nodeListConstructor':node.keywords.append(_blob_ast.keyword(arg='kernelExtent',value=_blob_ast.Constant(4.)))
            if node.func.id=='SpheralController':node.keywords.append(_blob_ast.keyword(arg='volumeType',value=_blob_ast.Name(id='RKSumVolume',ctx=_blob_ast.Load())))
        if isinstance(node.func,_blob_ast.Name) and node.func.id=='output' and node.args and isinstance(node.args[0],_blob_ast.Constant) and node.args[0].value=='hydro.HEvolution':
            node.args[0].value='hydro._smoothingScaleMethod.HEvolution'
        return node
    def visit_Assign(self,node):
        self.generic_visit(node)
        if any(isinstance(t,_blob_ast.Name) and t.id=='control' for t in node.targets):
            before=_blob_ast.parse('_blob_pre_controller(globals())').body[0]
            after=_blob_ast.parse('_blob_attach(control,globals())').body[0]
            return [before,node,after]
        return node


if __name__=='__main__':
    _blob_parser=_blob_argparse.ArgumentParser(description=__doc__)
    _blob_parser.add_argument('mode',choices=('run','render'));_blob_parser.add_argument('--output',type=_blob_Path,required=True)
    _blob_parser.add_argument('--strict-positive',action='store_true',help='Optional extra stop absent from native driver; default records negative energy without altering evolution.')
    _blob_parser.add_argument('--paper',action='store_true',help='Explicit paper configuration; requires isolated compiled overlay.')
    _blob_parser.add_argument('--nx',type=int,default=512);_blob_parser.add_argument('--tkh',type=float,default=8.)
    _blob_parser.add_argument('--frame-every',type=int,default=10)
    _blob_parser.add_argument('--integrator',choices=('cheap','fresh'),default='cheap')
    _blob_parser.add_argument('--partition',choices=('entropy','dudt'),default='entropy')
    _blob_parser.add_argument('--heating',choices=('one-sided','averaged'),default='one-sided',help='Native/paper Q_j heating or isolated averaged-Q predictor control')
    _blob_parser.add_argument('--av-gradient',choices=('lagged','sph','crk'),default='lagged',help='lagged native CRK, native rigorous fresh corrected SPH, or overlay fresh CRK Eq54')
    _blob_parser.add_argument('--time-grid',type=_blob_Path,help='Replay exact successive times from a recorded diagnosticHistory.json')
    _blob_args=_blob_parser.parse_args();_blob_output=_blob_args.output.resolve()
    if _blob_args.av_gradient=='crk' and not _blob_args.paper:
        _blob_parser.error('fresh CRK gradient requires the paper overlay')
    if _blob_args.heating!='one-sided' and not _blob_args.paper:
        _blob_parser.error('averaged heating requires the paper overlay')
    if _blob_args.nx<32 or _blob_args.nx%4 or _blob_args.tkh<=0 or _blob_args.frame_every<1:
        _blob_parser.error('nx must be >=32 and divisible by four; tkh/frame-every must be positive')
    if _blob_args.mode=='render':
        from study_blobPartition import context
        # Standard case plotters render the driver's recorded states passively.
        from warpSPHBootstrap import bootstrap
        bootstrap(precision='float32')
        from warpSPH.runner.media import encodeFrames
        from warpSPH.runner.display import closeWindow
        import h5py,torch
        settings=_blob_json.loads((_blob_output/'settings.json').read_text())
        # Native CLI calls the node-rate policy "dudt"; Warp's render context
        # calls the same choice "spheral". Older saved metadata uses the former.
        renderSettings=dict(settings,steps=1)
        if renderSettings['energyPartition']=='dudt':renderSettings['energyPartition']='spheral'
        case,ctx,*_=context(renderSettings)
        ctx.imagePath=str(_blob_output/'images');_blob_Path(ctx.imagePath).mkdir(exist_ok=True);ctx.exportPath=str(_blob_output)
        nativeConfig=_blob_json.loads((_blob_output/'nativeConfiguration.json').read_text())
        ctx.scratch['_titleProvenance']=('Paper policies | W7 | nh1 | '+str(nativeConfig['integrator'])+' | final '+str(settings['energyPartition'])+' | heating '+nativeConfig.get('heating','one-sided')) if settings.get('paperPolicies') else 'Actual bundled Spheral CRKSPH blob; native initialization / adaptive CheapRK2'
        plotter=None
        with h5py.File(_blob_output/'trajectory.h5','r') as f:
            for number,key in enumerate(sorted(f['states'])):
                group=f['states'][key];a={k:torch.as_tensor(v[()],device=ctx.device) for k,v in group.items()}
                system=_blob_NS(state=_blob_NS(**a),t=float(group.attrs['time']))
                if plotter is None:plotter=case.setupPlot(ctx,system)
                else:case.updatePlot(ctx,system,plotter,number)
        if plotter:
            if hasattr(plotter,'drain'):plotter.drain()
            closeWindow(plotter)
        print(encodeFrames(ctx.imagePath,ctx.exportPath),flush=True);raise SystemExit(0)
    _blob_output.mkdir(parents=True,exist_ok=False)
    _blob_source=_blob_Path(__file__).resolve().parents[1]/'results/materialEOS/spheral-build/source/tests/functional/Hydro/BlobTest/BlobTest-2d.py'
    _blob_text=_blob_source.read_text();_blob_tree=_blob_Compatibility().visit(_blob_ast.parse(_blob_text));_blob_ast.fix_missing_locations(_blob_tree)
    (_blob_output/'upstreamDriver.py').write_text(_blob_text)
    (_blob_output/'adaptedDriver.py').write_text(_blob_ast.unparse(_blob_tree)+'\n')
    _blob_manifest=dict(source=str(_blob_source),sourceSHA256=_blob_hashlib.sha256(_blob_text.encode()).hexdigest(),
        compatibilityRepairs=['Remove unused unavailable SpheralGnuPlotUtilities import','Correct undefined ASPH spelling to declared asph','Supply WT to current CRKSPH W argument','Supply WT to current Qconstructor kernel argument (q is unused by CRK branch)','Redirect HEvolution diagnostic to smoothing-scale package'],
        solverSelectionRepair='Join the independent fsisph/gsph/SPH dispatch to the preceding crksph/psph chain; otherwise SPH overwrites CRKSPH. Assert actual CRKSPH class before initialization.',
        instrumentation='Passive daemon/state/diagnostic hooks and finite/runaway watchdog; optional strict positivity stop; upstream source untouched.',
        numericalOverrides=['Select crksph=True; all numerical defaults otherwise supplied by driver.'],status='starting')
    _blob_manifest['watchdog']=dict(strictPositive=_blob_args.strict_positive,finite=True,maxVelocity=100,minSimulatedAdvancePerMinute=1e-3,negativeEnergy='Flag and retain; no repair or floor.')
    _blob_manifest['outputOverrides']=['Disable obsolete native visualization and periodic/final native restart outputs; use recorded HDF5/vispy and exported states instead.']
    _blob_manifest['integratorControl']=dict(mode=_blob_args.integrator,timeGrid=str(_blob_args.time_grid) if _blob_args.time_grid else None,
        rawPredictor='Native fresh SynchronousRK2 first update is redirected to raw increment; final update retains entropy-compatible work.' if _blob_args.integrator=='fresh' else 'Native CheapRK2 timeAdvanceOnly raw prediction')
    _blob_manifest['partitionControl']=_blob_args.partition
    _blob_manifest['avGradientControl']=_blob_args.av_gradient
    if _blob_args.paper:
        _blob_manifest['numericalOverrides']=['Paper W7/nh1/Cl2/Cq1/CFL.25; explicit kernelExtent4','Eq75 volume, Eq76 density and Eq63 final energy in isolated overlay',f'nx={_blob_args.nx}, ny={_blob_args.nx//4}, end={_blob_args.tkh} KH',
            'Published cold-cloud diagnostic counts all particles passing thresholds; original-tag diagnostic retained separately.']
        _blob_manifest['paperPolicies']=dict(kernel='W7',nPerh=1,radialNeighbors=4,Cl=2,Cq=1,CFL=.25,
            volume='Eq75 inverse gather number sum without native cap, refreshed before every RHS',density='Eq76 corrected scatter kernel with material mass replacement, once per step',
            energy='Eq63 entropy from latest RHS; raw RK2 predictor, compatible final work',
            retainedImplementationChoices=[('Fresh SynchronousRK2 RHS evaluations' if _blob_args.integrator=='fresh' else 'Native CheapRK2 derivative reuse')+'; bounded continuity density inside stages','Native Owen/IdealH implementation, support bounds and startup H iteration','Kernel table 1000 entries; paper does not specify table size'],
            explicitEOSRefresh='After per-step corrected density replacement')
        if _blob_args.partition=='dudt':
            _blob_manifest['paperPolicies']['energy']='Current native sign-filtered node-dudt fractions instead of paper Eq63; all other paper controls retained.'
    _blob_save('execution.json',_blob_manifest)
    _blob_sys.path[:0]=[str(_blob_source.parent),'/home/lu26029/dev/warpSPHRemote/python']
    _blob_sys.argv=[str(_blob_source),'--crksph','True','--vizCycle','None','--vizTime','None','--restartStep','None','--dataDir',str(_blob_output/'nativeOutputs')]
    if _blob_args.paper:
        _blob_sys.argv+=['--nx1',str(_blob_args.nx),'--ny1',str(_blob_args.nx//4),'--order','7','--nPerh','1.0','--Cl','2.0','--Cq','1.0','--cfl','.25','--goalTKH',str(_blob_args.tkh)]
    _blob_history=[];_blob_latest=None;_blob_publisher=None;_blob_file=None;_blob_g=None
    try:
        exec(compile(_blob_tree,str(_blob_source),'exec'),globals())
        _blob_manifest['status']='completed'
    except BaseException as _blob_error:
        _blob_manifest.update(status='failed',error=repr(_blob_error))
        if _blob_latest is not None:_blob_np.savez_compressed(_blob_output/'failedState.npz',**_blob_latest)
        raise
    finally:
        _blob_save('execution.json',_blob_manifest)
        if _blob_args.paper:_blob_save('paperPolicyCalls.json',_blob_paper_counts())
        if _blob_args.paper:
            import ctypes
            api=ctypes.CDLL(None)
            for name,columns in [('gradient',13),('affine',23)]:
                if not hasattr(api,'warpsph_paper_av_'+name+'_rows'):continue
                size=getattr(api,'warpsph_paper_av_'+name+'_rows');size.restype=ctypes.c_ulonglong
                data=getattr(api,'warpsph_paper_av_'+name+'_data');data.restype=ctypes.POINTER(ctypes.c_double)
                n=size()
                if n:_blob_np.savez_compressed(_blob_output/('av'+name.title()+'Audit.npz'),rows=_blob_np.ctypeslib.as_array(data(),shape=(n,columns)).copy())
            for name,columns in [('heating_node',40),('heating_pair',16),('partition',15)]:
                if not hasattr(api,'warpsph_paper_'+name+'_rows'):continue
                if name=='partition' and hasattr(api,'warpsph_paper_partition_columns'):
                    columns=api.warpsph_paper_partition_columns()
                size=getattr(api,'warpsph_paper_'+name+'_rows');size.restype=ctypes.c_ulonglong
                data=getattr(api,'warpsph_paper_'+name+'_data');data.restype=ctypes.POINTER(ctypes.c_double)
                n=size()
                if n:_blob_np.savez_compressed(_blob_output/(name+'Audit.npz'),rows=_blob_np.ctypeslib.as_array(data(),shape=(n,columns)).copy())
        if _blob_history:
            _blob_save('diagnosticHistory.json',[{k:v if _blob_np.isfinite(v) else None for k,v in r.items()} for r in _blob_history])
            _blob_clean=[{k:v if _blob_np.isfinite(v) else None for k,v in r.items()} for r in _blob_history]
            _blob_save('summary.json',dict(status=_blob_manifest['status'],time=_blob_history[-1]['time'],steps=len(_blob_history)-1,
                wallSeconds=_blob_time.perf_counter()-_blob_start,final=_blob_clean[-1],
                positiveEnergy=all(r['minEnergy']>0 for r in _blob_history)))
            if _blob_manifest['status']=='completed':_blob_np.savez_compressed(_blob_output/'finalState.npz',**_blob_latest)
        if _blob_file is not None:_blob_file.close()
        if _blob_publisher is not None:_blob_publisher.close(_blob_manifest['status'],_blob_manifest.get('error',''))
