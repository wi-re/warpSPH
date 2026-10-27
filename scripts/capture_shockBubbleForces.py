"""Passive native-RHS exports for the shock-bubble force audit.

Physical capture trajectories always use the existing daemon/video harness.
Spheral inputs are saved after initializeDerivatives; the following virtual
initialize call reads the preceding RHS outputs before derivatives are zeroed.
Warp inputs/pair forces are copied before compatible work clears pair arrays.
"""
import argparse
import functools
import json
import shutil
from pathlib import Path
import numpy as np
from compare_spheralMaterials import save, spheral_run, warp_run, host
from compare_spheralShockBubble import Measures, summary, context, build

TARGETS=(0.,.05,.1,.15)


def selected(t,dt):
    return next((v for v in TARGETS if abs(t-v)<dt*.1),None)


def spheral_factory(s,db,hydro,nodes,settings,root):
    folder=root/'rhs';folder.mkdir();initial=dict(np.load(root.parent/'input.npz'))
    vector=lambda v:[v.x,v.y]
    tensor=lambda v:[[v.xx,v.xy],[v.yx,v.yy]]
    class Capture(s.SynchronousRK2Integrator):
        def __init__(self):
            super().__init__(db);self.pending=None;self.done=set()
        def initializeDerivatives(self,t,dt,state,derivs):
            if self.pending is not None:
                a,target=self.pending;acc=hydro.pairAccelerations
                assert len(acc)==len(a['i'])
                a['nativePairAcceleration']=np.asarray([vector(acc[k]) for k in range(len(acc))])
                a['nativeAcceleration']=np.concatenate([np.asarray([vector(hydro.DvDt[k][j]) for j in range(n.numNodes)]) for k,n in enumerate(nodes)])
                a['nativeDudt']=np.concatenate([np.asarray([hydro.DspecificThermalEnergyDt[k][j] for j in range(n.numNodes)]) for k,n in enumerate(nodes)])
                a['currentGradient']=np.concatenate([np.asarray([tensor(hydro.DvDx[k][j]) for j in range(n.numNodes)]) for k,n in enumerate(nodes)])
                # First-half-step energy after the compiled compatible policy;
                # validate its partition independently from the raw dudt RHS.
                next_u=np.concatenate([np.asarray(n.specificThermalEnergy().internalValues()) for n in nodes])
                a['nativeNextEnergyReal']=next_u;a['nativeNextTime']=np.array(float(t))
                np.savez_compressed(folder/('t'+format(target,'.3f')+'.npz'),**a)
                print('Exported Spheral RHS',target,len(a['i']),'pairs',flush=True)
                self.pending=None
            super().initializeDerivatives(t,dt,state,derivs)
            target=selected(float(t),settings['dt'])
            if target is None or target in self.done:return
            self.done.add(target)
            rk=next(p for p in self.physicsPackages() if hasattr(p,'corrections'))
            C=rk.corrections(s.LinearOrder);WR=rk.WR(s.LinearOrder);vol=state.scalarFields(s.HydroFieldNames.volume)
            counts=[n.numNodes for n in nodes];offset=np.r_[0,np.cumsum(counts)]
            out={k:[] for k in ('positions','velocities','masses','densities','pressures','internalEnergies','supports','H','volumes','coefficients','qGradient','soundspeeds','materials','UIDs','real')}
            for k,n in enumerate(nodes):
                ids=initial['UIDs'][initial['materials']==k]
                for j in range(n.numNodes):
                    H=n.Hfield()[j]
                    values=dict(positions=vector(n.positions()[j]),velocities=vector(n.velocity()[j]),masses=n.mass()[j],
                        densities=n.massDensity()[j],pressures=hydro.pressure[k][j],internalEnergies=n.specificThermalEnergy()[j],
                        supports=2./H.xx,H=tensor(H),volumes=vol[k][j],coefficients=list(C[k][j]),qGradient=tensor(hydro.Q.DvDx[k][j]),
                        soundspeeds=hydro.soundSpeed[k][j],materials=k,UIDs=int(ids[j]) if j<len(ids) else -1-int(offset[k]+j),real=j<n.numInternalNodes)
                    for key,value in values.items():out[key].append(value)
            a={k:np.asarray(v) for k,v in out.items()};pairs=db.connectivityMap().nodePairList
            a['i']=np.asarray([offset[p.i_list]+p.i_node for p in pairs],dtype=np.int64)
            a['j']=np.asarray([offset[p.j_list]+p.j_node for p in pairs],dtype=np.int64)
            # Evaluate the actual compiled RK gradient, not a NumPy imitation.
            gi=[];gj=[]
            for p in pairs:
                ni,nj=nodes[p.i_list],nodes[p.j_list];i,j=p.i_node,p.j_node
                r=ni.positions()[i]-nj.positions()[j]
                gi.append(vector(WR.evaluateGradient(r,nj.Hfield()[j],C[p.i_list][i])))
                gj.append(vector(WR.evaluateGradient(-r,ni.Hfield()[i],C[p.j_list][j])))
            a.update(gradientI=np.asarray(gi),gradientJ=np.asarray(gj),time=np.array(float(t)),dt=np.array(float(dt)))
            self.pending=(a,target)
    return Capture()


def install_warp_capture(folder,settings):
    import warpSPH.schemes.builder as builder
    import importlib
    module=importlib.import_module('warpSPH.schemes.compressible.crkSPH')
    original_step=builder.crkSPH_step;original_acc=module.computeCRKAccel;original_sum=module.accumulateStage
    active={};done=set();folder.mkdir()
    def array(x):return x.detach().cpu().numpy().copy()
    @functools.wraps(original_step)
    def step(system,dt,config,schemeConfig,*args,**kwargs):
        t=float(system.t);target=selected(t,settings['dt'])
        active.clear();active.update(time=t,dt=float(dt),target=target if target not in done else None)
        result=original_step(system,dt,config,schemeConfig,*args,**kwargs)
        if active.get('data') is not None:
            a=active['data'];a['nativeAcceleration']=array(result[0].dvdt);a['nativeDudt']=array(result[0].dudt)
            np.savez_compressed(folder/('t'+format(active['target'],'.3f')+'.npz'),**a)
            done.add(active['target']);print('Exported Warp RHS',t,len(a['i']),'directed pairs',flush=True)
        return result
    def accel(state,config,schemeConfig,adjacency,stage,gradH):
        result=original_acc(state,config,schemeConfig,adjacency,stage,gradH)
        if active.get('target') is not None:
            a=host(state);i,j=adjacency.livePairs()
            a.update(i=array(i),j=array(j),volumes=array(stage.volumes),qGradient=array(stage.velocityGradient),
                A=array(stage.crkState.A),B=array(stage.crkState.B),gradA=array(stage.crkState.gradA),gradB=array(stage.crkState.gradB),
                soundspeeds=array(state.soundspeeds),pressurePairAcceleration=array(result[1])[:len(i)],viscousPairAcceleration=array(result[2])[:len(i)],
                time=np.array(active['time']),dt=np.array(active['dt']),real=np.ones(len(state.masses),dtype=bool))
            active['data']=a
        return result
    def accumulate(work,stageIndex,adjacency,state,energyScheme):
        if active.get('data') is not None:active['data']['energyFraction']=array(state.f_ij)[:len(active['data']['i'])]
        return original_sum(work,stageIndex,adjacency,state,energyScheme)
    builder.crkSPH_step=step;module.computeCRKAccel=accel;module.accumulateStage=accumulate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','spheral','warp','render'))
    p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--publisher-path',type=Path,default=Path('/home/lu26029/dev/warpSPHRemote/python'));args=p.parse_args()
    if args.mode=='prepare':
        args.output.mkdir(parents=True,exist_ok=False)
        for name in ('input.npz','warpInput.npz'):shutil.copyfile(args.input/name,args.output/name)
        settings=json.loads((args.input/'settings.json').read_text());settings['end']=.1504
        settings['gammaVariant']='n'+str(settings['nx'])+'-rhsCapture';save(args.output/'settings.json',settings);return 0
    settings=json.loads((args.output/'settings.json').read_text())
    if args.mode=='spheral':return spheral_run(args,settings,Measures,summary,'Force audit',spheral_factory)
    if args.mode=='render':
        from compare_spheralMaterials import render
        return render(args,settings,context)
    # Install before context/build resolves the scheme function.
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    folder=args.output/'warp-rhs';install_warp_capture(folder,settings)
    return warp_run(args,settings,context,Measures,summary,'Force audit',build,'warpInput.npz')


if __name__=='__main__':raise SystemExit(main())
