"""Resume full blob trajectory matching, isolating final compatible fractions.

Native-style support replacement is an explicit reference-match control only.
The preferred fresh before-RHS support semantics and all defaults stay intact.
"""
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from compare_spheralMaterials import save, warp_run, host
from compare_spheralBlobPaper import initialize as blob_initialize
from study_densityEvolution import install_audit as density_audit
from study_idealH import install_support, study
from study_blobThermal import Measures, summarize


def reference_fraction(si, sj, work):
    """Pinned native entropyWeighting: exact thresholds, positive ideal-gas s."""
    import torch
    denominator=si.abs()+sj.abs()
    fraction=torch.where(work>0.,sj.abs(),si.abs())/denominator.clamp_min(torch.finfo(si.dtype).tiny)
    active=(work.abs()>1.e-15)&((si-sj).abs()>1.e-15)
    return torch.where(active,fraction,torch.full_like(fraction,.5))


def reference_rate(adjacency,pairAcceleration,entropy,gamma,baseVelocity,totalAcceleration,multiplier):
    from warpSPH.modules.compSPH.workReduction import reduceReferenceEntropyWork
    half=baseVelocity+.5*multiplier*totalAcceleration
    return reduceReferenceEntropyWork(adjacency,half,pairAcceleration,entropy)


def matching_options(settings):
    options=copy.deepcopy(settings)
    options.setdefault('matrixControls',{}).setdefault('solver',{}).update(
        crkEvolution=settings['schedule'],crkEnergyPartition=settings['matchingEnergyPartition'],crkEnergyPredictor='raw')
    return options


def context(settings):
    # The density-study context deliberately replaces matrixControls; retain
    # this study's energy control while selecting the same density schedule.
    from study_blobThermal import context as blob_context
    return blob_context(matching_options(settings))


def prepare(root):
    root.mkdir(parents=True,exist_ok=False);parent=Path('results/materialEOS/idealH_2026-10-20').resolve();rows=[]
    table=parent/'tables.npz';shutil.copyfile(table,root/'tables.npz')
    references={}
    for split,grid in ((1,'grid'),(2,'halfGrid')):
        old=parent/f'blob-dt{split}-replacement';base=json.loads((old/'settings.json').read_text())
        references[str(split)]=str(Path(f'results/materialEOS/crk_heating_v2_2026-10-20/spheral-{grid}-averaged').resolve())
        for policy in ('native','reference'):
            name=f'blob-dt{split}-{policy}';folder=root/name;folder.mkdir();shutil.copyfile(old/'input.npz',folder/'input.npz')
            s=copy.deepcopy(base);s.update(gammaVariant=name,matchingPolicy=policy,
                matchingEnergyPartition='native' if policy=='native' else 'entropy',
                inputPath=str((folder/'input.npz').resolve()),idealTable=str((root/'tables.npz').resolve()),
                comparisonPolicy='Full matching control: native weighted IdealH target/state replacement, paper density/geometric volumes, fresh midpoint, averaged Q/current gradient; '+policy+' final compatible fraction',
                differences=['Direct versus tabulated hydrodynamic kernel, reductions and traversal remain.',
                    'native policy uses Warp RHS-estimated compatible fraction; reference uses actual final-work sign and pinned native thresholds.'])
            s['matrixControls']['solver'].update(crkEnergyPartition=s['matchingEnergyPartition'],crkEnergyPredictor='raw')
            assert len(set(s['gamma']))==1,'This reference fraction control is qualified only for common gamma'
            save(folder/'settings.json',s);rows.append(name)
    paths=[Path(p) for p in (__file__,'scripts/study_idealH.py','scripts/crkIdealH.py','scripts/study_densityEvolution.py',
        'scripts/study_blobThermal.py','scripts/compare_spheralBlobPaper.py','scripts/compare_spheralMaterials.py',
        'src/warpSPH/modules/crk/energyControl.py','src/warpSPH/modules/crk/terms.py',
        'src/warpSPH/modules/crk/evolution.py','src/warpSPH/modules/compSPH/pairWork.py',
        'src/warpSPH/schemes/compressible/crkSPH.py','src/warpSPH/systems/compSPH.py',
        'results/materialEOS/spheral-build/source/src/Hydro/entropyWeightingFunction.hh')]
    sources={};(root/'sources').mkdir()
    for p in paths:
        h=hashlib.sha256(p.read_bytes()).hexdigest();sources[str(p)]=h;shutil.copyfile(p,root/'sources'/(p.stem+'_'+h[:12]+p.suffix))
    reference_hashes={}
    for split,path in references.items():
        reference_hashes[split]={name:hashlib.sha256((Path(path)/name).read_bytes()).hexdigest() for name in
            ('initializedState.npz','finalState.npz','heating_nodeAudit.npz','trajectory.h5','execution.json','paperOverlayLoaded.json')}
    save(root/'provenance.json',dict(rows=rows,parent=str(parent),references=references,sources=sources,
        referenceHashes=reference_hashes,tableSHA256=hashlib.sha256(table.read_bytes()).hexdigest(),
        scope='Four Warp trajectories, paired final compatible fractions at two dt grids. Reuse completed compiled native fresh-RK2/CRK-gradient/averaged-Q references. Native support state replacement is diagnostic, not the preferred before-RHS policy. No production defaults or physics source changes.'))


def install_observer(folder):
    """Read-only first two steps: RHS components, updates and accepted states."""
    module=importlib.import_module('warpSPH.schemes.compressible.crkSPH')
    systems=importlib.import_module('warpSPH.systems.compSPH')
    accel,du=module.computeCRKAccel,module.computeCRKdudt
    apply,finalize=systems.CompSPHSystem.apply_state_update,systems.CompSPHSystem.finalize
    stats=dict(rhs=0,updates=0,accepted=0);pending={};out=folder/'earlyStates';out.mkdir()
    cpu=lambda x:x.detach().cpu().numpy().copy()
    def acceleration(*args,**kwargs):
        result=accel(*args,**kwargs)
        if stats['rhs']<4:pending['acceleration']=result[0].clone()
        return result
    def derivative(state,c,sc,adj,stage,*args):
        result=du(state,c,sc,adj,stage,*args);stats['rhs']+=1
        if stats['rhs']<=4:
            a=host(state);crk=stage.crkState
            a.update(volumes=cpu(stage.volumes),velocityGradient=cpu(stage.velocityGradient),
                acceleration=cpu(pending.pop('acceleration')),dudt=cpu(result),A=cpu(crk.A),B=cpu(crk.B),
                gradA=cpu(crk.gradA),gradB=cpu(crk.gradB))
            np.savez_compressed(out/f'rhs_{stats["rhs"]:02d}.npz',**a)
        return result
    def update(system,derivative,spec,**kwargs):
        result=apply(system,derivative,spec,**kwargs);stats['updates']+=1
        if stats['updates']<=4:
            np.savez_compressed(out/f'update_{stats["updates"]:02d}.npz',**host(system.state))
        return result
    def finish(system,*args,**kwargs):
        result=finalize(system,*args,**kwargs)
        if system.rerunRequest is None:
            stats['accepted']+=1
            if stats['accepted']<=2:np.savez_compressed(out/f'accepted_{stats["accepted"]:02d}.npz',**host(system.state))
        return result
    module.computeCRKAccel=acceleration;module.computeCRKdudt=derivative
    systems.CompSPHSystem.apply_state_update=update;systems.CompSPHSystem.finalize=finish
    return stats


def row(args):
    sys.path.insert(0,str(args.publisher_path));s=json.loads((args.output/'settings.json').read_text())
    density=density_audit(s);support=install_support(s)
    if s['matchingPolicy']=='reference':
        import warpSPH.modules.crk.energyControl as energy
        energy.entropyRate=reference_rate
    observer=install_observer(args.output)
    def initialize(c,system,a,s):
        blob_initialize(c,system,a,s);system.state.crkVolumes=None
        c.scratch['_titleProvenance']='Spheral matching | '+s['gammaVariant']+' | native timing diagnostic'
    try:return warp_run(args,s,context,Measures,summarize,'Matched Spheral blob',initialize_factory=initialize)
    finally:
        save(args.output/'densityAudit.json',density);save(args.output/'supportAudit.json',support);save(args.output/'observerAudit.json',observer)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','study','row'));p.add_argument('--output',required=True,type=Path)
    p.add_argument('--publisher-path',type=Path,default=Path('/home/lu26029/dev/warpSPHRemote/python'));a=p.parse_args();a.output=a.output.resolve()
    if a.mode=='prepare':prepare(a.output)
    elif a.mode=='study':study(a.output,row_script=__file__)
    else:raise SystemExit(row(a))
