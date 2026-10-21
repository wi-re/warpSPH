"""Local-table static audit and daemon-visible matched blob controls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace as NS
import numpy as np
from compare_spheralMaterials import save
from study_idealH import study,run_row


def prepare(root):
    root.mkdir(parents=True,exist_ok=False);parent=Path('results/materialEOS/idealH_2026-10-20').resolve();rows=[]
    for split in (1,2):
        source=parent/f'blob-dt{split}-weighted'
        for policy in ('generatedNative','generated'):
            name=f'blob-dt{split}-{policy}';folder=root/name;folder.mkdir();shutil.copyfile(source/'input.npz',folder/'input.npz')
            s=json.loads((source/'settings.json').read_text());s.update(gammaVariant=name,supportPolicy=policy,
                inputPath=str((folder/'input.npz').resolve()),comparisonPolicy='Local kernel/dimension-generated tables; same material weights / case bounds / before-RHS / paper density / fresh midpoint / energy / AV; '+policy)
            save(folder/'settings.json',s);rows.append(name)
    sources={};(root/'sources').mkdir()
    for f in (__file__,'scripts/study_idealH.py','scripts/study_densityEvolution.py',
        'src/warpSPH/modules/adaptiveSupport/owenHermite.py','src/warpSPH/modules/adaptiveSupport/wp_materialPsi.py',
        'src/warpSPH/modules/adaptiveSupport/optimalSupportOwen.py','src/warpSPH/configurations/compressibleConfig.py'):
        p=Path(f);h=hashlib.sha256(p.read_bytes()).hexdigest();sources[str(p)]=h;shutil.copyfile(p,root/'sources'/f'{p.stem}_{h[:12]}{p.suffix}')
    save(root/'provenance.json',dict(rows=rows,parent=str(parent),sources=sources,
        scope='Only locally generated target tables/moment vs retained imported-table controls; 1000-point linear native-compatible and 4096-point log refined construction. Both use native case R bounds, explicit before-RHS timing and weights. No native runtime imports/data are used to generate coefficients.'))


def audit(report):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import torch
    from warpSPHCore import KernelFunctions,ParticleState,DomainDescription,SupportScheme
    from warpSPH.configurations import CRKSPHConfig
    from warpSPH.modules.adaptiveSupport.owenHermite import coefficients,HermiteOwen,shell_values,hermite_values
    from warpSPH.modules.adaptiveSupport.optimalSupportOwen import evaluateOptimalSupportOwen,PsiLUTs
    report.mkdir(parents=True,exist_ok=True);result={};T=lambda a:torch.as_tensor(a,device='cuda',dtype=torch.float64).contiguous()
    for kernel in KernelFunctions:
        if kernel.name in ('ViscosityKernel','CohesionKernel','AdhesionKernel'):continue
        for dim in (1,2,3):
            start=time.perf_counter();a=coefficients(kernel,dim);x=np.linspace(3,12,400);z=shell_values(x,dim,a['gradData']);m=a['lookupMeta'];n=int(m[3]);y=a['lookupData']
            out=hermite_values(np.log(z),m[0],m[1],y[:n],y[n:])
            result[kernel.name+f'-{dim}d']=dict(buildAndAuditSeconds=time.perf_counter()-start,maxRelativeRoundtrip=float(abs(out/x-1).max()),
                minimum=1.1,maximum=256.,points=n,inverseSpace='log')
            print('LOCAL TABLE',kernel.name,dim,result[kernel.name+f'-{dim}d'],flush=True)
    n=32;axis=(np.arange(n)+.5)/n;x,y=np.meshgrid(axis,axis,indexing='xy');x=np.column_stack((x.ravel(),y.ravel()));tag=(x[:,0]>.5).astype(np.int32)
    rho=np.where(tag,10.,1.);a=dict(positions=x,supports=np.full(n*n,4/n),masses=rho/n**2,densities=rho,materials=tag)
    cases=dict(uniform=a,volumeJump=dict(a,masses=a['masses']*np.where(tag,2.,1.)),uniformSmallH=dict(a,supports=a['supports']*.7),uniformLargeH=dict(a,supports=a['supports']*1.4))
    paths=dict(blobInitial='results/materialEOS/blob_matrix_2026-10-20/matched/input.npz',blobWarp='results/materialEOS/density_evolution_2026-10-20/blob-dt2-paper/warp/finalState.npz',blobNative='results/materialEOS/crk_heating_v2_2026-10-20/spheral-halfGrid-averaged/finalState.npz')
    cases.update({k:dict(np.load(v)) for k,v in paths.items()});static={}
    for name,a in cases.items():
        box=np.array([40.,10.]) if name in paths else np.array([1.,1.]);domain=DomainDescription(T([0.,0.]),T(box),torch.tensor([True,True],device='cuda'),2)
        reference=np.load('reports/materialEOS/idealH/operators/'+name+'.npz')['native'][:,0]
        for mode in ('native','refined'):
            state=ParticleState(T(a['positions']%box),T(a['supports']),T(a['masses']),torch.zeros(len(a['supports']),device='cuda',dtype=torch.int32),T(a['densities']))
            state.materials=torch.as_tensor(a['materials'],device='cuda',dtype=torch.int32)
            cfg=CRKSPHConfig(owenTable='hermiteShell',owenMaterialWeights=True)
            PsiLUTs[(KernelFunctions.B8,2,'hermiteShell')]=HermiteOwen(KernelFunctions.B8,2,**(dict(points=1000,inverse_space='linear') if mode=='native' else {}))
            c=NS(kernel=KernelFunctions.B8,domain=domain,targetNeighbors=16*np.pi,verletScale=1.4,supportMode=SupportScheme.KernelMeanSymmetric)
            h=evaluateOptimalSupportOwen(state,c,cfg,validateOnce=True,computeDensity=False)[1].clamp(4.e-5,2.).cpu().numpy()
            static[name+'-'+mode]=dict(maxRelativeSupportDifference=float(abs(h/reference-1).max()),rmsRelativeSupportDifference=float(np.sqrt(np.mean((h/reference-1)**2))))
            print('STATIC LOCAL TARGET',name,mode,static[name+'-'+mode],flush=True)
    save(report/'operatorAudit.json',dict(kernelDimensions=result,staticNativeComparisons=static))
    for mode,opts in [('native',dict(points=1000,inverse_space='linear')),('refined',{})]:np.savez_compressed(report/('localB8-'+mode+'.npz'),**coefficients(KernelFunctions.B8,2,**opts))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','study','row','audit'))
    p.add_argument('--output',required=True,type=Path);p.add_argument('--publisher-path',type=Path,default=Path('/home/lu26029/dev/warpSPHRemote/python'));a=p.parse_args();a.output=a.output.resolve()
    if a.mode=='prepare':prepare(a.output)
    elif a.mode=='study':study(a.output,row_script=__file__)
    elif a.mode=='audit':audit(a.output)
    else:raise SystemExit(run_row(a))
