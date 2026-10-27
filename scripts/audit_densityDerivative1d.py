"""Static density-map derivatives versus continuity on common 1D states.

No time integration. Actual Warp operators and a compiled native Spheral
component audit; distinguish native stock density from paper mass replacement.
"""
import argparse
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from compare_spheralMaterials import save

NATIVE_FIELDS=('volumes','paperDensity','stockDensity','gradient','continuity','geometricContinuity',
    'differenceGradient','pressureDudt','partitionGradient','A','coefficient1','volumeCap','stockFloor','pressureEntropyRate')


def periodic_rows(a,domain):
    lo,hi=domain;length=hi-lo;x=(a['positions'][:,0]-lo)%length+lo;n=len(x)
    real=np.column_stack((x,a['supports'],a['masses'],a['densities'],a['velocities'][:,0],
        a['materials'],np.ones(n),np.arange(n),a['internalEnergies']))
    rows=[real];margin=a['supports'].max()
    for shift in (-length,length):
        moved=x+shift;keep=(moved>=lo-margin)&(moved<=hi+margin)
        ghosts=real[keep].copy();ghosts[:,0]=moved[keep];ghosts[:,6]=0.;rows.append(ghosts)
    return np.ascontiguousarray(np.concatenate(rows),dtype=np.float64)


class NativeMap:
    def __init__(self,path):
        self.build=json.loads(path.read_text());self.lib=ctypes.CDLL(self.build['libraryPath'])
        p=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.spheral_density_derivative.argtypes=[p,ctypes.c_int,ctypes.c_int,p]
        self.lib.spheral_density_derivative.restype=ctypes.c_int
        assert self.lib.spheral_density_derivative_version()==1
    def __call__(self,a,domain):
        rows=periodic_rows(a,domain);out=np.zeros((len(a['masses']),14),dtype=np.float64)
        assert self.lib.spheral_density_derivative(rows,len(rows),len(out),out)==0
        assert np.isfinite(out).all()
        return {k:out[:,i].copy() for i,k in enumerate(NATIVE_FIELDS)}


class WarpMap:
    def __init__(self):
        from warpSPHBootstrap import bootstrap
        bootstrap(precision='float64')
        import warp as wp
        wp.config.kernel_cache_dir='/tmp/warpsph-density-map-kernels'
        import torch
        self.torch=torch
    def make(self,a,domain):
        from warpSPHCore import ParticleState,DomainDescription,SupportScheme,buildVerletList
        # NumPy [:,None] can have inner stride 0 despite being C-contiguous;
        # Warp's length-one vector ABI requires an actual unit inner stride.
        t=lambda a:self.torch.tensor(np.array(a,copy=True,order='C'),dtype=self.torch.float64)
        s=ParticleState(t(a['positions']),t(a['supports']),t(a['masses']),self.torch.zeros(len(a['masses']),dtype=self.torch.int32),t(a['densities']))
        s.velocities=t(a['velocities']);s.internalEnergies=t(a['internalEnergies']);s.materials=self.torch.as_tensor(a['materials'],dtype=self.torch.int32)
        s.pressures=.4*s.densities*s.internalEnergies;s.soundspeeds=(1.4*s.pressures/s.densities).sqrt();s.alphas=self.torch.ones_like(s.densities)
        d=DomainDescription(t([domain[0]]),t([domain[1]]),self.torch.tensor([True]),1)
        adj=buildVerletList(s,d,supportMode=SupportScheme.SuperSymmetric,verletScale=1.01)
        return s,d,adj
    def __call__(self,a,domain):
        from warpSPHCore import KernelFunctions,OperationProperties,WarpOperation,SupportScheme,GradientScheme,warpOperation
        from warpSPH.modules.crk.terms import crkFactors
        from warpSPH.modules.crk.materialDensity import materialDensity
        from warpSPH.modules.crk import CRKStage,computeCRKdudt
        from warpSPH.configurations import CRKSPHConfig
        from warpSPH.configurations.crkSPH import resolveCRKLimiter
        s,d,adj=self.make(a,domain);V,rho,c=crkFactors(s,d,KernelFunctions.B8,adj,densityPolicy='massReplacement')
        stock=materialDensity(s,d,KernelFunctions.B8,V,c,adj,'sameMaterial')
        gradient=warpOperation(s,OperationProperties(kernel=KernelFunctions.B8,operation=WarpOperation.Gradient,
            supportMode=SupportScheme.Scatter,gradientMode=GradientScheme.Difference),queryValues=s.velocities,
            domain=d,adjacency=adj,queryVolumes=V,crkState=c).mT
        cfg=CRKSPHConfig();cfg.diffusionParams.C_l=cfg.diffusionParams.C_q=0.
        q=resolveCRKLimiter(cfg.crkViscosityParams,4.);q.viscosityKernelScale=4.
        stage=CRKStage(V,c,gradient,q)
        du=computeCRKdudt(s,NS(kernel=KernelFunctions.B8,domain=d),cfg,adj,stage)
        g=gradient[:,0,0];out=dict(volumes=V,paperDensity=rho,stockDensity=stock,gradient=g,
            continuity=-s.densities*g,geometricContinuity=-rho*g,pressureDudt=du,
            pressureEntropyRate=du/s.internalEnergies+.4*g)
        return {k:v.detach().numpy().copy() for k,v in out.items()}
    def relaxed_support(self,a,domain):
        from warpSPHCore import KernelFunctions,SupportScheme
        from warpSPH.configurations import CRKSPHConfig,MaterialConfig
        from warpSPH.modules.adaptiveSupport.optimalSupportOwen import evaluateOptimalSupportOwen
        s,d,_=self.make(a,domain)
        cfg=CRKSPHConfig(owenTable='hermiteShell',owenMaterialWeights=True,materials=[MaterialConfig(),MaterialConfig()])
        c=NS(kernel=KernelFunctions.B8,domain=d,targetNeighbors=8.,verletScale=1.4,supportMode=SupportScheme.KernelMeanSymmetric)
        return evaluateOptimalSupportOwen(s,c,cfg,validateOnce=True,computeDensity=False)[1].numpy().copy()


def derivative(mapping,a,domain,eps,support_rate=None,density_key='paperDensity'):
    plus={k:np.array(v,copy=True) for k,v in a.items()};minus={k:np.array(v,copy=True) for k,v in a.items()}
    plus['positions']+=eps*a['velocities'];minus['positions']-=eps*a['velocities']
    if support_rate is not None:
        plus['supports']+=eps*support_rate;minus['supports']-=eps*support_rate
        assert min(plus['supports'].min(),minus['supports'].min())>0
    return (mapping(plus,domain)[density_key]-mapping(minus,domain)[density_key])/(2*eps)


def controls():
    result={}
    for name in ('uniformAffine','materialAffine','unequalSpacingAffine'):
        if name=='unequalSpacingAffine':
            x=np.r_[-.5+(np.arange(100)+.5)*.005,(np.arange(400)+.5)*.00125]
            rho=np.where(x<0,1.,4.);m=np.full(len(x),.005);R=4*m/rho
        else:
            x=-.5+(np.arange(100)+.5)*.01;rho=np.where(x<0,1.,4.) if name=='materialAffine' else np.ones(len(x));m=rho*.01;R=np.full(len(x),.04)
        tag=(x>=0).astype(np.int32) if name!='uniformAffine' else np.zeros(len(x),np.int32)
        result[name]=(dict(positions=x[:,None],velocities=.1*x[:,None],supports=R,masses=m,densities=rho,
            materials=tag,internalEnergies=1/(.4*rho),supportRate=.1*R),[-.5,.5])
    return result


def discrepancy(value,reference,m,mask):
    a,b=value[mask],reference[mask];w=m[mask];delta=abs(a-b)
    return dict(weightedL1=float(np.dot(w,delta)/w.sum()),
        relativeL2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1.e-30)),maxAbsolute=float(delta.max()))


def audit(root,report,build):
    native=NativeMap(build);warp=WarpMap();report.mkdir(parents=True,exist_ok=True);arrays=report/'operators';arrays.mkdir(exist_ok=True)
    cases=controls();labels={}
    # Main standard derivative policies on both dt grids; final entropy-ODE
    # states provide a separate end-state control without expanding this into
    # a dense temporal reconstruction claim.
    for policy in ('compatible','raw','entropy','entropyCompatible'):
        for split in (1,2):
            folder=root/f'sod-dt{split}-{policy}'
            if not (folder/'frozenStates.json').exists():continue
            frozen=json.loads((folder/'frozenStates.json').read_text());s=json.loads((folder/'settings.json').read_text())
            wanted=(.02,.06,.2) if policy in ('compatible','raw') else (.2,)
            for t in wanted:
                item=min(frozen,key=lambda v:abs(v['time']-(t-.5*s['dt'])))
                a=dict(np.load(item['path']));name=f'sod-dt{split}-{policy}-t{t:g}'
                cases[name]=(a,s['domain']);labels[name]=item
    output={}
    for name,(a,domain) in cases.items():
        w=warp(a,domain);n=native(a,domain);mask=abs(a['positions'][:,0])<(.2 if name in controls() else .5);m=a['masses']
        record=dict(nativeVsWarp={k:discrepancy(n[k],w[k],m,mask) for k in
            ('volumes','paperDensity','stockDensity','gradient','pressureDudt','pressureEntropyRate')},
            nativeCaps=int(n['volumeCap'].sum()),nativeStockFloors=int(n['stockFloor'].sum()),
            nativeDifferenceVsGradient=float(abs(n['differenceGradient']-n['gradient']).max()),
            domain=domain,label=labels.get(name),samples={})
        data=dict(positions=a['positions'],supports=a['supports'],masses=m,materials=a['materials'],mask=mask)
        data.update({prefix+k:v for prefix,values in [('warp_',w),('native_',n)] for k,v in values.items()})
        for mode,rate in [('fixed',None),('measuredSupport',a.get('supportRate'))]:
            if mode=='measuredSupport' and rate is None:continue
            record['samples'][mode]={};fine={}
            for provider,mapping,base in [('warp',warp,w),('native',native,n)]:
                rows=[];values=[]
                for eps in (1.e-5,3.e-6,1.e-6):
                    d=derivative(mapping,a,domain,eps,rate);z=d/base['paperDensity'];continuity=-base['gradient'];residual=z-continuity
                    rows.append(dict(epsilon=eps,rate=discrepancy(z,continuity,m,mask),
                        signedMeanMismatch=float(np.dot(m[mask],residual[mask])/m[mask].sum())))
                    values.append(z)
                fine[provider]=values[-1];record['samples'][mode][provider]=dict(rows=rows,
                    fineDifference=discrepancy(values[-1],values[-2],m,mask))
                data[f'{provider}_{mode}_logDensityRate']=values[-1]
            record['samples'][mode]['nativeVsWarpDerivative']=discrepancy(fine['native'],fine['warp'],m,mask)
            # Native stock density is a different, positive, same-NodeList
            # formula. Test its own geometric derivative, not against the
            # corrected all-material paper density of the physical controls.
            for provider,mapping,base in [('warpStock',warp,w),('nativeStock',native,n)]:
                rows=[];values=[]
                for eps in (1.e-5,3.e-6,1.e-6):
                    z=derivative(mapping,a,domain,eps,rate,'stockDensity')/base['stockDensity']
                    residual=z+base['gradient'];values.append(z)
                    rows.append(dict(epsilon=eps,rate=discrepancy(z,-base['gradient'],m,mask),
                        signedMeanMismatch=float(np.dot(m[mask],residual[mask])/m[mask].sum())))
                record['samples'][mode][provider]=dict(rows=rows,fineDifference=discrepancy(values[-1],values[-2],m,mask))
                data[f'{provider}_{mode}_logDensityRate']=values[-1]
        newR=warp.relaxed_support(a,domain);relaxed=dict(a,supports=newR);rho_after=warp(relaxed,domain)['paperDensity']
        record['oneFurtherOwenAtFixedGeometry']=dict(maxRelativeSupportChange=float(abs(newR/a['supports']-1).max()),
            density=discrepancy(rho_after,w['paperDensity'],m,mask),
            signedEntropyChangeAtFixedU=float(np.dot(m[mask],-.4*np.log(rho_after[mask]/w['paperDensity'][mask]))/m[mask].sum()),
            scope='One further relaxation on the saved geometry, not an actual before/after-RHS trace')
        if name in controls():
            for provider in ('warp','native'):
                assert record['samples']['measuredSupport'][provider]['rows'][-1]['rate']['maxAbsolute']<2.e-6,(name,provider,record)
        np.savez_compressed(arrays/(name+'.npz'),**data);output[name]=record
        print('DENSITY MAP',name,'fixed mismatch',record['samples']['fixed']['warp']['rows'][-1]['rate']['weightedL1'],
            'with support',record['samples']['measuredSupport']['warp']['rows'][-1]['rate']['weightedL1'],flush=True)
    save(report/'densityOperatorAudit.json',dict(nativeBuild=native.build,cases=output,
        scope='Static finite differences along xdot=v, R fixed or captured Rdot. Geometric V and RK corrections recomputed at every displaced state. '
            'Native stock positive same-NodeList density is recorded separately; corrected all-material mass replacement is a declared paper formula using native RK kernels. '
            'Physical volume/continuity/pressure rates never overwritten. No time integration or native trajectory claim.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path)
    p.add_argument('--report',type=Path,default=Path('reports/materialEOS/densityEnergy'))
    p.add_argument('--native-build',type=Path,default=Path('results/materialEOS/spheral-density-derivative1d/build.json'));a=p.parse_args()
    audit(a.input.resolve(),a.report.resolve(),a.native_build.resolve())
