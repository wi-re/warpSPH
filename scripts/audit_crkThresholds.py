"""Static CPU pair sweeps: actual Warp limiter/Q versus compiled Spheral and paper."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from warpSPHBootstrap import bootstrap
bootstrap(precision='float64')
import numpy as np
import warp as wp
from warpSPHCore import scalar_t, domainData, KernelFunctions, sphKernelScale
from warpSPH.modules.reconstruction.limiters import computeVanLeer, crkLimiter
from warpSPH.modules.reconstruction.pairVelocity import limitedPairPhi, linearPairVelocity
from warpSPH.modules.dissipation import DiffusionParameters, computeFrontiereQ

Vec=wp.types.vector(length=2,dtype=scalar_t)
Mat=wp.types.matrix(shape=(2,2),dtype=scalar_t)


@wp.kernel
def evaluate(a:wp.array2d(dtype=scalar_t),domain:domainData,p:DiffusionParameters,out:wp.array2d(dtype=scalar_t)):
    n=wp.tid()
    x=Vec(a[n,0],a[n,1]);zero=Vec(scalar_t(0.),scalar_t(0.))
    gi=Mat(a[n,6],a[n,7],a[n,8],a[n,9]);gj=Mat(a[n,10],a[n,11],a[n,12],a[n,13])
    vi=Vec(a[n,14],a[n,15]);vj=Vec(a[n,16],a[n,17])
    crit=scalar_t(1.)/a[n,5];fold=scalar_t(.2)/a[n,5]
    phi=limitedPairPhi(x,a[n,2],a[n,3],vi,vj,gi,gj,wp.static(KernelFunctions.B8.value),2,True,True,crit,fold)
    dv=linearPairVelocity(vi,vj,gi,gj,x,phi)
    out[n,0]=phi
    out[n,1]=computeVanLeer(x,vi,vj,gi,gj)
    out[n,2]=crkLimiter(x,a[n,2],a[n,3],wp.static(KernelFunctions.B8.value),2,crit,fold)
    out[n,3]=computeFrontiereQ(x,zero,a[n,2],a[n,3],a[n,18],a[n,19],dv,domain,
        wp.static(KernelFunctions.B8.value),a[n,20],a[n,21],scalar_t(1.),scalar_t(1.),p,False,a[n,4])
    out[n,4]=computeFrontiereQ(x,zero,a[n,2],a[n,3],a[n,18],a[n,19],dv,domain,
        wp.static(KernelFunctions.B8.value),a[n,20],a[n,21],scalar_t(1.),scalar_t(1.),p,True,a[n,4])
    out[n,5]=computeFrontiereQ(x,zero,a[n,2],a[n,3],a[n,18],a[n,19],dv,domain,
        wp.static(KernelFunctions.B8.value),a[n,20],a[n,21],scalar_t(1.),scalar_t(1.),p,False)
    out[n,6]=computeFrontiereQ(x,zero,a[n,2],a[n,3],a[n,18],a[n,19],dv,domain,
        wp.static(KernelFunctions.B8.value),a[n,20],a[n,21],scalar_t(1.),scalar_t(1.),p,True)
    out[n,7]=sphKernelScale(wp.static(KernelFunctions.B8.value),2)


def build(root):
    base=Path('results/materialEOS/spheral-build').resolve()
    source=Path('benchmarks/eos/paperSpheral/limiterAudit.cc').resolve()
    key=hashlib.sha256(source.read_bytes()).hexdigest()
    library=(root/('libLimiterAudit_'+key[:12]+'.so')).resolve()
    flags=(base/'compiled/build/src/CRKSPH/CMakeFiles/Spheral_CRKSPH.dir/flags.make').read_text().splitlines()
    options={l.split(' = ',1)[0]:shlex.split(l.split(' = ',1)[1])for l in flags if ' = 'in l}
    command=['/usr/bin/g++',*options['CXX_DEFINES'],*options['CXX_INCLUDES'],*options['CXX_FLAGS'],'-shared',str(source),
        '-L'+str(base/'install/lib'),'-lSpheral_CXX','-Wl,-rpath,'+str(base/'install/lib'),'-o',str(library)]
    if not library.exists():subprocess.run(command,check=True)
    record=dict(command=command,source=str(source),sourceSHA256=key,library=str(library),
        librarySHA256=hashlib.sha256(library.read_bytes()).hexdigest(),
        nativeLibrarySHA256=hashlib.sha256((base/'install/lib/libSpheral_CXX.so').read_bytes()).hexdigest())
    (root/'build.json').write_text(json.dumps(record,indent=2)+'\n')
    shutil.copy2(source,root/'limiterAudit.cc')
    return library


def inputs():
    rows=[];labels=[]
    def add(nh,r,gi,gj,label='ordinary',unit=1.):
        rows.append([r*unit,0.,unit,.7*unit,4.,nh,gi,0.,0.,0.,gj,0.,0.,0.,
                     -r*unit,0.,0.,0.,1.,.182,2.*unit,1.3*unit])
        labels.append(label)
    for nh in (3.,4.,5.4,6.):
        for f in (.0,.01,.1,.5,.75,.9,.99,1.-1.e-8,1.,1.+1.e-8,1.1,2.):
            if f==0.:continue  # zero separation has no force; handle slope degeneracy at nonzero separation
            for gi in (-10.,-1.,-.1,1.e-8,.1,1.,10.):
                for gj in (-10.,-1.,-.1,1.e-8,.1,1.,10.):add(nh,f/nh,gi,gj)
        for gi,gj in ((0.,0.),(0.,-1.),(-1.,0.),(-1.e-40,-1.e-40),(-1.e-40,1.e-40),(-1.e-28,-1.e-29)):
            add(nh,.35,gi,gj,'guard')
        for unit in (1.e-16,1.e-12,1.,1.e12):add(nh,.35,-1.,-1.,'unit-scale',unit)
    rng=np.random.default_rng(419)
    for _ in range(1000):
        nh=float(rng.choice([3.,4.,5.4,6.]));R=rng.uniform(.3,2.,2);r=rng.uniform(.02,1.1)*max(R);angle=rng.uniform(-np.pi,np.pi)
        rows.append([r*np.cos(angle),r*np.sin(angle),*R,4.,nh,*rng.normal(size=8),*rng.normal(size=4),1.,.182,2.,1.3]);labels.append('random')
    return np.array(rows,dtype=np.float64),np.array(labels)


def main(root,report):
    root.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    library=build(root);a,labels=inputs();native=np.empty((len(a),8))
    api=ctypes.CDLL(str(library));f=api.spheral_limiter_pairs
    pointer=ctypes.POINTER(ctypes.c_double);f.argtypes=[pointer,ctypes.c_int,pointer]
    # A FieldListView snapshots NodeList metadata. Use one native NodeList per
    # nPerh batch, matching an actual solver's constant nPerh setup.
    for nh in np.unique(a[:,5]):
        idx=np.flatnonzero(a[:,5]==nh);v=np.ascontiguousarray(a[idx]);out=np.empty((len(v),8))
        f(v.ctypes.data_as(pointer),len(v),out.ctypes.data_as(pointer));native[idx]=out
    domain=domainData();domain.dim=2
    domain.domainMin=wp.array([-1.e30,-1.e30],dtype=scalar_t,device='cpu')
    domain.domainMax=wp.array([1.e30,1.e30],dtype=scalar_t,device='cpu')
    domain.periodicity=wp.array([False,False],dtype=wp.bool,device='cpu')
    p=DiffusionParameters();p.C_l=2.;p.C_q=1.
    output=wp.empty((len(a),8),dtype=scalar_t,device='cpu')
    wp.launch(evaluate,len(a),inputs=[wp.array(a,dtype=scalar_t,device='cpu'),domain,p,output],device='cpu')
    warp=output.numpy();r=np.linalg.norm(a[:,:2],axis=1);eta=r/np.maximum(a[:,2],a[:,3]);nh=a[:,5];K=a[:,4]
    # Independent source-level paper limiter, including its literal fixed .2 h
    # close-pair width. Undefined/tiny slope ratios are kept out of conformance.
    half=.5*a[:,:2];Gi=a[:,6:10].reshape(-1,2,2);Gj=a[:,10:14].reshape(-1,2,2)
    si=np.einsum('ni,nij,nj->n',half,Gi,half);sj=np.einsum('ni,nij,nj->n',half,Gj,half)
    normal=(np.abs(si)>1.e-30)&(np.abs(sj)>1.e-30)
    ratio=np.ones(len(a));ratio[normal]=np.minimum(si[normal]/sj[normal],sj[normal]/si[normal])
    slope=np.zeros(len(a));positive=ratio>0;slope[positive]=4*ratio[positive]/(1+ratio[positive])**2
    paperFold=np.where(eta<1/nh,np.exp(-((eta-1/nh)/(.2/K))**2),1.)
    paperPhi=slope*paperFold
    np.testing.assert_allclose(warp[normal,0],native[normal,0],rtol=2.e-12,atol=2.e-14)
    np.testing.assert_allclose(warp[normal,3:5],native[normal,1:3],rtol=2.e-11,atol=1.e-11)
    mask=normal&(nh==4.)
    np.testing.assert_allclose(warp[mask,0],paperPhi[mask],rtol=2.e-12,atol=2.e-14)
    np.savez_compressed(root/'pairs.npz',input=a,labels=labels,warp=warp,native=native,paperPhi=paperPhi,normal=normal)
    records={}
    for label in np.unique(labels):
        mask=labels==label
        records[str(label)]=dict(count=int(mask.sum()),phiDifferenceMax=float(np.max(np.abs(warp[mask,0]-native[mask,0]))),
            QDifferenceMax=float(np.max(np.abs(warp[mask,3:5]-native[mask,1:3]))))
    summary=dict(pairs=len(a),normalPairs=int(normal.sum()),groups=records,
        ordinaryPhiDifferenceMax=float(np.max(np.abs(warp[normal,0]-native[normal,0]))),
        ordinaryQDifferenceMax=float(np.max(np.abs(warp[normal,3:5]-native[normal,1:3]))),
        normalQScaledDifferenceMax=float(np.max(np.abs(warp[normal,3:5]-native[normal,1:3])/(1+np.abs(native[normal,1:3])))),
        B8KernelScale=float(warp[0,7]),paperKernelExtent=4.,defaultToPaperHratio=float(4./warp[0,7]),
        foldExamples=[dict(nPerh=n,Rcrit=1/(4*n),warpSpheralFoldR=.2/(4*n),paperFoldR=.2/4,
            warpSpheralFactorHalfSpacing=float(np.exp(-6.25)),paperFactorHalfSpacing=float(np.exp(-(.5/(.2*n))**2)))for n in (.75,1.,1.35,1.5)],
        scope='Static CPU operator checks; no trajectory, time integration or default changes.')
    unitMask=(labels=='unit-scale')&(nh==4.)
    summary['unitScaleExamples']=[dict(supportR=float(row[2]),warpPhi=float(w[0]),nativePhi=float(s[0]),
        warpQiScaled=float(w[3]/row[2]**2),nativeQiScaled=float(s[1]/row[2]**2))
        for row,w,s in zip(a[unitMask],warp[unitMask],native[unitMask])]
    (report/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    z=np.linspace(0,1.2,400)
    axes[0].plot(z,np.where(z<1,np.exp(-((z-1)/.2)**2),1),label='Current Warp/Spheral',color='black')
    for n in (1.,1.35,1.5):
        axes[0].plot(z,np.where(z<1,np.exp(-((z-1)/(.2*n))**2),1),ls='--',label=f'Paper nPerh={n:g}')
    axes[0].set(xlabel='Separation / nominal spacing',ylabel='Close-pair factor multiplying reconstruction');axes[0].legend(fontsize=7)
    for nh in (4.,5.4,6.):
        mask=(labels=='ordinary')&(a[:,6]==-1)&(a[:,10]==-1)&(a[:,5]==nh)
        rr=np.linspace(0,.5,500);crit=1/nh
        line,=axes[1].plot(rr,np.where(rr<crit,np.exp(-((rr-crit)/(.2/nh))**2),1),label=f'Warp/Spheral n_h={nh:g}')
        axes[1].plot(rr,np.where(rr<crit,np.exp(-((rr-crit)/.05)**2),1),ls='--',color=line.get_color(),label=f'Paper n_h={nh:g}')
        axes[1].scatter(eta[mask],warp[mask,0],s=8,color=line.get_color())
    axes[1].set(xlabel='Separation / maximum support radius',ylabel='Full phi for equal compressive slopes');axes[1].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(report/'thresholds.png',dpi=180);plt.close(fig)
    for name in (__file__,'src/warpSPH/modules/reconstruction/limiters.py','src/warpSPH/configurations/crkSPH.py',
        'src/warpSPH/modules/dissipation/pi/terms.py'):
        shutil.copy2(name,root/Path(name).name)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/materialEOS/limiter_threshold_audit_2026-10-20'))
    parser.add_argument('--report',type=Path,default=Path('reports/materialEOS/spheral/limiterThresholds'))
    args=parser.parse_args();main(args.output.resolve(),args.report.resolve())
