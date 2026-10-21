"""Independent NumPy/KD-tree audit of the compiled paper density/volume snapshot."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def w7(q):
    q=np.asarray(q)
    return 2268/(1487*np.pi)/math.factorial(7)*sum((-1)**k*math.comb(8,k)*np.maximum(4-k-q,0)**7 for k in range(5))


def audit(folder,rhs=False):
    a=np.load(folder/('firstRHSAudit.npz' if rhs else 'firstDensityAudit.npz'))['rows'];x=a[:,:2];m,V=a[:,2],a[:,3];H=a[:,4:7];rho=a[:,7];C=a[:,8:11];material=a[:,11];real=a[:,12]>0
    assert np.max(abs(H[:,1]))<1e-12 and np.max(abs(H[:,0]-H[:,2]))<1e-12
    invh=H[:,0];realIndex=np.flatnonzero(real)
    # Cover both interface and uniform regions without an O(N^2) dense matrix.
    if len(realIndex)>4096:
        shell=realIndex[abs(np.linalg.norm(x[realIndex]-[5,5],axis=1)-1)<.2]
        realIndex=np.unique(np.r_[realIndex[::max(1,len(realIndex)//2048)],shell[::max(1,len(shell)//2048)]])
    tree=cKDTree(x);neighbors=tree.query_ball_point(x[realIndex],float(4/invh.min())*(1+1e-12))
    volumeError=[];densityError=[];constantError=[];linearError=[]
    for i,js in zip(realIndex,neighbors):
        js=np.asarray(js);r=x[i]-x[js];distance=np.linalg.norm(r,axis=1)
        gather=w7(distance*invh[i])*invh[i]**2
        vref=1/gather.sum();volumeError.append(abs(vref/V[i]-1))
        scatter=w7(distance*invh[js])*invh[js]**2
        corrected=(C[i,0]+r@C[i,1:])*scatter
        mij=np.where(material[js]==material[i],m[js],m[i])
        rref=np.sum(mij*V[js]*corrected)/np.sum(V[js]**2*corrected)
        if not rhs:densityError.append(abs(rref/rho[i]-1))
        constantError.append(abs(np.sum(V[js]*corrected)-1))
        linearError.append(float(np.linalg.norm(np.sum((V[js]*corrected)[:,None]*r,axis=0))))
    result=dict(rows=len(a),checkedRealPoints=len(realIndex),method='Independent KD-tree candidate search, analytic B7 basis and direct NumPy Eq75/Eq76/constant-linear sums; Spheral RK coefficients from captured RHS.',
        phase='interim RHS (density is continuity prediction)' if rhs else 'primary density replacement',
        maxVolumeRelativeError=float(max(volumeError)),maxDensityRelativeError=float(max(densityError)) if densityError else None,
        maxConstantError=float(max(constantError)),maxLinearError=float(max(linearError)),
        tolerance=2e-5,limits='Analytic spline compared to native 1000-entry interpolated kernel table; verifies formulas, not nonlinear solution accuracy.')
    result['passed']=all(result[k] is None or result[k]<result['tolerance'] for k in ('maxVolumeRelativeError','maxDensityRelativeError','maxConstantError','maxLinearError'))
    (folder/('operatorAuditRHS.json' if rhs else 'operatorAudit.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not result['passed']:raise RuntimeError('Paper operator audit failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--rhs',action='store_true');a=p.parse_args();audit(a.input,a.rhs)
