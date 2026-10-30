#!/usr/bin/env python3
"""Evaluate the native wall-collision displacement on retained physical frames.

Only the collision term of getContribShift3/shiftandInterpolateD is evaluated.
This is not a native step replay, a complete shift, or an evolved remedy trial.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial import cKDTree


def collision_shift(x, mass, wall_x, wall_mass, dx):
    radius=.9*dx
    pairs=cKDTree(x).sparse_distance_matrix(cKDTree(wall_x),radius,output_type='coo_matrix')
    i,j,d=pairs.row,pairs.col,pairs.data
    if np.any(d==0):
        raise ValueError('Coincident fluid/wall rows: native formula divides by zero')
    mask=d<radius;i,j,d=i[mask],j[mask],d[mask]
    denominator=mass.copy()
    np.add.at(denominator,i,wall_mass[j])
    numerator=np.zeros_like(x)
    np.add.at(numerator,i,wall_mass[j,None]*(d-radius)[:,None]*(x[i]-wall_x[j])/d[:,None])
    return -numerator/denominator[:,None], len(i)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--trial',type=Path,help='Completed collision-only diagnostic run')
    p.add_argument('--native',type=Path,help='Native H5 for common-clock trial comparison')
    a=p.parse_args();dx=.00075
    # Exact equal-mass one-pair oracle: half the separation deficit, outward.
    trial,_=collision_shift(np.array([[0.,.75]]),np.ones(1),np.zeros((1,2)),np.ones(1),1.)
    np.testing.assert_allclose(trial,[[0.,.075]],atol=1e-15,rtol=0)
    records=[]
    with h5py.File(a.recording) as h:
        clock=h['time'][:];f=h['particles/kinds'][:]==0;mass=h['particles/masses'][:]
        wall_x=h['frames/positions'][0][~f];wall_mass=mass[~f]
        for target in [.01,.02,.05,.1,.4,1.3,1.9184199183159294]:
            k=int(np.argmin(abs(clock-target)));x=h['frames/positions'][k][f]
            delta,pairs=collision_shift(x,mass[f],wall_x,wall_mass,dx)
            magnitude=np.linalg.norm(delta,axis=1)/dx
            signed=np.column_stack((x[:,0],1.61-x[:,0],x[:,1],1.4-x[:,1])).min(1)/dx
            # Most-negative signed wall distance fixes each local inward normal.
            face=np.column_stack((x[:,0],1.61-x[:,0],x[:,1],1.4-x[:,1])).argmin(1)
            normals=np.array([[1,0],[-1,0],[0,1],[0,-1]])[face]
            outward=np.einsum('ij,ij->i',delta,normals)/dx
            affected=magnitude>1e-12;shallow=(signed<-.05)&(signed>-.4)
            row=dict(t=float(clock[k]),pairs=pairs,affectedRows=int(affected.sum()),
                maxCollisionDisplacementDx=float(magnitude.max()),
                affectedP50Dx=float(np.median(magnitude[affected])) if affected.any() else 0,
                shallowRows=int(shallow.sum()),shallowOutwardRows=int((shallow&(outward>1e-12)).sum()),
                shallowMeanOutwardDx=float(outward[shallow].mean()) if shallow.any() else 0)
            records.append(row);print(json.dumps(row),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(recording=str(a.recording.resolve()),
        formula='-sum_b m_b (r_ib - 0.9 dx) (x_i-x_b)/r_ib / (m_i+sum_b m_b), r_ib<0.9 dx',
        onePairOracle='passed',scope='same recorded input collision term only, no field remap or physical advance',
        frames=records)
    if a.trial:
        if not a.native:p.error('--trial requires --native')
        from render import read_snapshot
        comparisons=[]
        def stats(x,v,p,m):
            d=np.column_stack((x[:,0],1.61-x[:,0],x[:,1],1.4-x[:,1])).min(1)/dx
            return dict(minSignedDistanceDx=float(d.min()),deeperThanOneDx=int((d<-1).sum()),
                ke=float(.5*np.sum(m*np.sum(v*v,axis=1))),meanPressure=float(np.average(p,weights=m)))
        with h5py.File(a.recording) as baseline,h5py.File(a.native) as native:
            f=native['particles/kinds'][:]==0;clock=native['time'][:]
            np.testing.assert_array_equal(baseline['time'],clock)
            for path in sorted((a.trial/'snapshots').glob('*.bin')):
                sample=read_snapshot(path);frame=int(np.argmin(abs(clock-sample['t'])))
                if abs(clock[frame]-sample['t'])>1e-12:continue
                np.testing.assert_array_equal(sample['bc']==1,f)
                row=dict(t=float(sample['t']))
                native_p=native['frames/pressures'][frame][f]
                for label,h in [('native',native),('ramp',baseline)]:
                    x=h['frames/positions'][frame][f];v=h['frames/velocities'][frame][f];p=h['frames/pressures'][frame][f];m=h['particles/masses'][:][f]
                    row[label]=stats(x,v,p,m);row[label]['pressureRMSvsNativePa']=float(np.mean((p-native_p)**2)**.5)
                x,v,p=sample['x'][f],sample['v'][f],sample['p'][f];m=(sample['rho']*sample['volume'])[f]
                assert all(np.isfinite(a).all() for a in (x,v,p))
                row['collision']=stats(x,v,p,m);row['collision']['pressureRMSvsNativePa']=float(np.mean((p-native_p)**2)**.5)
                comparisons.append(row)
        result=json.loads((a.trial/'result.json').read_text())
        report['evolvedTrial']=dict(source=str(a.trial.resolve()),steps=result['nSteps'],
            wallSeconds=result['wall_seconds'],msPerStep=1000*result['wall_seconds']/result['nSteps'],
            stopReason=result['stop_reason'],diverged=result['diverged'],commonFrames=comparisons)
        print('EVOLVED TRIAL',json.dumps(report['evolvedTrial']),flush=True)
    a.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
