#!/usr/bin/env python3
"""Three-trajectory, offline Lobovsky D5 qualification; no solver imports.

Geometry statistics use recorded frames, not unobserved substeps. Physical rho
in the recording is constant and is not a sampling-density estimator.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import h5py
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'native': '#222222', 'hard': '#d55e00', 'ramp': '#0072b2'}
LABELS = {'native': 'dynGPUSPH FP64', 'hard': 'warpSPH hard', 'ramp': 'warpSPH ramp 0.5'}
DX = .00075


def mean_window(t, y, lo, hi):
    inside = (t > lo) & (t < hi)
    clock = np.r_[lo, t[inside], hi]
    return float(np.trapezoid(np.interp(clock, t, y), clock)/(hi-lo))


def groups(mask, clock):
    indices = np.flatnonzero(mask)
    blocks = np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1) if len(indices) else []
    return [dict(first=int(g[0]+1), last=int(g[-1]+1), count=len(g),
                 start=float(clock[g[0]]), end=float(clock[g[-1]])) for g in blocks]


def analyze(paths, out):
    started = time.monotonic(); out.mkdir(parents=True, exist_ok=True)
    data = {}; costs = {}; caps = {}; dense = {}; endpoint = {}
    for name, path in paths.items():
        with h5py.File(path) as h:
            assert h.attrs['complete']
            clock = h['time'][:]
            if data:
                np.testing.assert_allclose(clock, data['native']['t'], atol=1e-12, rtol=0)
                np.testing.assert_array_equal(h['particles/UIDs'], uids)
                np.testing.assert_array_equal(h['particles/kinds'], kinds)
            else:
                uids = h['particles/UIDs'][:]; kinds = h['particles/kinds'][:]
            fluid = kinds == 0
            mass = h['particles/masses'][:][fluid]
            assert np.isfinite(mass).all() and (mass>0).all()
            source = Path(h.attrs['source'])
            result = json.loads((source/'result.json').read_text())
            if name == 'native':
                records = result['steps']; its = np.array([r['iterations'] for r in records])
                real_clock = np.array([r['t'] for r in records]); solve_ms = None
            else:
                records = [json.loads(s) for s in (source/'solve-timing.jsonl').read_text().splitlines()]
                progress = [json.loads(s) for s in (source/'progress.jsonl').read_text().splitlines()]
                assert [r['step'] for r in records] == list(range(1,10001))
                assert len(progress) == 10001
                its = np.array([r['iterations'] for r in records]); real_clock = np.array([r['t'] for r in progress[1:]])
                solve_ms = np.array([r['solve_wall_ms'] for r in records])
                dense[name] = dict(t=[r['t'] for r in progress], probes=[r['pressureProbes'] for r in progress])
                np.testing.assert_allclose(real_clock, native_clock, atol=1e-12, rtol=0)
                assert all(r['captured'] for r in records)
            if name == 'native': native_clock = real_clock.copy()
            cap = its >= 200
            costs[name] = dict(wallSeconds=result['wall_seconds'], msPerStep=result['wall_seconds']/10,
                medianIterations=float(np.median(its)), meanIterations=float(its.mean()), caps=int(cap.sum()),
                solveSeconds=float(solve_ms.sum()/1000) if solve_ms is not None else None,
                capSolveFraction=float(solve_ms[cap].sum()/solve_ms.sum()) if solve_ms is not None else None)
            cap_groups = groups(cap & (real_clock > .005), real_clock)
            caps[name] = dict(groups=cap_groups, first=cap_groups[0] if cap_groups else None,
                longest=max(cap_groups, key=lambda g:g['count']) if cap_groups else None,
                windows=[dict(start=lo, end=hi, count=int((cap & (real_clock>lo) & (real_clock<=hi)).sum()))
                         for lo,hi in [(0,.005),(.005,.4),(.4,.8),(.8,1.2),(1.2,1.5),(1.5,float(real_clock[-1]))]])
            rows=[]; seen=np.zeros(fluid.sum(),dtype=bool); endpoint[name]={}
            for frame,t in enumerate(clock):
                x=h['frames/positions'][frame][fluid]; v=h['frames/velocities'][frame][fluid]
                p=h['frames/pressures'][frame][fluid]
                assert all(np.isfinite(a).all() for a in (x,v,p)), (name,frame)
                distances=np.column_stack((x[:,0],1.61-x[:,0],x[:,1],1.4-x[:,1]))
                d=distances.min(axis=1)/DX
                speed=np.linalg.norm(v,axis=1); energy=.5*mass*speed**2
                outside=d<0;seen |= outside
                fastest=speed>=np.quantile(speed,.99)
                row=dict(t=float(t), ke=float(energy.sum()), pe=float(9.81*np.dot(mass,x[:,1])),
                    pmean=float(np.average(p,weights=mass)), vmax=float(speed.max()),
                    v99=float(np.quantile(speed,.99)), minDistanceDx=float(d.min()),
                    penetrationCount=int(outside.sum()), penetrationBeyondQuarterDx=int((d<-.25).sum()),
                    penetrationBeyondOneDx=int((d<-1).sum()), everPenetratedCount=int(seen.sum()),
                    wallBandKE=float(energy[d<1].sum()), wallBandMassFraction=float(mass[d<1].sum()/mass.sum()),
                    fastestOnePercentKEFraction=float(energy[fastest].sum()/energy.sum()) if energy.sum() else 0.)
                rows.append(row)
                if frame == len(clock)-1:
                    endpoint[name] = dict(speedQuantiles=np.quantile(speed,[.5,.9,.95,.99,.999,1]).tolist(),
                        mass=float(mass.sum()), everPenetratedCount=int(seen.sum()),
                        maxRecordedPenetrationDx=float(max(0,-min(r['minDistanceDx'] for r in rows))),
                        maxRecordedPenetratingCount=max(r['penetrationCount'] for r in rows))
                    all_x=h['frames/positions'][frame]
                    tree=cKDTree(all_x); physical_fluid=np.flatnonzero(fluid)
                    selected=np.unique(np.r_[np.argsort(d)[:3],np.argsort(speed)[-10:]])
                    neighborhoods=tree.query_ball_point(x[selected],4*DX)
                    all_mass=h['particles/masses'][:]
                    entries=[]
                    for particle,neighbors in zip(selected,neighborhoods):
                        neighbors=np.array(neighbors)
                        q=np.linalg.norm(all_x[neighbors]-x[particle],axis=1)/(4*DX)
                        weights=7/(np.pi*(4*DX)**2)*np.maximum(1-q,0)**4*(1+4*q)*all_mass[neighbors]/1000
                        ff=fluid[neighbors]
                        entries.append(dict(uid=int(uids[physical_fluid[particle]]),position=x[particle].tolist(),
                            velocity=v[particle].tolist(),pressure=float(p[particle]),
                            fluidNeighborsIncludingSelf=int(ff.sum()),wallNeighbors=int((~ff).sum()),
                            fluidKernelVolume=float(weights[ff].sum()),totalKernelVolume=float(weights.sum())))
                    endpoint[name]['selectedNeighborhoods']=entries
                if frame % 50 == 0: print('FRAMES',name,frame,'/',len(clock),flush=True)
            data[name]=dict(t=clock.tolist(),rows=rows,probes=h['probes/pressure'][:].tolist(),
                realClock=real_clock.tolist(),iterations=its.tolist())
    clock=np.array(data['native']['t']); phases=[]; pressure=[]
    for lo,hi in [(0,.4),(.4,1.),(1.,1.25),(1.25,1.35),(1.35,1.6),(1.6,float(clock[-1]))]:
        phase=dict(start=lo,end=hi)
        for name in paths:
            rows=data[name]['rows'];ke=np.array([r['ke'] for r in rows]);pe=np.array([r['pe'] for r in rows])
            phase[name]=dict(meanKE=mean_window(clock,ke,lo,hi), meanMechanical=mean_window(clock,ke+pe,lo,hi))
        phases.append(phase)
    for lo,hi in [(.4,.7),(1.25,1.6)]:
        for k in range(4):
            row=dict(start=lo,end=hi,probe=k+1)
            for name in paths:
                p=np.array(data[name]['probes'])[:,k];mask=(clock>=lo)&(clock<=hi);indices=np.flatnonzero(mask)
                j=indices[np.argmax(p[mask])]
                row[name]=dict(commonPeakPa=float(p[j]),commonPeakTime=float(clock[j]),
                    commonImpulsePaSeconds=mean_window(clock,p,lo,hi)*(hi-lo))
                if name in dense:
                    t=np.array(dense[name]['t']);p=np.array(dense[name]['probes'])[:,k];mask=(t>=lo)&(t<=hi);indices=np.flatnonzero(mask)
                    j=indices[np.argmax(p[mask])]
                    row[name].update(everyStepPeakPa=float(p[j]),everyStepPeakTime=float(t[j]),
                        everyStepImpulsePaSeconds=mean_window(t,p,lo,hi)*(hi-lo))
            pressure.append(row)
    metrics=dict(costs=costs, caps=caps, phases=phases, pressure=pressure, geometry=endpoint,
                 endpoint={name:data[name]['rows'][-1] for name in paths}, sources={k:str(v) for k,v in paths.items()})
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (out/'frame-metrics.json').write_text(json.dumps(data,indent=2)+'\n')
    fig,axes=plt.subplots(3,2,figsize=(13,11),layout='constrained')
    for name in paths:
        rows=data[name]['rows'];ke=np.array([r['ke'] for r in rows]);pe=np.array([r['pe'] for r in rows])
        for ax,y,label in [(axes[0,0],ke,'Kinetic energy [J/m]'),(axes[0,1],ke+pe,'K + gravity PE [J/m]'),
                (axes[1,0],[r['v99'] for r in rows],'99th-percentile speed [m/s]'),
                (axes[1,1],[r['minDistanceDx'] for r in rows],'Minimum signed wall distance [dx]'),
                (axes[2,0],[r['penetrationCount'] for r in rows],'Fluid centres outside tank (recorded)'),
                (axes[2,1],[r['wallBandKE'] for r in rows],'KE within 1 dx of/outside wall [J/m]')]:
            ax.plot(clock,y,label=LABELS[name],color=COLORS[name]);ax.set_ylabel(label)
    axes[0,0].legend()
    for ax in axes.flat:ax.set_xlabel('Physical time [s]');ax.grid(alpha=.25)
    fig.savefig(out/'bulk-wall.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,9),layout='constrained')
    for name in paths:
        t=np.array(data[name]['realClock']);it=np.array(data[name]['iterations']);mask=it>=200
        axes[0].scatter(t[mask],np.full(mask.sum(),list(paths).index(name)),s=5,label=LABELS[name],color=COLORS[name])
        for k,ax in [(0,axes[1]),(3,axes[2])]:
            ax.plot(clock,np.array(data[name]['probes'])[:,k]/1000,label=LABELS[name],color=COLORS[name])
            ax.set_ylabel(f'Post-shift P{k+1} [kPa]')
    axes[0].set_yticks(range(3),[LABELS[n] for n in paths]);axes[0].set_ylabel('Steps at 200 cap')
    axes[1].legend()
    for ax in axes:ax.set_xlabel('Physical time [s]');ax.grid(alpha=.25)
    fig.savefig(out/'caps-pressure.png',dpi=150);plt.close(fig)
    selected=[.42,1.30,1.50,float(clock[-1])]
    fig,axes=plt.subplots(3,4,figsize=(17,8),layout='constrained')
    for row,(name,path) in enumerate(paths.items()):
        with h5py.File(path) as h:
            f=h['particles/kinds'][:]==0
            for col,target in enumerate(selected):
                frame=int(np.argmin(abs(clock-target)));x=h['frames/positions'][frame][f];v=h['frames/velocities'][frame][f]
                a=axes[row,col];sc=a.scatter(x[:,0],x[:,1],c=np.linalg.norm(v,axis=1),s=.15,vmin=0,vmax=4,rasterized=True)
                a.set_xlim(-.003,1.613);a.set_ylim(-.005,.66);a.set_aspect('equal');a.set_title(f'{LABELS[name]} t={clock[frame]:.3f}')
    fig.colorbar(sc,ax=axes,label='Speed [m/s]',shrink=.6);fig.savefig(out/'spatial.png',dpi=180);plt.close(fig)
    print('OFFLINE COMPLETE',out,'wall',time.monotonic()-started,flush=True)
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('native','hard','ramp','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();analyze({k:getattr(a,k).resolve() for k in ('native','hard','ramp')},a.out.resolve())


if __name__=='__main__':main()
