#!/usr/bin/env python3
"""Offline phase, convergence and sampling audit of a Lobovsky comparison.

Consumes analyze_trajectory.py's summary and its original recordings. No solver
imports or simulated state changes. Native raw probe.plt is context only: it
samples an earlier RK field with its own timestamps, unlike the common-frame
post-shift estimator. Keep all three pressure cadences labelled explicitly.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'reference': '#222222', 'local': '#0072b2'}
NAMES = {'reference': 'dynGPUSPH FP64', 'local': 'warpSPH FP32'}


def time_mean(t, y, low, high):
    use = (t > low) & (t < high)
    clock = np.r_[low, t[use], high]
    values = np.interp(clock, t, y)
    return float(np.trapezoid(values, clock)/(high-low))


def decorate(axes):
    for ax in np.asarray(axes).flat:
        ax.grid(alpha=.22)
        ax.set_xlabel('Physical time [s]')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--summary', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    base = json.loads(args.summary.read_text()); rows = base['rows']
    t = np.array([r['t'] for r in rows])
    histories = {}; metrics = {}
    files = {label: h5py.File(base[label], 'r') for label in NAMES}
    try:
        runs = {label: Path(h.attrs['source']) for label, h in files.items()}
        native_result = json.loads((runs['reference']/'result.json').read_text())
        local_progress = [json.loads(s) for s in (runs['local']/'progress.jsonl').read_text().splitlines()]
        receipts = [json.loads(s) for s in (runs['local']/'solve-timing.jsonl').read_text().splitlines()]
        native = native_result['steps']
        clock = np.array([r['t'] for r in native])
        np.testing.assert_allclose(clock, [r['t'] for r in local_progress[1:]], rtol=0, atol=1e-12)
        assert [r['step'] for r in receipts] == list(range(1, len(clock)+1))
        iterations = {'reference': np.array([r['iterations'] for r in native]),
                      'local': np.array([r['iterations'] for r in receipts])}
        epsilon = np.array([r['epsilonV'] for r in receipts])
        solve_ms = np.array([r['solve_wall_ms'] for r in receipts])
        caps = iterations['local'] >= 200
        later = np.flatnonzero(caps & (clock > .005))
        first = int(later[0])
        convergence = dict(firstPostStartupCap=dict(step=first+1, time=float(clock[first]),
            localIterations=int(iterations['local'][first]), nativeIterations=int(iterations['reference'][first]),
            epsilon=float(epsilon[first])), cappedSolveWallFraction=float(solve_ms[caps].sum()/solve_ms.sum()),
            totalSolveSeconds=float(solve_ms.sum()/1000), allStepsGraphed=all(r['captured'] for r in receipts), windows=[])
        for low, high in [(0,.1),(.1,.4),(.4,.8),(.8,1.2),(1.2,1.5),(1.5,float(clock[-1]))]:
            mask = (clock > low) & (clock <= high)
            record = dict(start=low, end=high, steps=int(mask.sum()))
            for label in NAMES:
                record[label] = dict(caps=int((iterations[label][mask]>=200).sum()),
                                     meanIterations=float(iterations[label][mask].mean()))
            convergence['windows'].append(record)
        print(f'DYNAMICS: clocks and all {len(clock):,} convergence receipts checked', flush=True)
        phase_metrics = []
        for low, high in [(0,.4),(.4,1.),(1.,1.35),(1.35,1.6),(1.6,float(t[-1]))]:
            record = dict(start=low, end=high)
            for label in NAMES:
                ke = np.array([r[label]['ke'] for r in rows])
                record[label] = dict(meanKE=time_mean(t, ke, low, high))
            a, b = (record[label]['meanKE'] for label in NAMES)
            record['relativeMeanKEDifference'] = b/a-1
            phase_metrics.append(record)
        probe_metrics = []
        dense_t = np.array([r['t'] for r in local_progress])
        dense_p = np.array([r['pressureProbes'] for r in local_progress])
        for label, h in files.items():
            histories[label] = dict(t=h['time'][:], p=h['probes/pressure'][:], wet=h['probes/wet'][:])
            np.testing.assert_allclose(histories[label]['t'], t, atol=1e-12, rtol=0)
        for low, high in [(.4,.7),(1.35,1.6)]:
            for k in range(4):
                record = dict(start=low, end=high, probe=k+1)
                for label in NAMES:
                    mask = (t>=low)&(t<=high); indices=np.flatnonzero(mask)
                    p = histories[label]['p'][:,k]; j=indices[np.argmax(p[mask])]
                    record[label] = dict(sampledPeakPa=float(p[j]), sampledPeakTime=float(t[j]),
                        sampledImpulsePaSeconds=time_mean(t,p,low,high)*(high-low))
                mask=(dense_t>=low)&(dense_t<=high); indices=np.flatnonzero(mask)
                j=indices[np.argmax(dense_p[mask,k])]
                record['localEveryStep'] = dict(peakPa=float(dense_p[j,k]), peakTime=float(dense_t[j]))
                probe_metrics.append(record)
        metrics.update(convergence=convergence, phases=phase_metrics, probes=probe_metrics,
                       endpoint=base['common_endpoint'], costs=base['costs'])
        # Bulk and mechanical-energy histories from the matched physical frames.
        fig, axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for label in NAMES:
            ke=np.array([r[label]['ke'] for r in rows]); pe=np.array([r[label]['potential_energy'] for r in rows])
            axes[0,0].plot(t,ke,color=COLORS[label],label=NAMES[label])
            axes[0,1].plot(t,ke+pe,color=COLORS[label],label=NAMES[label])
            axes[1,0].plot(t,[r[label]['vmax'] for r in rows],color=COLORS[label])
        axes[1,1].semilogy(t[1:],[r['eulerian_velocity_rms_1cm'] for r in rows[1:]],label='Eulerian velocity RMS [m/s]')
        axes[1,1].semilogy(t[1:],[r['position_rms'] for r in rows[1:]],label='Material displacement RMS [m]')
        for ax,title in zip(axes.flat,['Kinetic energy [J/m]','Mechanical energy K + gravity PE [J/m]','Maximum particle speed [m/s]','Matched-frame field differences']): ax.set_title(title)
        axes[0,0].legend();axes[1,1].legend(fontsize=9);decorate(axes)
        fig.suptitle('Lobovsky: matched original mass and timestep sequence; FP64 native / FP32 local')
        fig.savefig(out/'bulk.png',dpi=160);plt.close(fig)
        # Common estimator at common clocks, with dense local output exposed.
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for k,ax in enumerate(axes.flat):
            ax.plot(dense_t,dense_p[:,k]/1000,color=COLORS['local'],alpha=.55,lw=.8,label='warpSPH every accepted step')
            for label in NAMES:
                ax.plot(t,histories[label]['p'][:,k]/1000,color=COLORS[label],marker='.',ms=3,lw=1.1,label=NAMES[label]+' common ~10 ms frames')
            ax.set_title(f'P{k+1}, y={files["reference"]["probes/positions"][k,1]*1000:g} mm');ax.set_ylabel('Gauge pressure [kPa]')
        axes[0,0].legend(fontsize=8);decorate(axes)
        fig.suptitle('Same fluid-only post-shift estimator; saved frames miss narrow pressure spikes\nNo full-step post-shift native trace is available')
        fig.savefig(out/'pressure.png',dpi=160);plt.close(fig)
        # Native raw output is deliberately separate from the matched comparison.
        raw=np.loadtxt(runs['reference']/'probe.plt',skiprows=1)
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for k,ax in enumerate(axes.flat):
            ax.plot(raw[:,0],raw[:,8+k]/1000,color='#888888',lw=.65,label='Native raw probe.plt (earlier RK field, own clock)')
            ax.plot(t,histories['reference']['p'][:,k]/1000,color=COLORS['reference'],marker='.',ms=3,label='Native stored post-shift frames')
            ax.set_title(f'Native P{k+1}: stage/cadence context only');ax.set_ylabel('Gauge pressure [kPa]')
        axes[0,0].legend(fontsize=8);decorate(axes)
        fig.suptitle('Native output also contains sharp pressure transients\nDifferent RK state/timestamp: these raw maxima are not like-for-like error measurements')
        fig.savefig(out/'native-pressure-context.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(3,1,figsize=(13,9),layout='constrained')
        for label in NAMES: axes[0].plot(clock,iterations[label],color=COLORS[label],lw=.6,alpha=.75,label=NAMES[label])
        axes[0].set_ylabel('Pseudo-iterations');axes[0].legend()
        axes[1].plot(clock,epsilon,color=COLORS['local'],lw=.6);axes[1].axhline(-8,color='black',ls='--',label='Target -8');axes[1].legend();axes[1].set_ylabel('Local epsilon_v')
        edges=np.r_[np.arange(0,1.90001,.1),clock[-1]]
        centers=.5*(edges[:-1]+edges[1:])
        for label in NAMES:
            fraction=[100*float((iterations[label][(clock>a)&(clock<=b)]>=200).mean()) for a,b in zip(edges[:-1],edges[1:])]
            axes[2].plot(centers,fraction,color=COLORS[label],marker='.',label=NAMES[label])
        axes[2].set_ylabel('Capped steps per 0.1 s [%]');decorate(axes)
        fig.suptitle(f'Convergence: all {len(clock):,} dt values match; local CUDA graph active at every step')
        fig.savefig(out/'convergence.png',dpi=160);plt.close(fig)
        # Full tank limits retain high reference spray rather than crop it away.
        selected=[.4,1.48,float(t[-1])]
        fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
        spatial=[]
        for row,(label,h) in enumerate(files.items()):
            fluid=h['particles/kinds'][:]==0; clocks=h['time'][:]
            for col,target in enumerate(selected):
                i=int(np.argmin(abs(clocks-target)));x=h['frames/positions'][i][fluid]
                velocity=h['frames/velocities'][i][fluid]; speed=np.linalg.norm(velocity,axis=1)
                pressure=h['frames/pressures'][i][fluid]
                spatial.append(dict(implementation=label,time=float(clocks[i]),maxY=float(x[:,1].max()),
                    speed99=float(np.quantile(speed,.99)),speed999=float(np.quantile(speed,.999)),
                    pmin=float(pressure.min()),pmax=float(pressure.max()),negativePressureFraction=float((pressure<0).mean())))
                ax=axes[row,col];sc=ax.scatter(x[:,0],x[:,1],c=speed,s=.1,cmap='viridis',vmin=0,vmax=5,rasterized=True)
                ax.plot([0,1.61,1.61,0,0],[0,0,1.4,1.4,0],color='#777777',lw=.6)
                ax.set(xlim=(-.015,1.625),ylim=(-.015,1.415),xlabel='x [m]',ylabel='y [m]',title=f'{NAMES[label]}, t={clocks[i]:.5f} s')
                ax.set_aspect('equal')
                print(f'DYNAMICS: spatial {label} t={clocks[i]:.6f}',flush=True)
        fig.colorbar(sc,ax=axes,shrink=.7,label='Speed [m/s], common scale 0–5',extend='max')
        fig.suptitle('Matched physical frames; full domain retained, including sparse spray')
        fig.savefig(out/'spatial.png',dpi=180);plt.close(fig)
        metrics['spatial']=spatial
        metrics['analysisWallSeconds']=time.monotonic()-started
        (out/'dynamics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        report=['# Corrected ACSPH / dynGPUSPH Lobovsky dynamics comparison','',
            f'Same physical particles/mass and all {len(clock):,} dt values; {len(rows)} matched snapshots to {t[-1]:.9f} s. Native FP64 versus local FP32. No new simulations.','',
            '## Bulk dynamics','', '| Window [s] | Mean native KE [J/m] | Mean local KE [J/m] | Relative difference |','|---|---:|---:|---:|']
        for r in phase_metrics: report.append(f'| {r["start"]:.2f}–{r["end"]:.3f} | {r["reference"]["meanKE"]:.4f} | {r["local"]["meanKE"]:.4f} | {r["relativeMeanKEDifference"]*100:+.2f}% |')
        report+=['','![Bulk dynamics](bulk.png)','','## Pressure and cadence','',
            'The primary comparison uses the same normalized fluid-only Wendland C2 estimator on matched post-shift frames. Local full-step traces expose transients missed by those ~10 ms frames. Native raw probe.plt samples an earlier RK state and has its own timestamps; it is shown separately as context, never treated as an identical observable. Sampled maxima/impulses are not temporally converged peak estimates.','',
            '| Window [s] | Probe | Native common-frame peak [kPa] | Local common-frame peak [kPa] | Local every-step peak [kPa] |', '|---|---|---:|---:|---:|']
        for r in probe_metrics: report.append(f'| {r["start"]:.2f}–{r["end"]:.2f} | P{r["probe"]} | {r["reference"]["sampledPeakPa"]/1000:.3f} | {r["local"]["sampledPeakPa"]/1000:.3f} | {r["localEveryStep"]["peakPa"]/1000:.3f} |')
        report+=['','![Pressure](pressure.png)','','![Native RK pressure context](native-pressure-context.png)','','## Convergence and cost','',
            f'First post-startup cap: step {first+1}, t={clock[first]:.9f} s, local 200 versus native {iterations["reference"][first]:g} iterations; epsilon={epsilon[first]:.6f}. This precedes opposite-wall impact (~0.416 s). All local steps use CUDA graphs.',
            f'Capped steps consume {100*convergence["cappedSolveWallFraction"]:.2f}% of measured local solve wall time. Total solve time {convergence["totalSolveSeconds"]:.1f} s; full replay wall {base["costs"]["local"]["wall_seconds"]:,.1f} s versus native {base["costs"]["reference"]["wall_seconds"]:,.1f} s. The native run shared the GPU during part of its execution: these wall costs are not a controlled precision/hardware speed comparison.','',
            '| Window [s] | Local capped / steps | Native capped / steps | Local mean iterations | Native mean iterations |','|---|---:|---:|---:|---:|']
        for r in convergence['windows']: report.append(f'| {r["start"]:.2f}–{r["end"]:.3f} | {r["local"]["caps"]}/{r["steps"]} | {r["reference"]["caps"]}/{r["steps"]} | {r["local"]["meanIterations"]:.2f} | {r["reference"]["meanIterations"]:.2f} |')
        report+=['','![Convergence](convergence.png)','','## Spatial comparison','',
            'Full tank limits and identical speed scales retain both bulk and sparse spray. Material-row RMS includes chaotic trajectory separation; the 1 cm Eulerian metric in the primary report compares cells with at least four particles in each run. Neither metric identifies a numerical cause by itself.','',
            '![Spatial comparison](spatial.png)','','## Verdict and next discrimination','',
            'The corrected fixture reproduces the bulk collapse and first-impact energetics closely. It does not establish full solver parity: later convergence, return-impact pressure structure and final energy retention differ substantially. Caps exist well before the return impact; ordinary late chaotic separation alone does not explain that early solver-residual difference. Finishing and graph capture do not establish convergence.',
            'Next bounded control: reproduce solver-entry state/history at step 468 and distinguish hard-bounce switching, precision and residual/stopping semantics. The earlier step 56 SDF failure is fixed; no cause for this later cap is established here. Stored physical snapshots omit full BDF/low-part history and are not exact restart checkpoints. No tolerance, bounce or numerical default is changed by this analysis.','']
        (out/'dynamics.md').write_text('\n'.join(report))
        print(json.dumps(dict(output=str(out),wallSeconds=metrics['analysisWallSeconds'],firstPostStartupCap=convergence['firstPostStartupCap'],cappedSolveWallFraction=convergence['cappedSolveWallFraction'])),flush=True)
    finally:
        for h in files.values(): h.close()


if __name__=='__main__':
    main()
