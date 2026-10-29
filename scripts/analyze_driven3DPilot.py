#!/usr/bin/env python3
"""CPU-only endpoint, recording, provenance and performance audit of 3D pilots."""
from pathlib import Path
import argparse, hashlib, json, subprocess
import numpy as np


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root', type=Path);args=ap.parse_args()
    root=args.root.resolve();manifest=json.loads((root/'manifest.json').read_text())
    output=root/'analysis';output.mkdir(exist_ok=True)
    reports=[];configs=[];inventories=[];knots=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for job in manifest['jobs']:
        if not job.get('result'):continue
        run=Path(job['result']).parent
        if not (run/'result.json').exists():continue
        result=json.loads((run/'result.json').read_text())
        if result.get('status')!='completed':
            reports.append(dict(name=job['name'],status='failed',result=result));continue
        config=json.loads((run/'input.json').read_text());nx=config['simulation']['nx']
        assert result['actualTime']==4 and result['particles']==nx**3
        assert result['acceptedPositiveFinite'] and result['sourcesStable'] and result['stageChecks']['bad']==0 and result['extraRHS']==0
        with np.load(run/'initial.npz') as z:
            assert z['positions'].shape==(nx**3,3) and z['Q'].shape==(nx**3,5)
            assert np.all(z['velocities']==0)
            np.testing.assert_allclose(z['internalEnergies'],.9,rtol=1e-6)
            assert abs(float(z['masses'].astype(float).sum())-1)<1e-6
        snapshotAudit=[]
        for p in sorted((run/'samples').glob('*.npz')):
            with np.load(p) as z:
                assert z['positions'].shape==(nx**3,3) and z['Q'].shape==(nx**3,5)
                assert all(np.all(np.isfinite(z[k])) for k in z.files)
                assert all(np.min(z[k])>0 for k in ('masses','supports','densities','pressures','internalEnergies'))
                snapshotAudit.append(dict(name=p.name,time=float(z['t']),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
        times=np.array([r['time'] for r in snapshotAudit]);sampling=json.loads((run/'sampling.json').read_text())
        assert len(times)==17 and times[0]==0 and times[-1]==4 and np.all(np.diff(times)>0) and sampling['missedTargets']==0
        history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
        dt=np.diff([r['t'] for r in history]);assert np.all(times-np.arange(17)*.25>=-1e-8)
        assert np.max(times-np.arange(17)*.25)<=max(dt)+1e-7
        perf=json.loads((run/'performance.json').read_text())
        telemetry=root/'logs'/(job['name']+'.gpu.jsonl')
        samples=[json.loads(x) for x in telemetry.read_text().splitlines()] if telemetry.exists() else []
        process=[x['processMiB'] for x in samples if x.get('processMiB') is not None]
        total=[x['usedMiB'] for x in samples if x.get('usedMiB') is not None]
        row=dict(name=job['name'],status='passed',nx=nx,particles=nx**3,steps=result['nSteps'],
                 seconds=result['loopWallSeconds'],msPerStep=1000*result['loopWallSeconds']/result['nSteps'],
                 secondsPerSimulatedUnit=result['loopWallSeconds']/4,
                 peakTorchAllocatedMiB=result['peakTorchAllocatedBytes']/2**20,
                 peakTorchReservedMiB=result['peakTorchReservedBytes']/2**20,
                 sampledPeakProcessMiB=max(process) if process else None,
                 sampledPeakDeviceUsedMiB=max(total) if total else None,
                 memoryLimitations='Process/device samples include startup/warmup; 2-second samples can miss transient peaks. Torch peaks exclude allocations owned by other libraries.',
                 performance=perf,final=result['final'],stageChecks=result['stageChecks'],
                 maxRelativeBudgetResidual=result['maxAbsoluteEnergyBudgetResidual'],snapshots=snapshotAudit)
        videos=[p for p in run.glob('exports/**/*.mp4') if p.parent.name!='images']
        assert len(videos)==1
        video=videos[0];info=json.loads(subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height','-of','json',str(video)],capture_output=True,text=True,check=True).stdout)['streams'][0]
        from PIL import Image,ImageDraw
        n=int(info['nb_frames']);indices=sorted(set([0,n//2,n-1]));sheet=Image.new('RGB',(960,len(indices)*505),'white')
        directory=output/job['name'];directory.mkdir(exist_ok=True)
        for i,index in enumerate(indices):
            image=directory/f'frame-{index}.png'
            subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vf',f'select=eq(n\\,{index})','-frames:v','1','-y',str(image)],check=True)
            im=Image.open(image).convert('RGB');im.thumbnail((960,480));sheet.paste(im,(0,i*505+25));ImageDraw.Draw(sheet).text((8,i*505+5),f'{job["name"]}, frame {index}/{n-1}',fill='black')
        sheet.save(directory/'contact-sheet.png')
        row['video']=dict(path=str(video),metadata=info,sha256=hashlib.sha256(video.read_bytes()).hexdigest(),visualReview='Contact sheet prepared; model/user visual inspection pending')
        reports.append(row)
        c=json.loads(json.dumps(config));c['simulation'].pop('nx');c['simulation'].pop('exportRoot')
        for key in ('store','storeMode','storeInterval'):c['simulation'].pop(key,None)
        configs.append(c);inventories.append(json.loads((run/'sourceHashes.json').read_text()))
        import torch
        k=torch.load(run/'forcing.pt',map_location='cpu',weights_only=False)['knots'].numpy();knots.append(k)
        t=np.array([r['t'] for r in history])
        for ax,key in zip(axes.flat,('kineticEnergy','rmsMachMass','relativeEnergyBudgetResidual','minEnergy')):
            ax.plot(t,[r[key] for r in history],label=f'{nx}³',lw=.9);ax.set(xlabel='Simulated time',title=key);ax.grid(alpha=.2)
        write(directory/'summary.json',row)
    successful=[r for r in reports if r['status']=='passed']
    parity=dict(configurationEqualExceptResolutionOutput=all(c==configs[0] for c in configs),sourceInventoriesEqual=all(c==inventories[0] for c in inventories),commonForcingKnotsBitwiseEqual=all(np.array_equal(k[:min(len(k),len(knots[0]))],knots[0][:min(len(k),len(knots[0]))]) for k in knots))
    assert all(parity.values())
    for ax in axes.flat:ax.legend()
    fig.suptitle('CRKSPH 3D startup feasibility: identical physics, full exports0.25, native fresh midpoint')
    fig.savefig(output/'histories.png',dpi=150);fig.savefig(output/'histories.pdf');plt.close(fig)
    write(output/'summary.json',dict(runs=reports,parity=parity,scope='Short startup capacity/throughput only; no stationarity, cascade or resolution convergence qualification.'))
    lines=['# Driven3D performance pilot (2026-10-28)','',
           'CRKSPH from rest, seed1234;32³,64³,96³ to t4; full accepted-state exports every0.25. Unchanged finite-gamma OU/bath/B7/n_h4/native fresh midpoint model. Runs launch serially only after unrelated GPU work is terminal and GPU compute is free.','',
           '|Resolution|Particles|Steps|Loop seconds|ms/step|Seconds/simulated second|Torch peak GiB|Sampled process peak GiB|Snapshot MB|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in successful:
        process=r['sampledPeakProcessMiB'];lines.append(f'|{r["nx"]}³|{r["particles"]:,}|{r["steps"]:,}|{r["seconds"]:.2f}|{r["msPerStep"]:.2f}|{r["secondsPerSimulatedUnit"]:.2f}|{r["peakTorchAllocatedMiB"]/1024:.3f}|{process/1024 if process is not None else float("nan"):.3f}|{r["performance"]["compressedSnapshotBytes"]/1e6:.2f}|')
    lines+=['',f'{len(successful)}/3 endpoints audited. Configuration/source/common-forcing-knot parity: {parity}.',
            '', 'Loop wall includes setup plotting, accepted/RHS diagnostics, sparse exports and final video/checkpoint handling after hydro construction. Setup/first compilation and warmup are separate from this loop measurement. Per-step medians include the stated observers. Whole-device/process memory samples can miss transient peaks; exact Torch high-water marks do not measure all device allocations.',
            '', 'Every retained particle frame has been checked for finite fields, positive mass/support/density/pressure/u, actual accepted timestamps and shape. Raw RHS and accepted guards, no-extra-RHS and source stability are required. Compression and export cost are in each performance.json; recording cadence is not a training-data temporal-resolution qualification.',
            '', 'The first32³ run also saves native initial/final restart artifacts; complete3D restart-continuation parity remains a separate gate. Slice video shows z=0 interpolated density and speed, not a projection or the full volume. Contact sheets require visual review.',
            '', '![Histories]('+str(output/'histories.png')+')', '', 'No stationarity, broad inertial range, native viscosity or production training qualification is claimed. Evolved-state throughput and peak neighbour memory can exceed these startup measurements.', '', 'Evidence: '+str(root)]
    report=root.parents[1]/'docs/driven_3d_pilot_2026-10-28.md';report.write_text('\n'.join(lines)+'\n')
    plans=root.parents[1]/'PLANS.md';text=plans.read_text();marker='**3D pilot authorized/preparing:** serial CRK32³/64³/96³ to t4, exports0.25, unchanged physics/native steps; extend/CPU-check3D before launch, wait for current ACSPH GPU work. Record capacity, runtime and guards.'
    status=f'**3D startup pilot terminal:**{len(successful)}/3 endpoints audited; [performance report](docs/{report.name}), runtime/memory/exports recorded. Video review and3D restart continuation remain pending; no stationarity claim.'
    if marker in text:plans.write_text(text.replace(marker,status,1))
    roadmap=root.parents[1]/'docs/roadmap/driven_compressible_turbulence.md'
    with roadmap.open('a') as f:f.write('\n\n3D pilot terminal: '+status.replace('docs/','../')+'\n')
    print(json.dumps(dict(audited=len(successful),report=str(report),parity=parity)),flush=True)
    if len(successful)!=3:raise SystemExit(1)


if __name__=='__main__':main()
