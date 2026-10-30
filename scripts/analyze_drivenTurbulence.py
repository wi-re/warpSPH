#!/usr/bin/env python3
"""Qualify saved 2D compressible particle flows without launching simulations.

Accept native trajectory.h5 plus caseSpec.json, or a run with input.json,
result.json/history.jsonl (optional while running),
and samples/*.npz with scalar t, positions[N,2], masses[N], densities[N],
velocities[N,2], internalEnergies[N], pressures[N], Q[N,4], supports[N].
See docs/driven_turbulence.md for measurement and acceptance conventions.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from warpSPH.utils.turbulence import (particleStatistics, weightedPDF, depositCIC2D,
    spectra2D, structureFunctions2D, stationarityScreen, fieldCorrelationTime, depositKernel2D)


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def timeMean(times, values, low, high):
    selected = (times > low) & (times < high)
    t = np.r_[low, times[selected], high]
    values = np.asarray(values)
    flat = values.reshape(len(times), -1)
    interp = np.stack([np.interp(t, times, column) for column in flat.T], axis=1)
    return (np.trapezoid(interp, t, axis=0)/(high-low)).reshape(values.shape[1:])


def frameKeys(group):
    return sorted(group, key=lambda name: int(name.rsplit('_', 1)[-1]))


def snapshots(path):
    """Read retained NPZ observations or native driven trajectory output."""
    if (path/'samples').is_dir():
        refs = {}
        if (path/'initial.npz').exists():
            with np.load(path/'initial.npz') as initial:
                refs = dict(_initialEnergy=float(initial['Q'][:, -1].astype(float).sum()),
                            _initialMass=float(initial['Q'][:, 0].astype(float).sum()))
        for file in sorted((path/'samples').glob('*.npz')):
            with np.load(file) as state:
                yield file, dict(state, **refs)
        return
    import h5py
    with h5py.File(path/'trajectory.h5') as f:
        required = ('positions', 'masses', 'velocities', 'densities', 'pressures', 'internalEnergies', 'supports')
        if any(name not in f for name in required):
            raise ValueError('trajectory lacks per-frame driven particle measurements')
        for key in frameKeys(f['positions']):
            state = {name: f[name][key][:] for name in required}
            state['t'] = float(f['times'][key][0])
            for name in ('Q', 'sourceEnergyBudget'):
                if name in f:
                    state[name] = f[name][key][:]
            for name in ('initialEnergy', 'initialMass'):
                if name in f.attrs:
                    state['_'+name] = float(f.attrs[name])
            yield f'{path}/trajectory.h5:{key}', state


def forcingObservation(path):
    """Audit complete native conditional inputs, including between-frame knots."""
    if not (path/'trajectory.h5').exists():
        return dict(scope='investigation forcing checkpoint', available=(path/'forcing.pt').exists())
    import h5py
    with h5py.File(path/'trajectory.h5') as f:
        if 'frameData' not in f:
            return dict(timeVaryingCoefficients=False, completeKnots=False)
        frames = [f['frameData'][key] for key in frameKeys(f['positions'])]
        if not frames or any('forcingCoefficients' not in frame for frame in frames):
            return dict(timeVaryingCoefficients=False, completeKnots=False)
        if any('forcingKnots' not in frame or 'forcingKnotStart' not in frame.attrs for frame in frames):
            return dict(timeVaryingCoefficients=True, completeKnots=False)
        raw, nextKnot = [], 0
        for frame in frames:
            if int(frame.attrs['forcingKnotStart']) != nextKnot:
                raise ValueError('forcing knot history has missing or duplicate chunks')
            knot = frame['forcingKnots'][:]
            raw.append(knot); nextKnot += len(knot)
        raw = np.concatenate(raw)
        if not np.isfinite(raw).all():
            raise ValueError('forcing knot history is nonfinite')
        parameters = dict(frames[0]['forcingParameters'].attrs)
        modes = frames[0]['forcingModes'][:]
        normal = modes/np.linalg.norm(modes, axis=1, keepdims=True)
        longitudinal = normal[:, :, None]*normal[:, None, :]
        z = parameters['solenoidalWeight']
        projection = z*(np.eye(parameters['dim'])-longitudinal)+(1-z)*longitudinal
        scale = parameters['accelerationRMS']/np.sqrt(len(modes)*(z*z*(parameters['dim']-1)+(1-z)**2))
        error = 0.
        for key, frame in zip(frameKeys(f['positions']), frames):
            if dict(frame['forcingParameters'].attrs) != parameters or not np.array_equal(frame['forcingModes'][:], modes):
                raise ValueError('forcing parameters/modes change across the saved trajectory')
            if 'forcingKnotCoefficients' in frame:
                projected = scale*np.einsum('mij,kpmj->kpmi', projection, frame['forcingKnots'][:])
                if projected.size and np.max(abs(projected-frame['forcingKnotCoefficients'][:])) > 1e-12*max(1., parameters['accelerationRMS']):
                    raise ValueError('projected knot coefficients disagree with raw forcing knots')
            time = float(f['times'][key][0])
            if time != frame.attrs['forcingTime']:
                raise ValueError('frame time and applied forcing time disagree')
            clock = time/parameters['updateInterval'];index = int(np.floor(clock));alpha = clock-index
            if index+1 >= len(raw):
                raise ValueError('forcing history does not cover the fluid observation window')
            expected = scale*np.einsum('mij,pmj->pmi', projection, (1-alpha)*raw[index]+alpha*raw[index+1])
            error = max(error, float(abs(expected-frame['forcingCoefficients'][:]).max()))
        if error > 1e-12*max(1., parameters['accelerationRMS']):
            raise ValueError('forcing coefficients disagree with the saved knot history')
        return dict(timeVaryingCoefficients=True, completeKnots=True, knots=nextKnot, maxCoefficientError=error)


def analyze(path, output, grids, blocks, deposition='cic'):
    if (path/'input.json').exists():
        config = json.loads((path/'input.json').read_text())
    else:
        spec = json.loads((path/'caseSpec.json').read_text())
        config = dict(simulation=spec, params=spec.get('params', {}))
    params = config.get('params', {})
    gamma, length = params.get('gamma', 5/3), config['simulation'].get('L', 1.)
    nx = config['simulation']['nx']
    edges = np.linspace(-8, 8, 161)
    p0 = params.get('rho0', 1.)*params.get('soundSpeed', 1.)**2/gamma
    times, rows, pdfs, snapshotBudgets, data = [], [], [], [], {n: [] for n in grids}
    for file, s in snapshots(path):
        t = float(s['t'])
        mass, rho, velocity, u = (s[k] for k in ['masses', 'densities', 'velocities', 'internalEnergies'])
        row = particleStatistics(mass, rho, velocity, u, gamma=gamma, length=length)
        volume, logrho = mass/rho, np.log(rho/(mass.sum()/length**2))
        logpressure = np.log(s['pressures']/p0)
        if not np.isfinite(logpressure).all():
            raise ValueError(f'{file}: invalid pressure')
        row['kineticEnergy'] = float(.5*np.sum(mass*np.sum(velocity.astype(float)**2, axis=1)))
        row['thermalEnergy'] = float(np.sum(mass.astype(float)*u))
        row['minDensity'], row['minEnergy'] = float(rho.min()), float(u.min())
        pdf = {}
        for name, weights in [('Mass', mass), ('Volume', volume)]:
            pdf['logDensity'+name] = weightedPDF(logrho, weights, edges)
            pdf['logPressure'+name] = weightedPDF(logpressure, weights, edges)
        for n in grids:
            grid = (depositCIC2D(s['positions'], mass, velocity, gridSize=n, length=length) if deposition == 'cic' else
                    depositKernel2D(s['positions'], mass, velocity, s['supports'], gridSize=n, length=length))
            item = {k: grid[k] for k in ['emptyCellFraction', 'relativeMassError', 'momentumError', 'unresolvedKineticFraction']}
            if deposition == 'kernel':
                item.update({k: grid[k] for k in ['minFilterRadius', 'maxFilterRadius', 'broadenedParticleFraction']})
            if not grid['emptyCellFraction']:
                item.update(spectra=spectra2D(grid['velocity'], grid['density']),
                            structure=structureFunctions2D(grid['velocity'], length=length),
                            velocity=grid['velocity'].astype(np.float32))
            data[n].append(item)
        if all(k in s for k in ('Q', 'sourceEnergyBudget', '_initialEnergy', '_initialMass')):
            snapshotBudgets.append(dict(relativeEnergyBudgetResidual=float((s['Q'][:, -1].astype(float).sum()-s['_initialEnergy']-s['sourceEnergyBudget'].sum())/s['_initialEnergy']),
                relativeMassChange=float(s['Q'][:, 0].astype(float).sum()/s['_initialMass']-1), meanSpeed=float(np.linalg.norm(row['bulkVelocityMass']))))
        times.append(t); rows.append(row); pdfs.append(pdf)
    if len(times) < 8:
        raise ValueError(f'{path}: need at least8 snapshots')
    times = np.asarray(times)
    if np.any(np.diff(times) <= 0):
        raise ValueError(f'{path}: snapshots must be strictly chronological')
    complete = bool(times[0] <= blocks[0] and times[-1] >= blocks[-1])
    window = (blocks[0], blocks[-1]) if complete else (times[0], times[-1])
    metrics = ['rmsMachMass', 'kineticEnergy', 'logDensityStdMass', 'logDensityStdVolume']
    screens = {k: stationarityScreen(times, [r[k] for r in rows], blocks=blocks) for k in metrics} if complete else {}
    output.mkdir(parents=True, exist_ok=True)
    series = dict(t=times, **{k: np.asarray([r[k] for r in rows]) for k in rows[0]})
    for key in pdfs[0]:
        series['pdf_'+key] = np.asarray([p[key] for p in pdfs])
    np.savez_compressed(output/'particle-statistics.npz', **series, pdfEdges=edges)
    summaries = {}
    for n, measurements in data.items():
        valid = np.array(['spectra' in item for item in measurements])
        arrays = {k: np.array([item[k] for item in measurements]) for k in ['emptyCellFraction', 'relativeMassError', 'momentumError', 'unresolvedKineticFraction']}
        if deposition == 'kernel':
            arrays.update({k: np.array([item[k] for item in measurements]) for k in ['minFilterRadius', 'maxFilterRadius', 'broadenedParticleFraction']})
        arrays['boxModeEnergy'] = np.array([item['spectra']['boxModeEnergy'] if valid[i] else np.nan for i,item in enumerate(measurements)])
        indices = np.flatnonzero(valid)
        for kind, key in [('spectra', 'velocityEnergy'), ('spectra', 'compressionalEnergy'), ('spectra', 'solenoidalEnergy'), ('spectra', 'densityPower'), ('structure', 'longitudinal'), ('structure', 'transverse')]:
            if len(indices):
                shape = measurements[indices[0]][kind][key].shape
                arrays[key] = np.array([item[kind][key] if valid[i] else np.full(shape, np.nan) for i, item in enumerate(measurements)])
        gridSummary = dict(maxEmptyCellFraction=float(arrays['emptyCellFraction'].max()),
                           maxMassError=float(abs(arrays['relativeMassError']).max()),
                           maxMomentumError=float(abs(arrays['momentumError']).max()),
                           comparisonKMax=n//4,
                           medianUnresolvedKineticFraction=float(np.median(arrays['unresolvedKineticFraction'])))
        if deposition == 'kernel':
            gridSummary.update(minFilterRadius=float(arrays['minFilterRadius'].min()), maxFilterRadius=float(arrays['maxFilterRadius'].max()), meanBroadenedParticleFraction=float(arrays['broadenedParticleFraction'].mean()))
        selected = (times >= window[0]) & (times <= window[1])
        # Quadrature needs the bracketing observations, even just outside the
        # assessment window. Invalid earlier startup frames are not interpolants.
        low = max(0, np.searchsorted(times, window[0], side='right')-1)
        high = min(len(times), np.searchsorted(times, window[1], side='left')+1)
        stencil = slice(low, high)
        gridSummary['comparisonCoverageComplete'] = bool(valid[stencil].all())
        if gridSummary['comparisonCoverageComplete']:
            if np.count_nonzero(selected) >= 8:
                fields = np.array([measurements[i]['velocity'] for i in np.flatnonzero(selected)])
                corr = fieldCorrelationTime(times[selected], fields)
                gridSummary['fieldCorrelation'] = {k: corr[k] for k in ['tau', 'effectiveSamples', 'censored']}
                arrays.update(correlationLag=corr['lag'], velocityFieldACF=corr['acf'])
            else:
                gridSummary['fieldCorrelation'] = dict(tau=None, effectiveSamples=None, censored=True)
            gridSummary['means'] = {key: timeMean(times[stencil], arrays[key][stencil], *window) for key in ['velocityEnergy', 'compressionalEnergy', 'solenoidalEnergy', 'densityPower', 'longitudinal', 'transverse']}
            velocityE = arrays['velocityEnergy']
            denominator = velocityE[:, 1:].sum(axis=1).clip(min=np.finfo(float).tiny)
            fraction = arrays['boxModeEnergy']/denominator
            arrays['lowestShellFraction'] = velocityE[:, 1]/denominator
            gridSummary['lowestShellFractionMean'] = float(timeMean(times[stencil], arrays['lowestShellFraction'][stencil], *window))
            arrays['boxModeFraction'] = fraction
            gridSummary['boxModeFractionMean'] = float(timeMean(times[stencil], fraction[stencil], *window))
            if complete:
                gridSummary['boxModeScreen'] = stationarityScreen(times[stencil], fraction[stencil], blocks=blocks)
        else:
            gridSummary['invalidSpectralSnapshots'] = int(np.count_nonzero(~valid[stencil]))
        np.savez_compressed(output/f'grid-{n}.npz', t=times, **arrays)
        summaries[str(n)] = gridSummary
    means = {k: timeMean(times, series[k], *window) for k in rows[0]}
    meanPDFs = {k: timeMean(times, series['pdf_'+k], *window) for k in pdfs[0]}
    turnovers = float(np.trapezoid(np.interp(np.linspace(*window, 1001), times, series['rmsVelocityMass']), np.linspace(*window, 1001))/length)
    passed = complete and all(s['passed'] for s in screens.values()) and turnovers >= 10
    result = json.loads((path/'result.json').read_text()) if (path/'result.json').exists() else None
    history = [json.loads(line) for line in (path/'history.jsonl').read_text().splitlines() if line.strip()] if (path/'history.jsonl').exists() else []
    if not history:
        history = snapshotBudgets
    budget = dict(maxRelativeEnergyResidual=max(abs(r['relativeEnergyBudgetResidual']) for r in history),
                  maxRelativeMassChange=max(abs(r['relativeMassChange']) for r in history),
                  maxMeanSpeed=max(r['meanSpeed'] for r in history)) if history else None
    reference = path/'sourceHashes.json'
    sourceChanges, additions = [], []
    if reference.exists():
        # The archive is the authority; do not silently call a changed inventory identical.
        hashes = json.loads(reference.read_text())
        for key, expected in hashes.items():
            if key.startswith('warpSPH/'):
                file = Path(__file__).resolve().parents[1]/key.removeprefix('warpSPH/')
                if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                    sourceChanges.append(key)
    report = dict(measurementVersion=2, path=str(path), completeAssessmentWindow=complete, samples=len(times),
                  timeRange=[times[0], times[-1]], window=window, nx=nx, params=params, deposition=deposition,
                  stationaryScalarScreenPassed=bool(passed), screens=screens, boxCrossings=turnovers,
                  means=means, meanPDFs=meanPDFs, grid=summaries, budget=budget,
                  runtimeResult=result, budgetScope='all accepted states' if (path/'history.jsonl').exists() else 'saved snapshots only',
                  sourceManifestAvailable=reference.exists(), conditionalForcing=forcingObservation(path),
                  changedArchivedWarpSPHSources=sourceChanges)
    (output/'summary.json').write_text(json.dumps(plain(report), indent=2, allow_nan=False)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--grids', type=int, nargs='+', default=[16, 32])
    p.add_argument('--deposition', choices=['cic', 'kernel'], default='cic')
    p.add_argument('--blocks', type=float, nargs='+', default=[10., 20., 30., 40.])
    args = p.parse_args()
    if len({r.name for r in args.runs}) != len(args.runs):
        p.error('run directory basenames must be unique')
    reports = {r.name: analyze(r.resolve(), args.output/r.name, args.grids, args.blocks, args.deposition) for r in args.runs}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(plain(reports), indent=2, allow_nan=False)+'\n')
    for name, r in reports.items():
        print(name, 'complete=', r['completeAssessmentWindow'], 'stationaryScalarScreen=', r['stationaryScalarScreenPassed'], flush=True)


if __name__ == '__main__':
    main()
