#!/usr/bin/env python3
"""Qualify saved 2D compressible particle flows without launching simulations.

Each run contains input.json, result.json/history.jsonl (optional while running),
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
    spectra2D, structureFunctions2D, stationarityScreen, fieldCorrelationTime)


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


def analyze(path, output, grids, blocks):
    config = json.loads((path/'input.json').read_text())
    params = config.get('params', {})
    gamma, length = params.get('gamma', 5/3), config['simulation'].get('L', 1.)
    nx = config['simulation']['nx']
    edges = np.linspace(-8, 8, 161)
    p0 = params.get('rho0', 1.)*params.get('soundSpeed', 1.)**2/gamma
    times, rows, pdfs, data = [], [], [], {n: [] for n in grids}
    for file in sorted((path/'samples').glob('*.npz')):
        with np.load(file) as s:
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
                grid = depositCIC2D(s['positions'], mass, velocity, gridSize=n, length=length)
                item = {k: grid[k] for k in ['emptyCellFraction', 'relativeMassError', 'momentumError', 'unresolvedKineticFraction']}
                if not grid['emptyCellFraction']:
                    item.update(spectra=spectra2D(grid['velocity'], grid['density']),
                                structure=structureFunctions2D(grid['velocity'], length=length),
                                velocity=grid['velocity'].astype(np.float32))
                data[n].append(item)
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
        selected = (times >= window[0]) & (times <= window[1])
        if valid[selected].all():
            fields = np.array([measurements[i]['velocity'] for i in np.flatnonzero(selected)])
            corr = fieldCorrelationTime(times[selected], fields)
            gridSummary['fieldCorrelation'] = {k: corr[k] for k in ['tau', 'effectiveSamples', 'censored']}
            arrays.update(correlationLag=corr['lag'], velocityFieldACF=corr['acf'])
            gridSummary['means'] = {key: timeMean(times, arrays[key], *window) for key in ['velocityEnergy', 'compressionalEnergy', 'solenoidalEnergy', 'densityPower', 'longitudinal', 'transverse']}
            velocityE = arrays['velocityEnergy']
            fraction = velocityE[:, 1]/velocityE[:, 1:].sum(axis=1).clip(min=np.finfo(float).tiny)
            arrays['boxModeFraction'] = fraction
            gridSummary['boxModeFractionMean'] = float(timeMean(times, fraction, *window))
            if complete:
                gridSummary['boxModeScreen'] = stationarityScreen(times, fraction, blocks=blocks)
        else:
            gridSummary['invalidSpectralSnapshots'] = int(np.count_nonzero(~valid & selected))
        np.savez_compressed(output/f'grid-{n}.npz', t=times, **arrays)
        summaries[str(n)] = gridSummary
    means = {k: timeMean(times, series[k], *window) for k in rows[0]}
    meanPDFs = {k: timeMean(times, series['pdf_'+k], *window) for k in pdfs[0]}
    turnovers = float(np.trapezoid(np.interp(np.linspace(*window, 1001), times, series['rmsVelocityMass']), np.linspace(*window, 1001))/length)
    passed = complete and all(s['passed'] for s in screens.values()) and turnovers >= 10
    result = json.loads((path/'result.json').read_text()) if (path/'result.json').exists() else None
    history = [json.loads(line) for line in (path/'history.jsonl').read_text().splitlines() if line.strip()] if (path/'history.jsonl').exists() else []
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
    report = dict(path=str(path), completeAssessmentWindow=complete, samples=len(times),
                  timeRange=[times[0], times[-1]], window=window, nx=nx, params=params,
                  stationaryScalarScreenPassed=bool(passed), screens=screens, boxCrossings=turnovers,
                  means=means, meanPDFs=meanPDFs, grid=summaries, budget=budget,
                  runtimeResult=result, changedArchivedWarpSPHSources=sourceChanges)
    (output/'summary.json').write_text(json.dumps(plain(report), indent=2, allow_nan=False)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--grids', type=int, nargs='+', default=[16, 32])
    p.add_argument('--blocks', type=float, nargs='+', default=[10., 20., 30., 40.])
    args = p.parse_args()
    if len({r.name for r in args.runs}) != len(args.runs):
        p.error('run directory basenames must be unique')
    reports = {r.name: analyze(r.resolve(), args.output/r.name, args.grids, args.blocks) for r in args.runs}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(plain(reports), indent=2, allow_nan=False)+'\n')
    for name, r in reports.items():
        print(name, 'complete=', r['completeAssessmentWindow'], 'stationaryScalarScreen=', r['stationaryScalarScreenPassed'], flush=True)


if __name__ == '__main__':
    main()
