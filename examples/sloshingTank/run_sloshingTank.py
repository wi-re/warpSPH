#!/usr/bin/env python
"""Run the sloshing-tank case (SPHERIC Test Case 10) and compare the wall
Sensor-1 pressure history against the measured record.

    python examples/sloshingTank/run_sloshingTank.py --scheme wcsph
    python examples/sloshingTank/run_sloshingTank.py --scheme dfsph --tLimit 7

`--scheme wcsph`  -> weakly compressible `deltaSPH`.
`--scheme dfsph`  -> incompressible `divergenceFree`, with the integrator /
                     kernel / CFL preset that scheme needs.

Writes, into `examples/sloshingTank/output/`:
  <scheme>_sensor_pressure.{png,pdf}  simulated vs measured Sensor-1 pressure
                                      (+ prescribed roll angle, + repeatability
                                      peak band)
  <scheme>_series.npz                 raw t / pressures / roll angle / health
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from warpSPHBootstrap import bootstrap

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPHERIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SPHERIC_TestCase10')
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

#: Extra spec fields each scheme needs. `wcsph` keeps the case defaults.
SCHEME_PRESETS = {
    'wcsph': dict(scheme='deltaSPH'),
    'dfsph': dict(
        scheme='divergenceFree',
        integrationScheme='semiImplicitEuler',
        kernel='Wendland2',
        supportMode='SuperSymmetric',
        cflFactor=0.2,
        dt=1.0e-3,
        maxDt=2.0e-3,
    ),
}


def parseArgs(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scheme', choices=sorted(SCHEME_PRESETS), default='wcsph')
    p.add_argument('--nx', type=int, default=None, help='particles across the tank width')
    p.add_argument('--tLimit', type=float, default=7.0, help='simulated end time [s]')
    p.add_argument('--nSteps', type=int, default=None,
                   help='fixed step count (smoke tests); overrides --tLimit')
    p.add_argument('--rollDataFile', type=str, default=None,
                   help='roll-history table (default: bundled lateral_water_1x.txt)')
    p.add_argument('--targetDt', type=float, default=None,
                   help='WCSPH: acoustic dt target; c_s scales as ~1/targetDt '
                        '(case default 2e-4 -> c_s ~16.6; diffSPH uses c_s=20)')
    p.add_argument('--alpha', type=float, default=None,
                   help='WCSPH artificial-viscosity coefficient (case default 0.02)')
    p.add_argument('--smoothSigma', type=float, default=0.01,
                   help='Gaussian smoothing width for the simulated pressure [s]')
    p.add_argument('--replot', action='store_true', default=False,
                   help='skip the sim; rebuild the figure from <scheme>_series.npz')
    p.add_argument('--plot', dest='plot', action='store_true', default=False,
                   help='open the live field window during the run')
    p.add_argument('--store', dest='store', action='store_true', default=False,
                   help='write the HDF5 trajectory')
    p.add_argument('--video', action='store_true', default=False,
                   help='render velocity/density frames every --plotInterval steps '
                        'and encode <scheme>_field.{mp4,gif} (headless-safe)')
    p.add_argument('--plotInterval', type=int, default=None,
                   help='steps between rendered frames (video); case default 50')
    p.add_argument('--out', type=str, default=OUTDIR, help='output directory')
    return p.parse_args(argv)


def buildSpec(case, args):
    from warpSPH.runner import CaseSpec

    spec = CaseSpec(caseName=case.name, scheme=case.scheme,
                    params=dict(case.params)).merged(**case.defaults)
    overrides = dict(SCHEME_PRESETS[args.scheme])
    overrides.update(
        caseName=f'16-sloshingTank-{args.scheme}',
        tLimit=args.tLimit,
        plot=args.plot or args.video, store=args.store,
        video=args.video, show=False,
        progress=True,
    )
    if args.video and args.plotInterval is None:
        overrides['plotInterval'] = 50
    if args.plotInterval is not None:
        overrides['plotInterval'] = args.plotInterval
    if args.nx is not None:
        overrides['nx'] = args.nx
    if args.nSteps is not None:
        overrides['nSteps'] = args.nSteps
    params = {}
    if args.rollDataFile is not None:
        params['rollDataFile'] = os.path.abspath(args.rollDataFile)
    if args.targetDt is not None:
        params['targetDt'] = args.targetDt
    if args.alpha is not None:
        params['alpha'] = args.alpha
    if params:
        overrides['params'] = params
    return spec.merged(**overrides)


def experimentalRecord(path):
    import numpy as np
    raw = np.genfromtxt(path, delimiter='\t', skip_header=1)
    return raw[:, 0], raw[:, 1] * 100.0, raw[:, 2]        # t[s], p[Pa], roll[deg]


def repeatabilityBand():
    """(lo, hi) of the measured first-four impact-pressure peaks [Pa]."""
    import numpy as np
    path = os.path.join(SPHERIC, 'Repeatability_Files',
                        'Water_4first_peak_lateral_impact_tto_0_85_H93_B1X.txt')
    if not os.path.exists(path):
        return None
    peaks = np.genfromtxt(path, skip_header=2)
    peaks = peaks[np.isfinite(peaks).all(axis=1)] * 100.0
    return float(peaks.min()), float(peaks.max())


def resample(t, y, sigma):
    """Uniform-grid resample of (t, y) + Gaussian smoothing of width `sigma` [s].

    Non-finite samples (a diverged tail) are dropped before smoothing.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if t.size < 3:
        return t, y
    dt = np.median(np.diff(t))
    grid = np.arange(t[0], t[-1], dt)
    on = np.interp(grid, t, y)
    if sigma > 0 and dt > 0:
        on = gaussian_filter1d(on, sigma / dt)
    return grid, on


def makeFigure(scheme, nx, series, smoothSigma, tLimit, out):
    """Draw simulated-vs-measured Sensor-1 pressure + roll, save PNG/PDF."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = series['t']; p = series['sensorPressure']; probe = series['sensorPressureProbe']
    roll = series['rollAngleDeg']
    diverged = bool(series['diverged'])
    rollFile = os.path.join(SPHERIC, 'data_files', 'lateral_water_1x.txt')
    expT, expP, expRoll = experimentalRecord(rollFile)
    band = repeatabilityBand()

    noField = np.allclose(np.nan_to_num(p), 0.0) and np.allclose(np.nan_to_num(probe), 0.0)
    gridT, smoothP = resample(t, p, smoothSigma)
    tEnd = float(np.nanmax(t[np.isfinite(t)])) if np.isfinite(t).any() else tLimit

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1]))
    if band is not None:
        ax[0].axhspan(band[0], band[1], color='tab:orange', alpha=0.12,
                      label=f'measured impact-peak band ({band[0]:.0f}-{band[1]:.0f} Pa)')
    ax[0].plot(expT, expP, lw=1.0, ls='--', color='0.4', label='measured (Sensor 1)')
    ax[0].plot(t, p, lw=0.5, color='tab:blue', alpha=0.3, label='simulated (raw)')
    if gridT.size:
        ax[0].plot(gridT, smoothP, lw=1.6, color='tab:blue',
                   label=f'simulated (smoothed {smoothSigma*1e3:.0f} ms)')
    if probe.size == t.size and not noField:
        ax[0].plot(t, probe, lw=0.8, color='tab:green', alpha=0.6,
                   label='simulated (fluid-probe)')
    if diverged:
        ax[0].axvline(tEnd, color='tab:red', lw=1.2, ls=':')
        ax[0].text(tEnd, 0.96, ' diverged', color='tab:red', fontsize=9,
                   ha='right', va='top', rotation=90, transform=ax[0].get_xaxis_transform())
    if noField:
        ax[0].text(0.5, 0.5, 'this scheme carries no wall-pressure field\n'
                   '(VD+PS applies a position shift, not a stored p)',
                   transform=ax[0].transAxes, ha='center', va='center',
                   fontsize=10, color='0.4', style='italic')
    # Clip to the measured range so a numerical spike does not flatten it.
    if np.isfinite(expP).any():
        lim = 2.2 * max(abs(np.nanmin(expP)), abs(np.nanmax(expP)), 1.0)
        ax[0].set_ylim(-0.6 * lim, lim)
    ax[0].set_ylabel('Sensor 1 pressure [Pa]')
    ax[0].set_title(f'Sloshing tank (SPHERIC TC10) -- {scheme}, nx={nx}'
                    + ('  [DIVERGED at t={:.2f}s]'.format(tEnd) if diverged else
                       '  [ran to t={:.1f}s]'.format(tEnd)))
    ax[0].legend(fontsize=8, loc='upper left')
    ax[0].grid(alpha=0.3)

    ax[1].plot(expT, expRoll, lw=1.4, ls='--', color='0.4', label='prescribed roll')
    ax[1].plot(t, roll, lw=1.0, color='tab:red', label='applied roll')
    ax[1].set_xlabel('time [s]'); ax[1].set_ylabel('roll [deg]')
    ax[1].set_xlim(0, min(tLimit, float(expT[-1])))
    ax[1].legend(fontsize=8, loc='upper right'); ax[1].grid(alpha=0.3)
    fig.tight_layout()

    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'{scheme}_sensor_pressure.{ext}'), dpi=150)
    plt.close(fig)
    return band, noField, tEnd


def loadSeries(out, scheme):
    import numpy as np
    return dict(np.load(os.path.join(out, f'{scheme}_series.npz')))


def main(argv=None):
    args = parseArgs(sys.argv[1:] if argv is None else argv)
    import numpy as np
    os.makedirs(args.out, exist_ok=True)

    if args.replot:
        series = loadSeries(args.out, args.scheme)
        nx = args.nx or 0
        band, noField, tEnd = makeFigure(args.scheme, nx, series, args.smoothSigma,
                                         args.tLimit, args.out)
        print(f'replotted {args.out}/{args.scheme}_sensor_pressure.pdf  '
              f'(noPressureField={noField}, t_end={tEnd:.2f})')
        return 0

    bootstrap(precision='float32')
    from warpSPH.cases import importAll
    importAll()
    from warpSPH.runner import getCase, run

    case = getCase('sloshingTank')
    spec = buildSpec(case, args)

    print(f'== sloshingTank / {args.scheme} ==  nx={spec.nx}  tLimit={spec.tLimit}  '
          f'scheme={spec.scheme}')
    t0 = time.perf_counter()
    result = run(case, spec)
    wall = time.perf_counter() - t0

    series = dict(
        t=result.series('t'),
        sensorPressure=result.series('sensorPressure'),
        sensorPressureProbe=result.series('sensorPressureProbe'),
        sensorPressureCD=result.series('sensorPressureCD'),
        sensorPressureDF=result.series('sensorPressureDF'),
        rollAngleDeg=result.series('rollAngleDeg'),
        minDensity=result.series('minDensity'),
        maxDensity=result.series('maxDensity'),
        sensorRho=result.series('sensorRho'),
        kineticEnergy=result.series('kineticEnergy'),
        nx=spec.nx, diverged=result.diverged, nSteps=result.nSteps, wallTime=wall,
    )
    np.savez(os.path.join(args.out, f'{args.scheme}_series.npz'), **series)

    if result.videoPath and os.path.exists(result.videoPath):
        import shutil
        vdir = os.path.dirname(result.videoPath)
        for src, ext in ((result.videoPath, 'mp4'),
                         (os.path.join(vdir, 'out.gif'), 'gif')):
            if os.path.exists(src):
                dst = os.path.join(args.out, f'{args.scheme}_field.{ext}')
                shutil.copy(src, dst)
                print(f'   video -> {dst}')

    band, noField, tEnd = makeFigure(args.scheme, spec.nx, series, args.smoothSigma,
                                     spec.tLimit, args.out)

    t, p = series['t'], series['sensorPressure']
    m = (t >= 2.0) & np.isfinite(p)
    peakSim = float(np.nanmax(np.abs(p[m]))) if m.any() else float('nan')
    print(f'\n-- done in {wall:.1f}s  ({result.nSteps} steps, diverged={result.diverged}) --')
    print(f'   ran to t = {tEnd:.2f} s')
    print(f'   peak |p_sim| after t=2s : {peakSim:.1f} Pa'
          + ('   (NO wall-pressure field for this scheme)' if noField else ''))
    if band is not None:
        print(f'   measured impact-peak band: {band[0]:.0f} .. {band[1]:.0f} Pa')
    print(f'   density range over run   : '
          f'[{np.nanmin(series["minDensity"]):.3f}, {np.nanmax(series["maxDensity"]):.3f}]')
    print(f'   wrote {args.out}/{args.scheme}_sensor_pressure.pdf and _series.npz')
    return 1 if result.diverged else 0


if __name__ == '__main__':
    raise SystemExit(main())
