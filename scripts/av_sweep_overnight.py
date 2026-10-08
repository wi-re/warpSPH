#!/usr/bin/env python3
"""AV_PLAN Phases 3-7: the one overnight validation sweep (user, 2026-10-07: build first, sweep once).

Runs every phase's expensive validation back to back, one GPU job at a time, each `scripts/av_report.py` job in its
own process (a crash or a hang loses one job, not the night), with video on (`--profile full` defaults it), the
runner's velocity alarm and stall watchdog on (`av_report` passes `stallProgress=1e-3`). Then:

* `maps`   -- the detector maps (AV_PLAN §6.5): every detector on the same 2D Sod and 2D Sedov frame (same IC, same
             end time), per-particle alpha / div v / detector scalar dumped to npz and drawn side by side;
* `timing` -- Phase 5B's ms/step (CUDA-synchronised, after a warm-up run in the same process; video OFF, since it
             is a timing measurement);
* the verdict (`scripts/av_sweep_verdict.py`), which scores each phase's thresholds into `verdict.md`.

    python scripts/av_sweep_overnight.py                    # the full sweep -> results/av_sweep_<date>/
    python scripts/av_sweep_overnight.py --smoke            # every job at smoke size: the pre-flight check
    python scripts/av_sweep_overnight.py --dry              # print the job list and the time estimate
    python scripts/av_sweep_overnight.py --only phase4 kh256

Jobs are ordered by value, so a night cut short still leaves the most informative results.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

ALL10 = ['sod', 'sod2d', 'sod3d', 'sedov', 'noh', 'gresho', 'yee', 'linearWave', 'kelvinHelmholtz', 'rayleighTaylor']

#: (name, av_report --config, cases, extra av_report args, estimated minutes at the full profile)
JOBS: List[Dict[str, Any]] = [
    # Phase 4 first: the reconstruction AV is the strategic result (Garcia-Senz & Cabezon Table 1 rows 1-6)
    dict(name='phase4', config='phase4', cases=['sod', 'sod2d', 'sedov', 'noh', 'gresho', 'kelvinHelmholtz',
                                                'rayleighTaylor'], minutes=90),
    # Phase 6: Wadsley vs C&D, incl. the cylindrical Noh (uniform-compression blindness)
    dict(name='phase6', config='wadsley2017', cases=ALL10 + ['noh2d', 'shearBox'], minutes=20),
    dict(name='noh2d', config='cullenDehnen2010', cases=['noh2d'], minutes=4),
    dict(name='noh2dOthers', config='rosswog2020', cases=['noh2d'], minutes=4),
    # Phase 5B: Sphenix (shocks, Gresho, KH; robustness at cfl x 4)
    dict(name='phase5b', config='sphenix', cases=ALL10 + ['noh2d', 'shearBox'], minutes=16),
    dict(name='cfl4', config='cfl4', cases=['sod', 'sedov', 'noh', 'gresho'], minutes=12),
    # Phase 5A: dynamic beta, on the shock cases, Gresho, the shearing Noh and the shear box
    dict(name='phase5a', config='phase5a', cases=['sod', 'noh', 'sedov', 'gresho', 'shearingNoh', 'shearBox'],
         minutes=60),
    # Phase 7 cross terms (rows 10-12) on the full ten
    dict(name='phase7cross', config='phase7cross', cases=ALL10 + ['shearBox'], minutes=45),
    # KH at nx 256 to t = 3 on the smooth-density IC, as the Monaghan references (docs/av/kh256_2026-10-07: C&D /
    # Rosswog peak 0.21 -> 0.105, CRK 0.23-0.24 holds 0.17-0.19): does reconstruction close the gap to CRKSPH?
    *(dict(name=f'kh256_{c}', config=c, cases=['kelvinHelmholtz'],
           extra=['--runParam', 'nx=256', 'tLimit=3.0', '--caseParam', 'smoothDensity=1'],
           minutes=12, smokeExtra=['--caseParam', 'smoothDensity=1'])
      for c in ('gsAVSLRB2', 'gsAVSLR', 'noneLimited', 'sphenix', 'wadsley2017', 'rosswogLimited')),
    # Phase 3: the CRKSPH extraction against its references, and the plain reconstruction on the Monaghan host
    dict(name='phase3crk', config='crk', cases=ALL10, minutes=50),
    dict(name='phase3', config='phase3', cases=['sod', 'sod2d', 'sedov', 'noh', 'gresho', 'kelvinHelmholtz'],
         minutes=20),
    # baseline columns re-taken on this code (the M0g references predate Phases 3-6; differences are round-off,
    # see the Phase 3 notes) plus the new cases for them
    dict(name='baselineNew', config='baseline', cases=['shearBox', 'shearingNoh', 'noh2d'], minutes=10),
    dict(name='phase2New', config='rosswog2020', cases=['shearBox', 'shearingNoh'], minutes=5),
]

#: AV_PLAN Phase 7 bake-off (`--bakeoff`): every matrix row (av_report BAKEOFF_ROWS) on the ten cases plus the
#: shear box and the cylindrical Noh, one job per row; the CompSPH cross-check of the shortlist; KH at nx 256 for
#: the rows that changed after the 2026-10-07 sweep (the default, Sphenix ell_V 5).
BAKEOFF_CASES = ALL10 + ['shearBox', 'noh2d']


def _bakeoffJobs():
    sys.path.insert(0, str(REPO / 'scripts'))
    import av_report as A
    jobs = [dict(name=f'row{row:02d}_{cfg}', config=cfg, cases=BAKEOFF_CASES,
                 minutes=22 if 'Limited' in cfg or 'SLR' in cfg else 15)
            for row, cfg, *_ in A.BAKEOFF_ROWS]
    jobs.append(dict(name='compSPH', config='compSPH',
                     cases=['sod', 'sod2d', 'sedov', 'noh', 'gresho', 'kelvinHelmholtz', 'noh2d'], minutes=40))
    jobs += [dict(name=f'kh256_{c}', config=c, cases=['kelvinHelmholtz'],
                  extra=['--runParam', 'nx=256', 'tLimit=3.0', '--caseParam', 'smoothDensity=1'],
                  smokeExtra=['--caseParam', 'smoothDensity=1'], minutes=25)
             for c in ('rosswogLimitedCoupled', 'sphenix')]
    return jobs


#: The detector maps (AV_PLAN §6.5): every detector, same IC, same end time.
MAP_CONFIGS = ['cullenDehnen2010', 'readHayfield2012', 'rosswog2020', 'sphenix', 'wadsley2017', 'gsAVSLRB2',
               'rosswogLimitedCoupled']


def _log(msg: str) -> None:
    print(f'[av_sweep {datetime.datetime.now():%H:%M:%S}] {msg}', flush=True)


def runJob(job: Dict[str, Any], out: Path, smoke: bool, timeoutMin: float) -> Dict[str, Any]:
    jobDir = out / job['name']
    extra = job.get('smokeExtra', job.get('extra', [])) if smoke else job.get('extra', [])
    cmd = [PY, str(REPO / 'scripts' / 'av_report.py'), '--config', job['config'], '--cases', *job['cases'],
           '--profile', 'smoke' if smoke else 'full', '--out', str(jobDir), *extra]
    if smoke:
        cmd.append('--video')            # the pre-flight also exercises the video path
    log = out / f'{job["name"]}.log'
    _log(f'{job["name"]}: {" ".join(cmd[1:])}')
    started = time.time()
    with open(log, 'w') as fh:
        # carriage returns of the progress rows -> newlines, line-buffered, so the log can be tailed / monitored
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        tr = subprocess.Popen(['stdbuf', '-oL', 'tr', '\r', '\n'], stdin=proc.stdout, stdout=fh)
        proc.stdout.close()
        try:
            rc = proc.wait(timeout=timeoutMin * 60)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = 'timeout'
        tr.wait()
    rec = dict(name=job['name'], returncode=rc, minutes=(time.time() - started) / 60, log=str(log), dir=str(jobDir))
    _log(f'{job["name"]}: rc={rc} after {rec["minutes"]:.1f} min')
    return rec


# --------------------------------------------------------------------------- detector maps

def runMaps(out: Path, smoke: bool) -> Dict[str, Any]:
    """Each detector on 2D Sod and 2D Sedov from the same IC to the same time; per-particle npz + one figure each."""
    import dataclasses
    import numpy as np
    sys.path.insert(0, str(REPO / 'scripts'))
    import av_report as A
    from warpSPH.runner import run
    from warpSPH.cases.sodND import sod2dCase
    from warpSPH.cases.sedov import sedovCase

    mapDir = out / 'maps'
    mapDir.mkdir(parents=True, exist_ok=True)
    frames = {
        'sod2d': (sod2dCase, dict(nSteps=20) if smoke else dict(tLimit=0.15)),
        'sedov2d': (sedovCase, dict(dim=2, nx=40, nSteps=20) if smoke else dict(dim=2, nx=128, tLimit=0.2)),
    }
    done = []
    for frame, (case, kw) in frames.items():
        for name in MAP_CONFIGS:
            cfg = A.CONFIGS[name]

            def configure(ctx, _orig=case.configureScheme, _cfg=cfg):
                _orig(ctx)
                for k, v in _cfg.switchParams.items():
                    setattr(ctx.schemeConfig.viscositySwitchParams, k, v)
                for k, v in A.effectiveDiffusion(_cfg).items():
                    setattr(ctx.schemeConfig.diffusionParams, k, v)

            _log(f'maps: {frame} / {name}')
            res = run(dataclasses.replace(case, configureScheme=configure), scheme=cfg.scheme,
                      params=dict(viscositySwitch=cfg.switch), plot=True, video=True,
                      exportRoot=str(mapDir / 'runs' / f'{frame}_{name}'), progress=True, quiet=True,
                      velocityAlarmPlotInterval=1, stallProgress=1e-3, **kw)
            st = res.state.state
            cpu = lambda x: None if x is None else x.detach().cpu().numpy()   # noqa: E731
            scalar = {'rosswog2020': getattr(st, 'entropyRates', None),
                      'wadsley2017': getattr(res.state.state, 'wadsleyD', None)}.get(name)
            phi = None
            if A.effectiveDiffusion(cfg).get('velocityPairPolicy', 0):
                from warpSPH.modules.reconstruction import computePairPhiMean
                phi = computePairPhiMean(res.state, res.ctx.config, res.ctx.schemeConfig.diffusionParams)
            np.savez(mapDir / f'{frame}_{name}.npz', t=float(res.state.t), uid=cpu(st.UIDs), positions=cpu(st.positions),
                     alpha=cpu(st.alphas), divergence=cpu(st.divergence), density=cpu(st.densities),
                     detector=cpu(scalar), phi=cpu(phi))
            done.append(f'{frame}_{name}')
        _drawMaps(mapDir, frame)
    return dict(name='maps', returncode=0, files=done)


def _drawMaps(mapDir: Path, frame: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    files = [(n, mapDir / f'{frame}_{n}.npz') for n in MAP_CONFIGS if (mapDir / f'{frame}_{n}.npz').exists()]
    if not files:
        return
    fig, axes = plt.subplots(2, len(files), figsize=(3.2 * len(files), 6.4), squeeze=False)
    for k, (name, path) in enumerate(files):
        d = np.load(path, allow_pickle=True)
        x = d['positions']
        for row, (key, cmap, vmax) in enumerate((('alpha', 'magma', 2.0), ('divergence', 'RdBu_r', None))):
            v = d[key]
            ax = axes[row, k]
            if v.ndim == 0 or v.dtype == object:
                ax.set_axis_off()
                continue
            lim = vmax if vmax is not None else np.percentile(np.abs(v), 99)
            sc = ax.scatter(x[:, 0], x[:, 1], c=v, s=1.5, cmap=cmap, vmin=0 if vmax else -lim, vmax=lim)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{name}\n{key}' if row == 0 else key, fontsize=8)
            fig.colorbar(sc, ax=ax, shrink=0.7)
    fig.suptitle(f'{frame}: detectors on the same frame (t = {float(np.load(files[0][1])["t"]):.3g})')
    fig.tight_layout()
    fig.savefig(mapDir / f'{frame}_detectors.png', dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- Phase 5B timing

def runTiming(out: Path, smoke: bool) -> Dict[str, Any]:
    """ms/step on Gresho for the switch configs: a warm-up run first (compile, caches), then the timed run, CUDA
    synchronised. Video OFF: this is a timing measurement."""
    import dataclasses
    import torch
    sys.path.insert(0, str(REPO / 'scripts'))
    import av_report as A
    from warpSPH.runner import run
    from warpSPH.cases.greshoVortex import greshoVortexCase

    nx, nWarm, nTimed = (32, 5, 20) if smoke else (128, 30, 300)
    rows = {}
    for name in ('none', 'cullenDehnen2010', 'sphenix', 'wadsley2017', 'rosswog2020', 'gsAVSLRB2', 'rosswogLimitedCoupled'):
        cfg = A.CONFIGS[name]

        def configure(ctx, _orig=greshoVortexCase.configureScheme, _cfg=cfg):
            _orig(ctx)
            for k, v in _cfg.switchParams.items():
                setattr(ctx.schemeConfig.viscositySwitchParams, k, v)
            for k, v in A.effectiveDiffusion(_cfg).items():
                setattr(ctx.schemeConfig.diffusionParams, k, v)

        case = dataclasses.replace(greshoVortexCase, configureScheme=configure)
        kw = dict(scheme=cfg.scheme, nx=nx, params=dict(viscositySwitch=cfg.switch), progress=False, quiet=True)
        run(case, nSteps=nWarm, **kw)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = run(case, nSteps=nTimed, **kw)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        rows[name] = dict(msPerStep=1e3 * wall / max(int(res.nSteps), 1), nSteps=int(res.nSteps), nx=nx)
        _log(f'timing {name}: {rows[name]["msPerStep"]:.2f} ms/step')
    (out / 'timing.json').write_text(json.dumps(rows, indent=2))
    return dict(name='timing', returncode=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--out', default=None)
    ap.add_argument('--smoke', action='store_true', help='every job at smoke size (pre-flight)')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--only', nargs='*', default=None, help='job names (plus maps, timing)')
    ap.add_argument('--timeoutFactor', type=float, default=3.0, help='a job is killed after factor x its estimate')
    ap.add_argument('--bakeoff', action='store_true', help='the AV_PLAN Phase 7 bake-off instead of the Phases 3-6 sweep')
    args = ap.parse_args()

    jobs = _bakeoffJobs() if args.bakeoff else JOBS
    names = [j['name'] for j in jobs] + ['maps', 'timing']
    selected = args.only or names
    total = sum(j['minutes'] for j in jobs if j['name'] in selected) + (20 if 'maps' in selected else 0) \
        + (10 if 'timing' in selected else 0)
    if args.dry:
        for j in jobs:
            if j['name'] in selected:
                print(f'{j["name"]:22s} {j["config"]:22s} {len(j["cases"]):2d} cases  ~{j["minutes"]:3d} min')
        print(f'total ~{total / 60:.1f} h (full profile)')
        return 0

    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    kind = 'av_bakeoff' if args.bakeoff else 'av_sweep'
    out = Path(args.out) if args.out else REPO / 'results' / f'{kind}_{"smoke_" if args.smoke else ""}{stamp}'
    out.mkdir(parents=True, exist_ok=True)
    _log(f'{len(selected)} jobs, ~{total / 60:.1f} h at the full profile -> {out}')
    status = dict(started=stamp, smoke=args.smoke, jobs=[])
    statusFile = out / 'status.json'
    for job in jobs:
        if job['name'] not in selected:
            continue
        status['jobs'].append(runJob(job, out, args.smoke, job['minutes'] * args.timeoutFactor
                                     + (10 if args.smoke else 0)))
        statusFile.write_text(json.dumps(status, indent=2))
    for name, fn in (('maps', runMaps), ('timing', runTiming)):
        if name in selected:
            try:
                status['jobs'].append(fn(out, args.smoke))
            except Exception as exc:          # noqa: BLE001 -- the sweep must survive a failing extra
                _log(f'{name} FAILED: {type(exc).__name__}: {exc}')
                status['jobs'].append(dict(name=name, returncode=f'{type(exc).__name__}: {exc}'))
            statusFile.write_text(json.dumps(status, indent=2))
    status['finished'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    statusFile.write_text(json.dumps(status, indent=2))
    if args.bakeoff:
        rc = subprocess.call([PY, str(REPO / 'scripts' / 'av_bakeoff_report.py'), str(out)], cwd=REPO)
        _log(f'done; bake-off report rc={rc}: {out / "report.md"}')
    else:
        rc = subprocess.call([PY, str(REPO / 'scripts' / 'av_sweep_verdict.py'), str(out)], cwd=REPO)
        _log(f'done; verdict rc={rc}: {out / "verdict.md"}')
    print('AV_SWEEP_DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
