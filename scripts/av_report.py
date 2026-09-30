#!/usr/bin/env python3
"""The AV report (AV_PLAN.md Part 0): one table of metrics per (config, case).

    scripts/av_report.py --list
    scripts/av_report.py --config baseline --profile smoke
    scripts/av_report.py --config baseline --profile full           # video on
    scripts/av_report.py --config baseline --cases sod --repeat 2   # repro lock
    scripts/av_report.py --compare results/av_A results/av_B

A *config* is a named dissipation setup on a host scheme (`baseline` = Monaghan
with NoneSwitch / Cullen-Dehnen / Read-Hayfield). A *case* knows how to run itself
at each profile and how to turn the finished run into Part 0 metrics. Output goes
to `results/av_<stamp>/{results.json, report.md}` with the `meta` block of
`benchmarks/common/report.py` (versions + git SHAs).

Runs go one at a time, in this process (GPU runs are launch-bound; parallel ones
only slow each other down). `--profile full` records video and streams the
runner's progress rows; `smoke` (coarse, ~tens of steps) does neither -- it is a
plumbing check, not a physics run.

`--repeat N` runs every (config, case) N times and compares every scalar in the
records: the M0 reproducibility lock is `max relative difference < 1e-6`.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:  # `warpSPHBootstrap` resolves the sibling repos when run from the repo root
    import warpSPHBootstrap  # noqa: F401
except ImportError:
    pass

from benchmarks.common import report as rep                       # noqa: E402
from benchmarks.common.metrics import L1, fmt                      # noqa: E402

#: `tests/test_physics.py:_ENERGY_DRIFT` -- the existing per-scheme budget (Group E).
ENERGY_DRIFT = {'CompSPH': 1e-5, 'CRKSPH': 1e-4, 'Monaghan': 5e-3}
LOCK_TOL = 1e-6


# --------------------------------------------------------------------------- configs

@dataclass
class AVConfig:
    """A named dissipation setup: the host scheme and the switch, plus any case params."""
    name: str
    scheme: str = 'Monaghan'
    switch: str = 'NoneSwitch'
    params: Dict[str, Any] = field(default_factory=dict)


#: R&H 2012 is designed with a narrower, higher-baseline alpha range
#: (`probe_cullen_sod.py`, `tests/test_shockCapturing.py`).
_RH = dict(alpha_min=0.2, alpha_max=1.0)

CONFIGS: Dict[str, AVConfig] = {c.name: c for c in (
    AVConfig('none', switch='NoneSwitch'),
    AVConfig('cullenDehnen2010', switch='CullenDehnen2010'),
    AVConfig('readHayfield2012', switch='ReadHayfield2012', params=_RH),
)}

#: Named groups, so `--config baseline` runs the three M0 columns.
GROUPS: Dict[str, List[str]] = {
    'baseline': ['none', 'cullenDehnen2010', 'readHayfield2012'],
}


# --------------------------------------------------------------------------- run plumbing

@dataclass
class RunSpec:
    """What one case run needs: the case object, run kwargs, extra params."""
    case: Any
    kwargs: Dict[str, Any]
    params: Dict[str, Any] = field(default_factory=dict)


def _execute(spec: RunSpec, cfg: AVConfig, outDir: Path, tag: str, video: bool,
             avStride: int) -> Any:
    """Run one case under `cfg`; returns the `RunResult`, with the AV extras
    (`avReportDiagnostics`) recorded every `avStride` steps."""
    import dataclasses
    from warpSPH.runner import run
    from warpSPH.cases.compressible import avReportDiagnostics

    base = spec.case.diagnostics
    counter = [0]

    def diagnostics(ctx, state):
        row = base(ctx, state)
        if counter[0] % avStride == 0:
            row.update(avReportDiagnostics(ctx, state))
        counter[0] += 1
        return row

    case = dataclasses.replace(spec.case, diagnostics=diagnostics)
    kw = dict(spec.kwargs)
    kw.setdefault('scheme', cfg.scheme)
    params = dict(spec.params, viscositySwitch=cfg.switch, **cfg.params)
    if video:
        kw.update(plot=True, video=True, exportRoot=str(outDir / 'runs' / tag),
                  velocityAlarmPlotInterval=1)
    kw.update(progress=video, stallProgress=1e-3, quiet=True)
    return run(case, params=params, **kw)


def _trap(t: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, t)) if len(t) > 1 else float('nan')


def _common(res, cfg: AVConfig) -> Dict[str, Any]:
    """Metrics every case reports: Group B (alpha, AV energy), E (drift), F (cost)."""
    traj = res.trajectory
    out: Dict[str, Any] = {'diverged': bool(res.diverged), 'nSteps': int(res.nSteps),
                           'tFinal': float(res.state.t)}
    E = np.array([r['totalEnergy'] for r in traj])
    out['energyDrift'] = float(abs(E[-1] - E[0]) / abs(E[0]))
    out['energyDriftBudget'] = ENERGY_DRIFT.get(cfg.scheme, float('nan'))
    out['entropyGain'] = float(traj[-1]['entropy'] - traj[0]['entropy'])
    last = traj[-1]
    for k in ('alphaMean', 'alphaMax', 'alphaActiveFraction'):
        out[k] = float(last[k])
    rows = [r for r in traj if 'avPowerTotal' in r]
    if rows and all(math.isfinite(r['avPowerTotal']) for r in rows):
        t = np.array([r['t'] for r in rows])
        for k in ('Total', 'Linear', 'Quadratic'):
            out[f'avEnergy{k}'] = _trap(t, np.array([r[f'avPower{k}'] for r in rows]))
    nb = [r for r in traj if 'neighboursMean' in r]
    if nb:
        out['neighboursMean'] = float(np.mean([r['neighboursMean'] for r in nb]))
        out['neighboursMax'] = float(max(r['neighboursMax'] for r in nb))
    out['wallMsPerStep'] = 1e3 * res.wallTime / max(res.nSteps, 1)   # coarse: setup included
    return out


# --------------------------------------------------------------------------- Sod (1D)

SOD_WINDOW = (0.0, 1.0, 0.5)           # exact-solution window [xl, xr], interface
SOD_L1_RANGE = (0.05, 0.868)           # García-Senz Eq. (20) window
SOD_IC = ((1.0, 1.0, 0.0), (0.1, 0.125, 0.0))   # (p, rho, u) left / right
GAMMA = 5 / 3


def _sodSpec(profile: str) -> RunSpec:
    from warpSPH.cases.sod import sodCase
    nx, nSteps = (100, 60) if profile == 'smoke' else (400, 400)
    return RunSpec(sodCase, dict(nx=nx, nSteps=nSteps),
                   dict(right_rho=0.125, right_pressure=0.1))


def _sodMetrics(res, cfg: AVConfig) -> Dict[str, Any]:
    from warpSPH.caseUtils.compressible.sod.sodSolution import solve
    out = _common(res, cfg)
    st = res.state.state
    t = float(res.state.t)
    x = st.positions[:, 0].detach().cpu().numpy()
    rho = st.densities.detach().cpu().numpy()
    P = st.pressures.detach().cpu().numpy()
    u = st.velocities[:, 0].detach().cpu().numpy()
    A = P / rho ** GAMMA
    pos, regions, vals = solve(SOD_IC[0], SOD_IC[1], SOD_WINDOW, t, gamma=GAMMA, npts=2000)

    m = (x >= SOD_L1_RANGE[0]) & (x <= SOD_L1_RANGE[1])
    out['L1_vx'] = L1(torch.tensor(u), torch.tensor(np.interp(x, vals['x'], vals['u'])),
                      torch.tensor(m))

    # contact spike: max P / A in a window around the exact contact, vs the exact R4 value
    cx = pos['Contact Discontinuity']
    sel = (vals['x'] > cx) & (vals['x'] < cx + 0.04)
    p4, rho4 = float(np.median(vals['p'][sel])), float(np.median(vals['rho'][sel]))
    w = (x > cx - 0.07) & (x < cx + 0.07)
    out['contactSpikeP'] = float(P[w].max() / p4 - 1.0)
    out['contactSpikeA'] = float(A[w].max() / (p4 / rho4 ** GAMMA) - 1.0)

    def regionErr(key, arr, lo, hi, ex):
        s = (x >= lo) & (x <= hi)
        return float(arr[s].mean() / ex - 1.0) if s.any() else float('nan')
    d = 0.03
    R3, R4 = regions['Region 3'], regions['Region 4']          # (p, rho, u)
    out['R3_rho_err'] = regionErr('rho', rho, pos['Foot of Rarefaction'] + d, cx - d, R3[1])
    out['R4_rho_err'] = regionErr('rho', rho, cx + d, pos['Shock'] - d, R4[1])
    out['R4_P_err'] = regionErr('P', P, cx + d, pos['Shock'] - d, R4[0])
    return out


# --------------------------------------------------------------------------- Gresho (2D)

def _greshoSpec(profile: str) -> RunSpec:
    from warpSPH.cases.greshoVortex import greshoVortexCase
    if profile == 'smoke':
        return RunSpec(greshoVortexCase, dict(nx=32, nSteps=40))
    return RunSpec(greshoVortexCase, dict(nx=100, tLimit=3.0))


def _greshoExact(r: np.ndarray) -> np.ndarray:
    return np.where(r < 0.2, 5 * r, np.where(r < 0.4, 2 - 5 * r, 0.0))


def _greshoMetrics(res, cfg: AVConfig) -> Dict[str, Any]:
    out = _common(res, cfg)
    st = res.state.state
    pos = st.positions.detach().cpu().numpy()
    vel = st.velocities.detach().cpu().numpy()
    r = np.linalg.norm(pos, axis=1)
    vphi = (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]) / np.maximum(r, 1e-12)
    m = r <= 0.5
    out['L1_vphi'] = L1(torch.tensor(vphi), torch.tensor(_greshoExact(r)), torch.tensor(m))
    speed = np.linalg.norm(vel, axis=1)
    speed0 = res.trajectory[0]
    out['peakSpeed'] = float(speed.max())
    out['angularMomentumLoss'] = float(1.0 - res.trajectory[-1]['angularMomentum']
                                       / res.trajectory[0]['angularMomentum'])
    return out


@dataclass
class CaseDef:
    name: str
    spec: Callable[[str], RunSpec]
    metrics: Callable[[Any, AVConfig], Dict[str, Any]]
    avStride: Callable[[str], int] = lambda profile: 1


CASES: Dict[str, CaseDef] = {c.name: c for c in (
    CaseDef('sod', _sodSpec, _sodMetrics),
    CaseDef('gresho', _greshoSpec, _greshoMetrics, lambda p: 1 if p == 'smoke' else 5),
)}


# --------------------------------------------------------------------------- driver

def runOne(cfg: AVConfig, case: CaseDef, profile: str, outDir: Path, video: bool,
           tag: str) -> Dict[str, Any]:
    spec = case.spec(profile)
    started = time.perf_counter()
    res = _execute(spec, cfg, outDir, tag, video, case.avStride(profile))
    metrics = case.metrics(res, cfg)
    metrics['videoPath'] = res.videoPath
    return dict(config=cfg.name, case=case.name, profile=profile, scheme=cfg.scheme,
                switch=cfg.switch, seconds=time.perf_counter() - started, metrics=metrics)


def _scalars(rec: Dict[str, Any]) -> Dict[str, float]:
    return {k: v for k, v in rec['metrics'].items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and k not in ('wallMsPerStep',)}          # timing is not part of the lock


def lockCheck(reps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Max relative difference over every scalar between repeated runs."""
    worst, where = 0.0, None
    a = _scalars(reps[0])
    for other in reps[1:]:
        b = _scalars(other)
        for k, va in a.items():
            vb = b.get(k)
            if vb is None or (math.isnan(va) and math.isnan(vb)):
                continue
            rel = abs(va - vb) / max(abs(va), abs(vb), 1e-300) if va != vb else 0.0
            if rel > worst:
                worst, where = rel, k
    return dict(maxRelDiff=worst, worstMetric=where, locked=worst < LOCK_TOL)


COLUMNS = {
    'sod': ['L1_vx', 'contactSpikeP', 'contactSpikeA', 'R3_rho_err', 'R4_rho_err', 'R4_P_err'],
    'gresho': ['L1_vphi', 'peakSpeed', 'angularMomentumLoss'],
}
COMMON_COLUMNS = ['alphaMean', 'alphaMax', 'alphaActiveFraction', 'avEnergyLinear',
                  'avEnergyQuadratic', 'entropyGain', 'energyDrift', 'neighboursMean',
                  'wallMsPerStep']


def renderMarkdown(records: List[Dict[str, Any]], locks: Dict[str, Any], meta: Dict[str, Any],
                   profile: str) -> List[tuple]:
    sections = []
    for case in dict.fromkeys(r['case'] for r in records):
        cols = COLUMNS.get(case, []) + COMMON_COLUMNS
        rows = []
        for r in (x for x in records if x['case'] == case):
            m = r['metrics']
            flag = ' **DIVERGED**' if m.get('diverged') else ''
            rows.append([r['config'] + flag] + [fmt(m.get(c), '.4g') for c in cols])
        sections.append((f'{case} ({profile})', rep.mdTable(['config'] + cols, rows), []))
        budgets = [(r['config'], r['metrics']['energyDrift'], r['metrics']['energyDriftBudget'])
                   for r in records if r['case'] == case]
        over = [f'{c} ({d:.2e} > {b:.0e})' for c, d, b in budgets if d > b]
        sections.append((f'{case}: energy-drift budget (Group E)',
                         'all within budget' if not over else 'OVER: ' + ', '.join(over), []))
    if locks:
        rows = [[k, f"{v['maxRelDiff']:.2e}", v['worstMetric'] or '-',
                 'LOCKED' if v['locked'] else 'NOT LOCKED'] for k, v in locks.items()]
        sections.append((f'Reproducibility lock (tol {LOCK_TOL:g})',
                         rep.mdTable(['run', 'max rel diff', 'worst metric', 'state'], rows), []))
    sections.append(('meta', '```\n' + '\n'.join(f'{k}: {v}' for k, v in meta.items()) + '\n```', []))
    return sections


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--config', default='baseline', help='config or group name (see --list)')
    ap.add_argument('--profile', choices=('smoke', 'full'), default='smoke')
    ap.add_argument('--cases', nargs='*', default=None)
    ap.add_argument('--repeat', type=int, default=1, help='runs per (config, case); >1 checks the lock')
    ap.add_argument('--video', dest='video', action='store_true', default=None)
    ap.add_argument('--noVideo', dest='video', action='store_false')
    ap.add_argument('--out', default=None)
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    if args.list:
        print('configs:', ', '.join(CONFIGS), '\ngroups: ',
              ', '.join(f'{g}={v}' for g, v in GROUPS.items()), '\ncases:  ', ', '.join(CASES))
        return 0

    names = GROUPS.get(args.config, [args.config])
    cases = [CASES[c] for c in (args.cases or CASES)]
    video = (args.profile == 'full') if args.video is None else args.video
    outDir = rep.outDirFor('av', args.out)
    meta = rep.environmentMeta(extra=dict(profile=args.profile, config=args.config, repeat=args.repeat,
                                          video=video))
    print(f'[av_report] {names} x {[c.name for c in cases]}  profile={args.profile} '
          f'repeat={args.repeat} video={video}\n[av_report] -> {outDir}', flush=True)

    records: List[Dict[str, Any]] = []
    locks: Dict[str, Any] = {}
    for name in names:
        for case in cases:
            reps = []
            for i in range(args.repeat):
                print(f'[av_report] {name} / {case.name}  (run {i + 1}/{args.repeat})', flush=True)
                reps.append(runOne(CONFIGS[name], case, args.profile, outDir,
                                   video and i == 0, f'{name}_{case.name}'))
            records.append(reps[0])
            if args.repeat > 1:
                locks[f'{name}/{case.name}'] = lockCheck(reps)

    rep.writeResults(outDir, 'av', meta, records)
    rep.writeSummary(outDir, f'AV report ({args.profile})', renderMarkdown(records, locks, meta, args.profile))
    (outDir / 'summary.md').rename(outDir / 'report.md')
    if locks:
        (outDir / 'locks.json').write_text(__import__('json').dumps(locks, indent=2))
    print(f'[av_report] wrote {outDir}/report.md')
    return 0 if all(v['locked'] for v in locks.values()) else (0 if not locks else 2)


if __name__ == '__main__':
    sys.exit(main())
