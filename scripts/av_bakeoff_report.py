#!/usr/bin/env python3
"""AV_PLAN Phase 7: the bake-off report -- Part 0's table for every matrix row (`av_report.BAKEOFF_ROWS`).

Reads a `scripts/av_sweep_overnight.py --bakeoff` output directory and writes `report.md` there and to
`results/av_foundation/report.md` (the plan's deliverable path): the matrix, one table per case with the Part 0
groups (A shock quality, B dissipation accounting, C smooth flow, D instability growth, E conservation), the plan's
hard gates flagged, the best row per headline metric, the CompSPH cross-check, KH at nx 256, timing (Group F) and the
detector maps. The per-row prose is written by hand into AV_PLAN after reading this.

    python scripts/av_bakeoff_report.py results/av_bakeoff_<date>
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import av_report as A   # noqa: E402

#: KH nx 256 rows from earlier runs (smooth IC, t = 3) for the configs the bake-off does not re-run at nx 256
KH256_EARLIER = [
    ('docs/av/kh256_2026-10-07', 'Monaghan + C&D (alpha range default, C_q 0)', 0.1374, 0.2124, 0.1071),
    ('docs/av/kh256_2026-10-07', 'Monaghan + Rosswog 2020 (C_q 0)', 0.1311, 0.2086, 0.1045),
    ('docs/av/kh256_2026-10-07', 'CRKSPH default', 0.1446, 0.2364, 0.1873),
    ('results/av_sweep_2026-10-07_17-30', 'gsAVSLRB2 (row 9)', 0.1381, 0.2082, 0.1177),
    ('results/av_sweep_2026-10-07_17-30', 'gsAVSLR (row 8)', 0.1393, 0.2089, 0.1214),
    ('results/av_sweep_2026-10-07_17-30', 'rosswogLimited (row 10)', 0.1511, 0.2240, 0.1230),
    ('results/av_sweep_2026-10-07_17-30', 'wadsley2017 (row 7)', 0.1330, 0.2120, 0.1090),
]

GROUP_B = ['alphaMean', 'alphaActiveFraction', 'avEnergyTotal', 'avEnergyQuadratic', 'chenNixonRatioMedian']
GROUP_E = ['energyDrift', 'entropyGain']
#: headline metric per case: (metric, 'min' | 'max' | 'absmin', label)
HEADLINE = {
    'sod': ('L1_vx', 'min', 'Sod L1(v_x)'),
    'sod2d': ('L1_vx', 'min', 'Sod 2D L1(v_x)'),
    'sod3d': ('L1_vx', 'min', 'Sod 3D L1(v_x)'),
    'sedov': ('shockRadiusErr', 'absmin', 'Sedov |shock radius error|'),
    'noh': ('postShockRhoErr', 'absmin', 'Noh |post-shock rho error|'),
    'noh2d': ('preShockAlphaMean', 'min', 'cylindrical Noh pre-shock alpha'),
    'gresho': ('L1_vphi', 'min', 'Gresho L1(v_phi)'),
    'yee': ('L1_speed', 'min', 'Yee L1(speed)'),
    'linearWave': ('waveVelocityErr', 'absmin', 'linear wave |velocity error|'),
    'kelvinHelmholtz': ('khAmplitudeAt1p5', 'max', 'KH A(1.5) (nx 128)'),
    'rayleighTaylor': ('mixingWidth', 'max', 'RT mixing width'),
    'shearBox': ('modeAmplitudeFinal', 'max', 'shear box surviving mode'),
}


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return 'nan' if isinstance(v, float) and math.isnan(v) else f'{v:.4g}'
    return '-' if v is None else str(v)


def _load(out: Path) -> Dict[tuple, Dict[str, Any]]:
    rec: Dict[tuple, Dict[str, Any]] = {}
    for res in sorted(out.glob('*/results.json')):
        for r in json.loads(res.read_text())['records']:
            key = (r['config'], r['case'], 'kh256' if res.parent.name.startswith('kh256_') else 'std')
            rec[key] = r['metrics']
    return rec


def _gates(case: str, m: Dict[str, Any]) -> List[str]:
    out = []
    if m.get('diverged'):
        out.append('DIVERGED')
    if case == 'sedov' and m.get('peakOvershoots'):
        out.append('Sedov overshoot (gate: approach 4 from below)')
    if case in ('noh', 'noh2d') and isinstance(m.get('postShockRhoErr'), float) and abs(m['postShockRhoErr']) > 0.03:
        out.append(f'post-shock rho {m["postShockRhoErr"]:+.1%} (gate +-3 %)')
    drift, budget = m.get('energyDrift'), m.get('energyDriftBudget')
    if isinstance(drift, float) and isinstance(budget, float) and drift > budget:
        out.append(f'energy drift {drift:.1e} > {budget:.0e}')
    return out


def main() -> int:
    out = Path(sys.argv[1])
    rec = _load(out)
    status = json.loads((out / 'status.json').read_text()) if (out / 'status.json').exists() else {}
    timing = json.loads((out / 'timing.json').read_text()) if (out / 'timing.json').exists() else {}
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    rows = A.BAKEOFF_ROWS
    L: List[str] = [
        '# AV_PLAN Phase 7 -- the AV bake-off', '',
        f'Run `{out.name}` (code {commit}); Monaghan host, every row with video; CompSPH cross-check on the shortlist. '
        'Part 0 groups per case: **A** shock quality, **B** dissipation accounting, **C** smooth flow, '
        '**D** instability growth, **E** conservation; **F** (timing) at the end. Gates from Part 0 are flagged in '
        'the last column.', '',
        '## The matrix', '', '| row | config | detector | pair velocity | beta | runs | diverged |', '|---|---|---|---|---|---|---|']
    for row, cfg, det, pv, beta in rows:
        cases = [k for k in rec if k[0] == cfg and k[2] == 'std']
        div = sum(1 for k in cases if rec[k].get('diverged'))
        L.append(f'| {row} | `{cfg}` | {det} | {pv} | {beta} | {len(cases)} | {div} |')

    # best row per headline metric
    L += ['', '## Best row per headline metric', '', '| metric | best | value | runner-up | value | row 12 (default) |',
          '|---|---|---|---|---|---|']
    for case, (metric, mode, label) in HEADLINE.items():
        vals = []
        for row, cfg, *_ in rows:
            m = rec.get((cfg, case, 'std'))
            v = None if m is None or m.get('diverged') else m.get(metric)
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                vals.append((row, cfg, float(v)))
        if not vals:
            continue
        key = {'min': lambda t: t[2], 'max': lambda t: -t[2], 'absmin': lambda t: abs(t[2])}[mode]
        vals.sort(key=key)
        dflt = next((v for r, c, v in vals if r == 12), None)
        second = vals[1] if len(vals) > 1 else (None, '-', float('nan'))
        L.append(f'| {label} | {vals[0][0]} `{vals[0][1]}` | {_fmt(vals[0][2])} | {second[0]} `{second[1]}` | '
                 f'{_fmt(second[2])} | {_fmt(dflt)} |')

    # Part 0 table per case
    cases = [c for c in A.DEFAULT_CASES + ['shearBox', 'noh2d'] if any(k[1] == c for k in rec)]
    for case in cases:
        cols = A.COLUMNS.get(case, []) + GROUP_B + GROUP_E
        L += ['', f'### {case}', '', '| row | config | ' + ' | '.join(cols) + ' | gates |', '|' + '---|' * (len(cols) + 3)]
        for row, cfg, *_ in rows:
            m = rec.get((cfg, case, 'std'))
            if m is None:
                continue
            L.append(f'| {row} | `{cfg}` | ' + ' | '.join(_fmt(m.get(c)) for c in cols) + ' | '
                     + ('; '.join(_gates(case, m)) or '-') + ' |')

    # CompSPH cross-check
    comp = sorted({k[0] for k in rec if k[0].startswith('comp')})
    if comp:
        L += ['', '## CompSPH cross-check (switch only)', '']
        for case in ('sod', 'sod2d', 'sedov', 'noh', 'gresho', 'kelvinHelmholtz', 'noh2d'):
            cols = A.COLUMNS.get(case, []) + ['alphaMean', 'energyDrift']
            L += [f'**{case}**', '', '| config | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 1)]
            for cfg in comp:
                m = rec.get((cfg, case, 'std'))
                if m is not None:
                    L.append(f'| `{cfg}` | ' + ' | '.join(_fmt(m.get(c)) for c in cols) + ' |')
            L.append('')

    # KH nx 256
    L += ['', '## Kelvin-Helmholtz at nx 256 (smooth IC, t = 3)', '', '| run | A(1.5) | peak A | A(t = 3) | source |',
          '|---|---|---|---|---|']
    for (cfg, case, kind), m in sorted(rec.items()):
        if kind == 'kh256':
            L.append(f'| `{cfg}` (this run) | {_fmt(m.get("khAmplitudeAt1p5"))} | {_fmt(m.get("khAmplitudeMax"))} | '
                     f'{_fmt(m.get("khAmplitudeFinal"))} | {out.name} |')
    for src, name, a15, amax, a3 in KH256_EARLIER:
        L.append(f'| {name} | {a15:.4g} | {amax:.4g} | {a3:.4g} | {src} |')
    L.append('| McNally et al. (2012) reference | 0.1479 | | | |')

    # timing
    if timing:
        L += ['', '## Group F -- cost (Gresho nx 128, warmed up, CUDA-synchronised, no video)', '',
              '| config | ms/step |', '|---|---|']
        for cfg, v in timing.items():
            L.append(f'| `{cfg}` | {v["msPerStep"]:.2f} |')

    maps = sorted((out / 'maps').glob('*_detectors.png')) if (out / 'maps').exists() else []
    if maps:
        L += ['', '## Detector maps (same IC, same end time)', ''] + [f'![{m.stem}](maps/{m.name})' for m in maps]
    if status.get('jobs'):
        L += ['', '## Jobs', '', '| job | rc | minutes |', '|---|---|---|']
        L += [f'| {j["name"]} | {j.get("returncode")} | {j.get("minutes", float("nan")):.1f} |' for j in status['jobs']]

    text = '\n'.join(L) + '\n'
    (out / 'report.md').write_text(text)
    dest = REPO / 'results' / 'av_foundation'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'report.md').write_text(text.replace('](maps/', f'](../{out.name}/maps/'))
    print(f'bake-off report -> {out / "report.md"} and {dest / "report.md"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
