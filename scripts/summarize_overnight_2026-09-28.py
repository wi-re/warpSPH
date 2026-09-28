"""Summarise scratchpad/run_overnight_batch_2026-09-28.sh -> <base>/SUMMARY.md.

The question the batch answers: after the fourtakas2019 sign fix (68a9a6d),
does the fixed `fourtakas2019` DDT still earn its place as the default over
`deltaSPH`? Tables per case, one row per run, plus the gallery's pass/fail
lines and every velocity-alarm / stop line from the logs.

    python scripts/summarize_overnight_2026-09-28.py scripts/out_overnight_2026-09-28
"""
import glob
import json
import os
import re
import sys

import numpy as np


def _meta(d):
    try:
        m = d['meta']
        m = m.item() if m.shape == () else m
        return json.loads(m) if isinstance(m, str) else dict(m)
    except Exception:  # noqa: BLE001
        return {}


def _ddt(name):
    return 'deltaSPH' if 'ddt-deltaSPH' in name else ('fourtakas2019' if 'ddt-fourtakas2019' in name else '?')


def m31(base, out):
    out += ['## Marrone 3.1 (P1 = pProbe0Star, P2 = pProbe1Star; Buchner: P1 plateau ~0.55, P2 peak ~0.28 at t* ~5.5)', '',
            '| run | DDT | diverged | t* reached | max\\|v\\| | max\\|v\\| t*>5 | P1 plateau (3.2-4.8) | P1 peak | P2 peak | P2 peak t* | rho min/max |',
            '|---|---|---|---|---|---|---|---|---|---|---|']
    for f in sorted(glob.glob(os.path.join(base, 'm31', '*.npz'))):
        d = np.load(f, allow_pickle=True)
        name = os.path.basename(f)[:-4]
        ts, v = d['tStar'], d['maxVelocity']
        p1, p2 = d['pProbe0Star'], d['pProbe1Star']
        late, plat = ts > 5.0, (ts > 3.2) & (ts < 4.8)
        m = _meta(d)
        i2 = int(np.nanargmax(p2)) if np.isfinite(p2).any() else 0
        out.append(f"| {name} | {_ddt(name)} | {m.get('diverged', '?')} | {np.nanmax(ts):.2f} | {np.nanmax(v):.2f} | "
                   f"{np.nanmax(v[late]) if late.any() else float('nan'):.2f} | {np.nanmean(p1[plat]) if plat.any() else float('nan'):.3f} | "
                   f"{np.nanmax(p1):.3f} | {np.nanmax(p2):.3f} | {ts[i2]:.2f} | {np.nanmin(d['minDensity']):.3f}/{np.nanmax(d['maxDensity']):.3f} |")
    out.append('')


def m34(base, out):
    out += ['## Marrone 3.4 (PASS/FAIL lines from each log; key series maxima)', '']
    for f in sorted(glob.glob(os.path.join(base, 'm34', '*.npz'))):
        d = np.load(f, allow_pickle=True)
        name = os.path.basename(f)[:-4]
        m = _meta(d)
        probes = [k for k in d.files if re.fullmatch(r's_pSurf\d', k)]
        pk = ', '.join(f"{k[2:]} {np.nanmax(d[k]):.1f}" for k in probes)
        out.append(f"- **{name}** ({_ddt(name)}): diverged={m.get('diverged', '?')}, t* {m.get('tStarReached', float('nan')):.2f}, "
                   f"max|v| {np.nanmax(d['s_maxVelocity']):.2f}, rho [{np.nanmin(d['s_minDensity']):.3f}, {np.nanmax(d['s_maxDensity']):.3f}], "
                   f"obstacle pen {np.nanmax(d['s_maxObstaclePenDx']):.2f} dx; probe peaks (raw): {pk}")
    for log in sorted(glob.glob(os.path.join(base, 'D_m34_*.log'))):
        lines = [l.strip() for l in open(log, errors='replace') if re.match(r'\s+(PASS|FAIL)\s', l)]
        out.append(f"  - `{os.path.basename(log)}`: " + ('; '.join(dict.fromkeys(lines)) or 'no PASS/FAIL lines'))
    out.append('')


def slosh(base, out):
    out += ['## sloshingTank t=7 (Sensor 1; measured impact-peak band 2.2-13.1 kPa)', '',
            '| DDT | per-0.5 s window max, wall sensor (kPa) from t=2 | fluid probe (kPa) | max\\|v\\| |', '|---|---|---|---|']
    for f in sorted(glob.glob(os.path.join(base, 'sloshing_*', 'wcsph_series.npz'))):
        d = np.load(f, allow_pickle=True)
        t = d['t']

        def win(k, scale=1e3):
            v = d[k]
            tt = t if len(v) == len(t) else np.linspace(t[0], t[-1], len(v))
            return ' '.join(f"{np.nanmax(np.abs(v[(tt >= a) & (tt < a + 0.5)])) / scale:.1f}" for a in np.arange(2.0, 7.0, 0.5))
        ddt = os.path.basename(os.path.dirname(f)).replace('sloshing_', '')
        out.append(f"| {ddt} | {win('sensorPressure')} | {win('sensorPressureProbe')} | {win('maxVelocity', 1.0)} |")
    out.append('')


def logs(base, out):
    out += ['## Alarms, stops and crashes in the logs', '']
    for log in sorted(glob.glob(os.path.join(base, '*.log'))):
        hits = [l.strip()[:200] for l in open(log, errors='replace')
                if re.search(r'VELOCITY ALARM (raised|stop)|stops the run|non-finite|Traceback|Error:|DIVERGED', l)]
        if hits:
            out.append(f"- `{os.path.basename(log)}`: {len(hits)} line(s); first: {hits[0]}")
    status = os.path.join(base, 'status.txt')
    if os.path.exists(status):
        bad = [l.strip() for l in open(status) if ' end ' in l and not l.strip().endswith('rc=0')]
        out.append('')
        out.append('Non-zero exits: ' + ('; '.join(bad) if bad else 'none'))
    out.append('')


def gallery(base, out):
    out += ['## Gallery (render_examples.py, shipped settings)', '']
    log = os.path.join(base, 'F_gallery.log')
    if os.path.exists(log):
        keep = [l.rstrip() for l in open(log, errors='replace')
                if re.search(r'(FAIL|OK|ok|failed|timed out|error|diverged|DIVERGED|summary|SUMMARY|\[\d+/\d+\])', l)]
        out += ['```'] + keep[-80:] + ['```']
    else:
        out.append('(no gallery log)')
    out.append('')


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else 'scripts/out_overnight_2026-09-28'
    out = [f'# Overnight 2026-09-28 -- fixed fourtakas2019 vs deltaSPH DDT, and the gallery', '']
    for part in (m31, m34, slosh, logs, gallery):
        try:
            part(base, out)
        except Exception as ex:  # noqa: BLE001 -- a partial summary beats none
            out += [f'({part.__name__} failed: {type(ex).__name__}: {ex})', '']
    path = os.path.join(base, 'SUMMARY.md')
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')
    print('\n'.join(out))
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
