"""Summary of the delta-SPH lone-particle reset batch (MDBC_CONTACT_LINE_PLAN.md
§12-13): every `scripts/out_contactLine/sloshing_default_nx0_cavoff*_trace`
run, grouped by variant (baseline / loneReset / loneReset + oneSidedHydro)
and realisation (unperturbed, `_j<seed>` = seeded IC jitter).

Per run: completion, fliers, ceiling riders, wall-sparse, lone-at-wall rows,
maxV / min rho, and Sensor 1 as the peak of the 0.05 s moving mean in the
two impact windows (t 1.5-2.5, 3.2-4.2), read two ways: `sensorPressure`
(the wall particle's own Tait pressure: goes through the mDBC closure) and
`sensorPressureProbe` (Gaussian fluid-side probe: does not). Then per
variant: mean and range over realisations.

Usage: python scripts/summarize_loneBatch.py [--out FILE]
"""
import argparse
import bisect
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out_contactLine')
WINDOWS = ((1.5, 2.5), (3.2, 4.2))
T_END = 4.2


def windowPeak(t, p, a, b, w=0.05):
    c = [0.0]
    for v in p:
        c.append(c[-1] + v)
    best = 0.0
    i0, i1 = bisect.bisect_left(t, a), bisect.bisect_right(t, b)
    for i in range(i0, i1):
        j = bisect.bisect_left(t, t[i] + w)
        best = max(best, (c[j] - c[i]) / max(1, j - i))
    return best / 1000.0


def summarize(d):
    r = json.load(open(os.path.join(d, 'record.json')))
    births = json.load(open(os.path.join(d, 'births.json'))) if os.path.exists(os.path.join(d, 'births.json')) else []
    trace = json.load(open(os.path.join(d, 'trace.json'))) if os.path.exists(os.path.join(d, 'trace.json')) else []
    g = lambda k: [x.get(k, 0) or 0 for x in r]
    t = [x['t'] for x in r]
    s = dict(tEnd=r[-1]['t'], steps=len(r),
             fliers=len({x['uid'] for x in births}), births=len(births),
             freeVmax=max(g('freeVmax')),
             ceilMax=max(g('nCeiling')), ceilMean=sum(g('nCeiling')) / len(r),
             wsMean=sum(g('nWallSparse')) / len(r),
             loneRows=sum(1 for x in trace if x['nF'] == 0),
             maxV=max(g('maxVelocity')), minRho=min(g('minDensity')),
             spikes=sum(1 for v in g('sensorPressure') if v > 2e4))
    for key, tag in (('sensorPressure', 'wall'), ('sensorPressureProbe', 'probe')):
        p = g(key)
        for k, (a, b) in enumerate(WINDOWS, 1):
            s[f'{tag}{k}'] = windowPeak(t, p, a, b)
    return s


def variantOf(name):
    v = 'loneReset + oneSidedHydro' if '_lone0_hyd1' in name else 'loneReset' if '_lone0' in name else 'baseline'
    m = re.search(r'_j(\d+)', name)
    return v, (f'seed {m.group(1)}' if m else 'unperturbed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(OUT, 'lone_batch_summary.md'))
    args = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(OUT, 'sloshing_default_nx0_cavoff*_trace'))):
        if not os.path.exists(os.path.join(d, 'record.json')):
            continue
        v, real = variantOf(os.path.basename(d))
        try:
            rows.append((v, real, os.path.basename(d), summarize(d)))
        except Exception as e:  # a half-written record from an interrupted run
            rows.append((v, real, os.path.basename(d), dict(error=str(e))))
    order = {'baseline': 0, 'loneReset': 1, 'loneReset + oneSidedHydro': 2}
    rows.sort(key=lambda x: (order[x[0]], x[1]))

    L = ['# delta-SPH lone-particle reset batch: sloshingTank t = 0-4.2', '',
         'Sensor 1 = peak of the 0.05 s moving mean (kPa), impact 1 (t 1.5-2.5) / impact 2 (t 3.2-4.2); '
         '`wall` = the wall particle\'s own Tait pressure (through the mDBC closure), `probe` = fluid-side '
         'Gaussian probe (not). TC10 measured band 2.2-13.1 kPa. "incomplete" = the run did not reach t = 4.2.', '',
         '| variant | realisation | tEnd | fliers (uids/births, vmax) | ceiling max/mean | wall-sparse mean | lone rows | maxV / min rho | S1 wall 1/2 | S1 probe 1/2 | steps > 20 kPa |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for v, real, name, s in rows:
        if 'error' in s:
            L.append(f'| {v} | {real} | ERROR {s["error"][:60]} |||||||||')
            continue
        done = '' if s['tEnd'] >= T_END - 1e-3 else ' (incomplete)'
        L.append(f"| {v} | {real} | {s['tEnd']:.2f}{done} | {s['fliers']}/{s['births']}, {s['freeVmax']:.1f} | "
                 f"{s['ceilMax']}/{s['ceilMean']:.2f} | {s['wsMean']:.2f} | {s['loneRows']} | "
                 f"{s['maxV']:.2f} / {s['minRho']:.3f} | {s['wall1']:.2f} / {s['wall2']:.2f} | "
                 f"{s['probe1']:.2f} / {s['probe2']:.2f} | {s['spikes']} |")
    L += ['', '## Per variant over complete realisations (mean [min, max])', '',
          '| variant | n | ceiling mean | wall-sparse mean | fliers | S1 wall impact 2 | S1 probe impact 2 | S1 probe impact 1 |',
          '|---|---|---|---|---|---|---|---|']
    for v in order:
        ss = [s for vv, _, _, s in rows if vv == v and 'error' not in s and s['tEnd'] >= T_END - 1e-3]
        if not ss:
            continue
        f = lambda k: f"{sum(x[k] for x in ss) / len(ss):.2f} [{min(x[k] for x in ss):.2f}, {max(x[k] for x in ss):.2f}]"
        L.append(f"| {v} | {len(ss)} | {f('ceilMean')} | {f('wsMean')} | {f('fliers')} | {f('wall2')} | {f('probe2')} | {f('probe1')} |")
    L += ['', 'Run folders (videos inside): ' + ', '.join(f'`{n}`' for _, _, n, _ in rows)]
    text = '\n'.join(L) + '\n'
    with open(args.out, 'w') as fh:
        fh.write(text)
    print(text)


if __name__ == '__main__':
    main()
