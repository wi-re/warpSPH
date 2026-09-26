"""Per-contact analysis of `probe_contactLine.py`'s flier trace
(`trace.json`, MDBC_CONTACT_LINE_PLAN.md §11.1).

A tracked particle's trace is split into *contacts*: maximal runs of steps
with nF > 2 (supported by fluid), each preceded by a free step (nF <= 2). For
every contact it prints

  pIn      the particle's own pressure on the last free step before contact
           (what it carried in: frozen, since an isolated row's p equation
           is an empty sum)
  vnIn     normal velocity relative to its fluid neighbours on the last free
           step (if it had any neighbour), else at first contact (< 0 approaching)
  vnOut    the same on the first free step after the contact (> 0 departing;
           None if the contact has not ended)
  e        restitution -vnOut / vnIn (1 = elastic, 0 = merged)
  pMax     its own peak pressure during the contact
  band     mean fraction of its fluid neighbours inside the wall-adjacent
           band that `cavitationProjection = 'contact'` clamps
  nW       max wall neighbours during the contact

Usage: python scripts/analyze_flierTrace.py <runDir> [<runDir> ...]
"""
import json
import os
import sys
from collections import defaultdict


def contacts(rows):
    out, cur, prevFree = [], None, None
    for r in rows:
        supported = r['nF'] > 2
        if supported:
            if cur is None:
                vnIn = prevFree.get('vn') if prevFree and prevFree.get('vn') is not None else r.get('vn')
                cur = dict(tIn=r['t'], pIn=prevFree['p'] if prevFree else None,
                           vnIn=vnIn, pMax=r['p'], band=[], nW=0, prevFree=prevFree)
            cur['pMax'] = max(cur['pMax'], r['p'])
            if r.get('nNbr'):
                cur['band'].append(r.get('nBand', 0) / r['nNbr'])
            cur['nW'] = max(cur['nW'], r['nW'])
        else:
            if cur is not None:
                cur['tOut'] = r['t']
                cur['vnOut'] = r.get('vn')
                out.append(cur)
                cur = None
            prevFree = r
    if cur is not None:
        cur['tOut'], cur['vnOut'] = None, None
        out.append(cur)
    return out


def main():
    for d in sys.argv[1:]:
        trace = json.load(open(os.path.join(d, 'trace.json')))
        byUid = defaultdict(list)
        for r in trace:
            byUid[r['uid']].append(r)
        print(f'== {d}: {len(byUid)} tracked particles, {len(trace)} rows')
        allC = []
        for uid, rows in sorted(byUid.items()):
            for c in contacts(sorted(rows, key=lambda r: r['t'])):
                if c['prevFree'] is None:
                    continue           # never free before: the birth itself
                c['uid'] = uid
                allC.append(c)
        for c in allC:
            e = (-c['vnOut'] / c['vnIn']) if (c['vnOut'] is not None and c['vnIn'] and c['vnIn'] < 0) else None
            band = sum(c['band']) / len(c['band']) if c['band'] else 0.0
            fmt = lambda x, f='{:+.2f}': 'None' if x is None else f.format(x)
            print(f"uid {c['uid']:6d} t {c['tIn']:.3f}->{fmt(c['tOut'], '{:.3f}')} "
                  f"pIn {fmt(c['pIn'], '{:+8.1f}')} vnIn {fmt(c['vnIn'])} vnOut {fmt(c['vnOut'])} "
                  f"e {fmt(e)} pMax {c['pMax']:+8.1f} band {band:.2f} nW {c['nW']}")
        ended = [c for c in allC if c['vnOut'] is not None and c['vnIn'] and c['vnIn'] < 0]
        if ended:
            hot = [c for c in ended if (c['pIn'] or 0) > 0]
            cold = [c for c in ended if (c['pIn'] or 0) <= 0]
            inBand = [c for c in ended if c['band'] and sum(c['band']) / len(c['band']) > 0.5]
            mean = lambda xs: sum(xs) / len(xs) if xs else float('nan')
            e = lambda cs: [-c['vnOut'] / c['vnIn'] for c in cs]
            print(f"-- ended contacts {len(ended)}: mean e {mean(e(ended)):.2f}; "
                  f"arrived with p>0: {len(hot)} (e {mean(e(hot)):.2f}), p<=0: {len(cold)} (e {mean(e(cold)):.2f}); "
                  f"mostly-band partners: {len(inBand)} (e {mean(e(inBand)):.2f})")


if __name__ == '__main__':
    main()
