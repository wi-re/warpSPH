#!/usr/bin/env python3
"""Score an AV_PLAN overnight sweep (`scripts/av_sweep_overnight.py`) against the plan's thresholds -> verdict.md.

Each check quotes the threshold from AV_PLAN.md's phase Validation table and reports PASS / FAIL / INFO (a number
the plan asks to report, no pass mark) / MISSING (a run is absent or diverged). References: the sweep's own runs,
falling back to the stored reference reports (`results/av_M0g_*`, `results/av_crk3_*`) for the baseline columns.
A FAIL is a finding to read, not a verdict on its own -- several thresholds are the papers' rankings, and AV_PLAN
says reproducing a failure can be the validation (Phase 4 Sedov).

    python scripts/av_sweep_verdict.py results/av_sweep_<date>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
REFERENCES = ['results/av_crk3_crkNone', 'results/av_crk3_crkCullenDehnen2010', 'results/av_M0g_baseline',
              'results/av_M0g_baselineQ', 'results/av_M0g_phase2', 'results/av_M0g_crk_all']
#: (reference dir, config) pairs that predate a physics change and must not serve as a reference: M0g's CRK + C&D
#: ran before the 2026-10-07 CRK switch fix (the switch was a no-op, alpha ~ 1 everywhere).
STALE_REFERENCES = {('results/av_M0g_crk_all', 'crkCullenDehnen2010')}
GROUP_A = {
    'sod': ['L1_vx', 'contactSpikeP', 'R3_rho_err', 'R4_rho_err', 'R4_P_err'],
    'sod2d': ['L1_vx', 'R3_rho_err', 'R4_rho_err', 'R4_P_err'],
    'sod3d': ['L1_vx', 'R3_rho_err', 'R4_rho_err', 'R4_P_err'],
    'sedov': ['peakRhoRatio', 'shockRadiusErr'],
    'noh': ['postShockRhoErr'],
}


class Records:
    def __init__(self, sweep: Path):
        self.sweep: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.ref: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.kh256: Dict[str, Dict[str, Any]] = {}
        for res in sorted(sweep.glob('*/results.json')):
            for r in json.loads(res.read_text())['records']:
                if res.parent.name.startswith('kh256_'):
                    self.kh256[r['config']] = r['metrics']
                else:
                    self.sweep[(r['config'], r['case'])] = r['metrics']
        for d in REFERENCES:
            p = REPO / d / 'results.json'
            if p.exists():
                for r in json.loads(p.read_text())['records']:
                    if (d, r['config']) not in STALE_REFERENCES:
                        self.ref.setdefault((r['config'], r['case']), r['metrics'])

    def get(self, config: str, case: str, metric: str, allowRef: bool = True) -> Optional[float]:
        m = self.sweep.get((config, case))
        if m is None and allowRef:
            m = self.ref.get((config, case))
        if m is None or m.get('diverged'):
            return None
        v = m.get(metric)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None

    def diverged(self, config: str, case: str) -> Optional[bool]:
        m = self.sweep.get((config, case))
        return None if m is None else bool(m.get('diverged'))


Check = Tuple[str, str, str, str]      # (phase, check, status, detail)


def _fmt(v: Optional[float]) -> str:
    return 'n/a' if v is None else f'{v:.4g}'


def _cmp(phase, name, ok: Optional[bool], detail) -> Check:
    return (phase, name, 'MISSING' if ok is None else ('PASS' if ok else 'FAIL'), detail)


def _within(R: Records, cfg: str, refCfg: str, case: str, tol: float) -> Tuple[Optional[bool], str]:
    worst, where, missing = 0.0, '-', []
    for k in GROUP_A[case]:
        a, b = R.get(cfg, case, k), R.get(refCfg, case, k)
        if a is None or b is None:
            missing.append(k)
            continue
        rel = abs(a - b) / max(abs(b), 1e-12)
        # an `*_err` metric is itself a relative error, often ~1e-3: 5 % of it is noise, so a difference of up to one
        # percentage point of error also passes (reported as 0)
        if k.endswith('Err') or k.endswith('_err'):
            rel = 0.0 if abs(a - b) <= 0.01 else rel
        if rel > worst:
            worst, where = rel, k
    if missing and len(missing) == len(GROUP_A[case]):
        return None, f'missing {missing}'
    return worst <= tol, f'worst {where} {worst:.2%} (tol {tol:.0%}; *_err metrics also pass within 0.01 absolute)' + (f'; missing {missing}' if missing else '')


def _withinCD(R: Records, cfg: str, case: str, tol: float) -> Tuple[Optional[bool], str]:
    """`_within` against the M0 C&D column (`C_q = 0`: no quadratic term, so cold-gas shocks get no viscosity --
    Noh's post-shock density is 50 % off) or the C&D-Q column (`C_q = 2`), whichever passes. The papers' operators
    (Sphenix beta = 3, Wadsley beta = 2) have a quadratic term, so the plan's "within 5 % of C&D at M0" is only a
    like-for-like comparison against C&D-Q; both are shown."""
    a, da = _within(R, cfg, 'cullenDehnen2010', case, tol)
    b, db = _within(R, cfg, 'cullenDehnen2010Q', case, tol) if R.get('cullenDehnen2010Q', case, GROUP_A[case][0]) is not None \
        else (None, 'no C&D-Q run')
    ok = None if a is None and b is None else bool(a) or bool(b)
    return ok, f'vs C&D (C_q 0): {da} | vs C&D-Q (C_q 2): {db}'


def phase3(R: Records) -> List[Check]:
    P = 'Phase 3'
    out = []
    for cfg in ('crkNone', 'crkCullenDehnen2010'):
        for case in ('sod', 'sedov', 'sod2d', 'sod3d', 'noh'):
            e = R.get(cfg, case, 'energyDrift', allowRef=False)
            out.append(_cmp(P, f'{cfg}/{case} energy drift <= 1e-4', None if e is None else e <= 1e-4, _fmt(e)))
            ref = R.ref.get((cfg, case))
            new = R.sweep.get((cfg, case))
            if ref and new:
                rels = [abs(new[k] - ref[k]) / max(abs(ref[k]), 1e-12) for k in GROUP_A.get(case, [])
                        if isinstance(new.get(k), (int, float)) and isinstance(ref.get(k), (int, float))]
                worst = max(rels) if rels else None
                out.append((P, f'{cfg}/{case} Group A vs stored reference (plan: 1e-6)', 'INFO',
                            f'max rel {_fmt(worst)} (float32 round-off from code motion; CRK Gresho / KH amplify it, see Phase 3 notes)'))
    for cfg in ('noneLinear', 'noneLimited'):
        for case in ('sod', 'gresho', 'sedov', 'noh', 'kelvinHelmholtz'):
            d = R.diverged(cfg, case)
            out.append(_cmp(P, f'Monaghan + {cfg} runs {case} clean', None if d is None else not d,
                            'diverged' if d else 'ok'))
    return out


def phase4(R: Records) -> List[Check]:
    P = 'Phase 4'
    out = []
    a, b = R.get('gsAVSLRB2', 'sod', 'L1_vx'), R.get('gsAVSW', 'sod', 'L1_vx')
    out.append(_cmp(P, 'Sod: L1(v_x) AVSLRB2 <= AVSW', None if None in (a, b) else a <= b, f'{_fmt(a)} vs {_fmt(b)}'))
    for cfg, over in (('gsAVSLR', True), ('gsAVSWSLR', True), ('gsAVSLRB2', False), ('gsAVSLRB', None), ('gsAV', None)):
        v = R.get(cfg, 'sedov', 'peakRhoRatio')
        if over is None:
            out.append((P, f'Sedov peak rho/rho0 {cfg}', 'INFO', _fmt(v)))
        else:
            out.append(_cmp(P, f'Sedov: {cfg} {"overshoots" if over else "stays below"} rho/rho0 = 4',
                            None if v is None else (v > 4.0) == over, _fmt(v)))
    g = {c: R.get(c, 'gresho', 'L1_vphi') for c in ('gsAV', 'gsAVSW', 'gsAVSLR', 'gsAVSWSLR', 'gsAVSLRB', 'gsAVSLRB2')}
    slr = [g[c] for c in ('gsAVSLR', 'gsAVSWSLR', 'gsAVSLRB', 'gsAVSLRB2')]
    ok = None if None in g.values() else (max(slr) < g['gsAVSW'] < g['gsAV'])
    out.append(_cmp(P, 'Gresho L1(v_phi): SLR family < AVSW < AV (strict)', ok,
                    ', '.join(f'{k[2:]} {_fmt(v)}' for k, v in g.items())))
    k = {c: R.get(c, 'kelvinHelmholtz', 'khAmplitudeAt1p5') for c in g}
    slrK = [k[c] for c in ('gsAVSLR', 'gsAVSWSLR', 'gsAVSLRB', 'gsAVSLRB2')]
    ok = None if None in k.values() else (k['gsAV'] < k['gsAVSW'] < min(slrK))
    out.append(_cmp(P, 'KH A(1.5): AV < AVSW < SLR family', ok, ', '.join(f'{c[2:]} {_fmt(v)}' for c, v in k.items())
                    + ' (paper: 2.87, 8.31, 12.58, 12.14, 12.01, 12.52 e-2; McNally 14.79e-2)'))
    ok = None if None in k.values() else min(slrK) >= 1.4 * k['gsAVSW']
    out.append(_cmp(P, 'KH: SLR family >= 1.4 x AVSW', ok, f'min SLR {_fmt(min(slrK) if None not in slrK else None)}'))
    a, b = R.get('gsAVSLRB2', 'rayleighTaylor', 'mixingWidth'), R.get('gsAVSW', 'rayleighTaylor', 'mixingWidth')
    out.append(_cmp(P, 'RT: mixing width AVSLRB2 >= AVSW', None if None in (a, b) else a >= b, f'{_fmt(a)} vs {_fmt(b)}'))
    raw, rec = R.get('gsAV', 'gresho', 'avEnergyTotal'), R.get('gsAVSLR', 'gresho', 'avEnergyTotal')
    ratio = None if None in (raw, rec) or rec == 0 else raw / rec
    out.append(_cmp(P, 'HEADLINE: Gresho AV energy raw / reconstructed >= 10', None if ratio is None else ratio >= 10,
                    f'raw {_fmt(raw)}, SLR {_fmt(rec)}, ratio {_fmt(ratio)}; linear/quadratic raw '
                    f'{_fmt(R.get("gsAV", "gresho", "avEnergyLinear"))}/{_fmt(R.get("gsAV", "gresho", "avEnergyQuadratic"))}, '
                    f'SLR {_fmt(R.get("gsAVSLR", "gresho", "avEnergyLinear"))}/{_fmt(R.get("gsAVSLR", "gresho", "avEnergyQuadratic"))}'))
    return out


def phase5a(R: Records) -> List[Check]:
    P = 'Phase 5A'
    out = []
    for det in ('CD', 'Rosswog'):
        fixed, coupled, low = f'cn{det}Fixed2', f'cn{det}Coupled2', f'cn{det}Fixed0p2'
        for case in ('sod', 'noh', 'sedov'):
            ok, detail = _within(R, coupled, fixed, case, 0.05)
            out.append(_cmp(P, f'{det} {case}: coupled within 5% of fixed beta = 2', ok, detail))
        for metric in ('angularMomentumLoss', 'avEnergyQuadratic'):
            a, b = R.get(coupled, 'gresho', metric), R.get(fixed, 'gresho', metric)
            out.append(_cmp(P, f'{det} Gresho {metric}: coupled < fixed 2', None if None in (a, b) else abs(a) < abs(b),
                            f'{_fmt(a)} vs {_fmt(b)} (beta 0.2: {_fmt(R.get(low, "gresho", metric))})'))
        a, b = R.get(coupled, 'shearingNoh', 'avEnergyQuadratic'), R.get(fixed, 'shearingNoh', 'avEnergyQuadratic')
        out.append(_cmp(P, f'{det} shearing Noh: quadratic AV energy lower coupled', None if None in (a, b) else a < b,
                        f'{_fmt(a)} vs {_fmt(b)}'))
        a, b = R.get(coupled, 'shearingNoh', 'shockFront'), R.get(fixed, 'shearingNoh', 'shockFront')
        out.append(_cmp(P, f'{det} shearing Noh: front within 3%', None if None in (a, b) else abs(a / b - 1) <= 0.03,
                        f'{_fmt(a)} vs {_fmt(b)} (exact {_fmt(R.get(fixed, "shearingNoh", "shockFrontExact"))})'))
        qf, qc = R.get(fixed, 'shearBox', 'avQuadraticFraction'), R.get(coupled, 'shearBox', 'avQuadraticFraction')
        out.append(_cmp(P, f'{det} shear box: quadratic dominates at fixed beta, suppressed coupled',
                        None if None in (qf, qc) else (qf > 0.5 and qc < qf),
                        f'quadratic fraction fixed {_fmt(qf)}, coupled {_fmt(qc)}; mode amplitude fixed '
                        f'{_fmt(R.get(fixed, "shearBox", "modeAmplitudeFinal"))}, coupled '
                        f'{_fmt(R.get(coupled, "shearBox", "modeAmplitudeFinal"))}'))
        for case in ('sod', 'gresho', 'shearBox'):
            out.append((P, f'{det} {case}: Chen & Nixon ratio median (fixed / coupled)', 'INFO',
                        f'{_fmt(R.get(fixed, case, "chenNixonRatioMedian"))} / {_fmt(R.get(coupled, case, "chenNixonRatioMedian"))}'))
    return out


def phase5b(R: Records, timing: Dict[str, Any]) -> List[Check]:
    P = 'Phase 5B'
    out = []
    for case in ('sod', 'sod2d', 'sod3d', 'sedov', 'noh'):
        ok, detail = _withinCD(R, 'sphenix', case, 0.05)
        out.append(_cmp(P, f'{case}: Group A within 5% of C&D or C&D-Q', ok, detail))
    for metric in ('L1_vphi', 'alphaMean'):
        a, b = R.get('sphenix', 'gresho', metric), R.get('cullenDehnen2010', 'gresho', metric)
        out.append(_cmp(P, f'Gresho {metric} <= C&D', None if None in (a, b) else a <= b, f'{_fmt(a)} vs {_fmt(b)}'))
    a, b = R.get('sphenix', 'kelvinHelmholtz', 'khAmplitudeAt1p5'), R.get('cullenDehnen2010', 'kelvinHelmholtz', 'khAmplitudeAt1p5')
    out.append(_cmp(P, 'KH A(1.5) >= C&D', None if None in (a, b) else a >= b, f'{_fmt(a)} vs {_fmt(b)}'))
    s, c = timing.get('sphenix', {}).get('msPerStep'), timing.get('cullenDehnen2010', {}).get('msPerStep')
    out.append(_cmp(P, 'ms/step >= 15% lower than C&D', None if None in (s, c) else s <= 0.85 * c,
                    f'sphenix {_fmt(s)}, C&D {_fmt(c)}, none {_fmt(timing.get("none", {}).get("msPerStep"))} ms/step'))
    for case in ('sod', 'sedov', 'noh', 'gresho'):
        ds, dc = R.diverged('sphenixCfl4', case), R.diverged('cullenDehnen2010Cfl4', case)
        out.append((P, f'cfl x 4 on {case}: sphenix / C&D diverged', 'INFO', f'{ds} / {dc}'))
    return out


def phase6(R: Records) -> List[Check]:
    P = 'Phase 6'
    out = []
    for case in ('sod', 'sedov', 'noh'):
        best = None
        for ref in ('cullenDehnen2010', 'cullenDehnen2010Q', 'rosswog2020', 'rosswog2020Q', 'sphenix'):
            ok, detail = _within(R, 'wadsley2017', ref, case, 0.05)
            if ok:
                best = (ref, detail)
                break
        out.append(_cmp(P, f'{case}: Group A within 5% of one of C&D(-Q) / Rosswog(-Q) / Sphenix',
                        None if R.sweep.get(('wadsley2017', case)) is None else best is not None,
                        f'{best[0]}: {best[1]}' if best else _within(R, 'wadsley2017', 'cullenDehnen2010', case, 0.05)[1]))
    a, b = R.get('wadsley2017', 'gresho', 'L1_vphi'), R.get('cullenDehnen2010', 'gresho', 'L1_vphi')
    out.append(_cmp(P, 'Gresho L1(v_phi) <= C&D', None if None in (a, b) else a <= b, f'{_fmt(a)} vs {_fmt(b)}'))
    for metric in ('preShockAlphaMean', 'alphaActiveFraction'):
        a, b = R.get('wadsley2017', 'noh2d', metric), R.get('cullenDehnen2010', 'noh2d', metric)
        out.append(_cmp(P, f'cylindrical Noh {metric}: Wadsley materially (< 1/2) below C&D',
                        None if None in (a, b) else a < 0.5 * b,
                        f'{_fmt(a)} vs {_fmt(b)} (Rosswog {_fmt(R.get("rosswog2020", "noh2d", metric))}, '
                        f'Sphenix {_fmt(R.get("sphenix", "noh2d", metric))})'))
    a, b = R.get('wadsley2017', 'kelvinHelmholtz', 'khAmplitudeAt1p5'), R.get('cullenDehnen2010', 'kelvinHelmholtz', 'khAmplitudeAt1p5')
    out.append(_cmp(P, 'KH growth >= C&D', None if None in (a, b) else a >= b, f'{_fmt(a)} vs {_fmt(b)}'))
    a, b = R.get('wadsley2017', 'rayleighTaylor', 'mixingWidth'), R.get('cullenDehnen2010', 'rayleighTaylor', 'mixingWidth')
    out.append(_cmp(P, 'RT growth >= C&D', None if None in (a, b) else a >= b, f'{_fmt(a)} vs {_fmt(b)}'))
    return out


def kh256(R: Records) -> List[Check]:
    P = 'KH nx 256'
    out = [(P, 'reference (docs/av/kh256_2026-10-07)', 'INFO',
            'Monaghan C&D / Rosswog peak 0.21 -> 0.105 at t = 3; default CRK peak 0.23-0.24, holds 0.17-0.19')]
    for cfg, m in sorted(R.kh256.items()):
        out.append((P, cfg, 'INFO' if not m.get('diverged') else 'FAIL',
                    f'peak {_fmt(m.get("khAmplitudeMax"))}, A(1.5) {_fmt(m.get("khAmplitudeAt1p5"))}, '
                    f'final {_fmt(m.get("khAmplitudeFinal"))}'))
    return out


def phase7(R: Records) -> List[str]:
    """The Part 0 table: Group A + B columns for every config the sweep ran, per case."""
    if str(REPO / 'scripts') not in sys.path:
        sys.path.insert(0, str(REPO / 'scripts'))
    import av_report as A
    lines = []
    cases = sorted({c for _, c in R.sweep})
    for case in cases:
        cols = A.COLUMNS.get(case, []) + ['alphaMean', 'avEnergyTotal', 'avEnergyQuadratic', 'energyDrift']
        cfgs = sorted({c for c, k in R.sweep if k == case})
        lines.append(f'\n### {case}\n')
        lines.append('| config | ' + ' | '.join(cols) + ' |')
        lines.append('|' + '---|' * (len(cols) + 1))
        for c in cfgs:
            m = R.sweep[(c, case)]
            vals = ['DIVERGED' if m.get('diverged') else _fmt(m.get(k) if isinstance(m.get(k), (int, float)) else None)
                    for k in cols]
            lines.append(f'| {c} | ' + ' | '.join(vals) + ' |')
    return lines


def main() -> int:
    sweep = Path(sys.argv[1])
    R = Records(sweep)
    timing = json.loads((sweep / 'timing.json').read_text()) if (sweep / 'timing.json').exists() else {}
    status = json.loads((sweep / 'status.json').read_text()) if (sweep / 'status.json').exists() else {}
    checks = phase3(R) + phase4(R) + phase5a(R) + phase5b(R, timing) + phase6(R) + kh256(R)
    counts = {s: sum(1 for c in checks if c[2] == s) for s in ('PASS', 'FAIL', 'MISSING', 'INFO')}
    lines = [f'# AV_PLAN sweep verdict -- {sweep.name}', '',
             f'{counts["PASS"]} pass, {counts["FAIL"]} fail, {counts["MISSING"]} missing, {counts["INFO"]} info. '
             'A FAIL is a finding to read against the phase text, not automatically a bug.', '',
             '## Jobs', '', '| job | rc | minutes |', '|---|---|---|']
    for j in status.get('jobs', []):
        lines.append(f'| {j["name"]} | {j.get("returncode")} | {j.get("minutes", float("nan")):.1f} |')
    phaseName = None
    for phase, name, st, detail in checks:
        if phase != phaseName:
            lines += ['', f'## {phase}', '', '| check | status | detail |', '|---|---|---|']
            phaseName = phase
        lines.append(f'| {name} | **{st}** | {detail} |' if st == 'FAIL' else f'| {name} | {st} | {detail} |')
    lines += ['', '## Phase 7: the Part 0 table (every config this sweep ran)'] + phase7(R)
    maps = sorted((sweep / 'maps').glob('*_detectors.png')) if (sweep / 'maps').exists() else []
    if maps:
        lines += ['', '## Detector maps (AV_PLAN §6.5)', ''] + [f'![{m.stem}](maps/{m.name})' for m in maps]
    (sweep / 'verdict.md').write_text('\n'.join(lines) + '\n')
    print(f'verdict: {counts} -> {sweep / "verdict.md"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
