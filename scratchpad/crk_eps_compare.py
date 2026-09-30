import json, math, sys
new = json.load(open('scripts/out_crk2d/sweep/new.json')); old = json.load(open('scripts/out_crk2d/sweep/old.json'))
def cur(rs):
    return {r['case']: r for r in rs if r.get('setting') in (None, 'current') and abs(r.get('eta_crit', 0) - 0.3333333) < 1e-6 and 'diff_rho_rel_l1' not in r}
N, O = cur(new), cur(old)
skip = {'case', 'eta_crit', 'eta_fold', 'setting', 'wall_s', 'n_steps'}
print(f'{"case":16s} {"metric":16s} {"old (eps~0)":>14s} {"new (eps^2=1e-2)":>17s} {"change":>9s}')
for c in N:
    if c not in O: continue
    for k, v in N[c].items():
        if k in skip or not isinstance(v, (int, float)) or isinstance(v, bool): continue
        o = O[c].get(k)
        if o is None or (isinstance(v, float) and math.isnan(v)): continue
        ch = (v - o) / abs(o) * 100 if o not in (0, 0.0) else float('nan')
        print(f'{c:16s} {k:16s} {o:14.4e} {v:17.4e} {ch:8.1f}%')
    print(f'{"":16s} steps {O[c]["n_steps"]} -> {N[c]["n_steps"]}, diverged {O[c]["diverged"]} -> {N[c]["diverged"]}')
