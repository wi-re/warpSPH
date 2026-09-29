import sys, glob, os, json
sys.path.insert(0, 'scripts')
import numpy as np
import probe_deltaSPHMarrone as P

def rows(pattern, tag):
    for f in sorted(glob.glob(pattern)):
        d = np.load(f, allow_pickle=True)
        col = {k: d[k] for k in d.files if k != 'meta'}
        meta = json.loads(str(d['meta']))
        checks, m = P._score(col)
        ts = col['tStar']; v = col['maxVelocity']
        late = ts > 5
        fails = [c[0] for c in checks if not c[1]]
        print(f"{tag:9s} {os.path.basename(f)[:-4][:58]:58s} p1pl={m.get('p1_plateau',np.nan):.2f} "
              f"(in1 {m.get('p1_plateau_In1',np.nan):.2f} shep {m.get('p1_plateau_Shep',np.nan):.2f}) "
              f"p1pk={m.get('p1_firstPeak',np.nan):.2f} p2pk={m.get('p2_peak',np.nan):.2f}@{m.get('p2_tPeak',np.nan):.2f} "
              f"p2raw={m.get('p2_rawmax',np.nan):.1f} p2post={m.get('p2_postMax',np.nan):.2f} "
              f"vmax={np.nanmax(v):.1f} vlate={np.nanmax(v[late]):.1f} rho5-95=[{m.get('rhoMin_p5',np.nan):.3f},{m.get('rhoMax_p95',np.nan):.3f}] "
              f"ext=[{m['rhoMin']:.2f},{m['rhoMax']:.2f}] {9-len(fails)}/9 fail={fails}")
O='scripts/out_overnight_2026-09-28/m31/'; B='scripts/out_baseline_2026-09-26/'
for lbl, pat in [('OVN', O+'*.npz'), ('BASE', B+'m31/*.npz'), ('BASEseed', B+'m31_seeds/*.npz')]:
    rows(pat, lbl)
