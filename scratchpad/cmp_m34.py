import numpy as np, glob, json, re
def pk(d):
    return ' '.join(f"{np.nanmax(d[f's_pSurf{i}']):.0f}" for i in range(9))
for tag,pat in [('OVN','scripts/out_overnight_2026-09-28/m34/*.npz'),('BASE','scripts/out_baseline_2026-09-26/m34/*.npz')]:
    for f in sorted(glob.glob(pat)):
        d=np.load(f,allow_pickle=True)
        print(tag,f.split('/')[-1][:60],'vmax %.1f rho[%.2f,%.2f] pen %.2f wallpen? KEend %s'%(np.nanmax(d['s_maxVelocity']),np.nanmin(d['s_minDensity']),np.nanmax(d['s_maxDensity']),np.nanmax(d['s_maxObstaclePenDx']), ''))
        print('    peaks',pk(d))
