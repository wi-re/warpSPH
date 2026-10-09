import numpy as np, glob
fs = sorted(glob.glob('scripts/out_overnight_2026-09-28/sloshing_*/*_series.npz')) + sorted(glob.glob('scripts/out_baseline_2026-09-26/sloshing/*_series.npz'))
for f in fs:
    d=np.load(f,allow_pickle=True); t=d['t']
    def win(k,scale=1e3):
        v=d[k]; tt=t if len(v)==len(t) else np.linspace(t[0],t[-1],len(v))
        return ' '.join(f"{np.nanmax(np.abs(v[(tt>=a)&(tt<a+0.5)]))/scale:5.1f}" for a in np.arange(2,7,.5))
    print(f.split('/')[-2], f.split('/')[-1][:30]); print(' wall :',win('sensorPressure')); print(' probe:',win('sensorPressureProbe')); print(' vmax :',win('maxVelocity',1.0))
    print(' rho', np.nanmin(d['minDensity']), np.nanmax(d['maxDensity']), 'diverged', d['diverged'], 'steps', d['nSteps'], 'wall', d['wallTime'])
