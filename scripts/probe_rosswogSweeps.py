#!/usr/bin/env python3
"""AV_PLAN Phase 2 sweeps for the Rosswog (2020) entropy trigger, Monaghan host, alpha in [0, 1]:

    scripts/probe_rosswogSweeps.py --sweep timestep     # Sod 1D, cfl 0.3 vs 0.15: alpha within 10 % (L1)
    scripts/probe_rosswogSweeps.py --sweep resolution   # Gresho, alphaMean must not grow with nx

Eq. (16) is non-dimensionalised by tau/dt so that the alpha field does not depend on the step
size; the resolution sweep checks the trigger does not sharpen into noise as dx -> 0. One run at a
time, video on, progress streamed.
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import warpSPHBootstrap  # noqa: F401,E402
from warpSPH.runner import run  # noqa: E402

PARAMS = dict(viscositySwitch='Rosswog2020', alpha_min=0.0, alpha_max=1.0)


def _configure(case):
    """alpha_min / alpha_max on every case (only Sod reads them as case params)."""
    import dataclasses
    configure = case.configureScheme

    def configureScheme(ctx):
        configure(ctx)
        ctx.schemeConfig.viscositySwitchParams.alpha_min = 0.0
        ctx.schemeConfig.viscositySwitchParams.alpha_max = 1.0
    return dataclasses.replace(case, configureScheme=configureScheme)


def _run(case, out, **kw):
    return run(_configure(case), scheme='Monaghan', plot=True, video=True, exportRoot=str(out), progress=True,
               stallProgress=1e-3, velocityAlarmPlotInterval=1, params=dict(PARAMS, **kw.pop('params', {})), **kw)


def timestep(out: Path, nx: int, tLimit: float):
    from warpSPH.cases.sod import sodCase
    alphas = {}
    for cfl in (0.3, 0.15):
        res = _run(sodCase, out / f'cfl{cfl}', nx=nx, tLimit=tLimit, cflFactor=cfl,
                   params=dict(right_rho=0.125, right_pressure=0.1))
        st = res.state.state
        order = torch.argsort(st.UIDs)
        alphas[cfl] = st.alphas[order].detach().cpu()
        print(f'[sweep] cfl {cfl}: t {res.state.t:.4f} steps {res.nSteps} alpha mean {alphas[cfl].mean():.4f} '
              f'max {alphas[cfl].max():.4f}', flush=True)
    a, b = alphas[0.3], alphas[0.15]
    rel = (a - b).abs().mean() / b.abs().mean()
    print(f'[sweep] timestep: L1(alpha_0.3 - alpha_0.15) / mean(alpha_0.15) = {rel:.4f} '
          f'({"PASS" if rel < 0.1 else "FAIL"} < 0.1)', flush=True)


def resolution(out: Path, nxs, tLimit: float):
    from warpSPH.cases.greshoVortex import greshoVortexCase
    means = []
    for nx in nxs:
        res = _run(greshoVortexCase, out / f'nx{nx}', nx=nx, tLimit=tLimit)
        a = res.state.state.alphas.detach().cpu()
        aMean = [r['alphaMean'] for r in res.trajectory if 'alphaMean' in r]
        means.append(float(a.mean()))
        print(f'[sweep] gresho nx {nx}: t {res.state.t:.3f} final alpha mean {a.mean():.4f} '
              f'max {a.max():.4f}; time-mean of alphaMean {sum(aMean) / max(len(aMean), 1):.4f}', flush=True)
    ok = all(m2 <= m1 * 1.05 for m1, m2 in zip(means, means[1:]))
    print(f'[sweep] resolution: alphaMean {means} ({"PASS" if ok else "FAIL"}: non-increasing within 5 %)', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sweep', choices=('timestep', 'resolution'), required=True)
    ap.add_argument('--out', default='results/av_M2_sweeps')
    ap.add_argument('--nx', type=int, nargs='+', default=None)
    ap.add_argument('--tLimit', type=float, default=None)
    a = ap.parse_args()
    out = Path(a.out) / a.sweep
    if a.sweep == 'timestep':
        timestep(out, (a.nx or [400])[0], a.tLimit or 0.2)
    else:
        resolution(out, a.nx or [50, 100, 200], a.tLimit or 3.0)


if __name__ == '__main__':
    main()
