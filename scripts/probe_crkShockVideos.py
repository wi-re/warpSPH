"""Video-backed CRKSPH runs of the cases `av_report.py` does not cover (triple point, shearing Noh), at the default limiter
(CRKSPH_LIMITER_PLAN close-out: Noh + 2D shock cases at the new (1/n_h, 0.2/n_h) default). One at a time, progress streamed.

    scripts/probe_crkShockVideos.py [--out results/crk_closeout_videos] [--cases triplePoint] [--nx 256] [--tLimit 7]
"""
import argparse
import sys
from pathlib import Path

from warpSPH.cases.shearingNoh import shearingNohCase
from warpSPH.cases.triplePoint import triplePointCase
from warpSPH.runner import run

CASES = {'triplePoint': (triplePointCase, dict(nx=64, tLimit=2.0)), 'shearingNoh': (shearingNohCase, dict(nx=64, tLimit=0.6))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='results/crk_closeout_videos')
    ap.add_argument('--cases', nargs='+', default=list(CASES))
    ap.add_argument('--nx', type=int, default=None, help='override the per-case resolution')
    ap.add_argument('--tLimit', type=float, default=None, help='override the per-case end time')
    a = ap.parse_args()
    for name in a.cases:
        case, kw = CASES[name]
        kw = {**kw, **{k: v for k, v in (('nx', a.nx), ('tLimit', a.tLimit)) if v is not None}}
        res = run(case, scheme='CRKSPH', plot=True, video=True, exportRoot=str(Path(a.out) / name), progress=True,
                  stallProgress=1e-3, velocityAlarmPlotInterval=1, **kw)
        E = res.series('totalEnergy')
        print(f'\n[crk-video] {name}: t={float(res.state.t):.4f} steps={res.nSteps} diverged={res.diverged} '
              f'stop={res.stopReason} dE/E={float((E[-1]-E[0])/E[0]):.2e}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
