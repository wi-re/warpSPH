"""Controlled native/extent AV length x current/cached gradient shock bubble.

Process one physical run at a time; every run publishes to the daemon and
keeps a video. No pressure-force or coefficient changes; fresh directories.
"""
import argparse
import json
import shutil
from pathlib import Path

from compare_spheralMaterials import save, warp_run
from compare_spheralShockBubble import context, Measures, summary, build

VARIANTS = {'native-current': (0., 'current'), 'extent-current': (2., 'current'),
            'native-cached': (0., 'cached'), 'extent-cached': (2., 'cached')}


def controlled_context(settings):
    case, ctx, sim, params, solver = context(settings)
    solver['crkViscosityGradient'] = settings['viscosityGradient']
    solver['crkViscosityParams'] = {'viscosityKernelScale': settings['viscosityKernelScale']}
    return case, ctx, sim, params, solver


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','run'));p.add_argument('--input',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=tuple(VARIANTS));p.add_argument('--time',type=float,default=.2)
    args=p.parse_args()
    if args.mode=='prepare':
        original=json.loads((args.input/'settings.json').read_text())
        args.output.mkdir(parents=True,exist_ok=False)
        for variant,(scale,gradient) in VARIANTS.items():
            folder=args.output/variant;folder.mkdir()
            for name in ('input.npz','warpInput.npz'):shutil.copyfile(args.input/name,folder/name)
            settings=dict(original,end=args.time,gammaVariant='n'+str(original['nx'])+'-'+variant,
                viscosityGradient=gradient,viscosityKernelScale=scale,
                comparisonControl='Only Q smoothing length and AV gradient timing vary. Pressure force, density/volume/support policy, Cl=1 Cq=.75 unchanged. Cached follows previous RHS gradient; zero initial velocity makes startup exact.')
            save(folder/'settings.json',settings)
        save(args.output/'study.json',dict(status='prepared',variants=VARIANTS,source=str(args.input),
            policy='Sequential daemon-visible videos. Retain failures; inspect each run. Compare early common times, no visual fitting.'))
        return 0
    if args.variant is None:p.error('--variant is required for run')
    args.output=args.output/args.variant
    settings=json.loads((args.output/'settings.json').read_text())
    return warp_run(args,settings,controlled_context,Measures,summary,'Shock bubble AV',build,'warpInput.npz')


if __name__=='__main__':raise SystemExit(main())
