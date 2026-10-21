"""Isolate midpoint predictor, work weights and raw Q work. Daemon/video on."""
import argparse
import json
import shutil
from pathlib import Path
from compare_spheralMaterials import save,warp_run
from compare_spheralShockBubble import context,Measures,summary,build

VARIANTS={
    'native-current':('native','raw',False),
    'one-sided':('native','raw',True),
    'predictor':('native','compatible',False),
    'weights':('spheral','raw',False),
    'policy':('spheral','compatible',False),
    'full':('spheral','compatible',True),
}


def controlled_context(settings):
    case,ctx,sim,params,solver=context(settings)
    solver.update(crkEnergyPartition=settings['energyPartition'],crkEnergyPredictor=settings['energyPredictor'],
                  crkViscosityParams={'oneSidedWork':settings['oneSidedWork'],'viscosityKernelScale':0.})
    return case,ctx,sim,params,solver


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','run'))
    p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=tuple(VARIANTS));p.add_argument('--time',type=float,default=.2)
    args=p.parse_args()
    if args.mode=='prepare':
        original=json.loads((args.input/'settings.json').read_text());args.output.mkdir(parents=True,exist_ok=False)
        for variant,(partition,predictor,one_sided) in VARIANTS.items():
            folder=args.output/variant;folder.mkdir()
            for name in ('input.npz','warpInput.npz'):shutil.copyfile(args.input/name,folder/name)
            settings=dict(original,end=args.time,gammaVariant='n'+str(original['nx'])+'-energy-'+variant,
                energyPartition=partition,energyPredictor=predictor,oneSidedWork=one_sided,
                viscosityGradient='current',viscosityKernelScale=0.,
                comparisonControl='Energy controls only. Pressure, volume/density, support and Q gradient/length fixed; Cl=1 Cq=.75, no switch. Spheral weights=sign-filtered node dudt. Full includes compatible predictor and one-sided raw Q heating.')
            save(folder/'settings.json',settings)
        save(args.output/'study.json',dict(variants=VARIANTS,input=str(args.input),status='prepared'))
        return 0
    if args.variant is None:p.error('--variant required for run')
    args.output=args.output/args.variant;settings=json.loads((args.output/'settings.json').read_text())
    return warp_run(args,settings,controlled_context,Measures,summary,'Shock bubble energy',build,'warpInput.npz')


if __name__=='__main__':raise SystemExit(main())
