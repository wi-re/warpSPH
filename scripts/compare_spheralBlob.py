"""Compiled Spheral blob on saved Warp inputs; daemon, watchdogs and video.

Native Spheral policies are recorded explicitly, not represented as Warp's
experimental policy flags. Render in the Warp environment after integration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from compare_spheralMaterials import PIN, SOURCE, save, spheral_run, render
from study_blobPartition import Measures, summary, context


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=False)
    settings=json.loads((args.input/'settings.json').read_text())
    if args.dt is not None:settings['dt']=args.dt
    if args.time is not None:
        import math
        settings['steps']=math.ceil(args.time/settings['dt']);settings['end']=settings['steps']*settings['dt']
    if args.frame_every is not None:settings['frameEvery']=args.frame_every
    shutil.copyfile(args.input/'input.npz',args.output/'input.npz')
    settings.update(gammaVariant='nx'+str(settings['nx'])+'-compiled-reference',referencePin=PIN,
        sourceRoot=str(SOURCE),nPerh=1.35,extent=3.,referenceViscosity={'Cl':1.,'Cq':1.},
        comparisonInput=str(args.input.resolve()),
        inputSHA256=hashlib.sha256((args.output/'input.npz').read_bytes()).hexdigest(),
        referencePolicies=dict(energyPartition='native sign-filtered node-dudt',predictor='native compatible',
            integrator='SynchronousRK2',volume='MassOverDensity',density='RigorousSumDensity',
            support='SPHSmoothingScale IdealH',XSPH=False,precision='float64'),
        source='Byte-for-byte copy of saved Warp blob input. Geometry, end time, dt, kernel, starting supports and volume type matched; compiled Spheral native evolution/Q/energy retained. Warp energyPartition/energyPredictor keys are render-context provenance only.')
    if args.source_q:
        settings.pop('referenceViscosity',None)
        settings['gammaVariant']+='-sourceQ'
    settings['referencePolicies']['viscosityCoefficients']='constructor defaults scaled by kernel extent' if args.source_q else 'Cl=Cq=1 explicitly matched to Warp'
    save(args.output/'settings.json',settings)
    print('Prepared exact saved-input Spheral blob',args.output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','spheral','render'))
    p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dt',type=float);p.add_argument('--time',type=float);p.add_argument('--frame-every',type=int)
    p.add_argument('--source-q',action='store_true',help='Retain current CRK constructor coefficients instead of matching Cl=Cq=1.')
    p.add_argument('--publisher-path',type=Path,default=Path('/home/lu26029/dev/warpSPHRemote/python'))
    a=p.parse_args()
    if a.mode=='prepare':
        if a.input is None:p.error('--input required for prepare')
        prepare(a)
    else:
        settings=json.loads((a.output/'settings.json').read_text())
        if a.mode=='spheral':raise SystemExit(spheral_run(a,settings,Measures,summary,'Blob reference'))
        render(a,settings,context)
