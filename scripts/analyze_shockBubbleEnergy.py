"""Common diagnostics for controlled energy-policy shock-bubble trajectories."""
import argparse
from pathlib import Path
from analyze_shockBubbleViscosity import analyze
from study_shockBubbleEnergy import VARIANTS

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variants',nargs='+',choices=tuple(VARIANTS),default=tuple(VARIANTS))
    p.add_argument('--times',type=float,nargs='+',default=(.1,.15,.2));a=p.parse_args()
    analyze(a.input,a.reference,a.output,a.variants,tuple(a.times),'Energy controls',
        'Only energy partition, predictor or one-sided raw viscous work vary. Support, volume/density, Q length/current gradient, Cl/Cq and no alpha switch fixed. Common saved frames can differ by one step, actual times retained. Mirror walls and Spheral positivity separately qualified. Not an exact nonlinear oracle.')
