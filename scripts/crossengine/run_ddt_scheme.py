"""DELTASPH_VALIDATION_PLAN §8.16: run any `DensityDiffusionScheme` member
against the diffSPH-matched dambreak IC (Part 8), pure warpSPH pipeline.
Generalizes `run_wrongsign_ddt.py` with a `--scheme` selector so
`moltenicolagrossi2009`/`fourtakas2019` can be validated the same way
`deltaSPH_wrongSign` was.
"""
import sys
import json
import argparse

sys.path.insert(0, '/home/lu26029/dev/warpSPH/examples/weaklyCompressible')

ap = argparse.ArgumentParser()
ap.add_argument('--scheme', type=str, required=True,
                help='DensityDiffusionScheme member name, e.g. moltenicolagrossi2009, fourtakas2019, deltaSPH_wrongSign, deltaSPH')
ap.add_argument('--nSteps', type=int, default=None)
ap.add_argument('--out', type=str, required=True)
ap.add_argument('--video', action=argparse.BooleanOptionalAction, default=True,
                help='export frames + encode a video (vispy), like the other '
                     'crossengine runs; on by default -- pass --no-video to opt out')
cliArgs = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

MATCHED_ARGS = [
    '--nx', '64', '--L', '2.0', '--n_h', '4.0',
    '--kernel', 'Wendland4', '--integrationScheme', 'symplecticEuler',
    '--supportMode', 'SuperSymmetric', '--tLimit', '4.0', '--targetDt', '0.0005',
    '--band', '7', '--fillRatio', '0.5', '--fluidWidth', str(5.0 / 12.0),
    '--gravityMagnitude', '10.0', '--wallBC', 'constant',
    '--caseName', f'dambreak-{cliArgs.scheme}DDT',
    '--quiet', '--precision', 'float32',
]
MATCHED_ARGS += ['--plot', '--video'] if cliArgs.video else ['--no-plot', '--no-video']
MATCHED_ARGS += ['--no-store']
if cliArgs.nSteps is not None:
    MATCHED_ARGS += ['--nSteps', str(cliArgs.nSteps)]

from warpSPH.cases.dambreak import dambreakCase
from warpSPH.runner import caseMain
from warpSPH.enumTypes import DensityDiffusionScheme

schemeMember = getattr(DensityDiffusionScheme, cliArgs.scheme)

_origConfigureScheme = dambreakCase.configureScheme


def _configureSchemeSelected(ctx):
    _origConfigureScheme(ctx)
    ctx.schemeConfig.diffusionParams.densityDiffusionTerm = schemeMember


dambreakCase.configureScheme = _configureSchemeSelected

try:
    result = caseMain(dambreakCase, MATCHED_ARGS)
except Exception as e:
    print(f"CRASHED: {type(e).__name__}: {e}")
    sys.exit(1)

print('diverged:', result.diverged, 'nSteps:', result.nSteps)
print('exportPath:', result.exportPath)
print('videoPath:', result.videoPath)

import numpy as np
if result.trajectory:
    maxV = np.array([r['maxVelocity'] for r in result.trajectory if 'maxVelocity' in r])
    ke = np.array([r['kineticEnergy'] for r in result.trajectory if 'kineticEnergy' in r])
    minD = np.array([r['minDensity'] for r in result.trajectory if 'minDensity' in r])
    maxD = np.array([r['maxDensity'] for r in result.trajectory if 'maxDensity' in r])
    summary = dict(
        scheme=cliArgs.scheme, diverged=result.diverged, nSteps=result.nSteps,
        maxVelocity_final=float(maxV[-1]), maxVelocity_max=float(maxV.max()),
        kineticEnergy_final=float(ke[-1]), kineticEnergy_max=float(ke.max()),
        minDensity_final=float(minD[-1]), minDensity_min=float(minD.min()),
        maxDensity_final=float(maxD[-1]), maxDensity_max=float(maxD.max()),
    )
    print(json.dumps(summary, indent=2))
    with open(cliArgs.out + '.summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)
