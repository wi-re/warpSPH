#!/usr/bin/env python
"""Resume a Sod Shock Tube 1D run from its trajectory export.

Reads the last frame of `sod_1d.py`'s `trajectory.h5` (see
`warpSPH.io.loadTrajectory`/`loadTrajectoryFrame`) and continues stepping from
there, appending new frames to that *same* growing file rather than starting a
parallel export -- the point of the trajectory scheme is one file per run,
resume included.

    python examples/compressible/01-sod/sod_resume.py --plot --store
"""

import argparse
import os

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

import h5py
import torch
from tqdm.autonotebook import tqdm
from warpSPHIntegrators import get_scheme

from warpSPH.cases.compressible.sod import sodCase
from warpSPH.io import importConfigs, latestExportPath, loadTrajectory, loadTrajectoryFrame, writeFrame
from warpSPH.runner import CaseSpec, RunContext, encodeFrames

argparser = argparse.ArgumentParser(
    description='Resume a Sod Shock Tube 1D simulation from its trajectory export.')
argparser.add_argument('--exportPath', type=str, default=None,
                       help='Export directory to resume from. Defaults to the newest run of the case.')
argparser.add_argument('--plot', action='store_true', help='Whether to plot the results during the simulation.')
argparser.add_argument('--store', action='store_true',
                       help='Whether to keep appending frames to trajectory.h5.')
argparser.add_argument('--plotInterval', type=int, default=10,
                       help='Interval (in steps) at which to plot the results.')
argparser.add_argument('--exportInterval', type=float, default=None,
                       help='Simulated-time interval between stored frames. Defaults to the interval '
                            'the run was originally exported with.')
argparser.add_argument('--t_limit', type=float, default=0.3, help='Time limit to run the simulation to.')
args = argparser.parse_args()

if args.exportPath is None:
    args.exportPath = latestExportPath('01-sodShockTube')
    print(f'resuming from {args.exportPath}')
exportPath = args.exportPath

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

trajectoryFile, meta = loadTrajectory(exportPath, device, extraFields=sodCase.extraFields)
config, schemeConfig = importConfigs(os.path.join(exportPath, 'config.json'), meta['bundle'].importFunction)

# Sod's own params (gamma, left/right states, ...) were written once as
# top-level attrs by `writeInitialData`'s `extraData` -- read them back so
# `sodCase`'s hooks (which all read `ctx.param(...)`) see the run they
# actually ran, not the case's bare defaults.
params = {k: trajectoryFile.attrs[k] for k in sodCase.params if k in trajectoryFile.attrs}
spec = CaseSpec(caseName=sodCase.name, scheme=sodCase.scheme, params=params)

lastFrameIndex = int(meta['frameKeys'][-1].split('_')[1])
system, t = loadTrajectoryFrame(trajectoryFile, meta, len(meta['frameKeys']) - 1, schemeConfig=schemeConfig)
trajectoryFile.close()

exportInterval = args.exportInterval if args.exportInterval is not None \
    else float(h5py.File(os.path.join(exportPath, 'trajectory.h5'), 'r').attrs['exportInterval'])

ctx = RunContext(
    spec=spec, case=sodCase, config=config, integrator=get_scheme(config.integrationScheme),
    schemeConfig=schemeConfig, scheme=meta['scheme'], device=device, dtype=config.dtype, bundle=meta['bundle'],
    exportPath=exportPath,
)

runningState = system.initializeNewState()
t_limit = args.t_limit
delta_t = t_limit - runningState.t
dt = config.dt if isinstance(config.dt, float) else config.dt.cpu().item()
nSteps = int(delta_t / dt)
print(f"Resuming from frame {lastFrameIndex} at time {runningState.t:.5f}. "
     f"Running {nSteps} steps to reach {t_limit:.5f} seconds.")

if args.plot:
    ctx.imagePath = os.path.join(exportPath, 'images')
    os.makedirs(ctx.imagePath, exist_ok=True)
    handle = sodCase.setupPlot(ctx, runningState)

if args.store:
    outFile = h5py.File(os.path.join(exportPath, 'trajectory.h5'), 'a')
    groups = (outFile['positions'], outFile['velocities'], outFile['densities'], outFile['times'],
             outFile['rigidBodyTrajectories']) + tuple(outFile[name] for name in sodCase.extraFields)
    storeSteps = max(1, int(exportInterval / dt))

startIndex = lastFrameIndex + 1
for i in (tq := tqdm(range(startIndex, startIndex + nSteps), leave=True)):
    stepResult = ctx.integrator.function(
        state=runningState, f=ctx.stepFunction, dt=ctx.config.dt,
        config=ctx.config, schemeConfig=ctx.schemeConfig, verbose=False,
    )
    runningState = stepResult.state

    tScalar = runningState.t.item() if torch.is_tensor(runningState.t) else runningState.t
    row = sodCase.diagnostics(ctx, runningState)
    tq.set_description(f"t: {tScalar:.4f}, " + ", ".join(f"{k}: {v:.4f}" for k, v in row.items()))

    if args.plot and (i % args.plotInterval == 0 or i == startIndex + nSteps - 1):
        sodCase.updatePlot(ctx, runningState, handle, i)

    if args.store and (i % storeSteps == 0 or i == startIndex + nSteps - 1):
        writeFrame(groups, i, stepResult.state, stepResult.stages, config=ctx.config,
                  schemeConfig=ctx.schemeConfig, uniqueParticles=True, writeStages=False,
                  extraFields=sodCase.extraFields)

if args.store:
    outFile.close()

if args.plot:
    encodeFrames(ctx.imagePath, exportPath)
