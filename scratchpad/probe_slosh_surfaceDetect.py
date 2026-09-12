"""Why is UID 4104 (the sloshingTank ceiling-hover particle) flagged as a
free-surface particle (surfaceIndicators == 1) while it is pinned against a
solid ceiling with plenty of nearby boundary particles? Resume the checkpoint,
rebuild the neighbour list, and print the exact colour-field-detector inputs
for this one particle: colorField, meanColorField, numNeighbors (AllToAll,
so boundary should count), the fluid-only count, and config.targetNeighbors *
colorFieldThreshold (the cutoff `detectFreeSurfaceColorField` compares
numNeighbors against)."""
import argparse
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument('--from-step', type=int, default=60000)
ap.add_argument('--uid', type=int, default=4104)
ap.add_argument('--dir', default=None)
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import h5py
import torch

from warpSPH.io.hdf5 import loadState
from warpSPH.systems.weaklyCompressible import WeaklyCompressibleState
from warpSPH.cases.sloshingTank import sloshingTankCase
from warpSPH.runner import run
from warpSPH.modules.surfaceDetection.colorFieldCompute import computeColorField
from warpSPH.modules.util.wp_numNeighbors import countNeighborsWarp
from warpSPHCore import (OperationProperties, WarpOperation, SupportScheme,
                          OperationDirection, GradientScheme, buildVerletList)

runDir = args.dir or sorted(glob.glob('export/16-sloshingTank-wcsph_*'), key=os.path.getmtime)[-1]
r = run(sloshingTankCase, scheme='deltaSPH', nx=225, nSteps=1, tLimit=1e9,
        quiet=True, store=False, progress=False)
ctx = r.ctx
cfg, scheme = ctx.config, ctx.schemeConfig
dev = cfg.domain.min.device

f = os.path.join(runDir, 'trajectory', f'state_{args.from_step:04d}.h5')
h = h5py.File(f, 'r')
ckpt = loadState(h['state'], dev, WeaklyCompressibleState)

import dataclasses
running = r.state
particles = getattr(running, 'state', running)
for fl in dataclasses.fields(ckpt):
    v = getattr(ckpt, fl.name, None)
    if torch.is_tensor(v):
        setattr(particles, fl.name, v.clone())
    elif v is not None and fl.name != 't':
        setattr(particles, fl.name, v)

adj = buildVerletList(particles, cfg.domain, verletScale=1.0,
                      supportMode=SupportScheme.SuperSymmetric,
                      priorNeighborhood=None, verbose=False)

colorField, colorFieldGrad = computeColorField(particles, cfg, scheme, adj)
numNeighbors = countNeighborsWarp(
    particles,
    OperationProperties(kernel=cfg.kernel, operation=WarpOperation.Interpolate,
                        supportMode=SupportScheme.SuperSymmetric,
                        operationMode=OperationDirection.AllToAll,
                        gradientMode=GradientScheme.Naive),
    domain=cfg.domain, adjacency=adj)
numNeighborsFluid = countNeighborsWarp(
    particles,
    OperationProperties(kernel=cfg.kernel, operation=WarpOperation.Interpolate,
                        supportMode=SupportScheme.SuperSymmetric,
                        operationMode=OperationDirection.FluidToFluid,
                        gradientMode=GradientScheme.Naive),
    domain=cfg.domain, adjacency=adj)

i = int((particles.UIDs == args.uid).nonzero()[0])
print(f'UID {args.uid} at index {i}, kind={int(particles.kinds[i])}, y={float(particles.positions[i,1]):.5f}')
print(f'colorField[i] = {float(colorField[i]):.4f}')
print(f'numNeighbors (AllToAll, excl. ghost) = {int(numNeighbors[i])}')
print(f'numNeighbors (FluidToFluid only)     = {int(numNeighborsFluid[i])}')
print(f'config.targetNeighbors = {cfg.targetNeighbors}, colorFieldThreshold = {scheme.surfaceDetectionConfig.colorFieldThreshold}')
print(f'cutoff = targetNeighbors * colorFieldThreshold = {cfg.targetNeighbors * scheme.surfaceDetectionConfig.colorFieldThreshold:.2f}')

# mean colour field over the same AllToAll neighbourhood (matches
# detectFreeSurfaceColorField's own meanColorField calc)
from warpSPH.modules.util.wp_sum import warpSum
meanColorField = warpSum(
    particles,
    OperationProperties(kernel=cfg.kernel, operation=WarpOperation.Interpolate,
                        supportMode=SupportScheme.SuperSymmetric,
                        operationMode=OperationDirection.AllToAll,
                        gradientMode=GradientScheme.Naive),
    queryValues=colorField, domain=cfg.domain, adjacency=adj) / numNeighbors
print(f'meanColorField[i] = {float(meanColorField[i]):.4f}')
print(f'colorField < meanColorField ? {float(colorField[i]) < float(meanColorField[i])}')
print(f'numNeighbors < cutoff ?        {int(numNeighbors[i]) < cfg.targetNeighbors * scheme.surfaceDetectionConfig.colorFieldThreshold}')

# how many of its AllToAll neighbours are boundary vs fluid vs ghost, by hand
xi = particles.positions[i]
support = float(particles.supports[i])
d = (particles.positions - xi).norm(dim=1)
within = d < support
for k, name in [(0, 'fluid'), (1, 'boundary'), (2, 'ghost')]:
    cnt = int(((particles.kinds == k) & within).sum()) - (1 if k == int(particles.kinds[i]) else 0)
    print(f'  neighbours of kind {name}: {cnt}')
