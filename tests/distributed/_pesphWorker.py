"""Bounded native PESPH parity, resident restart and coordinated failure controls.

Particle HDF5 plus the core search recipe exercise the native state contract;
this does not qualify the driven-flow checkpoint CLI or its source adapter.
Repeated tiny controls deliberately omit video and do not measure performance.
"""
from dataclasses import fields, replace
from datetime import timedelta
from pathlib import Path
import argparse
import contextlib
import importlib
import json
import sys
import time
from unittest.mock import patch

import h5py
import torch
import torch.distributed as dist
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_pesphOperators import setup, VARIANTS
from warpSPH.enumTypes import AdaptiveSupportScheme
from warpSPH.io.hdf5 import dumpState, loadState
from warpSPH.modules.pesph.initialization import initializePESPHState
from warpSPH.modules.pesph.chunked import checkPESPHChunks
from warpSPH.schemes.compressible.pesph import pesphStep
from warpSPH.systems.pesph import PESPHSystem
from warpSPHCore import SupportScheme
from warpSPHCore.distributed import RowChunkProcessGroup, DistributedAllocationError, DistributedContractError
from warpSPHCore.radiusSearch import (buildChunkedVerletList, RowChunkPolicy,
    rowChunkCheckpointState, restoreRowChunkCheckpointState)
from warpSPHIntegrators.schemes import get_scheme, IntegrationSchemeType


def tensors(state):
    return {f.name: getattr(state, f.name).detach().clone()
            for f in fields(state) if isinstance(getattr(state, f.name), torch.Tensor)}


def equal(actual, expected):
    assert type(actual) is type(expected)
    for f in fields(expected):
        a, b = getattr(actual, f.name), getattr(expected, f.name)
        if isinstance(b, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=f.name)
        else:
            assert a == b, f.name


def qualify(comm, folder):
    rows = []
    integrator = get_scheme(IntegrationSchemeType.rungeKutta2)
    dispatch = importlib.import_module('warpSPHCore.autograd.row_dispatch')
    build = importlib.import_module('warpSPHCore.radiusSearch.verlet.build')
    for dim, nx in ((2, 6), (3, 5)):
        for variant in VARIANTS:
            initial, cfg, scheme = setup(dim=dim, nx=nx, variant=variant, uniform=True)
            initial.velocities *= .05
            scheme = replace(scheme, adaptiveSupportScheme=AdaptiveSupportScheme.NumberDensity,
                adaptiveSupportCorrections=True, adaptiveSupportIterations=60,
                dissipation=variant.name == 'PressureEnergy')
            def step(system):
                return integrator.function(state=system, dt=1e-5, f=pesphStep,
                    config=cfg, schemeConfig=scheme).state
            reference = PESPHSystem(state=initial.clone(), domain=cfg.domain)
            initializePESPHState(reference, cfg, scheme)
            for _ in range(4):
                reference = step(reference)
            current = PESPHSystem(state=initial.clone(), domain=cfg.domain)
            current.adjacency = buildChunkedVerletList(current.state, cfg.domain,
                policy=RowChunkPolicy(edgeBudget=4096, maxRows=3, bytesPerEdge=8),
                verletScale=cfg.verletScale, supportMode=SupportScheme.Gather, communication=comm)
            def forbidden(*args, **kwargs):
                raise AssertionError('PESPH constructed a monolithic pair list')
            with patch.object(build, 'buildVerletList_', forbidden):
                initializePESPHState(current, cfg, scheme)
                for _ in range(2):
                    current = step(current)
                accepted = replace(current.state, **tensors(current.state))
                # All ranks must receive the same allocation classification, and
                # accepted particles must survive failure in a later force pass.
                native = dispatch._launchLocal
                def fail(spec, *args, **kwargs):
                    if spec.kernel.key.endswith('_pressureForce') and comm.rank == comm.size-1:
                        raise torch.OutOfMemoryError('injected PESPH force output allocation')
                    return native(spec, *args, **kwargs)
                with patch.object(dispatch, '_launchLocal', fail):
                    try:
                        step(current)
                    except DistributedAllocationError as error:
                        assert 'injected PESPH' in str(error)
                    else:
                        raise AssertionError('expected coordinated PESPH allocation failure')
                equal(current.state, accepted)
                comm.assertReplicated('pesph:accepted-after-failure', tensors(current.state), exact=True)
                # A real native particle file, preserving authoritative E/A and
                # all optional/scalar switch history, plus an edge-free recipe.
                path = folder / f'{dim}-{variant.name}-rank{comm.rank}.h5'
                with h5py.File(path, 'w') as file:
                    file.attrs['time'] = float(current.t)
                    dumpState(current.state, file.create_group('state'))
                recipe = rowChunkCheckpointState(current.adjacency)
                with h5py.File(path, 'r') as file:
                    state = loadState(file['state'], comm.device, type(current.state))
                    resumed = PESPHSystem(state=state, domain=cfg.domain, t=float(file.attrs['time']))
                equal(resumed.state, current.state)
                resumed.adjacency = restoreRowChunkCheckpointState(recipe, device=comm.device,
                    policy=replace(current.adjacency.plan.policy, maxRows=1), communication=comm)
                for _ in range(2):
                    resumed = step(resumed)
            equal(resumed.state, reference.state)
            assert float(resumed.t) == float(reference.t)
            # An unsupported rank-local context rejects before a particle pass;
            # the communicator must remain usable afterward.
            original = resumed.sourceTerms
            resumed.sourceTerms = object() if comm.rank == comm.size-1 else None
            try:
                checkPESPHChunks(resumed, cfg, scheme)
            except DistributedContractError as error:
                assert 'require ConservedSources' in str(error)
            else:
                raise AssertionError('expected unsupported source rejection')
            resumed.sourceTerms = original
            comm.agree('pesph:guard-recovered', variant.name)
            rows.append(dict(dim=dim, variant=variant.name, acceptedSteps=4,
                exactNativeFields=True, changedRowsOnRestart=True, allocationFailure=True))
            print(f'[pesph rank {comm.rank}] {dim}D {variant.name} passed', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--replica-checks', choices=('exact', 'contract'), default='exact')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.device.startswith('cuda'):
        torch.cuda.set_device(torch.device(args.device))
    dist.init_process_group('nccl' if args.device.startswith('cuda') else 'gloo', timeout=timedelta(seconds=60))
    comm = RowChunkProcessGroup(dist.group.WORLD, args.device, replicaChecks=args.replica_checks)
    folder = Path(args.output); folder.mkdir(parents=True, exist_ok=True)
    wp.init(); started = time.perf_counter()
    stream = torch.cuda.Stream(device=comm.device) if comm.device.type == 'cuda' else None
    try:
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            rows = qualify(comm, folder)
        if stream is not None:
            stream.synchronize()
        report = dict(status='passed', cases=rows, wallSeconds=time.perf_counter()-started, **comm.summary())
        if comm.device.type == 'cuda':
            import ctypes
            version = ctypes.c_int(); ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(version))
            report['runtimeNccl'] = version.value
        (folder / f'pesph-rank{comm.rank}.json').write_text(json.dumps(report, indent=2))
        print(f'[pesph rank {comm.rank}] all PESPH checks passed', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
