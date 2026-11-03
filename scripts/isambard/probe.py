#!/usr/bin/env python3
"""Real torchrun collective/device/shared-storage probe, independent of solver steps."""
import argparse
from datetime import timedelta
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('nccl', 'gloo'), default='nccl')
    parser.add_argument('--expected-nodes', type=int, required=True)
    parser.add_argument('--expected-architecture', default='aarch64')
    args = parser.parse_args()
    if platform.machine() != args.expected_architecture:
        raise RuntimeError(f'Expected {args.expected_architecture}; running {platform.machine()}. Rebuild for the target CPU architecture.')
    os.environ.setdefault('warpSPHCore_PRECISION', 'float32')
    import torch
    import torch.distributed as dist
    import warp as wp
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    gpu = args.backend == 'nccl'
    device = torch.device('cuda', local) if gpu else torch.device('cpu')
    if gpu:
        if int(os.environ['LOCAL_WORLD_SIZE']) > torch.cuda.device_count():
            raise RuntimeError('Physical GPU missing; shared-GPU debug mode is not a site qualification')
        torch.cuda.set_device(device)
    dist.init_process_group(args.backend, timeout=timedelta(seconds=600))
    try:
        size = dist.get_world_size()
        wp.init()
        # Compile a real Warp kernel on the Torch nondefault stream, then use
        # its output in a collective. This checks device and stream integration.
        if gpu:
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                value = torch.full((4096,), float(rank+1), dtype=torch.float64, device=device)
                wp.launch(twice, dim=value.numel(), inputs=[wp.from_torch(value)], stream=wp.stream_from_torch(stream))
                dist.all_reduce(value)
            stream.synchronize()
            torch.testing.assert_close(value, torch.full_like(value, size*(size+1)), rtol=0, atol=0)
        integer = torch.tensor([rank], device=device, dtype=torch.int64)
        gathered = [torch.empty_like(integer) for _ in range(size)]
        dist.all_gather(gathered, integer)
        assert [int(value.item()) for value in gathered] == list(range(size))
        value = torch.tensor([123 if rank == 0 else -1], device=device, dtype=torch.int64)
        dist.broadcast(value, 0)
        assert value.item() == 123
        if gpu:
            output = torch.empty(4, device=device, dtype=torch.float64)
            data = torch.arange(4*size, device=device, dtype=torch.float64) + rank+1.
            dist.reduce_scatter_tensor(output, data)
            expected = torch.arange(rank*4, (rank+1)*4, device=device, dtype=torch.float64)*size + size*(size+1)/2
            assert torch.equal(output, expected)
        if size > 1:
            received = torch.empty_like(integer)
            work = dist.batch_isend_irecv([dist.P2POp(dist.isend, integer, (rank+1)%size),
                                         dist.P2POp(dist.irecv, received, (rank-1)%size)])
            for item in work:
                item.wait()
            assert received.item() == (rank-1)%size
        args.output.mkdir(parents=True, exist_ok=True)
        # Each rank writes independently, then everybody reads everybody's file
        # after a barrier. A merely node-local output directory cannot pass.
        marker = args.output / f'visibility-{rank}.txt'
        marker.write_text(str(rank))
        dist.barrier()
        assert all((args.output / f'visibility-{other}.txt').read_text() == str(other) for other in range(size))
        from warpSPH.distributedRun import _sourcePins
        packages = {name: importlib.metadata.version(name) for name in
                    ('torch', 'warp-lang', 'warpSPH', 'warpSPHCore', 'warpSPHIntegrators')}
        record = dict(rank=rank, localRank=local, ranks=size, host=socket.gethostname(), pid=os.getpid(),
                      architecture=platform.machine(), backend=args.backend, packages=packages,
                      sources={name: item['digest'] for name, item in _sourcePins().items()},
                      cuda=torch.version.cuda, gpuUuid=str(torch.cuda.get_device_properties(device).uuid) if gpu else None,
                      gpuName=torch.cuda.get_device_name(device) if gpu else None,
                      nccl=list(torch.cuda.nccl.version()) if gpu else None,
                      network={key: value for key, value in os.environ.items() if key.startswith(('NCCL_', 'FI_', 'SLINGSHOT_'))})
        records = [None] * size
        dist.all_gather_object(records, record)
        if any(row['sources'] != records[0]['sources'] or row['packages'] != records[0]['packages']
               for row in records):
            raise RuntimeError('Application/runtime differs between ranks')
        if len({row['host'] for row in records}) != args.expected_nodes:
            raise RuntimeError('Expected number of physical hosts was not used')
        if len({(row['host'], row['pid']) for row in records}) != size:
            raise RuntimeError('Ranks are not separate processes')
        if gpu and len({row['gpuUuid'] for row in records}) != size:
            raise RuntimeError('Ranks did not use distinct physical GPUs')
        if rank == 0:
            temporary = args.output / '.probe.json.tmp'
            temporary.write_text(json.dumps(dict(status='passed', ranks=records, sharedStorage=True), indent=2)+'\n')
            temporary.replace(args.output / 'probe.json')
            print(f'Collectives, stream ordering, rank placement and shared storage passed: {size} ranks on {args.expected_nodes} hosts', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    # Warp needs module-level, file-backed kernels; do not define this in -c.
    import warp as wp
    @wp.kernel
    def twice(values: wp.array(dtype=wp.float64)):
        index = wp.tid()
        values[index] = values[index] * wp.float64(2.0)
    main()
