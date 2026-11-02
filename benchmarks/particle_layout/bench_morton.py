"""Small CPU boundary-cost probe; no GPU locality/performance claim.

Run with paired frontend/integrator/core PYTHONPATH, OMP/BLAS one thread.
Tiny repeated steps intentionally omit video; this is not a trajectory campaign.
"""
import argparse
from contextlib import redirect_stdout
from copy import copy
import io
import json
import platform
import os
import random
from pathlib import Path
import statistics
import timeit

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import torch
from warpSPH.runner import RunConfiguration
from warpSPH.cases.compressible.linearWave import linearWaveCase
from warpSPH.evaluation import MortonStepper, adopt_spatial_layout, canonical_view
from warpSPH.evaluation.layout import permute_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes',nargs='+',type=int,default=[32,4096])
    parser.add_argument('--methods',nargs='+',default=['CompSPH','CRKSPH'])
    parser.add_argument('--samples',type=int,default=7)
    parser.add_argument('--calls',type=int,default=6)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    affinity = sorted(os.sched_getaffinity(0))
    def frequency(cpu):
        try:return int(Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq').read_text())
        except (OSError,ValueError):return 0
    chosen = max(affinity, key=lambda cpu:(frequency(cpu), -cpu))
    os.sched_setaffinity(0,{chosen})
    report=dict(cpu=chosen,cpu_max_khz=frequency(chosen),mode_order='deterministic shuffled per sample',
                original_affinity=affinity,python=platform.python_version(),torch=torch.__version__,device='cpu',
                precision='float32',samples=args.samples,calls=args.calls,rows=[])
    for method in args.methods:
        for n in args.sizes:
            with redirect_stdout(io.StringIO()):
                ctx=RunConfiguration('linearWave',simulation=dict(device='cpu',nx=n,dim=1,
                    integrationScheme='rungeKutta4',supportMode='KernelMeanSymmetric',dt=1e-4,quiet=True),
                    solver=dict(scheme=method)).prepare(linearWaveCase)
                initial=linearWaveCase.buildSystem(ctx);linearWaveCase.initialConditions(ctx,initial)
                p=torch.randperm(n,generator=torch.Generator().manual_seed(17))
                initial.state=permute_state(initial.state,p);initial.adjacency=None
                stepper=MortonStepper(ctx.integrator)
                first=stepper(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
                accepted=adopt_spatial_layout(first).state
                def native(state):
                    # The audited initializer only assigns metadata on its input;
                    # stages own their tensor clones, as on the adapter identity path.
                    owned=copy(state);owned.state=copy(state.state)
                    return ctx.integrator(owned,1e-4,ctx.stepFunction,config=ctx.config,schemeConfig=ctx.schemeConfig)
                nativeInput=native(initial).state
                modes={'cold_input':lambda:native(initial),
                       'cold_morton':lambda:stepper(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig),
                       'retained_native_input':lambda:native(nativeInput),
                       'retained_native_compute':lambda:native(accepted),
                       'retained_morton_identity':lambda:stepper(accepted,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig),
                       'adoption':lambda:adopt_spatial_layout(first),
                       'canonical_particles':lambda:canonical_view(accepted,include_adjacency=False),
                       'canonical_view':lambda:canonical_view(accepted)}
                row=dict(method=method,particles=n,timings={})
                samples = {name: [] for name in modes}
                for fun in modes.values():
                    for _ in range(3):fun()
                rng = random.Random(17)
                for _ in range(args.samples):
                    names = list(modes);rng.shuffle(names)
                    for name in names:
                        samples[name].append(timeit.timeit(modes[name],number=args.calls)/args.calls*1e3)
                for name, values in samples.items():
                    row['timings'][name]=dict(median_ms=statistics.median(values),samples_ms=values)
            report['rows'].append(row)
            print(json.dumps(row),flush=True)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
