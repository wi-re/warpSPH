"""Sequential CUDA locality and adapter cost qualification, synchronized wall time.

Short repeated one-step numerical/throughput fixtures omit video and viewer.
No claim of a long physical trajectory or a CUDA-graph comparison.
"""
import argparse
from contextlib import redirect_stdout
from copy import copy
import io
import json
import random
import statistics
import time
from pathlib import Path

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import torch

from warpSPH.runner import RunConfiguration
from warpSPH.cases.compressible.linearWave import linearWaveCase
from warpSPH.configurations.simulationConfig import buildConfig
from warpSPH.sample.compressible import setupBasicCompressibleInitialState
from warpSPH.systems.compSPH import CompSPHState,CompSPHSystem
from warpSPH.modules.eos import refreshEOS
from warpSPH.utils.domain import buildDomainDescription
from warpSPH.enumTypes import AdaptiveSupportScheme,ViscositySwitch
from warpSPHCore import SupportScheme
from warpSPH.evaluation import MortonStepper,adopt_spatial_layout,canonical_view
from warpSPH.evaluation.layout import permute_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nx',nargs='+',type=int,default=[32,128,256,512,1024])
    parser.add_argument('--dim',type=int,default=2)
    parser.add_argument('--methods',nargs='+',default=['CompSPH','CRKSPH'])
    parser.add_argument('--samples',type=int,default=5)
    parser.add_argument('--calls',type=int,default=8)
    parser.add_argument('--native-only',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    report=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,dim=args.dim,support_mode='KernelMeanSymmetric',
                precision='float32',samples=args.samples,calls=args.calls,rows=[])
    for fluid in args.methods:
        for nx in args.nx:
            start=time.perf_counter()
            print(f'preparing {fluid} nx={nx} dim={args.dim}',flush=True)
            with redirect_stdout(io.StringIO()):
                ctx=RunConfiguration('linearWave',simulation=dict(device='cuda',nx=32,dim=1,
                    integrationScheme='rungeKutta4',supportMode='KernelMeanSymmetric',dt=1e-4,quiet=True),
                    solver=dict(scheme=fluid)).prepare(linearWaveCase)
                domain=buildDomainDescription(l=1,dim=args.dim,periodic=True,dtype=torch.float32,device='cuda')
                ctx.config,ctx.integrator=buildConfig(dim=args.dim,nx=nx,domain=domain,dtype=torch.float32,
                    device=torch.device('cuda'),dt=1e-4,integrationScheme=ctx.config.integrationScheme,n_h=3.,
                    supportMode=SupportScheme.KernelMeanSymmetric)
                initial=setupBasicCompressibleInitialState(nx,ctx.config,ctx.schemeConfig,CompSPHState,CompSPHSystem)
                initial.state.internalEnergies.fill_(2.5)
                initial.state.velocities[:,0]=.03*torch.sin(2*torch.pi*initial.state.positions[:,0])
                refreshEOS(initial.state,ctx.schemeConfig)
                initial.state.totalEnergies=initial.state.masses*(initial.state.internalEnergies+.5*initial.state.velocities.square().sum(-1))
                ctx.schemeConfig.compatibleEnergy=False
                ctx.schemeConfig.adaptiveSupportScheme=AdaptiveSupportScheme.NoScheme
                ctx.schemeConfig.viscositySwitchParams.scheme=ViscositySwitch.NoneSwitch
                p=torch.randperm(len(initial.state.UIDs),generator=torch.Generator().manual_seed(17),device='cpu').to('cuda')
                initial.state=permute_state(initial.state,p);initial.adjacency=None
                stepper=MortonStepper(ctx.integrator)
                def native(state):
                    owned=copy(state);owned.state=copy(state.state)
                    return ctx.integrator(owned,1e-4,ctx.stepFunction,config=ctx.config,schemeConfig=ctx.schemeConfig)
                nativeInput=native(initial).state
                modes={'cold_native_input':lambda:native(initial),
                       'retained_native_input':lambda:native(nativeInput)}
                if not args.native_only:
                    first=stepper(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
                    accepted=adopt_spatial_layout(first).state
                    a=canonical_view(nativeInput,include_adjacency=False)[0].state
                    b=canonical_view(first.state,include_adjacency=False)[0].state
                    for name in ('positions','velocities','internalEnergies','densities','supports'):
                        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=4e-5,atol=4e-6)
                    modes.update(cold_morton=lambda:stepper(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig),
                        retained_native_compute=lambda:native(accepted),
                        retained_morton_identity=lambda:stepper(accepted,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig),
                        adoption=lambda:adopt_spatial_layout(first),
                        canonical_particles=lambda:canonical_view(accepted,include_adjacency=False))
                samples={name:[] for name in modes}
                diagnostics={}
                if not args.native_only:
                    diagnostics['max_abs_sorted_native']={name:float((getattr(a,name)-getattr(b,name)).abs().max())
                        for name in ('positions','velocities','internalEnergies','densities','supports')}
                def energy(s):
                    return (s.masses.double()*(s.internalEnergies.double()+.5*s.velocities.double().square().sum(-1))).sum()
                e0=energy(initial.state)
                diagnostics['native_relative_energy_change']=float((energy(nativeInput.state)-e0)/e0)
                diagnostics['native_momentum_change_norm']=float(((nativeInput.state.masses[:,None]*nativeInput.state.velocities).double().sum(0)
                    -(initial.state.masses[:,None]*initial.state.velocities).double().sum(0)).norm())
                for fun in modes.values():
                    for _ in range(3):fun()
                torch.cuda.synchronize()
            print(f'ready {fluid} N={len(initial.state.UIDs)}; setup {time.perf_counter()-start:.3f}s',flush=True)
            rng=random.Random(17)
            with torch.no_grad():
                for trial in range(args.samples):
                    names=list(modes);rng.shuffle(names)
                    for name in names:
                        torch.cuda.synchronize();tick=time.perf_counter()
                        for _ in range(args.calls):modes[name]()
                        torch.cuda.synchronize()
                        elapsed=(time.perf_counter()-tick)/args.calls*1e3
                        samples[name].append(elapsed)
                    print(f'{fluid} N={len(initial.state.UIDs)} sample={trial+1}/{args.samples} complete',flush=True)
            row=dict(method=fluid,nx=nx,particles=len(initial.state.UIDs),wall_seconds=time.perf_counter()-start,
                     diagnostics=diagnostics,timings={n:dict(median_ms=statistics.median(v),samples_ms=v) for n,v in samples.items()})
            report['rows'].append(row)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
