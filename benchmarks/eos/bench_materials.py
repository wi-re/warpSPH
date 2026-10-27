"""End-to-end eager EOS latency, including dispatch/allocation and GPU completion.

Run from the repository root with the normal CUDA Python environment.
"""
import argparse
import json
from pathlib import Path
import time

from warpSPHBootstrap import bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--particles', type=int, default=163840)
    parser.add_argument('--repeats', type=int, default=200)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.particles < 1 or args.repeats < 1:
        parser.error('particles and repeats must be positive')
    bootstrap(precision='float32')
    import torch
    from warpSPH.configurations import CRKSPHConfig, EOSConfig, MaterialConfig
    from warpSPH.modules.eos import evaluateEOS, idealGasEOS
    rho = torch.linspace(.5, 3., args.particles, device='cuda')
    energy = torch.linspace(1., 4., args.particles, device='cuda')
    ids = torch.arange(args.particles, device='cuda', dtype=torch.int32) % 2
    cfg = CRKSPHConfig(materials=[MaterialConfig('gas0', EOSConfig(gamma=1.4)),
                                 MaterialConfig('gas1', EOSConfig(gamma=1.4))])

    def scalar():
        return idealGasEOS(A=None, u=energy, P=None, rho=rho, gamma=1.4)

    def material():
        return evaluateEOS(rho, ids, cfg, u=energy)

    for a, b in zip(scalar(), material()):
        torch.testing.assert_close(a, b)
    summary = dict(particles=args.particles, repeats=args.repeats, device=torch.cuda.get_device_name())
    for label, compute in (('legacyScalar', scalar), ('materialTable', material)):
        for _ in range(10):
            compute()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(args.repeats):
            output = compute()
        torch.cuda.synchronize()
        summary[label + 'Milliseconds'] = (time.perf_counter() - begin) * 1000 / args.repeats
    print(json.dumps(summary, indent=2), flush=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
