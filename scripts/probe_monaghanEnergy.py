#!/usr/bin/env python3
"""OPEN_PROBLEMS §16 probe: which term of the Monaghan RHS breaks total-energy conservation?

Runs a case for a few steps, then evaluates the RHS pieces of `schemes/monaghan.py` on the
resulting state (`--switch ReadHayfield2012` adds the SPHS entropy-dissipation source) and reports, for each, the *net energy rate it injects*

    E_dot = sum_i m_i ( v_i . dvdt_i + dudt_i )

which must vanish pair by pair for a conservative pair scheme (an antisymmetric pair gradient
cancels exactly, whatever the density). The four physical pieces are: the pressure force with its
work term, the viscous force with its heating, and the conductivity. Each is also reported
relative to the kinetic power `sum |m v . dvdt|` of the piece, so a violation reads as a fraction.

    python scripts/probe_monaghanEnergy.py [--case sedov|noh|sod] [--nSteps 120] [--Cq 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from warpSPHCore import OperationProperties, SupportScheme          # noqa: E402
from warpSPH.runner import run                                        # noqa: E402
from warpSPH.modules.dissipation import (computeConductivity, computeThermalDissipation,   # noqa: E402
                                         computeViscosity)
from warpSPH.modules.eos import idealGasEOS                           # noqa: E402
from warpSPH.modules.internalEnergy import computeDudtMonaghan       # noqa: E402
from warpSPH.modules.pressure import computePressureForceSymmetric   # noqa: E402

CASES = {
    'sedov': ('warpSPH.cases.sedov', 'sedovCase', dict(dim=3, nx=24)),
    'noh': ('warpSPH.cases.noh', 'nohCase', dict()),
    'sod': ('warpSPH.cases.sod', 'sodCase', dict(nx=100)),
}


def pieces(system, config, schemeConfig, switch='NoneSwitch'):
    st = system.state
    st.entropies, _, st.pressures, st.soundspeeds = idealGasEOS(
        A=None, u=st.internalEnergies, P=None, rho=st.densities, gamma=schemeConfig.gamma)
    adj = system.adjacency
    prop = OperationProperties(kernel=config.kernel, supportMode=SupportScheme.KernelMeanSymmetric)
    dp = schemeConfig.diffusionParams
    kw = dict(operationProperties=prop, domain=config.domain, adjacency=adj, queryAlphas=st.alphas)
    out = {}
    out['pressure force + work'] = (
        computePressureForceSymmetric(st, config, supportScheme=SupportScheme.KernelMeanSymmetric,
                                      adjacency=adj, gradH=None),
        computeDudtMonaghan(st, config, supportScheme=SupportScheme.KernelMeanSymmetric,
                            adjacency=adj, gradH=None))
    out['viscous force + heating'] = (
        computeViscosity(st, viscosityParams=dp, **kw),
        computeThermalDissipation(st, conductivityParams=dp, **kw))
    out['conductivity'] = (torch.zeros_like(st.velocities),
                           computeConductivity(st, conductivityParams=dp, **kw))
    if switch == 'ReadHayfield2012':
        # OPEN_PROBLEMS §17: the SPHS entropy-dissipation source (eqs. 33-35); heat only
        from warpSPH.modules.shockCapturing.ReadHayfield2012 import computeReadHayfieldTerms
        _, sw = computeReadHayfieldTerms(0.0, st, config, schemeConfig, SupportScheme.KernelMeanSymmetric, adj)
        out['R&H entropy dissipation'] = (torch.zeros_like(st.velocities), sw.dudt_diss)
    return st, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='sedov', choices=list(CASES))
    ap.add_argument('--nSteps', type=int, default=120)
    ap.add_argument('--Cq', type=float, default=None, help='override C_q (default: Monaghan 0)')
    ap.add_argument('--switch', default='NoneSwitch')
    args = ap.parse_args()

    import importlib
    mod, name, kw = CASES[args.case]
    case = getattr(importlib.import_module(mod), name)

    def configure(ctx, _orig=case.configureScheme):
        _orig(ctx)
        if args.Cq is not None:
            ctx.schemeConfig.diffusionParams.C_q = args.Cq

    import dataclasses
    case = dataclasses.replace(case, configureScheme=configure)
    res = run(case, scheme='Monaghan', nSteps=args.nSteps, progress=False, quiet=True,
              params=dict(viscositySwitch=args.switch), **kw)
    st, out = pieces(res.state, res.ctx.config, res.ctx.schemeConfig, args.switch)
    m, v = st.masses, st.velocities
    E = res.series('totalEnergy')
    print(f'{args.case}: {args.nSteps} steps, t={float(res.state.t):.4f}, N={len(m)}, '
          f'total energy {E[0]:.4f} -> {E[-1]:.4f}')
    print(f'{"piece":28s} {"E_dot injected":>16s} {"|kinetic power|":>16s} {"ratio":>10s}')
    total = 0.0
    for name_, (dvdt, dudt) in out.items():
        kin = (m * torch.einsum('ij,ij->i', v, dvdt)).sum().item()
        heat = (m * dudt).sum().item()
        e = kin + heat
        total += e
        scale = max((m * torch.einsum('ij,ij->i', v, dvdt)).abs().sum().item(), abs(heat), 1e-30)
        print(f'{name_:28s} {e:16.6e} {scale:16.6e} {e / scale:10.2e}   (kinetic {kin:+.4e}, heat {heat:+.4e})')
    print(f'{"total":28s} {total:16.6e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
