"""Call audit of any registered case: where do the torch ops, warp launches and host syncs of a step
come from? Prints the section tree (inclusive and self counts per call, syncs per step, how much of
the run is outside any section), the heaviest sections with their top ops / kernels / sync call sites,
and the worst steps, and writes the same to a file. See `warpSPHCore/callAudit.py` and the
`call-audit` skill.

    python scripts/audit_case.py dambreak --scheme artificialCompressible --nx 70 --steps 60 --warmup 10
    python scripts/audit_case.py sloshingTank --scheme artificialCompressible --nx 100 --set acBounceRamp=0.5
    python scripts/audit_case.py hydrostaticColumn --scheme deltaSPH --nx 64 --no-cudaGraph

`--set key=value` (repeatable) sets a CaseSpec field or case parameter (value read as a Python
literal, else a string). Warm-up steps (first-step compilation, graph capture, cache fills) are
discarded. No video, no per-step plots: this is a counting run; the dispatch hook makes it ~3x
slower, so read the counts, not the timings.
"""
import argparse
import ast
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('case', help='registered case name (e.g. dambreak, sloshingTank, hydrostaticColumn)')
ap.add_argument('--scheme', default=None)
ap.add_argument('--nx', type=int, default=None)
ap.add_argument('--steps', type=int, default=60, help='steps to run in total (default 60)')
ap.add_argument('--warmup', type=int, default=10, help='leading steps discarded from the counts (default 10)')
ap.add_argument('--set', dest='sets', action='append', default=[], metavar='KEY=VALUE')
ap.add_argument('--scheme-set', dest='schemeSets', action='append', default=[], metavar='KEY=VALUE',
                help='set an attribute of the scheme config after the case configured it (e.g. supportSolveOnGrid=True)')
ap.add_argument('--cudaGraph', action=argparse.BooleanOptionalAction, default=None,
                help="override the case's cudaGraph default")
ap.add_argument('--no-torch-ops', dest='torchOps', action='store_false',
                help='skip the aten-op counter (faster; syncs and launches only)')
ap.add_argument('--out', default=None, help='report file (default scripts/out_callAudit/<case>_<scheme>_<time>.txt)')
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

from warpSPHCore import callAudit
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase, listCases

importAll()
if args.case not in listCases():
    raise SystemExit(f'unknown case {args.case!r}; known: {", ".join(listCases())}')
kw = {}
for item in args.sets:
    key, _, raw = item.partition('=')
    try:
        kw[key] = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        kw[key] = raw
case = getCase(args.case)
if args.schemeSets:
    import dataclasses
    schemeKw = {}
    for item in args.schemeSets:
        key, _, raw = item.partition('=')
        try:
            schemeKw[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            schemeKw[key] = raw
    innerConfigure = case.configureScheme

    def _configure(ctx, *a, **k):
        out = innerConfigure(ctx, *a, **k) if innerConfigure else None
        for key, value in schemeKw.items():
            setattr(ctx.schemeConfig, key, value)
        return out
    case = dataclasses.replace(case, configureScheme=_configure)
specFields = {k: v for k, v in dict(scheme=args.scheme, nx=args.nx, cudaGraph=args.cudaGraph).items() if v is not None}

callAudit.enable(torchOps=args.torchOps, warmupSteps=args.warmup)
print(f'[audit] {args.case} scheme={args.scheme or case.scheme} steps={args.steps} warmup={args.warmup} '
      f'(no video: a counting run; the audit hooks slow it ~3x)', flush=True)
t0 = time.perf_counter()
# set --steps via nSteps; user overrides can still replace any spec field or case param
params = {k: v for k, v in kw.items() if k not in ('nSteps',)}
result = run(case, quiet=False, progress=True, nSteps=kw.get('nSteps', args.steps), **specFields, params=params)
print(f'[audit] {result.nSteps} steps in {time.perf_counter() - t0:.1f} s wall (audit overhead included)', flush=True)

out = args.out or os.path.join(HERE, 'out_callAudit',
                               f"{args.case}_{args.scheme or case.scheme}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
os.makedirs(os.path.dirname(out), exist_ok=True)
text = callAudit.writeReport(out)
print(text)
print(f'[audit] written to {out}')
