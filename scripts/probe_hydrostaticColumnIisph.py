"""Validation run of `IncompressibleSPHScheme.iisph` on `hydrostaticColumn`
(DFSPH_IMPROVEMENT_PLAN.md active track, validation ladder item 1).

Part 33 landed `iisph` (plain IISPH, [I]) as the first scheme to hold the
quiescent wall-bounded free-surface-under-gravity column, but only measured it
at `nx = 32`. This probe runs it at the case's default `nx = 128` and grades
the column with the spray-robust density FOMs added to `hydrostaticDiagnostics`
in the same part:

- `minDensity`         -- the plain fluid min: dominated by 1-3 particles the
                          bulk slosh throws 1-3 dx above the surface;
- `densityP05`         -- 5th percentile of the fluid density (spray cannot
                          move it);
- `embeddedMinDensity` -- min over fluid rows > 1 dx below the 95th-percentile
                          surface, i.e. the ballistic skin removed outright.

and the structural axes the case already reports: the 95th-percentile surface
height (geometry stability), `pressureSlopeRatio` (-> ~1.0 is the exact
hydrostatic gradient), `pressureResidual`, `maxVelocity` / `kineticEnergy`
(the bounded undamped free-slip slosh Part 33 leaves open).

Usage:
    python scripts/probe_hydrostaticColumnIisph.py                  # nx=128 -> tLimit
    python scripts/probe_hydrostaticColumnIisph.py --steps 1500     # fixed step count
    python scripts/probe_hydrostaticColumnIisph.py --nx 32 --steps 2000
    python scripts/probe_hydrostaticColumnIisph.py --scheme dfsphReference --steps 1500
"""
import argparse

args = argparse.ArgumentParser()
args.add_argument('--nx', type=int, default=128)
args.add_argument('--steps', type=int, default=0,
                 help='fixed step count; 0 (default) runs time-limited to tLimit')
args.add_argument('--scheme', type=str, default='iisph',
                 help='iisph (default), dfsphReference, or divergenceFree')
args.add_argument('--samples', type=str, default='',
                 help='comma list of step indices to tabulate; default is '
                      '10 evenly spaced over the run')
args = args.parse_args()

from warpSPH.cases import hydrostaticColumn
from warpSPH.runner import run

r = run(hydrostaticColumn.hydrostaticColumnCase, nx=args.nx,
        nSteps=(None if args.steps == 0 else args.steps),
        scheme=args.scheme, quiet=True, plot=False, store=False, progress=False,
        integrationScheme='semiImplicitEuler')

rows = [x for x in r.trajectory if x.get('step', -1) >= 0]
n = len(rows)
if args.samples:
    idxs = [int(s) - 1 for s in args.samples.split(',')]
else:
    idxs = sorted(set(min(n - 1, max(0, round(k * (n - 1) / 9))) for k in range(10)))


def g(row, key):
    return row.get(key, float('nan'))


h0 = g(rows[0], 'dispMax')  # just to touch the key; real geometry is surfaceY
print(f'hydrostaticColumn  scheme={args.scheme}  nx={args.nx}  '
      f'{"steps=" + str(args.steps) if args.steps else "-> tLimit"}  '
      f'ran {n} steps  diverged={r.diverged}')
print()
hdr = (f'{"step":>6} {"t":>7} {"|v|max":>8} {"KE":>9} {"minRho":>8} '
       f'{"rhoP05":>8} {"embMin":>8} {"maxRho":>8} {"slope":>8} {"pResid":>9} '
       f'{"dispMax":>8}')
print(hdr)
print('-' * len(hdr))
for i in idxs:
    row = rows[i]
    print(f'{row.get("step", i + 1):>6} {g(row, "t"):>7.4f} '
          f'{g(row, "maxVelocity"):>8.3f} {g(row, "kineticEnergy"):>9.4f} '
          f'{g(row, "minDensity"):>8.4f} {g(row, "densityP05"):>8.4f} '
          f'{g(row, "embeddedMinDensity"):>8.4f} {g(row, "maxDensity"):>8.4f} '
          f'{g(row, "pressureSlopeRatio"):>8.3f} {g(row, "pressureResidual"):>9.2e} '
          f'{g(row, "dispMax"):>8.3f}')

tail = [g(x, 'pressureSlopeRatio') for x in rows[3 * n // 4:]]
tail = [s for s in tail if s == s]
slopeLate = sum(tail) / len(tail) if tail else float('nan')
keTail = [g(x, 'kineticEnergy') for x in rows[3 * n // 4:]]
keTail = [k for k in keTail if k == k]
print()
print(f'late (last quarter):  slopeRatio ~ {slopeLate:.3f}   '
      f'KE in [{min(keTail):.4f}, {max(keTail):.4f}]' if keTail else '')
embTail = [g(x, 'embeddedMinDensity') for x in rows[3 * n // 4:]]
embTail = [e for e in embTail if e == e]
minTail = [g(x, 'minDensity') for x in rows[3 * n // 4:]]
minTail = [m for m in minTail if m == m]
if embTail:
    print(f'                      embeddedMinDensity in [{min(embTail):.4f}, '
          f'{max(embTail):.4f}]   plain minDensity in [{min(minTail):.4f}, '
          f'{max(minTail):.4f}]')
