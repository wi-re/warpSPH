"""Static packing/torque audit of saved research states; advances no particles.

The manufactured comparison is exact incompressible parallel shear, with the
KH seed removed. It is a geometric comparator, not an exact seeded KH solution.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree, Voronoi

from crkClosure2D import pressure_rhs
from crkClosureGPU2D import GPUClosure
from study_crkClosure2D import kh_initial


def quantiles(a):
    return dict(zip(('min', 'p01', 'p10', 'median', 'p90', 'p99', 'max'),
                    map(float, np.quantile(a, [0, .01, .1, .5, .9, .99, 1]))))


def packing(x, V):
    tree = cKDTree(x % 1, boxsize=1.)
    distances, neighbors = tree.query(x % 1, k=13)
    r = x[neighbors[:, 1:]] - x[:, None]
    r -= np.round(r)
    eigen = np.linalg.eigvalsh(np.einsum('nka,nkb->nab', r, r) / 12)
    shifts = np.array([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)])
    tiled = np.concatenate([x + shift for shift in shifts])
    voronoi = Voronoi(tiled)
    center = np.flatnonzero(np.all(shifts == 0, axis=1))[0] * len(x)
    areas = []
    for number in range(center, center + len(x)):
        region = voronoi.regions[voronoi.point_region[number]]
        if not region or -1 in region:
            raise RuntimeError('Central periodic Voronoi cell is unbounded')
        polygon = voronoi.vertices[region]
        areas.append(abs(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                                - polygon[:, 1] * np.roll(polygon[:, 0], -1))) / 2)
    areas = np.array(areas)
    if not np.isclose(areas.sum(), 1., atol=1.e-10):
        raise RuntimeError('Periodic Voronoi area does not tile the domain')
    normalized = distances[:, 1] / np.sqrt(V)
    closest = int(np.argmin(normalized))
    return dict(nearestDistanceOverSqrtV=quantiles(normalized),
                fractionNearestBelowQuarterSpacing=float(np.mean(normalized < .25)),
                fractionNearestBelowHalfSpacing=float(np.mean(normalized < .5)),
                closestPairParticleIDs=[closest, int(neighbors[closest, 1])],
                closestPairPositions=x[[closest, neighbors[closest, 1]]].tolist(),
                twelveNeighborCovarianceCondition=quantiles(eigen[:, 1] / eigen[:, 0]),
                voronoiAreaOverEvolvedV=quantiles(areas / V),
                voronoiAreaSum=float(areas.sum()))


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def operator_audit(x, V, mass, velocity, pressure, i, j, q):
    raw = x[j] - x[i]
    r = raw - np.round(raw)
    moment = np.zeros((len(x), 2, 2))
    edge_moment = q[:, :, None] * r[:, None, :]
    np.add.at(moment, i, edge_moment)
    np.add.at(moment, j, edge_moment)
    a, dV, du = pressure_rhs(V, mass, pressure, velocity, i, j, q)
    force = mass[:, None] * a
    pressure_sum = pressure[i] + pressure[j]
    intrinsic = float(np.sum(pressure_sum * cross(r, q)))
    from_moments = float(np.dot(pressure, moment[:, 1, 0] - moment[:, 0, 1]))
    wrapped = float(np.sum(cross(x, force)))
    seam = float(np.sum(pressure_sum * cross(raw - r, q)))
    scale = max(1., float(np.sum(abs(pressure_sum * cross(r, q)))))
    if max(abs(intrinsic - from_moments), abs(wrapped - intrinsic - seam)) > 1.e-11 * scale:
        raise RuntimeError('Independent pair/moment/wrapped torque identities disagree')
    return dict(momentResidual=float(np.max(abs(moment / V[:, None, None] - np.eye(2)))),
                totalForce=list(map(float, force.sum(axis=0))),
                pressureTorqueMinimumImage=intrinsic,
                pressureTorqueFromMomentAntisymmetry=from_moments,
                pressureTorqueWrappedCoordinates=wrapped, periodicSeamTorque=seam,
                absolutePairTorqueSum=float(np.sum(abs(pressure_sum * cross(r, q)))),
                weightedNoncentralFraction=float(np.sum(abs(cross(r, q))) /
                    np.sum(np.linalg.norm(r, axis=1) * np.linalg.norm(q, axis=1))),
                negativeRadialCoefficientFraction=float(np.mean(np.einsum('na,na->n', r, q) < 0)),
                maxAcceleration=float(np.linalg.norm(a, axis=1).max()),
                volumeDivergenceRMS=float(np.sqrt(np.dot(V, (dV / V)**2) / V.sum())),
                volumeDivergenceMax=float(np.max(abs(dV / V))),
                instantaneousTotalEnergyRate=float(np.sum(mass * (np.sum(velocity * a, axis=1) + du))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Completed KH run directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = np.load(args.input / 'finalState.npz')
    settings = json.loads((args.input / 'researchRun.json').read_text())
    if settings['case'] != 'kh':
        raise ValueError('This manufactured comparison requires a saved KH run')
    n = settings['n']
    x0, V0, rho0, v0 = kh_initial(n, settings['jitter'])
    assert np.array_equal(x0, data['initialPositions'])
    v_shear = v0.copy()
    v_shear[:, 1] = 0
    shear_x = (x0 + settings['time'] * v_shear) % 1
    states = dict(initialSeeded=(x0, V0, v0, np.full(len(x0), 2.5)),
                  exactUnseededShear=(shear_x, V0, v_shear, np.full(len(x0), 2.5)),
                  actualKHFinal=(data['positions'], data['volumes'], data['velocities'], data['pressures']))
    result = dict(sourceRun=settings['remoteRunID'], time=settings['time'], particles=len(x0),
                  scope='Static diagnostics only. No trajectories advanced. Unseeded shear is a geometry control, not an exact seeded KH reference.',
                  states={})
    for name, (x, V, velocity, pressure) in states.items():
        print(f'Static audit: {name}', flush=True)
        solver = GPUClosure(len(x), coarse_k=8)
        i, j, q, info = solver.solve(x, V)
        q0 = solver.q0.numpy()
        operators = {}
        for label, coefficient in [('rawCRK', q0), ('projected', q)]:
            operators[label] = operator_audit(x, V, data['masses'], velocity, pressure, i, j, coefficient)
            # A nonconstant periodic pressure exposes cancellations hidden by P=const.
            smooth = 2.5 + .25 * np.sin(2*np.pi*x[:, 0]) * np.cos(2*np.pi*x[:, 1])
            operators[label]['smoothPressureProbe'] = operator_audit(x, V, data['masses'], velocity, smooth, i, j, coefficient)
        result['states'][name] = dict(packing=packing(x, V), operators=operators, solver=info,
            relativeCoefficientChangeL2=float(np.linalg.norm(q-q0)/np.linalg.norm(q0)))
        print(json.dumps(result['states'][name]), flush=True)
    # Track the geometric warning through existing recorded frames, without
    # changing the simulation's historical 'passed' algebra/stability result.
    import h5py
    recorded_packing = []
    with h5py.File(args.input / 'trajectory.h5', 'r') as stream:
        for key in sorted(stream['states']):
            group = stream['states'][key]
            x, V = group['positions'][()], group['volumes'][()]
            distances = cKDTree(x % 1, boxsize=1.).query(x % 1, k=2)[0][:, 1] / np.sqrt(V)
            recorded_packing.append(dict(time=float(group.attrs['time']),
                nearestDistanceOverSqrtV=quantiles(distances),
                fractionNearestBelowQuarterSpacing=float(np.mean(distances < .25))))
    result['recordedPacking'] = recorded_packing
    # Unseeded parallel shear has exact a=0, dV=0, du=0 for constant pressure.
    result['states']['exactUnseededShear']['continuumReference'] = dict(acceleration=0., volumeDivergence=0., internalEnergyRate=0.)
    (args.output / 'qualification.json').write_text(json.dumps(result, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    titles = ['Initial seeded KH', f"Exact unseeded shear at t={settings['time']:g}",
              f"Experimental KH at t={settings['time']:g}"]
    for axis, title, (name, (x, V, _, _)) in zip(axes, titles, states.items()):
        axis.scatter(*x.T, c=data['materials'], s=2, cmap='coolwarm', vmin=0, vmax=1)
        axis.set(xlim=(0, 1), ylim=(0, 1), title=title, xlabel='x', ylabel='y', aspect='equal')
    fig.savefig(args.output / 'packingComparison.png', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    for statistic, label in [('min', 'Minimum'), ('p01', '1st percentile'), ('p10', '10th percentile')]:
        axes[0].plot([row['time'] for row in recorded_packing],
                     [row['nearestDistanceOverSqrtV'][statistic] for row in recorded_packing], label=label)
    axes[0].set(xlabel='Recorded time', ylabel='Nearest distance / sqrt(evolved V)',
                title='KH crowding despite passing internal gates')
    axes[0].legend()
    x = data['positions']
    axes[1].scatter(*x.T, c=data['materials'], s=18, cmap='coolwarm', vmin=0, vmax=1)
    ids = result['states']['actualKHFinal']['packing']['closestPairParticleIDs']
    center = x[ids].mean(axis=0)
    axes[1].scatter(*x[ids].T, s=110, facecolors='none', edgecolors='black')
    axes[1].set(xlim=(center[0]-.07, center[0]+.07), ylim=(center[1]-.07, center[1]+.07),
                title='Final closest pair (circled)', xlabel='x', ylabel='y', aspect='equal')
    fig.savefig(args.output / 'packingWarning.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
