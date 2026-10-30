#!/usr/bin/env python3
"""Fixed-field ghost-layout audit; no time integration or BDF restart.

Use a retained solver bundle only for material/configuration metadata. Load
physical fields from a trajectory snapshot, rebuild each geometry's adjacency,
and compare boundary closures, shifting and source-closed spatial operators.
These few operator evaluations deliberately omit video.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runtime', type=Path, required=True)
    ap.add_argument('--entry', type=Path, required=True)
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--dt', type=float, required=True, help='Real dt for the shift-only evaluation')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.runtime.resolve()/'src'),
                   str(args.runtime.resolve()/'scripts/acsph_reference')]
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import numpy as np
    import torch
    from diagnostic_state import map_tensors
    from render import read_snapshot
    from fixture_geometry import lobovsky_tank_sdf
    from audit_local import reference_wall_velocities, install
    from warpSPH.rigidBody.ghostParticles import _hybridGhostOffsets
    from warpSPHCore import buildVerletList, SupportScheme
    from warpSPH.configurations.compute import neighborSearchOptions
    from warpSPH.modules.mdbc.velocity import freeSlip
    from warpSPH.modules.shifting.wrapper import solveShifting
    from warpSPH.modules.artificialCompressible import dualTime
    from warpSPH.modules.artificialCompressible.surfaceFlags import detectSurface
    from warpSPH.modules.artificialCompressible.pressureSmoothing import computePressureSmoothing
    from warpSPH.modules.artificialCompressible.parameters import acParameters
    torch.set_num_threads(4)
    start = time.monotonic()
    payload = map_tensors(torch.load(args.entry, weights_only=False, map_location='cpu'), device='cuda:0')
    problem = payload['problem']
    config, scheme = problem.config, problem.schemeConfig
    install(args.out/'operators', observe=False, wall_diffusion='source',
            acoustic_damper='source', rk_treatment='source', wall_velocity='source')
    snap = read_snapshot(args.snapshot)
    base = copy.copy(problem.state)
    wall, ghost, fluid = [base.kinds == k for k in (1, 2, 0)]
    physical = ~ghost
    assert len(snap['x']) == int(physical.sum())
    np.testing.assert_array_equal(snap['bc'], np.where(base.kinds[physical].cpu().numpy()==0, 1, 3))
    for name, key in [('positions','x'), ('velocities','v'), ('pressures','p')]:
        field = getattr(base, name).clone()
        field[physical] = torch.as_tensor(snap[key], device='cuda:0', dtype=field.dtype)
        setattr(base, name, field)
    base.velocities[~fluid] = 0.
    report = dict(snapshot=str(args.snapshot.resolve()), t=snap['t'],
                  metadata=str(args.entry.resolve()), shiftDt=args.dt, controls={}, comparisons={},
                  scope='Fixed physical fields; no time advancement, no reconstructed BDF history')
    outputs = {}

    def stats(value, mask):
        value = value[mask]
        return dict(rms=float(value.square().mean().sqrt()), maximum=float(value.abs().max()))

    for geometry in ('legacy', 'volume'):
        state = copy.copy(base)
        state.positions = base.positions.clone()
        state.ghostOffsets = base.ghostOffsets.clone()
        if geometry == 'volume':
            offsets = _hybridGhostOffsets(state.positions[wall], [lobovsky_tank_sdf],
                                         float(config.dx), float(state.supports.mean()))
            state.ghostOffsets[wall] = offsets
            state.ghostOffsets[ghost] = -state.ghostOffsets[state.ghostIndices[ghost].long()]
        state.positions[ghost] = state.positions[state.ghostIndices[ghost].long()]+state.ghostOffsets[ghost]
        adjacency = buildVerletList(state, config.domain, verletScale=config.verletScale,
            supportMode=SupportScheme.SuperSymmetric, verbose=False, **neighborSearchOptions(config))
        source_v = reference_wall_velocities(state, config, scheme, adjacency)
        ghost_v = freeSlip(state, config, scheme, adjacency)
        wall_pos = state.positions[state.ghostIndices[ghost].long()]
        flat_bottom = ((wall_pos[:,1] < 0)&(wall_pos[:,0]>.05)&(wall_pos[:,0]<1.56))
        depth = state.positions[ghost][:,1][flat_bottom]
        report['controls'][geometry] = dict(bottomNodeDepthMin=float(depth.min()),
            bottomNodeDepthMax=float(depth.max()),
            wallVelocitySourceDifference=stats(ghost_v-source_v, wall))
        state.velocities = source_v
        flags = detectSurface(state, config, scheme, adjacency)
        state.pressures = dualTime.wallPressures(state, config, adjacency, problem.wallBodyForce,
            edgeRelax=scheme.acParams.wallPressureEdgeRelax, edgeFlags=flags.rawEdge, lambdas=flags.lambdas)
        diffusion = computePressureSmoothing(state, config, scheme, adjacency,
            flags.renormalization, pressures=state.pressures)
        _, _, k1, k2, nu = acParameters(state, config, scheme, args.dt)
        rp, rv = dualTime.spatialResidual(state, config, scheme, adjacency, k1,
            k2, diffusion, problem.bodyForce, nu, taper=flags.taper,
            preparedViscosity=problem.preparedViscosity)
        report['controls'][geometry]['sourcePressureResidualPaPerSecond'] = stats(rp, fluid)
        report['controls'][geometry]['sourceMomentumResidualMetersPerSecond2'] = stats(rv, fluid)
        outputs[geometry] = dict(source_v=source_v, ghost_v=ghost_v, lambdas=flags.lambdas,
                                  rp=rp, rv=rv, shifts={})
        for closure, velocity in [('source', source_v), ('ghost', ghost_v)]:
            state.velocities = velocity
            shift = solveShifting(systemState=state, config=config, schemeConfig=scheme,
                adjacency=adjacency, dt=args.dt)
            outputs[geometry]['shifts'][closure] = shift
            report['controls'][geometry][closure+'ShiftMeters'] = stats(shift, fluid)
        torch.cuda.synchronize()
        print('GEOMETRY COMPLETE', geometry, snap['t'], flush=True)
    for name in ('source_v', 'ghost_v', 'lambdas', 'rp', 'rv'):
        mask = wall if name.endswith('_v') else fluid
        report['comparisons'][name] = stats(outputs['volume'][name]-outputs['legacy'][name], mask)
    for closure in ('source', 'ghost'):
        difference = outputs['volume']['shifts'][closure]-outputs['legacy']['shifts'][closure]
        record = stats(difference, fluid)
        magnitude = difference[fluid].norm(dim=-1)/float(config.dx)
        record.update(maximumNormInDx=float(magnitude.max()),
            fluidParticlesOver001Dx=int((magnitude > .001).sum()),
            fluidCount=int(fluid.sum()))
        report['comparisons'][closure+'ShiftMeters'] = record
    report['wallSeconds'] = time.monotonic()-start
    report['physicalSteps'] = 0
    (args.out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
