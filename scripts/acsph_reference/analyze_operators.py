#!/usr/bin/env python3
"""Analyze observational Lobovsky RK dumps in stable physical UID order."""
import argparse
import json
from pathlib import Path
import numpy as np
from render import read_snapshot


def reference(root, iteration, stage):
    prefix = f'i{iteration:03d}-s{stage}-'
    index = np.fromfile(root/(prefix+'index.bin'), dtype='<u4')
    order = np.argsort(index)
    assert np.array_equal(index[order], np.arange(len(index)))
    data = {}
    for key in ('x', 'v', 'p', 'lambda', 'gradP', 'norm', 'wallV', 'rV', 'rP', 'terms'):
        a = np.fromfile(root/(prefix+key+'.bin'), dtype='<f8').reshape(len(index), -1).squeeze()
        # RV/RP and the diagnostic terms are already in original UID order.
        data[key] = a if key in ('rV', 'rP', 'terms') else a[order]
    return data


def stats(a):
    return {'max_abs': float(np.max(np.abs(a))), 'rms_component': float(np.sqrt(np.mean(a*a)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    r = reference(root/'reference-operators', 0, 0)
    b = np.load(root/'local-audited-3/operators/i000-s0.npz')
    c = np.load(root/'wall-diffusion-3/operators/i000-s0.npz')
    f = b['bc'] == 0
    x = b['x']
    regions = {'fluid': f, 'bottom': f & (x[:, 1] < .003),
               'left': f & (x[:, 0] < .003),
               'away_from_walls': f & (x[:, 0] > .003) & (x[:, 1] > .003)}
    report = {'initial': {}, 'reference_terms_last_iteration': {}}
    for name, mask in regions.items():
        report['initial'][name] = {
            'count': int(mask.sum()),
            'x_identical': bool(np.array_equal(r['x'][mask], b['x'][mask])),
            'v_identical': bool(np.array_equal(r['v'][mask], b['v'][mask])),
            'p_identical': bool(np.array_equal(r['p'][mask], b['p'][mask])),
            'lambda_error': stats(r['lambda'][mask]-b['lambdas'][mask]),
            'momentum_residual_error': stats(r['rV'][mask]-b['rV'][mask]),
            'reference_rP': stats(r['rP'][mask]),
            'local_rP': stats(b['rP'][mask]),
            'wall_variant_rP': stats(c['rP'][mask]),
            # Reference uses half dtau / double k2 in its first two iterations.
            'reference_rP_at_local_k2': stats(r['rP'][mask]/2),
        }
    grad_error = c['gradP'][f] - r['gradP'][f]
    report['initial']['fluid_gradient_error'] = stats(grad_error)
    for m in (0, 1, 199):
        for s in range(3):
            a = reference(root/'reference-operators', m, s)
            t = a['terms'][f]
            report['reference_terms_last_iteration'][f'i{m:03d}-s{s}'] = {
                name: stats(t[:, cols]) for name, cols in (
                    ('pressure_acceleration', slice(0, 2)), ('viscosity', slice(2, 4)),
                    ('momentum_JST', slice(4, 6)), ('acoustic_damper', slice(6, 8)),
                    ('velocity_divergence', 8), ('pressure_diffusion_before_taper', 9))}
    # Observation side effects: report actual floating-point differences.
    a = read_snapshot(root/'reference-3/snapshots/000001.bin')
    d = read_snapshot(root/'reference-audited-3/snapshots/000001.bin')
    report['reference_observation_error'] = {k: stats(a[k]-d[k]) for k in ('x', 'v', 'p')}
    a = read_snapshot(root/'rho-fixed-mapped-source-history-3/snapshots/000003.bin')
    d = read_snapshot(root/'local-audited-3/snapshots/000003.bin')
    report['local_observation_error'] = {k: stats(a[k]-d[k]) for k in ('x', 'v', 'p')}
    shift_root = root/'reference-shift-operators'
    if (shift_root/'i000-s9-index.bin').exists():
        order = np.argsort(np.fromfile(shift_root/'i000-s9-index.bin', dtype='<u4'))
        n = len(order)
        shift = {}
        for key in ('x', 'v', 'p'):
            pre = np.fromfile(shift_root/f'i000-s9-{key}_pre.bin', dtype='<f8').reshape(n, -1)[order]
            post = np.fromfile(shift_root/f'i000-s9-{key}_post.bin', dtype='<f8').reshape(n, -1)
            shift[key+'_change'] = stats((post-pre)[f])
            shift[key+'_change']['rms_magnitude'] = float(np.sqrt(np.mean(np.sum((post[f]-pre[f])**2, -1))))
            shift[key+'_change']['max_magnitude'] = float(np.linalg.norm(post[f]-pre[f], axis=-1).max())
        local = np.load(root/'wall-damper-rk-3/operators/shift-000.npz')
        shift['local_shift'] = stats(local['shift'][f])
        shift['local_shift']['rms_magnitude'] = float(np.sqrt(np.mean(np.sum(local['shift'][f]**2, -1))))
        shift['local_shift']['max_magnitude'] = float(np.linalg.norm(local['shift'][f], axis=-1).max())
        # The added shift observations must not perturb the already recorded
        # first-step spatial/RK dumps (same numerical executable source).
        files = list((root/'reference-operators').glob('*.bin'))
        shift['operator_dumps_bitwise_equal'] = all(p.read_bytes() == (shift_root/p.name).read_bytes() for p in files)
        report['first_step_shift'] = shift
    out = root/'operators-analysis'
    out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    sample = np.flatnonzero(f)[::3]
    for ax, data, title in ((axes[0, 0], b['rP'], 'Current AC-2L: initial pressure residual'),
                            (axes[0, 1], c['rP'], 'Fluid-side wall-gradient probe: same IC')):
        im = ax.scatter(x[sample, 0], x[sample, 1], c=data[sample], s=2,
                        cmap='coolwarm', vmin=-8e6, vmax=8e6, rasterized=True)
        ax.set_title(title); ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        fig.colorbar(im, ax=ax, label='Dp/Dtau [Pa/s]')
    mask = f & np.isclose(x[:, 0], .3)
    for label, data in [('Reference / 2 (common k2)', r['rP']/2),
                         ('Current', b['rP']), ('Wall-gradient probe', c['rP'])]:
        axes[1, 0].plot(x[mask, 1]/.00075, data[mask], label=label)
    axes[1, 0].set_xlim(0, 8); axes[1, 0].set_xlabel('Initial y / dx, x = 0.3 m')
    axes[1, 0].set_ylabel('Initial Dp/Dtau [Pa/s]'); axes[1, 0].legend()
    names = ['pressure_acceleration', 'viscosity', 'momentum_JST', 'acoustic_damper']
    t = report['reference_terms_last_iteration']['i199-s2']
    axes[1, 1].bar(np.arange(4), [t[k]['rms_component'] for k in names])
    axes[1, 1].set_xticks(np.arange(4), ['Pressure', 'Viscosity', 'JST', 'Div. damper'])
    axes[1, 1].set_ylabel('Reference acceleration component RMS [m/s²]')
    axes[1, 1].set_yscale('log'); axes[1, 1].set_title('First real step: iteration 199, RK stage 2')
    fig.suptitle('Exact Lobovsky IC: spatial-operator audit (CUDA FP64)')
    fig.savefig(out/'operators.png', dpi=160)
    print(json.dumps(report['initial'], indent=2))


if __name__ == '__main__':
    main()
