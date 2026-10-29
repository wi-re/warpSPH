#!/usr/bin/env python3
"""Offline audit of retained driven-flow forcing, periodic seams and input power.

No hydrodynamic steps. CPU FP32/FP64 evaluation is compared with an independent
NumPy FP64 real-Fourier sum, including spatial derivatives. Outputs are evidence,
not a claim of stationary turbulence or a measurement of spectral energy flux.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent(state, time):
    p = state['parameters']; d = p['dim']
    modes = np.array([n for n in itertools.product(range(-p['kMax'], p['kMax']+1), repeat=d)
                      if p['kMin']**2 <= sum(k*k for k in n) <= p['kMax']**2
                      and next(k for k in n if k) > 0], dtype=float)
    longitudinal = modes[:, :, None]*modes[:, None, :]/np.sum(modes*modes, axis=1)[:, None, None]
    z = p['solenoidalWeight']
    projection = z*(np.eye(d)-longitudinal)+(1-z)*longitudinal
    scale = p['accelerationRMS']/np.sqrt(len(modes)*(z*z*(d-1)+(1-z)**2))
    clock = time/p['updateInterval']; i = int(np.floor(clock)); alpha = clock-i
    raw = (1-alpha)*state['knots'][i].numpy()+alpha*state['knots'][i+1].numpy()
    coeff = scale*np.einsum('mij,pmj->pmi', projection, raw)
    return modes, coeff


def evaluate(x, modes, coeff, length):
    wave = modes*2*np.pi/length
    phase = x@wave.T
    a = np.cos(phase)@coeff[0]+np.sin(phase)@coeff[1]
    # J[n, output component, differentiation coordinate]
    jac = np.einsum('nm,mj,mk->njk', -np.sin(phase), coeff[0], wave)
    jac += np.einsum('nm,mj,mk->njk', np.cos(phase), coeff[1], wave)
    return a, jac


def field_checks(field, state, times):
    length = state['parameters']['length']
    rng = np.random.default_rng(19)
    random_x = rng.uniform(-length/2, length/2, (513, 2))
    shifts = rng.integers(-2, 3, random_x.shape)*length
    axis = np.linspace(-length/2, length/2, 129)
    records = []
    for dtype in (torch.float32, torch.float64):
        metrics = dict(independentValueMaxAbs=0., independentGradientMaxAbs=0.,
                       seamValueMaxAbs=0., seamGradientMaxAbs=0.,
                       translatedValueMaxAbs=0., divergenceMaxAbs=0.,
                       float32TimeValueMaxAbs=0., coefficientMaxAbs=0.)
        for time in times:
            modes, coeff = independent(state, float(time))
            assert np.array_equal(modes, field.modes.numpy())
            metrics['coefficientMaxAbs'] = max(metrics['coefficientMaxAbs'], float(np.max(abs(coeff-field.coefficients(time).numpy()))))
            points = torch.tensor(random_x, dtype=dtype, requires_grad=True)
            actual = field(points, time)
            jac = torch.stack([torch.autograd.grad(actual[:, j].sum(), points, retain_graph=True)[0] for j in range(2)], 1)
            expected, expected_jac = evaluate(points.detach().numpy().astype(float), modes, coeff, length)
            metrics['independentValueMaxAbs'] = max(metrics['independentValueMaxAbs'], float(np.max(abs(actual.detach().numpy()-expected))))
            metrics['independentGradientMaxAbs'] = max(metrics['independentGradientMaxAbs'], float(np.max(abs(jac.detach().numpy()-expected_jac))))
            metrics['divergenceMaxAbs'] = max(metrics['divergenceMaxAbs'], float(jac.diagonal(dim1=1, dim2=2).sum(1).abs().max()))
            translated = field(torch.tensor(random_x+shifts, dtype=dtype), time)
            metrics['translatedValueMaxAbs'] = max(metrics['translatedValueMaxAbs'], float((translated-actual.detach()).abs().max()))
            quantized = field(points, float(np.float32(time)))
            metrics['float32TimeValueMaxAbs'] = max(metrics['float32TimeValueMaxAbs'], float((quantized-actual).abs().max()))
            for dim in range(2):
                seam = np.zeros((258, 2)); seam[:, 1-dim] = np.tile(axis, 2)
                seam[:129, dim] = -length/2; seam[129:, dim] = length/2
                x = torch.tensor(seam, dtype=dtype, requires_grad=True)
                a = field(x, time)
                j = torch.stack([torch.autograd.grad(a[:, k].sum(), x, retain_graph=True)[0] for k in range(2)], 1)
                metrics['seamValueMaxAbs'] = max(metrics['seamValueMaxAbs'], float((a[:129]-a[129:]).abs().max()))
                metrics['seamGradientMaxAbs'] = max(metrics['seamGradientMaxAbs'], float((j[:129]-j[129:]).abs().max()))
        records.append(dict(dtype=str(dtype), **metrics))
    # All time intervals in the retained history: coefficient continuity at knots.
    jumps = []
    interval = state['parameters']['updateInterval']
    for i in range(1, 801):
        t = i*interval
        left = field.coefficients(np.nextafter(t, -np.inf)).numpy()
        right = field.coefficients(np.nextafter(t, np.inf)).numpy()
        jumps.append(float(np.max(abs(left-right))))
    # Periodic grid quadrature and spectral support (six modes, no aliases).
    grids = []
    modes, coeff = independent(state, 23.173)
    for nx in (16, 64, 128, 192, 256):
        axis = (np.arange(nx)+.5)/nx*length-length/2
        x = np.stack(np.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
        a = field(torch.tensor(x, dtype=torch.float32), 23.173).numpy().astype(float).reshape(nx, nx, 2)
        spectrum = np.fft.fftn(a, axes=(0, 1))/(nx*nx)
        support = np.zeros((nx, nx), bool)
        for n in modes.astype(int):
            support[n[0] % nx, n[1] % nx] = True
            support[-n[0] % nx, -n[1] % nx] = True
        power = np.sum(abs(spectrum)**2, axis=-1)
        grids.append(dict(nx=nx, mean=np.mean(a, axis=(0, 1)).tolist(), rms=float(np.sqrt(np.sum(a*a, axis=-1).mean())),
                          parsevalRMS=float(np.sqrt(.5*np.sum(coeff*coeff))),
                          undeclaredModePowerFraction=float(power[~support].sum()/power.sum())))
    return dict(evaluations=records, times=times.tolist(), seamPointsPerEdge=129, randomPositionsPerTime=513, translationBoxRange=[-2, 2],
                maxCoefficientKnotJumpAdjacentFloat64Times=max(jumps), grids=grids,
                scope='CPU kernels; FP32 time quantization tested separately; no CUDA claim')


def history_blocks(run):
    rows = [json.loads(line) for line in (run/'history.jsonl').open() if line.strip()]
    t = np.array([r['t'] for r in rows]); assert t[0] == 0 and t[-1] == 40 and np.all(np.diff(t) > 0)
    data = {k: np.array([r[k] for r in rows]) for k in ('forceWork', 'thermalExchange', 'kineticEnergy', 'totalEnergy')}
    edges = np.array([10., 20., 30., 40.])
    values = {k: np.interp(edges, t, v) for k, v in data.items()}
    blocks = []
    for i in range(3):
        a, b = edges[i:i+2]
        bt = np.r_[a, t[(t > a) & (t < b)], b]
        kinetic = np.interp(bt, t, data['kineticEnergy'])
        blocks.append(dict(start=a, end=b, meanK=float(np.trapezoid(kinetic, bt)/(b-a)), **{
            k: float(np.diff(v)[i]/(b-a)) for k, v in values.items()}))
    return dict(blocks=blocks, meanInputPower10to40=float((values['forceWork'][-1]-values['forceWork'][0])/30),
                meanBathPower10to40=float((values['thermalExchange'][-1]-values['thermalExchange'][0])/30))


def snapshot_power(run, field, state):
    rows = []
    for index in (0, 50, 100, 150, 200, 250, 300, 350, 400):
        path = run/'samples'/f'{index:05d}.npz'
        with np.load(path) as s:
            x = s['positions'].astype(float); m = s['masses'].astype(float); v = s['velocities'].astype(float); t = float(s['t'])
        modes, coeff = independent(state, t)
        a, _ = evaluate(x, modes, coeff, state['parameters']['length'])
        x32 = torch.tensor(x, dtype=torch.float32)
        actual = field(x32, t).double().numpy()
        length = state['parameters']['length']
        wrapped = torch.remainder(x32+length/2, length)-length/2
        wrapped_actual = field(wrapped, t).double().numpy()
        wrapped_reference, _ = evaluate(np.remainder(x+length/2, length)-length/2, modes, coeff, length)
        mean_a = np.sum(m[:, None]*a, axis=0)/m.sum(); mean_v = np.sum(m[:, None]*v, axis=0)/m.sum()
        corrected = a-mean_a
        rows.append(dict(time=t, independentValueMaxAbs=float(abs(actual-a).max()),
                         wrappedIndependentValueMaxAbs=float(abs(wrapped_actual-wrapped_reference).max()),
                         wrappedVsUnwrappedValueMaxAbs=float(abs(wrapped_actual-actual).max()),
                         independentWrapValueMaxAbs=float(abs(wrapped_reference-a).max()),
                         rawEvaluationPowerError=float(np.sum(m[:, None]*(v-mean_v)*(actual-a))),
                         wrappedEvaluationPowerError=float(np.sum(m[:, None]*(v-mean_v)*(wrapped_actual-a))),
                         positionMin=x.min(0).tolist(), positionMax=x.max(0).tolist(),
                         meanAcceleration=mean_a.tolist(), meanVelocity=mean_v.tolist(),
                         rawRMS=float(np.sqrt(np.sum(m[:, None]*a*a)/m.sum())),
                         appliedRMS=float(np.sqrt(np.sum(m[:, None]*corrected*corrected)/m.sum())),
                         instantaneousPower=float(np.sum(m[:, None]*v*corrected)),
                         meanRemovalPowerCorrection=float(-m.sum()*np.dot(mean_v, mean_a))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    from warpSPH.modules.forcing import FourierOU
    runs = sorted((args.baseline/'runs').glob('weak*-s1234'))+sorted((args.campaign/'runs').glob('weak*-s1234'))
    assert len(runs) == 16
    states = [torch.load(run/'forcing.pt', map_location='cpu', weights_only=True) for run in runs]
    state = states[0]; common = min(len(s['knots']) for s in states)
    assert common >= 802
    assert all(s['parameters'] == state['parameters'] and torch.equal(s['knots'][:common], state['knots'][:common]) for s in states)
    field = FourierOU.from_state_dict(state)
    times = np.unique(np.r_[np.arange(0, 40.001, .5), .000001, .049999, .05, .050001, 23.173, 39.999997])
    checks = field_checks(field, state, times)
    forcing_t = np.linspace(0, 40, 8001)
    modes = field.modes.numpy(); k2 = np.sum(modes*modes, axis=1).astype(int)
    force_power = np.stack([.5*np.sum(independent(state, t)[1]**2, axis=(0, 2)) for t in forcing_t])
    band_power = {str(k): force_power[:, k2 == k].sum(1) for k in sorted(set(k2))}
    blocks = []
    for a, b in ((0, 10), (10, 20), (20, 30), (30, 40), (10, 40)):
        select = (forcing_t >= a) & (forcing_t <= b)
        total = force_power.sum(1)[select]; t = forcing_t[select]
        blocks.append(dict(start=a, end=b, rmsSquaredMean=float(np.trapezoid(total, t)/(b-a)),
                           instantaneousRMSRange=[float(np.sqrt(total.min())), float(np.sqrt(total.max()))],
                           boxModePowerFraction=float(np.trapezoid(band_power['1'][select], t)/np.trapezoid(total, t))))
    results = []
    for run, saved in zip(runs, states):
        result = dict(name=run.name, run=str(run.resolve()), **history_blocks(run),
                      snapshots=snapshot_power(run, field, state),
                      evidence={n: digest(run/n) for n in ('forcing.pt', 'history.jsonl', 'resolved.json')})
        results.append(result)
        print(json.dumps({k: result[k] for k in ('name', 'meanInputPower10to40', 'meanBathPower10to40')}), flush=True)
    data = dict(parameters=state['parameters'], commonKnots=common, runCount=len(runs), modes=modes.tolist(),
                checks=checks, forcingBlocks=blocks, runs=results,
                interpolationEnsembleVarianceFactor=(2+np.exp(-state['parameters']['updateInterval']/state['parameters']['correlationTime']))/3,
                driverSHA256=digest(Path(__file__)))
    (out/'audit.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(out/'forcing-history.npz', t=forcing_t, modePower=force_power, modes=modes)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), layout='constrained')
    axes[0].plot(forcing_t, np.sqrt(force_power.sum(1)), lw=1)
    axes[0].axhline(.3, color='.5', ls='--', label='Ensemble knot RMS parameter')
    axes[0].set_ylabel('Spatial acceleration RMS'); axes[0].legend(fontsize=8)
    for k, y in band_power.items(): axes[1].plot(forcing_t, y, label=f'|n|²={k}', lw=1)
    axes[1].legend(ncol=3); axes[1].set_ylabel('Acceleration variance')
    for run in results:
        if 'CRKSPH' in run['name']:
            axes[2].plot([15, 25, 35], [b['forceWork'] for b in run['blocks']], 'o-', label=run['name'].split('-')[0][4:]+'²')
    axes[2].legend(ncol=4); axes[2].set_ylabel('Mean input power / mass')
    for ax in axes:
        ax.set_xlabel('Physical time'); ax.grid(alpha=.2)
    fig.suptitle('Retained forcing: one common field, state-dependent mechanical work')
    fig.savefig(out/'forcing-power.png', dpi=160); fig.savefig(out/'forcing-power.pdf'); plt.close(fig)
    print(json.dumps(dict(checks=checks, forcingBlocks=blocks)), flush=True)


if __name__ == '__main__':
    main()
