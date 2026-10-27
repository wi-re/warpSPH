"""Independent physical targets for the Spheral multi-material harness."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_spheralMaterials import prepare, Measurements, summarize
from warpSPH.caseUtils.compressible.sod.twoGasReference import sodProfile
from warpSPHCore import scalar_t
from warpSPHCore.kernels.kernelFunctions.quinticSpline import quinticSpline_k, quinticSpline_dkdq, quinticSpline_C_d


@wp.kernel
def _quintic_samples(q: wp.array(dtype=scalar_t), dim: int,
                     value: wp.array(dtype=scalar_t), gradient: wp.array(dtype=scalar_t)):
    i = wp.tid()
    value[i] = quinticSpline_C_d(dim)*quinticSpline_k(q[i], dim)/wp.pow(scalar_t(3.), scalar_t(dim))
    gradient[i] = quinticSpline_C_d(dim)*quinticSpline_dkdq(q[i], dim)/wp.pow(scalar_t(3.), scalar_t(dim+1))


@pytest.mark.parametrize('case', ['box', 'sod'])
@pytest.mark.parametrize('gamma', ['equal', 'unequal'])
def test_published_geometry_mass_and_equilibrium_initial_state(tmp_path, case, gamma):
    root = tmp_path/'case'
    prepare(NS(case=case, gamma=gamma, n=100, time=None, dt=None, support='adaptive', frame_every=100, output=root))
    settings = json.loads((root/'settings.json').read_text()); a = dict(np.load(root/'input.npz'))
    assert settings['volumePolicy'] == ('massOverDensity' if case=='box' else 'kernelSum')
    if case == 'box':
        assert len(a['positions']) == 10000
        assert np.count_nonzero(a['materials']==1) == 2500
        assert np.sum(a['masses'], dtype=np.float64) == pytest.approx(1.75, rel=1.e-7)
        np.testing.assert_array_equal(a['materials']==1, np.all(abs(a['positions'])<.25, axis=1))
        target_pressure = np.ones(len(a['masses']))
    else:
        assert len(a['positions']) == 250
        assert np.ptp(a['masses']) < 1.e-10  # Equal mass, unequal spacing.
        assert a['positions'].min() > -1 and a['positions'].max() < 1
        target_pressure = np.where(a['materials']==0, 1., .1795)
    g = np.array(settings['gamma'])[a['materials']]
    np.testing.assert_allclose((g-1)*a['densities']*a['internalEnergies'], target_pressure, rtol=1.e-7)
    row = Measurements(settings, a)(a, 0.)
    assert abs(row['relativeEnergyDrift']) < 1.e-14
    assert row['materialMassError'] == 0


@pytest.mark.parametrize('gamma', [[5/3, 5/3], [1.4, 5/3]])
def test_spheral_reference_agrees_with_independent_high_precision_riemann(tmp_path, gamma):
    from compare_spheralMaterials import SOURCE
    if not (SOURCE/'tests/functional/Hydro/Sod/SodAnalyticSolution.py').exists():
        pytest.skip('Pinned analytic reference source is not installed')
    settings = dict(case='sod', gamma=gamma, sourceRoot=str(SOURCE))
    initial = dict(masses=np.ones(2), materials=np.array([0, 1]), internalEnergies=np.ones(2), velocities=np.zeros((2, 1)))
    exact = Measurements(settings, initial).exact
    x = torch.linspace(-.5, .5, 1001, dtype=torch.float64)
    expected = sodProfile(x, .15, (1., 0., 1., gamma[0]), (.25, 0., .1795, gamma[1]))
    _, v, _, rho, p, _, _ = exact.solution(.15, x=x.numpy())
    for values, name in [(v, 'velocity'), (rho, 'density'), (p, 'pressure')]:
        np.testing.assert_allclose(values, expected[name].numpy(), atol=8.5e-5, rtol=0)


@pytest.mark.parametrize('dim', [1, 2])
def test_actual_warp_quintic_matches_compiled_spheral_samples(dim):
    source = Path(__file__).resolve().parents[1]/'reports/materialEOS/spheral/materialBaselines/kernelReference.json'
    if not source.exists(): pytest.skip('Compiled Spheral kernel samples not exported')
    reference = json.loads(source.read_text())[str(dim)]
    q = wp.array(np.array(reference['eta'])/3., dtype=scalar_t, device='cpu')
    value = wp.empty_like(q); gradient = wp.empty_like(q)
    wp.launch(_quintic_samples, len(reference['eta']), inputs=[q, dim, value, gradient], device='cpu')
    np.testing.assert_allclose(value.numpy(), reference['kernel'], atol=4.e-6, rtol=1.e-5)
    np.testing.assert_allclose(gradient.numpy(), reference['gradient'], atol=4.e-6, rtol=1.e-5)
    np.testing.assert_allclose(reference['tableKernel'], reference['kernel'], atol=1.e-8, rtol=1.e-6)


@pytest.mark.parametrize('pressure,expected', [(1., True), (1.02, False)])
def test_summary_serializes_and_retains_failed_equilibrium_gate(tmp_path, pressure, expected):
    root = tmp_path/'case'
    prepare(NS(case='box', gamma='unequal', n=32, time=None, dt=None, support='adaptive', frame_every=100, output=root))
    settings = json.loads((root/'settings.json').read_text()); initial = dict(np.load(root/'input.npz'))
    measure = Measurements(settings, initial); final = {k: v.copy() for k, v in initial.items()}
    final['pressures'][:] = pressure
    history = [measure(initial, 0.), measure(final, settings['end'])]
    summarize(root, 'Spheral', settings, history, initial, final, 1., 7000, 'unit-test', {})
    saved = json.loads((root/'comparisonRun.json').read_text())
    assert saved['passed'] is expected
    assert saved['gates']['pressure'] is expected
    assert saved['gates']['materialMass'] is True
