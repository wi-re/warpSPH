"""Observation operators must preserve signs, physical volume and affine curl."""
from types import SimpleNamespace

import pytest
import torch

from warpSPH.caseUtils.compressible.richtmyerMeshkov.diagnostics import markerMeasures, curlMeasures


def test_marker_envelope_keeps_the_initial_particle_band_offset():
    y = torch.arange(128, dtype=torch.float64) / 128
    cos, sin = torch.cos(2*torch.pi*y), torch.sin(2*torch.pi*y)
    x = torch.cat([.2*cos-.01, .2*cos+.01]) + .7
    values = markerMeasures(x, cos.repeat(2), sin.repeat(2))
    assert float(values['interfaceMean']) == pytest.approx(.7)
    assert float(values['interfaceCosAmplitude']) == pytest.approx(.2)
    assert float(values['interfaceSinAmplitude']) == pytest.approx(0, abs=1e-15)
    assert float(values['markerLayerWidth']) == pytest.approx(.42)
    asymmetric = markerMeasures(torch.tensor([0., .1, .5]), torch.zeros(3), torch.zeros(3))
    assert float(asymmetric['bubbleExtent']) == pytest.approx(.3)
    assert float(asymmetric['spikeExtent']) == pytest.approx(.2)


def test_curl_uses_physical_volume_and_separates_counter_rotating_halves():
    state = SimpleNamespace(positions=torch.tensor([[0., -.25], [0., .25], [5., .25]]),
        masses=torch.tensor([2., 6., 100.]), densities=torch.tensor([2., 3., 1.]),
        velocities=torch.tensor([[4., 2.], [4., -1.], [4., 10.]]), alphas=torch.tensor([.02, .5, 2.]))
    J = torch.zeros(3, 2, 2)
    J[:, 1, 0] = torch.tensor([-2., 1., 100.])
    values, omega = curlMeasures(state, J, torch.tensor([True, True, False]))
    assert values['circulationLower'] == -2
    assert values['circulationUpper'] == 2
    assert values['circulationPositive'] + values['circulationNegative'] == 0
    assert values['enstrophy'] == 3
    assert values['transverseKineticEnergy'] == 7
    assert float(values['interactionAlphaActiveVolume']) == pytest.approx(2/3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA SPH')
def test_corrected_diagnostic_gradient_recovers_rotation_across_materials():
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.richtmyerMeshkov import richtmyerMeshkovCase
    from warpSPH.modules.reconstruction.gradient import computeVelocityJacobian
    from warpSPHCore import SupportScheme
    ctx = RunConfiguration('richtmyerMeshkov', simulation={'nx': 16, 'tLimit': 1.}).prepare(richtmyerMeshkovCase)
    system = richtmyerMeshkovCase.buildSystem(ctx)
    state = system.state
    state.velocities = torch.stack([-state.positions[:, 1], state.positions[:, 0]], -1)
    before = state.densities.clone()
    J = computeVelocityJacobian(state, ctx.config, SupportScheme.KernelMeanSymmetric, system.adjacency)
    # Exclude periodic discontinuities of this manufactured, non-periodic field.
    mask = (state.positions[:, 0].abs() < 1) & (state.positions[:, 1].abs() < .15)
    values, omega = curlMeasures(state, J, mask)
    torch.testing.assert_close(omega[mask], torch.full_like(omega[mask], 2.), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(state.densities, before, rtol=0, atol=0)
    assert values['circulationUpper'] > 0 and values['circulationLower'] > 0
