"""Independent real-Fourier controls at the signed periodic seams.

These field-only checks exercise the FP32 path used by driven trajectories;
no hydrodynamic evolution or CUDA allocation is needed.
"""
import math

import numpy as np
import pytest
import torch

from warpSPH.modules.forcing import FourierOU


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('length', [1., 2.7])
def test_signed_seams_and_gradient_match_known_divergence_free_field(dtype, length):
    field = FourierOU(length=length, accelerationRMS=.3)
    raw = torch.zeros_like(field.knots[0])
    modes = [tuple(n) for n in field.modes.tolist()]
    # Three different wavelengths/directions, including a mixed x/y phase.
    raw[0, modes.index((0., 1.)), 0] = .2/field.scale
    raw[1, modes.index((2., 0.)), 1] = .3/field.scale
    raw[0, modes.index((1., -1.))] = .1/field.scale
    field.knots = [raw.clone() for _ in range(4)]
    edge = torch.linspace(-length/2, length/2, 65, dtype=dtype)
    x = torch.cat([torch.stack((torch.full_like(edge, sign*length/2), edge), -1)
                   for sign in (-1, 1)]+[
                  torch.stack((edge, torch.full_like(edge, sign*length/2)), -1)
                  for sign in (-1, 1)])
    x.requires_grad_(True)
    actual = field(x, .137)
    gradient = torch.stack([torch.autograd.grad(actual[:, j].sum(), x, retain_graph=True)[0]
                            for j in range(2)], 1)
    # Closed-form reference at the exact representable coordinates, in FP64.
    xx, yy = x.detach().double().T
    k = 2*math.pi/length
    mixed = k*(xx-yy)
    expected = torch.stack((.2*torch.cos(k*yy)+.1*torch.cos(mixed),
                            .3*torch.sin(2*k*xx)+.1*torch.cos(mixed)), -1)
    reference_gradient = torch.stack((
        -.1*k*torch.sin(mixed), -.2*k*torch.sin(k*yy)+.1*k*torch.sin(mixed),
        .6*k*torch.cos(2*k*xx)-.1*k*torch.sin(mixed), .1*k*torch.sin(mixed)), -1).reshape(-1, 2, 2)
    atol = 2e-6 if dtype == torch.float32 else 2e-14
    torch.testing.assert_close(actual.double(), expected, rtol=0, atol=atol)
    torch.testing.assert_close(gradient.double(), reference_gradient, rtol=0, atol=atol*10/length)
    for start in (0, 130):
        torch.testing.assert_close(actual[start:start+65], actual[start+65:start+130], rtol=0, atol=atol)
        torch.testing.assert_close(gradient[start:start+65], gradient[start+65:start+130], rtol=0, atol=atol*10/length)
    assert float(gradient.diagonal(dim1=1, dim2=2).sum(1).abs().max()) < atol*10/length


def test_fixed_clock_coefficients_are_continuous_at_late_knots():
    field = FourierOU(accelerationRMS=.3, correlationTime=.5, updateInterval=.05)
    # Query ahead before checking old boundaries, as in timestep bound/replay use.
    field.accelerationBound(40.)
    for i in (1, 2, 7, 123, 799, 800):
        t = i*.05
        before = field.coefficients(np.nextafter(t, -np.inf))
        at = field.coefficients(t)
        after = field.coefficients(np.nextafter(t, np.inf))
        torch.testing.assert_close(before, at, rtol=0, atol=1e-12)
        torch.testing.assert_close(after, at, rtol=0, atol=1e-12)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_many_box_crossings_preserve_the_same_physical_field(dtype):
    field = FourierOU(accelerationRMS=.3, correlationTime=.5, updateInterval=.05)
    generator = torch.Generator().manual_seed(89)
    x = 48*torch.rand((2048, 2), generator=generator, dtype=dtype)-24
    # Compare the exact represented coordinates modulo L, in high precision.
    # This isolates field evaluation from roundoff already lost in a trajectory.
    wrapped = torch.remainder(x.double()+.5, 1.)-.5
    for time in (.173, 17.321, 39.999):
        actual = field(x, time).double()
        expected = field(wrapped, time)
        torch.testing.assert_close(actual, expected, rtol=0,
                                   atol=2e-5 if dtype == torch.float32 else 4e-14)
