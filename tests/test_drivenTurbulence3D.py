"""3D source/recording controls before finite GPU feasibility trajectories."""
import math
from types import SimpleNamespace
import pytest
import torch
from warpSPH.modules.forcing import FourierOU
from warpSPH.modules.meshless.sources import ConservedSources
from warpSPH.runner import RunConfiguration
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase
from warpSPH.cases.common.plotting import Field, _plotOptions


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_manufactured_3d_field_projection_periodicity_and_divergence(dtype):
    field = FourierOU(dim=3, accelerationRMS=.3)
    raw = torch.zeros_like(field.knots[0])
    modes = [tuple(m) for m in field.modes.tolist()]
    raw[1, modes.index((1., 0., 0.)), 1] = .2/field.scale
    raw[0, modes.index((0., 1., 0.)), 2] = .3/field.scale
    raw[0, modes.index((1., 1., 0.))] = torch.tensor([.1, -.1, .2])/field.scale
    field.knots = [raw.clone() for _ in range(6)]
    x = torch.rand((101, 3), generator=torch.Generator().manual_seed(14), dtype=dtype).requires_grad_()
    k = 2*math.pi
    mixed = k*(x[:, 0].double()+x[:, 1].double())
    expected = torch.stack((.1*torch.cos(mixed), .2*torch.sin(k*x[:, 0].double())-.1*torch.cos(mixed),
                            .3*torch.cos(k*x[:, 1].double())+.2*torch.cos(mixed)), -1)
    actual = field(x, .13)
    atol = 2e-6 if dtype == torch.float32 else 2e-8  # prescribed raw tensor was FP32
    torch.testing.assert_close(actual.double(), expected, rtol=0, atol=atol)
    for axis in range(3):
        shift = torch.zeros(3, dtype=dtype);shift[axis] = 1
        torch.testing.assert_close(field(x+shift, .13), actual, rtol=0, atol=atol)
    gradient = torch.stack([torch.autograd.grad(actual[:, j].sum(), x, retain_graph=True)[0] for j in range(3)], 1)
    torch.testing.assert_close(gradient.diagonal(dim1=1, dim2=2).sum(1), torch.zeros(len(x), dtype=dtype), rtol=0, atol=atol*10)


def test_3d_parseval_normalization_and_checkpoint_query_order():
    field = FourierOU(dim=3, accelerationRMS=.3, correlationTime=.5, updateInterval=.05)
    axis = torch.arange(12, dtype=torch.float64)/12
    x = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    value = field(x, .173)
    coeff = field.coefficients(.173)
    torch.testing.assert_close(value.square().sum(-1).mean(), .5*coeff.square().sum(), rtol=1e-12, atol=1e-14)
    torch.testing.assert_close((coeff*field.modes).sum(-1), torch.zeros_like(coeff[..., 0]), rtol=0, atol=1e-15)
    restored = FourierOU.from_state_dict(field.state_dict())
    for t in (1.1, .173, 0., 2.):
        torch.testing.assert_close(field(x[:17], t), restored(x[:17], t), rtol=0, atol=0)
    assert field.parameters['dim'] == 3 and len(field.modes) == 16


def test_3d_mass_mean_and_signed_source_energy():
    mass = torch.tensor([1., 2., 4.], dtype=torch.float64)
    x = torch.tensor([[.1, .2, .3], [.2, .3, .4], [.4, .5, .6]], dtype=torch.float64)
    velocity = torch.tensor([[.2, -.1, .3], [-.4, .1, -.2], [.3, .2, -.1]], dtype=torch.float64)
    u = torch.tensor([.5, .9, 1.4], dtype=torch.float64)
    source = ConservedSources(FourierOU(dim=3), True, .9, .2)
    momentum, work, heat = source.fields(mass, x, velocity, u, .17)
    torch.testing.assert_close(momentum.sum(0), torch.zeros(3, dtype=mass.dtype), rtol=0, atol=1e-15)
    torch.testing.assert_close(work, (momentum*velocity).sum(-1))
    torch.testing.assert_close(heat, mass*(.9-u)/.2)
    assert heat[0] > 0 and heat[-1] < 0
    q = torch.cat((mass[:, None], mass[:, None]*velocity, (mass*(u+.5*velocity.square().sum(-1)))[:, None]), 1)
    rate, ledger = source.rates(SimpleNamespace(Q=q, pos=x, vel=velocity, u=u), .17)
    assert rate.shape == (3, 5)
    torch.testing.assert_close(rate[:, -1].sum(), ledger.sum())


@pytest.mark.parametrize('scheme', ['CRKSPH', 'MFM', 'MFV'])
def test_3d_case_configuration_and_slice_options(scheme):
    configuration = RunConfiguration('drivenTurbulence', simulation=dict(dim=3, nx=32, device='cpu'),
                                     solver=dict(scheme=scheme, temporalScheme='ssprk') if scheme != 'CRKSPH' else dict(scheme=scheme))
    assert configuration.resolve().dim == 3
    field = Field('densities', 'z=0', gridResolution=96, sliceAxis=2, slicePosition=0.)
    grid = _plotOptions(field, 2).gridVisualization
    assert (grid.sliceAxis, grid.slicePosition, grid.resolution) == (2, 0., 96)
