"""Check analytic oracles independently of any SPH/meshless implementation."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from _thermalReference import IdealGasReference, periodic_mode, insulated_mode, plate_mode


@pytest.fixture
def material():
    return IdealGasReference(density=2., cv=3., gamma=1.4, conductivity=1.5)


def test_caloric_units_and_ideal_gas_eos_agree(material):
    assert material.diffusivity == .25
    assert material.cp == pytest.approx(4.2)
    assert material.isobaric_diffusivity == pytest.approx(.25/1.4)
    assert material.gas_constant == pytest.approx(1.2)
    temperature = 7.
    pressure_from_u = (material.gamma-1)*material.density*material.cv*temperature
    assert pressure_from_u == pytest.approx(material.density*material.gas_constant*temperature)


@pytest.mark.parametrize('mode', [periodic_mode, insulated_mode])
def test_closed_modes_have_exact_decay_and_no_net_exchange(mode, material):
    length, mean, amplitude = 2., 4., .3
    x = np.linspace(0, length, 4097)
    args = dict(length=length, mean=mean, amplitude=amplitude, material=material)
    initial = mode(x, 0., **args)
    # Half-life from the supplied diffusivity, never fitted to measured rates.
    k = (2 if mode is periodic_mode else 1)*math.pi/length
    half_life = math.log(2)/(material.diffusivity*k*k)
    final = mode(x, half_life, **args)
    np.testing.assert_allclose(final.temperature-mean, .5*(initial.temperature-mean), atol=1e-15)
    np.testing.assert_allclose(final.specific_internal_energy, material.cv*final.temperature)
    assert final.energy_per_area == initial.energy_per_area == material.density*material.cv*mean*length
    assert final.cumulative_boundary_heat == 0
    assert final.heat_flux[0]-final.heat_flux[-1] == pytest.approx(0., abs=1e-15)
    if mode is insulated_mode:
        np.testing.assert_allclose(final.heat_flux[[0,-1]], 0., atol=1e-15)
    else:
        assert final.heat_flux[0] != 0.  # Periodic flux can pass through each face.


@pytest.mark.parametrize('amplitude', [-.3, 0., .3])
def test_plate_flux_balance_and_integrated_energy(amplitude, material):
    length, t = 1.7, .4
    x = np.linspace(0, length, 65537)
    args = dict(length=length, left=3., right=5., amplitude=amplitude, material=material)
    initial = plate_mode(x, 0., **args)
    final = plate_mode(x, t, **args)
    np.testing.assert_allclose(final.temperature[[0,-1]], [3., 5.], atol=1e-15)
    # Independent spatial quadrature of rho*u, not the analytic integral.
    energy_density = material.density*final.specific_internal_energy
    integral = np.sum((energy_density[1:]+energy_density[:-1])*.5*np.diff(x))
    assert integral == pytest.approx(final.energy_per_area, abs=2e-9)
    assert final.energy_per_area-initial.energy_per_area == pytest.approx(final.cumulative_boundary_heat, abs=1e-14)
    # Independent time quadrature of signed face fluxes.
    times = np.linspace(0, t, 4097)
    net_flux = np.array([np.subtract(*plate_mode([0.,length], time, **args).heat_flux) for time in times])
    heat = np.sum((net_flux[1:]+net_flux[:-1])*.5*np.diff(times))
    assert heat == pytest.approx(final.cumulative_boundary_heat, abs=2e-9)
    if amplitude == 0:
        np.testing.assert_allclose(final.heat_flux, -material.conductivity*2/length)
        assert final.cumulative_boundary_heat == 0.
    else:
        assert math.copysign(1., final.cumulative_boundary_heat) == -math.copysign(1., amplitude)


@pytest.mark.parametrize('mode', [periodic_mode, insulated_mode, plate_mode])
def test_oracles_satisfy_heat_equation_by_finite_differences(mode, material):
    x = np.linspace(.2, 1.8, 31)
    args = dict(length=2., amplitude=.3, material=material)
    args.update(dict(left=3., right=5.) if mode is plate_mode else dict(mean=4.))
    dx, dt, t = 1e-3, 1e-5, .3
    center = mode(x, t, **args)
    derivative_t = (mode(x,t+dt,**args).temperature-mode(x,t-dt,**args).temperature)/(2*dt)
    derivative_xx = (mode(x+dx,t,**args).temperature-2*center.temperature+mode(x-dx,t,**args).temperature)/dx**2
    np.testing.assert_allclose(derivative_t, material.diffusivity*derivative_xx, rtol=2e-6, atol=1e-9)
    gradient = (mode(x+dx,t,**args).temperature-mode(x-dx,t,**args).temperature)/(2*dx)
    np.testing.assert_allclose(center.heat_flux, -material.conductivity*gradient, rtol=2e-6, atol=1e-9)


@pytest.mark.parametrize('mode', [periodic_mode, insulated_mode, plate_mode])
def test_zero_conductivity_preserves_profile_and_has_no_flux(mode):
    args = dict(length=2., amplitude=.2, material=IdealGasReference(2.,3.,1.4,0.))
    args.update(dict(left=3.,right=5.) if mode is plate_mode else dict(mean=4.))
    initial, final = [mode([0.,.2,1.,2.], t, **args) for t in (0.,100.)]
    np.testing.assert_array_equal(final.temperature, initial.temperature)
    np.testing.assert_array_equal(final.heat_flux, np.zeros(4))
    assert final.cumulative_boundary_heat == 0.


@pytest.mark.parametrize('values', [(0.,3.,1.4,1.), (2.,0.,1.4,1.), (2.,3.,1.,1.), (2.,3.,1.4,-1.), (2.,3.,1.4,float('nan'))])
def test_invalid_material_is_rejected(values):
    with pytest.raises(ValueError):
        IdealGasReference(*values)


@pytest.mark.parametrize('override', [dict(t=-1.), dict(length=0.), dict(mean=.2), dict(x=[-1.]), dict(x=[float('nan')])])
def test_invalid_mode_is_rejected(material, override):
    args = dict(x=[0.,1.], t=.1, length=2., mean=4., amplitude=.3, material=material)
    args.update(override)
    with pytest.raises(ValueError):
        periodic_mode(**args)
