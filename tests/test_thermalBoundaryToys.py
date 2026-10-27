"""Independent analytic checks of the thermal toy diagnostics."""
import math
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_thermalBoundaries import thermostat_error,mode_response,reservoir_toy


def test_thermostat_records_mass_weighted_missing_heat_and_kinetic_energy():
    mass=torch.tensor([1.,3.],dtype=torch.float64)
    velocity=torch.tensor([[2.],[-1.]],dtype=torch.float64)
    energy=torch.tensor([1.,2.],dtype=torch.float64)
    row=thermostat_error(mass,velocity,energy,2.5)
    assert row['missingResetHeat']==3.
    assert row['totalEnergy']==10.5
    assert row['maxURelativeError']==.6
    assert row['mass']==4.


@pytest.mark.parametrize('length,decay',[(1.,.2),(2.,3.),(4.,12.)])
def test_fourier_diagnostic_recovers_a_known_heat_equation_mode(length,decay):
    n=64; x=(torch.arange(n,dtype=torch.float64)+.5)*length/n
    mode=.1*torch.sin(2*math.pi*x/length)
    row=mode_response(x,1.+mode,-decay*mode,torch.full_like(x,length/n),length)
    assert row['fittedDecayRate']==pytest.approx(decay,rel=1e-13)
    assert row['fittedDiffusivity']==pytest.approx(decay*(length/(2*math.pi))**2,rel=1e-13)
    assert row['nonFourierRateFraction']<1e-13
    assert row['relativeEnergyRateResidual']<1e-14
    assert row['entropyRate']>0
    assert row['projectedAmplitude']==pytest.approx(.1,rel=1e-13)


def test_zero_transport_and_spurious_harmonics_are_distinguished():
    x=(torch.arange(64,dtype=torch.float64)+.5)/32
    mode=.1*torch.sin(math.pi*x);mass=torch.ones_like(x)/32
    off=mode_response(x,1.+mode,torch.zeros_like(x),mass)
    assert off['fittedDiffusivity']==0 and off['nonFourierRateFraction']==0
    distorted=mode_response(x,1.+mode,-mode+.05*torch.sin(3*math.pi*x),mass)
    assert distorted['fittedDecayRate']==pytest.approx(1.,rel=1e-13)
    assert distorted['nonFourierRateFraction']==pytest.approx(1/math.sqrt(5),rel=1e-13)


def test_pressure_reservoir_has_the_analytic_volume_dependence(tmp_path):
    row=reservoir_toy(tmp_path)
    for state in row['rows']:
        assert state['pressureRelativeError']<1e-6
        assert state['uMean']==pytest.approx(state['volumeRatio'],rel=1e-6)
        assert state['totalEnergy']==pytest.approx(state['volumeRatio'],rel=1e-6)


@pytest.mark.parametrize('scheme,variant',[('MFM',None),('MFV',None),('PESPH','PressureEnergy')])
def test_uniform_caloric_prescription_is_measured_through_native_stages(scheme,variant,tmp_path):
    from probe_thermalBoundaries import boundary_toy
    row=boundary_toy(scheme,variant,16,.002,.004,'rungeKutta2','uniform',tmp_path)
    assert row['steps']==2 and not row['diverged']
    assert row['t']==pytest.approx(.004,abs=1e-9)
    assert row['acceptedURelativeError']<2e-6
    assert row['maxStageURelativeError']<2e-6
    assert abs(row['caloricResidualIncludingInitialReset'])<3e-6
