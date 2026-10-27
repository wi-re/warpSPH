"""Independent weak-shock checks; this approximation must not be called exact."""
import pytest

from warpSPH.caseUtils.compressible.richtmyerMeshkov.reference import weakShockReference


def test_weak_shock_reference_is_linear_in_amplitude_and_scales_with_wavelength():
    ref = weakShockReference(1, 3, 1, 1.4, 5/3, 1.2, .01)
    assert ref['growthRate'] == pytest.approx(.0070291877255716055)
    assert ref['circulationUpper'] == pytest.approx(.005388461180332558, rel=1e-6)
    assert ref['circulationLower'] == -ref['circulationUpper']
    twice = weakShockReference(1, 3, 1, 1.4, 5/3, 1.2, .02)
    long = weakShockReference(1, 3, 1, 1.4, 5/3, 1.2, .01, wavelength=2.)
    assert twice['growthRate'] == pytest.approx(2*ref['growthRate'])
    assert long['growthRate'] == pytest.approx(ref['growthRate'])
    assert long['circulationUpper'] == pytest.approx(2*ref['circulationUpper'])


@pytest.mark.parametrize('rho1,mach', [(3, 1), (1, 1.2)])
def test_contact_or_identical_gases_do_not_deposit_circulation(rho1, mach):
    ref = weakShockReference(1, rho1, 1, 1.4, 1.4, mach, .01)
    assert ref['growthRate'] == pytest.approx(0, abs=1e-14)
    assert ref['circulationUpper'] == pytest.approx(0, abs=1e-14)


def test_reference_rejects_reflected_rarefaction():
    with pytest.raises(ValueError, match='reflected shock'):
        weakShockReference(1, .25, 1, 1.4, 1.4, 1.2, .01)
