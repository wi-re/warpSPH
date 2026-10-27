"""Exact-wave and reflected-geometry checks independent of a simulation run."""
import pytest
import torch
from warpSPH.caseUtils.compressible.sod.twoGasReference import sodProfile, sodWaves
from warpSPH.cases.compressible.shockBubble import foldPositions
from warpSPH.runner.configuration import RunConfiguration
from warpSPH.cases.compressible.twoGasSod import twoGasSodCase
from warpSPH.cases.compressible.hydrostatic import hydrostaticCase
from warpSPH.cases.compressible.shockBubble import shockBubbleCase
from warpSPH.caseUtils.compressible.materialValidation import hostScalars


def test_exact_standard_sod_star_and_shock_conservation():
    left, right = (1., 0., 1., 1.4), (.125, 0., .1, 1.4)
    w = sodWaves(left, right)
    assert w['pressure'] == pytest.approx(.303130178, rel=1e-8)
    assert w['velocity'] == pytest.approx(.92745262, rel=1e-8)
    speed = w['waveRight']
    assert right[0]*speed == pytest.approx(w['densityRight']*(speed-w['velocity']))
    assert right[2]+right[0]*speed**2 == pytest.approx(w['pressure']+w['densityRight']*(speed-w['velocity'])**2)


@pytest.mark.parametrize('g0,g1', [(5/3, 5/3), (1.4, 5/3)])
def test_two_gamma_fan_continuity_and_contact_jump(g0, g1):
    left, right = (1., 0., 1., g0), (.25, 0., .1795, g1)
    w = sodWaves(left, right)
    x = torch.tensor([w['fanHead']-1e-8, w['fanHead']+1e-8,
                      w['fanTail']-1e-8, w['fanTail']+1e-8,
                      w['velocity']-1e-8, w['velocity']+1e-8], dtype=torch.float64)
    q = sodProfile(x, 1., left, right)
    for name in ('density', 'pressure', 'velocity'):
        assert q[name][0] == pytest.approx(float(q[name][1]), abs=1e-7)
        assert q[name][2] == pytest.approx(float(q[name][3]), abs=1e-7)
    assert q['pressure'][4] == q['pressure'][5]
    assert q['velocity'][4] == q['velocity'][5]
    assert q['density'][4] != q['density'][5]


def test_four_reflected_tiles_fold_to_identical_physical_points():
    p = torch.tensor([[.3, .1], [1.9, .1], [.3, .7], [1.9, .7]], dtype=torch.float64)
    torch.testing.assert_close(foldPositions(p), torch.tensor([[.3, .1]], dtype=p.dtype).expand_as(p))


def test_material_diagnostics_follow_the_runner_host_scalar_contract():
    values = hostScalars({'error': torch.tensor(.25), 'step': 4., 'energy': torch.tensor(1.)})
    assert values == {'error': .25, 'step': 4., 'energy': 1.}
    assert all(isinstance(v, float) for v in values.values())


def test_two_gas_reference_includes_unforced_stationary_contact():
    x = torch.tensor([-.4, -.001, .001, .4])
    q = sodProfile(x, .02, (1., 0., 1., 1.4), (.25, 0., 1., 5/3))
    torch.testing.assert_close(q['density'], torch.tensor([1., 1., .25, .25]))
    torch.testing.assert_close(q['pressure'], torch.ones_like(x))
    torch.testing.assert_close(q['velocity'], torch.zeros_like(x))


@pytest.mark.parametrize('case', [hydrostaticCase, twoGasSodCase, shockBubbleCase])
def test_new_case_variants_resolve(case):
    for variant in ('default', *case.variants):
        spec = RunConfiguration(case.name, variant=variant, solver={'scheme': 'CRKSPH'}).resolve(case)
        assert spec.nx > 0
