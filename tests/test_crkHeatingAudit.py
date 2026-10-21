"""Pair-power contracts, including the declared unequal-support Comp predictor."""
from pathlib import Path
import pytest
import torch
from warpSPHCore import get_torch_precision


@pytest.fixture(scope='module')
def audit(tmp_path_factory):
    if get_torch_precision()!=torch.float64:pytest.skip('Scientific heating audit requires float64')
    import sys
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    import audit_crkHeatingOperators
    return audit_crkHeatingOperators.audit(tmp_path_factory.mktemp('heating'))


def test_both_crk_local_partitions_have_the_same_instantaneous_pair_power(audit):
    for label in ('CRK-pressure','CRK-viscous'):
        assert audit['summary'][label]['maximumScaledPowerResidual']<1.e-12
        assert max(abs(r['thermalPartitionSumDifference']) for r in audit['operators'][label]['cases'])<1.e-12
    assert audit['summary']['CRK-viscous']['maximumLocalRateDifference']>.1
    assert audit['summary']['CRK-pressure']['maximumLocalRateDifference']==0


def test_comp_gather_predictor_balances_equal_supports_but_is_not_a_general_energy_rate(audit):
    rows=audit['operators']['Comp-pressure']['cases']
    equal=[r for r in rows if r['case'].startswith('Rj/Ri=1 ')]
    unequal=[r for r in rows if not r['case'].startswith('Rj/Ri=1 ')]
    assert max(abs(r['rawPowerScaledResidual']) for r in equal)<1.e-12
    assert max(abs(r['rawPowerScaledResidual']) for r in unequal)>.1


def test_final_compatible_work_closes_energy_for_each_measured_operator(audit):
    for row in audit['summary'].values():
        assert abs(row['compatibleEnergyResidual'])<1.e-12
        assert row['maximumPairMomentumResidual']<1.e-11
        assert row['maximumPairTorqueResidual']<1.e-10
