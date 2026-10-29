"""Quantitative controls for the long-run statistical analysis, without hydro."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('long_analysis',REPO/'scripts/analyze_drivenStationarity.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
utility=analysis.load(REPO/'src/warpSPH/utils/turbulence.py','stationarity_reference')


def test_irregular_time_quadrature_retains_a_known_drift():
    t=np.array([0.,.03,.2,.9,1.2,1.8,2.]);v=2+3*t
    assert float(analysis.mean(t,v,.1,1.9))==pytest.approx(5.,abs=2e-15)
    assert analysis.power_rate(t,v,.1,1.9)==pytest.approx(3.,abs=2e-15)
    with pytest.raises(ValueError):analysis.mean(t,v,-.1,1.)


def test_block_bootstrap_deterministic_mean_and_guards():
    report=analysis.block_bootstrap(np.full(200,3.5),25)
    assert report['meanCI95']==[3.5,3.5]
    assert report['approximateBlocks']==8
    with pytest.raises(ValueError):analysis.block_bootstrap(np.arange(20),21)
    with pytest.raises(ValueError):analysis.block_bootstrap(np.ones((20,2)),2)


def test_sustained_linear_drift_is_not_detrended_or_given_short_block_confidence():
    t=np.linspace(0,160,1601);v=1+.01*t
    estimates=[analysis.estimate(t,v,lo,hi,utility) for lo,hi in ((40,80),(80,120),(120,160))]
    np.testing.assert_allclose([r['mean'] for r in estimates],[1.6,2.,2.4],atol=1e-14)
    screen=utility.stationarityScreen(t,v,blocks=[40,80,120,160])
    assert screen['relativeSpread']==pytest.approx(.4,abs=1e-14)
    assert not screen['passed']
    assert all('unavailable' in e['bootstrap'] for e in estimates)


def test_periodic_fluctuations_have_equal_block_means_with_conditional_uncertainty():
    t=np.linspace(0,160,1601);v=2+.2*np.sin(np.pi*t)
    screen=utility.stationarityScreen(t,v,blocks=[40,80,120,160])
    assert screen['passed']
    np.testing.assert_allclose(screen['blockMeans'],2,atol=1e-14)
    estimate=analysis.estimate(t,v,40,80,utility)
    assert estimate['variance']==pytest.approx(.02,abs=1e-14)
    assert estimate['bootstrap']['meanCI95'][0]<2<estimate['bootstrap']['meanCI95'][1]
    assert 'Conditional' in estimate['bootstrap']['interpretation']


def test_companion_rejects_a_failed_physical_guard_before_analysis(tmp_path):
    import json
    watcher=analysis.load(REPO/'scripts/watch_drivenStationarity.py','stationarity_watcher')
    (tmp_path/'result.json').write_text(json.dumps(dict(status='completed',actualTime=160,
        acceptedPositiveFinite=True,sourcesStable=True,stageChecks=dict(bad=1),extraRHS=0)))
    with pytest.raises(ValueError,match='physical result'):
        watcher.verify_run(tmp_path,tmp_path,1234)


def test_missing_seeds_cannot_pass_the_ensemble_screen(tmp_path,monkeypatch):
    watcher=analysis.load(REPO/'scripts/watch_drivenStationarity.py','stationarity_watcher_plot')
    monkeypatch.setenv('MPLCONFIGDIR',str(tmp_path/'mpl'))
    directory=tmp_path/'analysis/example';directory.mkdir(parents=True)
    t=np.arange(161,dtype=float);ones=np.ones_like(t)
    np.savez(directory/'native-series.npz',t=t,kineticEnergy=ones,rmsMachMass=ones,forceWork=.01*t)
    np.savez(directory/'statistics.npz',t=t,boxModeEnergy=.8*ones,boxModeFraction=.8*ones,logDensityStdMass=.1*ones)
    report=dict(run=str(tmp_path/'example'),seed=1234,windows=[dict(native=dict(kineticEnergy=dict(mean=1.))) for _ in range(3)],scalarScreenPassed=True,fieldCorrelation=dict(tau=1.),cost=dict(loopWallSeconds=1,msPerStep=1))
    result=watcher.ensemble(tmp_path,[report])
    assert result['completedRuns']==1
    assert result['expectedRuns']==3
    assert not result['allScalarScreensPassed']
    assert result['betweenSeedKStandardDeviation'] is None
