"""Mixed physical endpoints and fixed-step probes must keep distinct completion goals."""
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('evaluation_goals',Path(__file__).resolve().parents[1]/'scripts/evaluate_pesph.py')
evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)


@pytest.mark.parametrize('requested,actual,time,quick,expected',[
    (4,4,.004,False,'complete'),(4,3,.003,True,'failed'),
    (None,4,.004,False,'failed'),(None,4,1.,False,'complete'),
    (None,4,.004,True,'failed'),
])
def test_completion_uses_each_rows_declared_goal(tmp_path,monkeypatch,requested,actual,time,quick,expected):
    monkeypatch.setattr(evaluation,'profile_context',lambda *args:None)
    row=dict(id='probe__PESPH-E',name='probe',case='probe',scheme='PESPH-E',
        configuration={'simulation':{} if requested is None else {'nSteps':requested}})
    d=tmp_path/row['id'];d.mkdir()
    raw=dict(id=row['id'],name='probe',case='probe',scheme='PESPH-E',status='complete',
        metrics={'nSteps':actual,'tFinal':time,'targetTime':1.,'wallSeconds':1.})
    (d/'result.json').write_text(json.dumps(raw))
    records=evaluation.collect(tmp_path,dict(rows=[row],quick=quick))
    assert records[0]['status']==expected
    assert records[0]['metrics']['reachedRequestedGoal']==(expected=='complete')
    assert records[0]['metrics']['reachedTarget']==(time>=1.)
    assert json.loads((d/'result.json').read_text())==raw,'Reporting must preserve the worker evidence'
